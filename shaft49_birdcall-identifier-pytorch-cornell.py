import pandas as pd
import numpy as np
import cv2
import os
import librosa # package for music and audio analysis

from tqdm.notebook import tqdm, trange

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from csv import writer

import warnings
warnings.filterwarnings('ignore')


PATH = '../input/birdsong-recognition'
IMG = '../input/birdsongspectrograms'
transformers = transforms.Compose([
    transforms.RandomCrop((128, 512), pad_if_needed = True, padding_mode = "reflect"),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
def load_image(path, rescale = True, normalize = True):
    image = Image.open(path)
    image = transformers(image)
    return image
df = pd.read_csv(os.path.sep.join([PATH,'train.csv']), skiprows = 0)
le = LabelEncoder() # encode the
le.fit(df['ebird_code'].to_numpy())
len(le.classes_) # shows the number of classes present in the csv file
def append_list_as_rows(file_name, list_of_elem):
    with open(file_name, 'a+', newline = '') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)
csv_file_name = 'train_test_data.csv'
def remove_previous_csv_file():
    try:
        os.remove(csv_file_name)
        print('[INFO] CSV file removed successfully')
    except OSError as error:
        print(f'[ERROR] {error}')
        print(f'[INFO] {csv_file_name} cannot be removed')
remove_previous_csv_file()
header = ['target', 'filepath']
append_list_as_rows(csv_file_name, header)
for index, row in tqdm(df.iterrows()):
    bird = row['ebird_code']
    audio = row['filename'].replace('.mp3', '.jpg')
    filepath = f'{audio}'
    
    target = le.transform([bird])[0] # get the encoded class name
    
    if os.path.isfile(os.path.sep.join([IMG, filepath])):
        append_list_as_rows(csv_file_name, [target, filepath])

print('[INFO] Complete writing to the csv file')
df2 = pd.read_csv(csv_file_name, skiprows = 0)

df2.head() # prints first five rows

VALIDATION_SIZE = 0.1
df2 = df2.sample(frac = 1).reset_index(drop = True)

total_len = len(df2)
train_size = int(total_len * (1.0 - VALIDATION_SIZE))
val_size = int(total_len - train_size)

print(f'[INFO] Total Data: {total_len}, Train Data: {train_size}, Val Data: {val_size}')

def get_features(option):
    data = None
    if option == 'train':
        data = df2[:train_size]
    elif option == 'test':
        data = df2[train_size:]
    
    for index, row in tqdm(data.iterrows()):
        filepath = row['filepath']
        spectrogram = load_image(os.path.sep.join([IMG, filepath]))
        
        yield spectrogram, row['target']

print(df2.head())
BATCH_SIZE = 32
def get_batch(data_generator):
    X, Y = [], []
    cnt = 0
    for x, y in data_generator:
        X.append(x)
        Y.append(y)
        cnt += 1
        if cnt >= BATCH_SIZE:
            break
    return torch.stack(X), torch.tensor(Y)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'[INFO] Device: {device}')
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3) # in_channels, out_channels, kernerl_size
        self.conv2 = nn.Conv2d(2, 4, 3)
        self.conv3 = nn.Conv2d(4, 8, 3)
        
        fn = 6944
        self.fc1 = nn.Linear(fn, fn * 2) # in_features, out_features
        self.fc2 = nn.Linear(fn * 2, fn)
        self.fc3 = nn.Linear(fn, fn // 2)
        self.output = nn.Linear(fn // 2, len(le.classes_))
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        x = self.output(x)
        
        return x
    
    def flatten(self, x):
        res = 1
        for sz in x.size()[1:]:
            res *= sz
        return x.view(-1, res)
LR = 0.0001

net = model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = LR)
def get_number_of_correct_for_this_batch(y_pred, y):
    y_pred = torch.nn.Softmax(dim = 1)(y_pred)
    y_pred = torch.argmax(y_pred, dim = 1)
    correct = torch.eq(y_pred, y).sum()
    return correct.item()
BEST_MODEL_PATH = 'best_model.pth'

EPOCHS = 40
best_loss = 1000000
patience = 4

for epoch in range(EPOCHS):
    # Training
    net.train()
    gen = get_features('train')
    steps = math.ceil(train_size / BATCH_SIZE)
    total_loss = 0
    total_correct = 0
    loop = tqdm(range(steps), total = steps)
    
    for i, _ in enumerate(loop):
        X, Y = get_batch(gen)
        X, Y = X.to(device), Y.to(device)
        
        # Forward Propagation
        optimizer.zero_grad()
        y_pred = net(X)
        loss = criterion(y_pred, Y.view(-1))
        total_loss += loss.item()
        
        # Backward Propagation
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            # Get Stats
            correct = get_number_of_correct_for_this_batch(y_pred, Y)
            total_correct += correct
            
            # Update Stats
            loop.update(1)
            loop.set_description(f'Epoch {epoch + 1}/{EPOCHS}')
            loop.set_postfix(loss = loss.item(), acc = total_correct/((i + 1) * BATCH_SIZE))
    
    # Validation
    with torch.no_grad():
        net.eval()
        gen = get_features('test')
        steps = math.ceil(val_size / BATCH_SIZE)
        total_loss = 0
        total_correct = 0
        loop = tqdm(range(steps), total = steps)
        
        for i, _ in enumerate(loop):
            X, Y = get_batch(gen)
            X, Y = X.to(device), Y.to(device)
            
            y_pred = net(X)
            
            loss = criterion(y_pred, Y.view(-1))
            total_loss += loss.item()
            
            correct = get_number_of_correct_for_this_batch(y_pred, Y)
            total_correct += correct
            
            loop.update(1)
            loop.set_description(f'Epoch {epoch + 1}/{EPOCHS}')
            loop.set_postfix(loss = loss.item(), acc = total_correct/((i + 1) * BATCH_SIZE))
        
        # Early Stopping
        
        if total_loss < best_loss:
            best_loss = total_loss
            patience = 4
            torch.save(net, BEST_MODEL_PATH)
        else:
            patience -= 1
        
        if patience <= 0:
            print(f'[INFO] Early Stopping at {epoch}')
            break
net = torch.load(BEST_MODEL_PATH)
#from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
def mono_to_color(X, mean = None, std = None, norm_max = None, norm_min = None, eps = 1e-6):
    # Standardize
    mean = mean or X.mean()
    X = X - mean
    
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    
    if (_max - _min) > eps:
        # Normlize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.unint8)
    else:
        # Just Zero
        V = np.zeros_like(Xstd, dtype = np.unit8)
    return V

def build_spectrogram(path, offset, duration):
    y, sr = librosa.load(path, offset = offset, duration = duration)
    total_secs = y.shape[0] / sr
    M = librosa.feature.melspectrogram(y = y, sr = sr)
    M = librosa.power_to_db(M)
    M = mono_to_color(M)
    
    filename = path.split('/')[-1][:-4]
    path = 'test.jpg'
    cv2.imwrite(path, M, [int(cv2.IMWRITE_JPEG_QUALITY, 85)])
    return path
def make_prediction(x):
    net.eval()
    y_pred = net(x)
    y_pred. nn.Softmax(dim = 1)(y_pred)
    y_pred = torch.argmax(y_pred, dim = 1)
    return le.inverse_transform(y_pred)[0]
TEST_FOLDER = '../input/birdsong-recognition/test_audio' # hidden folder

try:
    preds = []
    test = pd.read_csv(os.path.sep.join([PATH, 'test.csv']))
    
    for index, row in tqdm(test.iterrows()):
        # Get test row information
        site = row['site']
        start_time = row['seconds']
        row_id = row['row_id']
        audio_id = row['audio_id']
        
        # Get the test sound clip
        audio_file = os.path.sep.join([TEST_FOLDER, audio_id + '.mp3'])
        if os.path.isfile(audio_file):
            if site == 'site_1' or site == 'site_2':
                path = build_spectrogram(audio_file, start_time, 5)
                y = load_image(path)
            else:
                path = build_spectrogram(audio_file, 0, duration = None)
                image = load_image(path)

            # Make the predictions
            pred = make_prediction(image)

            # Store predictions
            preds.append([row_id, pred])
        else:
            preds = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')
            break
except Exception as e:
    preds = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')
    print(f'[Reason] {e}')
    
# Convert to dataframe
pred_df = pd.DataFrame(preds, columns = ['row_id', 'birds'])
pred_df.head()
pred_df.fillna('nocall', inplace = True) # fill the columns with nocall that are empty
pred_df.to_csv('submission.csv', index = False)
