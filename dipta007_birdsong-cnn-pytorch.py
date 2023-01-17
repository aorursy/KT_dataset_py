import pandas as pd

import numpy as np

import os

import librosa



from tqdm.notebook import tqdm, trange

import subprocess



import math

import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision import transforms

from PIL import Image





import warnings

warnings.filterwarnings('ignore')



PATH = '../input/birdsong-recognition/'

IMG = '../input/birdsongspectrograms/'

os.listdir(PATH)
transformers = transforms.Compose([

    transforms.RandomCrop((128, 512), pad_if_needed=True, padding_mode="reflect"),

    transforms.ToTensor(),

    transforms.Normalize((0.5), (0.5)),

])



def load_img(path, rescale=True, normalize=True):

    img = Image.open(path)

    img = transformers(img)

    return img
df = pd.read_csv(os.path.join(PATH, 'train.csv'), skiprows=0)



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(df['ebird_code'].to_numpy())



df.head()
from csv import writer

def append_list_as_row(file_name, list_of_elem):

    with open(file_name, 'a+', newline='') as write_obj:

        csv_writer = writer(write_obj)

        csv_writer.writerow(list_of_elem)
DURATION = 10



import os

try: 

    os.remove("train_val.csv")

    print("removed successfully") 

except OSError as error: 

    print(error) 

    print("File path can not be removed") 

    

header = ['target', 'filepath']

append_list_as_row('train_val.csv', header)



for index, row in tqdm(df.iterrows()):

    bird = row['ebird_code']

    audio = row['filename'].replace('mp3', 'jpg')

    filepath = f'{audio}'

    

    target = le.transform([bird])[0]

    duration = row['duration']

    

    if os.path.isfile(f"{IMG}{filepath}"):

        tmp = []

        tmp.append(target)

        tmp.append(filepath)



        append_list_as_row('train_val.csv', tmp)

    

#     if duration > 10:

#         now = load_clip(filepath, 0, DURATION)

#         print(now, now.size())

#         break
del df

import gc

gc.collect()



df = pd.read_csv('train_val.csv', skiprows=0)
df.head()
VALIDATION_SIZE = 0.1



df = df.sample(frac=1).reset_index(drop=True)



total_len = len(df)

train_sz = int(total_len * (1-VALIDATION_SIZE))

val_sz = int(total_len - train_sz)

print(train_sz, total_len - train_sz, len(df[:train_sz]), len(df[train_sz:]))





def get_features(train):

    data = None

    if train:

        data = df[:train_sz]

    else:

        data = df[train_sz:]



    for index, row in tqdm(data.iterrows()):

        filepath = row['filepath']

        spectrogram = load_img(IMG + filepath)



        yield spectrogram, row['target']

    

df.head()
BATCH_SIZE = 128



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



print(device)
class model(nn.Module):

    def __init__(self):

        super(model, self).__init__()

        self.conv1 = nn.Conv2d(1, 2, 3)

        self.conv2 = nn.Conv2d(2, 4, 3)

        self.conv3 = nn.Conv2d(4, 8, 3)

        

        fn = 6944

        self.fc1 = nn.Linear(fn, fn*2)

        self.fc2 = nn.Linear(fn*2, fn)

        self.fc3 = nn.Linear(fn, fn//2)

        self.output = nn.Linear(fn//2, 264)

        

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
LR=0.0001



net = model().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
def get_number_of_correct_for_this_batch(y_pred, y):

    y_pred = torch.nn.Softmax(dim=1)(y_pred)

    y_pred = torch.argmax(y_pred, dim=1)

    correct_now = torch.eq(y_pred, y).sum()

    return correct_now.item()
EPOCHS = 40

BEST_MODEL_PATH = 'best_model.pth'



best_loss = 1000000

patience = 4



for epoch in range(EPOCHS):

    #### Training

    net.train()

    gen = get_features(True)

    steps = math.ceil(train_sz / BATCH_SIZE)

    total_loss = 0

    total_correct = 0

    loop = tqdm(range(steps), total=steps)

    for i, _ in enumerate(loop):

        X, Y = get_batch(gen)

        X, Y = X.to(device), Y.to(device)



        # Forward propagation

        optimizer.zero_grad()

        y_pred = net(X)

        loss = criterion(y_pred, Y.view(-1))

        total_loss += loss.item()

        

        # Backward propagation

        loss.backward()

        optimizer.step()

        

        with torch.no_grad():

            # Get stats

            correct_now = get_number_of_correct_for_this_batch(y_pred, Y)

            total_correct += correct_now



            # Update stats

            loop.update(1)

            loop.set_description('Epoch {}/{}'.format(epoch + 1, EPOCHS))

            loop.set_postfix(loss=loss.item(), acc=total_correct/((i+1) * BATCH_SIZE))

    

    

    #### Validation

    with torch.no_grad():

        net.eval()

        gen = get_features(False)

        steps = math.ceil(val_sz / BATCH_SIZE)

        total_loss = 0

        total_correct = 0

        loop = tqdm(range(steps), total=steps)

        for i, _ in enumerate(loop):

            X, Y = get_batch(gen)

            X, Y = X.to(device), Y.to(device)



            y_pred = net(X)



            loss = criterion(y_pred, Y.view(-1))

            total_loss += loss.item()



            correct_now = get_number_of_correct_for_this_batch(y_pred, Y)

            total_correct += correct_now



            loop.update(1)

            loop.set_description('Epoch {}/{}'.format(epoch + 1, EPOCHS))

            loop.set_postfix(loss=loss.item(), acc=total_correct/((i+1) * BATCH_SIZE))



        # Early Stopping

        if total_loss < best_loss:

            best_loss = total_loss

            patience = 4

            torch.save(net, BEST_MODEL_PATH)

        else:

            patience -= 1



        if patience <= 0:

            print(f"Early stopping at {epoch}")

            break
best = torch.load(BEST_MODEL_PATH)
import librosa

import cv2

#from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data

def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):

    # Stack X as [X,X,X]

#     X = np.stack([X, X, X], axis=-1)



    # Standardize

    mean = mean or X.mean()

    X = X - mean

    std = std or X.std()

    Xstd = X / (std + eps)

    _min, _max = Xstd.min(), Xstd.max()

    norm_max = norm_max or _max

    norm_min = norm_min or _min

    if (_max - _min) > eps:

        # Normalize to [0, 255]

        V = Xstd

        V[V < norm_min] = norm_min

        V[V > norm_max] = norm_max

        V = 255 * (V - norm_min) / (norm_max - norm_min)

        V = V.astype(np.uint8)

    else:

        # Just zero

        V = np.zeros_like(Xstd, dtype=np.uint8)

    return V



def build_spectrogram(path, offset, duration):

    y, sr = librosa.load(path, offset=offset, duration=duration)

    total_secs = y.shape[0] / sr

    M = librosa.feature.melspectrogram(y=y, sr=sr)

    M = librosa.power_to_db(M)

    M = mono_to_color(M)

    

    filename = path.split("/")[-1][:-4]

    path = 'test.jpg'

    cv2.imwrite(path, M, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    return path
def make_prediction(x):

    best.eval()

    y_pred = best(x)

    y_pred = nn.Softmax(dim=1)(y_pred)

    y_pred = torch.argmax(y_pred, dim=1)

    return le.inverse_transform(y_pred)[0]
TEST_FOLDER='../input/birdsong-recognition/test_audio/'



try:

    preds = []

    test = pd.read_csv(os.path.join(PATH, 'test.csv'))

    for index, row in test.iterrows():

        # Get test row information

        site = row['site']

        start_time = row['seconds'] - 5

        row_id = row['row_id']

        audio_id = row['audio_id']



        # Get the test sound clip

        if site == 'site_1' or site == 'site_2':

            path = build_spectrogram(TEST_FOLDER + audio_id + '.mp3', start_time, 5)

            y = load_img(path)

        else:

            path = build_spectrogram(TEST_FOLDER + audio_id + '.mp3', 0, duration=None)

            y = load_img(path)



        # Make the prediction

        pred = make_prediction(y, le, model)



        # Store prediction

        preds.append([row_id, pred])

except Exception as e:

    preds = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')

    print('why', e)

        

# print(preds)

preds = pd.DataFrame(preds, columns=['row_id', 'birds'])
preds.head()
preds.fillna('nocall', inplace=True)

preds.to_csv('submission.csv', index=False)