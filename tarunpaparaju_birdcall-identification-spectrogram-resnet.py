!pip install -q pydub

!pip install -q colored
import os

import gc

import cv2

import time

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')



import IPython

import IPython.display as ipd

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

from colored import fg, bg, attr

from sklearn.utils import shuffle



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import pydub

import librosa

import librosa.display

from pydub import AudioSegment as AS

from librosa.feature import melspectrogram

from librosa.core import power_to_db as ptdb



import torch

import torch.nn as nn

from torch.optim import Adam

from albumentations import Normalize

from torchvision.models import resnet34

from torch.utils.data import Dataset, DataLoader

from torch import FloatTensor, LongTensor, DoubleTensor



from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences as pad
torch.backends.cudnn.benchmark = True
F = 512

DROP = 0.2

EPOCHS = 100

LR = 1e-3, 1e-2

VAL_BATCH_SIZE = 64

TRAIN_BATCH_SIZE = 64



MAX_OUTPUTS = 3

THRESHOLD = 6e-3

MIN_THRESHOLD = 5e-3



SR = 44100

CHUNKS = 1

TSR = 32000

SPLIT = 0.8

N_MELS = 256

MEL_LEN = 313

POP_FRAC = 0.25

MAXLEN = 2000000

AMPLITUDE = 1000

CHUNK_SIZE = 160000
os.listdir('../input')
!unzip ../input/birdcall-identification-spectrogram-loader/train_1.zip

!unzip ../input/birdcall-identification-spectrogram-loader/train_2.zip

!unzip ../input/birdcall-identification-spectrogram-loader/train_3.zip

!unzip ../input/birdcall-identification-spectrogram-loader/train_4.zip

!unzip ../input/birdcall-identification-spectrogram-loader/train_5.zip
TEST_DATA_PATH = '../input/birdsong-recognition/test.csv'

TRAIN_DATA_PATH = '../input/birdsong-recognition/train.csv'

TEST_AUDIO_PATH = '../input/birdsong-recognition/test_audio/'

TRAIN_AUDIO_PATH = '../input/birdsong-recognition/train_audio/'

CHECKING_PATH = '../input/prepare-check-dataset/birdcall-check/'

IMG_PATHS = ['train_1', 'train_2', 'train_3', 'train_4', 'train_5']
PATH_DICT = {}

for folder_path in tqdm(IMG_PATHS):

    for img_path in os.listdir(folder_path):

        PATH_DICT[img_path] = folder_path + '/'
sub = os.path.exists(TEST_AUDIO_PATH)

TEST_DATA_PATH = TEST_DATA_PATH if sub else CHECKING_PATH + 'test.csv'

TEST_AUDIO_PATH = TEST_AUDIO_PATH if sub else CHECKING_PATH + 'test_audio/'
test_df = pd.read_csv(TEST_DATA_PATH)

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df.head()
train_df.head()
keys = set(train_df.ebird_code)

values = np.arange(0, len(keys))

code_dict = dict(zip(sorted(keys), values))
counts = [list(train_df.ebird_code).count(code) for code in set(train_df.ebird_code)]

idx = np.argsort(counts)

df = pd.DataFrame(np.transpose([np.array(list(set(train_df.ebird_code)))[idx], np.array(counts)[idx]]))

df.columns = ["Bird species", "Count"]

fig = px.bar(df, x="Count", y="Bird species", title="Bird species vs. Count", template="simple_white")

fig.data[0].orientation = 'h'

fig.update_layout(height=1800, paper_bgcolor="#edebeb")

fig.update_traces(textfont=dict(

        color="white"

    ))

fig.show()
nums_1 = train_df.duration

nums_1 = nums_1.fillna(nums_1.mean())



fig = ff.create_distplot(hist_data=[nums_1],

                         group_labels=["0"],

                         colors=["seagreen"], show_hist=False)



fig.update_layout(title_text="Duration distribution", xaxis_title="Duration",

                  template="plotly_white", paper_bgcolor="#edebeb")

fig.show()
counts = [list(train_df.pitch).count(code) for code in set(train_df.pitch)]

df = pd.DataFrame(np.transpose([list(set(train_df.pitch)), counts]))

df.columns = ["Pitch", "Count"]

fig = px.pie(df, names="Pitch", values="Count", title="Pitch", color_discrete_sequence=list(reversed(px.colors.cyclical.Edge)))

fig.update_layout(paper_bgcolor="#edebeb")

fig.update_traces(textfont=dict(

        color="white"

    ))

fig.show()
counts = [list(train_df.channels).count(code) for code in set(train_df.channels)]

df = pd.DataFrame(np.transpose([list(set(train_df.channels)), counts]))

df.columns = ["Channels", "Count"]

colors = px.colors.qualitative.Plotly

fig = px.pie(df, names="Channels", values="Count", title="Channels", color_discrete_sequence=["darkblue", "SteelBlue"])

fig.update_layout(paper_bgcolor="#edebeb")

fig.update_traces(textfont=dict(

        color="white"

    ))

fig.show()
def normalize(x):

    return np.float32(x)/2**15



def read(file, norm=False):

    try:

        a = AS.from_mp3(file)

        a = a.set_frame_rate(TSR)

    except:

        return TSR, np.zeros(MAXLEN)



    y = np.array(a.get_array_of_samples())

    if a.channels == 2: y = y.reshape((-1, 2))

    if norm: return a.frame_rate, normalize(y)

    if not norm: return a.frame_rate, np.float32(y)



def write(file, sr, x, normalized=False):

    birds_audio_bitrate, file_format = '320k', 'mp3'

    ch = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1

    y = np.int16(x * 2 ** 15) if normalized else np.int16(x)

    song = AS(y.tobytes(), frame_rate=sr, sample_width=2, channels=ch)

    song.export(file, format=file_format, bitrate=birds_audio_bitrate)
def to_tensor(data):

    return [FloatTensor(point) for point in data]
def get_idx(length):

    length = get_len(length)

    max_idx = MAXLEN - CHUNK_SIZE

    idx = np.random.randint(length + 1)

    chunk_range = idx, idx + CHUNK_SIZE

    chunk_idx = max([0, chunk_range[0]])

    chunk_idx = min([chunk_range[1], max_idx])

    return (chunk_idx, chunk_idx + CHUNK_SIZE)



def get_len(length):

    if length > MAXLEN: return MAXLEN

    if length <= MAXLEN: return int(length*POP_FRAC)
def get_chunk(data, length):

    index = get_idx(length)

    return data[index[0]:index[1]]



def get_signal(data):

    length = max(data.shape)

    data = data.T.flatten().reshape(1, -1)

    data = np.float32(pad(data, maxlen=MAXLEN).reshape(-1))

    return [get_chunk(data, length) for _ in range(CHUNKS)]
def to_imagenet(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):

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

        V = (V - norm_min) / (norm_max - norm_min)

    else:

        # Just zero

        V = np.zeros_like(Xstd, dtype=np.uint8)

    return np.stack([V]*3, axis=-1)
def get_melsp(data):

    melsp = melspectrogram(data, n_mels=N_MELS)

    return to_imagenet(librosa.power_to_db(melsp))



def get_melsp_img(data):

    data = get_signal(data)

    return np.stack([get_melsp(point) for point in data])
def viz(species):

    data = train_df.query('ebird_code == "{}"'.format(species))

    

    data = data.reset_index(drop=True)

    ebird_code, filename = data.ebird_code.loc[0], data.filename.loc[0]

    

    ipd.Audio(TRAIN_AUDIO_PATH + species + '/' + filename)



    f, pth = (30, 5), TRAIN_AUDIO_PATH

    fig, ax = plt.subplots(nrows=CHUNKS, ncols=2, figsize=f)



    ax[0].set_title(species + " signal", fontsize=16)

    ax[1].set_title(species + " melspectrogram", fontsize=16)



    sr, data = read(pth + ebird_code + '/' + filename)

    signals, melsp_features = get_signal(data), get_melsp_img(data)



    values = zip(signals, melsp_features)

    for i, (signal, melsp_feature) in enumerate(values):

        ax[0].plot(signal, 'crimson'); ax[1].imshow(cv2.resize(melsp_feature, (4096, 1024)))

    

    display(ipd.Audio(pth + ebird_code + '/' + filename))

    plt.show()
viz('aldfly')
viz('canwre')
viz('lesgol')
class BirdDataset(Dataset):

    def __init__(self, df, path):

        self.code_dict = code_dict

        self.classes = len(code_dict)

        self.df, self.path = df, path

        self.dataset_length = len(df)

        

    def __len__(self):

        return self.dataset_length

    

    def __getitem__(self, i):

        file_name = self.df.filename[i]

        ebird_code = self.df.ebird_code[i]

        num_code = self.code_dict[ebird_code]

        default_signal = np.random.random(MAXLEN)*AMPLITUDE

        default_values = SR, np.int32(np.round(default_signal))



        values = read(self.path + ebird_code + '/' + file_name)

        _, data = values if len(values) == 2 else default_values

        code = to_categorical([num_code], num_classes=self.classes)

        return to_tensor([np.nan_to_num(get_melsp_img(data)), np.repeat(code, CHUNKS, 0)])
class MelDataset(Dataset):

    def __init__(self, df):

        self.aug = Normalize(p=1)

        self.code_dict = code_dict

        self.classes = len(code_dict)

        self.df, self.dataset_length = df, len(df)

        

    def __len__(self):

        return self.dataset_length

    

    def __getitem__(self, i):

        file_name = self.df.filename[i]

        image_name = file_name + '.jpg'

        ebird_code = self.df.ebird_code[i]

        num_code = self.code_dict[ebird_code]

        image = cv2.imread(PATH_DICT[image_name] + image_name)

        code = to_categorical([num_code], num_classes=self.classes)

        return to_tensor([self.aug(image=image)['image'], np.repeat(code, CHUNKS, 0)])
train_df = shuffle(train_df)



split = int(SPLIT*len(train_df))

train_df = train_df.reset_index(drop=True)

valid_df = train_df[split:].reset_index(drop=True)

train_df = train_df[:split].reset_index(drop=True)



train_set = MelDataset(train_df)

valid_set = MelDataset(valid_df)

valid_loader = tqdm(DataLoader(valid_set, batch_size=VAL_BATCH_SIZE))

train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
class BirdNet(nn.Module):

    def __init__(self, f, o):

        super(BirdNet, self).__init__()

        self.dropout = nn.Dropout(p=DROP)

        self.dense_output = nn.Linear(f, o)

        self.resnet = resnet34(pretrained=True)

        self.resnet_head = list(self.resnet.children())

        self.resnet_head = nn.Sequential(*self.resnet_head[:-1])



    def forward(self, x):

        x = self.resnet_head(x)

        return self.dense_output(self.dropout(x.view(-1, F)))
O = len(code_dict)

network = BirdNet(f=F, o=O)

optimizer = Adam([{'params': network.resnet.parameters(), 'lr': LR[0]},

                  {'params': network.dense_output.parameters(), 'lr': LR[1]}])
def cel(y_true, y_pred):

    y_true = torch.argmax(y_true, axis=-1)

    return nn.CrossEntropyLoss()(y_pred, y_true.squeeze())



def accuracy(y_true, y_pred):

    y_true = torch.argmax(y_true, axis=-1).squeeze()

    y_pred = torch.argmax(y_pred, axis=-1).squeeze()

    return (y_true == y_pred).float().sum()/len(y_true)
def print_metric(data, batch,

                 epoch, start,

                 end, metric, typ):



    t = typ, metric, "%s", data, "%s"

    if typ == "Val": pre = "\nEPOCH %s" + str(epoch+1) + "%s  "

    if typ == "Train": pre = "BATCH %s" + str(batch-1) + "%s  "

    time = np.round(end - start, 1); time = "Time: %s{}%s s".format(time)

    fonts = [(fg(211), attr('reset')), (fg(212), attr('reset')), (fg(213), attr('reset'))]

    print(pre % fonts[0] + "{} {}: {}{}{}".format(*t) % fonts[1] + "  " + time % fonts[2])
def get_shuffle_idx(tensor):

    return shuffle(np.arange(len(tensor)))
D = (3, N_MELS, MEL_LEN)

network = network.cuda()

device = torch.device('cuda')



start = time.time()

print("STARTING TRAINING ...\n")



for epoch in range(EPOCHS):

    fonts = (fg(48), attr('reset'))

    print(("EPOCH %s" + str(epoch+1) + "%s") % fonts)

    

    batch = 1

    network.train()

    for minibatch in train_loader:

        train_X, train_y = minibatch

        train_y = train_y.view(-1, O)

        train_X = train_X.view(-1, *D)

        idx = get_shuffle_idx(train_X)

        train_X = train_X[idx].to(device)

        train_y = train_y[idx].to(device)

        train_preds = network.forward(train_X)

        train_loss = cel(train_y, train_preds)

        train_accuracy = accuracy(train_y, train_preds)

        

        optimizer.zero_grad()

        train_loss.backward()



        optimizer.step()

        end = time.time()

        batch = batch + 1

        is_print = batch % 100 == 1

        acc = np.round(train_accuracy.item(), 3)

        if is_print: print_metric(acc, batch, 0, start, end, "Acc", "Train")

    

    valid_loss = 0

    valid_points = 0

    valid_accuracy = 0

    

    network.eval()

    with torch.no_grad():

        for minibatch in valid_loader:

            valid_X, valid_y = minibatch

            valid_y = valid_y.view(-1, O)

            valid_X = valid_X.view(-1, *D)

            idx = get_shuffle_idx(valid_X)

            valid_X = valid_X[idx].to(device)

            valid_y = valid_y[idx].to(device)

            valid_preds = network.forward(valid_X)

            valid_points = valid_points + len(valid_y)

            valid_loss += cel(valid_y, valid_preds).item()*len(valid_y)

            valid_accuracy += accuracy(valid_y, valid_preds).item()*len(valid_y)

    

    end = time.time()

    valid_loss /= valid_points

    valid_accuracy /= valid_points

    acc = np.round(valid_accuracy, 3)

    print_metric(acc, 0, epoch, start, end, "Acc", "Val"); print("")

    

print("ENDING TRAINING ...")
def get_time(site, start_time):

    if site == 'site_3': return 0, None

    if site == 'site_1' or site == 'site_2':

        return int((start_time - 5)*TSR), int(start_time*TSR)



class BirdTestDataset(Dataset):

    def __init__(self, df, path):

        self.df, self.path = df, path

        self.dataset_length = len(df)

        self.normalize = Normalize(p=1)

        

    def __len__(self):

        return self.dataset_length

    

    def __getitem__(self, i):

        site = self.df.site[i]

        start_time = self.df.seconds[i]

        s, e = get_time(site, start_time)

        default_values = TSR, np.zeros(MAXLEN)

        values = read(self.path + self.df.audio_id[i])

        out = values if len(values) == 2 else default_values

        return FloatTensor(self.normalize(image=get_melsp_img(out[1][s:e]))['image'])
def softmax(x):

    return np.exp(x)/np.sum(np.exp(x), axis=1)[:, None]



def decision_fn(probs):

    probs[np.argsort(probs)[:-MAX_OUTPUTS]] = 0

    if np.max(probs) < MIN_THRESHOLD: return [-1]

    condition = probs == np.max(probs), probs >= THRESHOLD

    return np.where(np.logical_or(*condition))[0].tolist()
network.eval()

test_preds = []

test_set = BirdTestDataset(test_df, TEST_AUDIO_PATH)

test_loader = DataLoader(test_set, batch_size=VAL_BATCH_SIZE)



if os.path.exists(TEST_AUDIO_PATH):

    for test_X in tqdm(test_loader):

        test_pred = network.forward(test_X.view(-1, *D).to(device))

        test_preds.extend(softmax(test_pred.detach().cpu().numpy()).flatten())
code_dict[-1] = 'nocall'

keys = code_dict.keys()

values = code_dict.values()

code_dict = dict(zip(values, keys))



if os.path.exists(TEST_AUDIO_PATH):

    test_preds = np.array(test_preds)

    test_preds = test_preds.reshape(-1, CHUNKS, O).mean(axis=1)

    test_preds = [decision_fn(test_pred) for test_pred in test_preds]

    test_preds = [[code_dict[pred] for pred in test_pred] for test_pred in test_preds]
submission = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')
if os.path.exists(TEST_AUDIO_PATH):

    submission.birds = [' '.join(pred) for pred in test_preds][:len(submission)]
submission.head(10)
submission.to_csv('submission.csv', index=False)

torch.save(network.cpu().state_dict(), 'resnet.pt')