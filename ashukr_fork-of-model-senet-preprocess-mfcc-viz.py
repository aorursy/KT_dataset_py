import torch.nn as nn
!ls ../input/train/
import os

from os.path import isdir, join

import shutil

from tqdm import tqdm

# TODO: create train and test dataset



f = open('../input/train/validation_list.txt', 'r')

validation_list = f.readlines()

f.close()



audio_path = "../input/train/audio"



validation_path_list = []

for n in validation_list:

    validation_path_list.append(n[:-1])



validation_path = "../validation/"

os.makedirs(validation_path)

i=0

for file in validation_path_list:

    #print((audio_path+"/"+ file))

    shutil.copy((audio_path+"/"+ file), "../validation/"+str(i)+".wav")

    # if isdir((audio_path +"/" + file)):

    i+=1







dirs = [f for f in os.listdir(audio_path) if isdir(join(audio_path, f))]

dirs.sort()

recordings = []

for direct in dirs:

    waves = [f for f in os.listdir(join(audio_path, direct)) if f.endswith('.wav')]

    for wave in waves:

        recordings.append(str(direct)+"/"+wave)



print("total data available is {}".format(len(recordings)))



train_path = "../train/"

for cat in dirs:

    path = train_path + str(cat)

    os.makedirs(path)





print("creating validation and train dataset...")



for direct in tqdm(dirs):

    for f in os.listdir(join(audio_path, direct)):

        if f.endswith('.wav'):

            # print(str(direct)+"/"+f)

            if (str(direct)+"/"+f) not in validation_path_list:

                

            

                shutil.copy(str(audio_path)+"/"+str(direct)+"/"+f, '../train/'+str(direct))



print("number of validation points {}".format(i))

print("counting the train dataset...")

train_recordings = []

for direct in tqdm(dirs):

    waves = [f for f in os.listdir(join(train_path, direct)) if f.endswith('.wav')]

    for wave in waves:

        train_recordings.append(str(direct)+"/"+wave)

print("total number of files in train is {}".format(len(train_recordings)))
!ls ../
!ls ../train
import torch

import torch.nn.functional as F

from torch.nn import MaxPool2d
"""

test dataset for initializing weights

"""



import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader

import librosa

from glob import glob

import random

from skimage.transform import resize

from random import choice, sample

import pandas as pd



SR = 16000





class PreDataset(Dataset):

    """

    1. add background noise

    2. generate silent data

    3. cache some parts to speed up iterating

    """



    def __init__(self, label_words_dict, add_noise, preprocess_fun, preprocess_param = {}, sr=SR, resize_shape=128, is_1d=False):

        self.add_noise = add_noise

        self.sr = sr

        self.label_words_dict = label_words_dict

        self.preprocess_fun = preprocess_fun

        self.preprocess_param = preprocess_param



        # read all background noise here

        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob("../train/_background_noise_/*.wav")]



        self.resize_shape = resize_shape

        self.is_1d = is_1d

        pre_list = pd.read_csv("sub/base_average.csv")

        self.semi_dict = dict(zip(pre_list['fname'], pre_list['label']))

        self.wav_list = ['../test/' + x for x in self.semi_dict]

        self.wav_list = sample(self.wav_list, len(self.wav_list))



    def get_one_noise(self):

        """generates one single noise clip"""

        selected_noise = self.background_noises[random.randint(0, len(self.background_noises) - 1)]

        # only takes out 16000

        start_idx = random.randint(0, len(selected_noise)-1-self.sr)

        return selected_noise[start_idx:(start_idx + self.sr)]



    def get_mix_noises(self, num_noise=1, max_ratio=0.1):

        result = np.zeros(self.sr)

        for _ in range(num_noise):

            result += random.random() * max_ratio * self.get_one_noise()

        return result / num_noise if num_noise > 0 else result



    def get_one_word_wav(self, idx, speed_rate=None):

        wav = librosa.load(self.wav_list[idx], sr=self.sr)[0]

        if speed_rate:

            wav = librosa.effects.time_stretch(wav, speed_rate)

        if len(wav) < self.sr:

            wav = np.pad(wav, (0, self.sr - len(wav)), 'constant')

        return wav[:self.sr]



    def get_silent_wav(self, num_noise=1, max_ratio = 0.5):

        return self.get_mix_noises(num_noise=num_noise, max_ratio=max_ratio)



    def timeshift(self, wav, ms = 100):

        shift = (self.sr * ms) // 1000

        shift = random.randint(-shift, shift)

        a = -min(0, shift)

        b = max(0, shift)

        data = np.pad(wav, (a, b), "constant")

        return data[:len(data) - a] if a else data[b:]



    def get_noisy_wav(self, idx):

        scale = random.uniform(0.75, 1.25)

        num_noise = random.choice([1,2])

        max_ratio = random.choice([0.1, 0.5, 1, 2])

        mix_noise_proba = 0.25

        shift_range = random.randint(80, 120)

        if random.random() < mix_noise_proba:

            return scale * (self.timeshift(self.get_one_word_wav(idx), shift_range) + self.get_mix_noises(

                num_noise, max_ratio))

        else:

            return scale * self.timeshift(self.get_one_word_wav(idx), shift_range)



    def __len__(self):

        return len(self.wav_list)



    def __getitem__(self, idx):

        """reads one sample"""

        wav_numpy = self.preprocess_fun(self.get_noisy_wav(idx), **self.preprocess_param)

        if self.resize_shape:

            wav_numpy = resize(wav_numpy, (self.resize_shape, self.resize_shape), preserve_range=True)

        wav_tensor = torch.from_numpy(wav_numpy).float()

        if not self.is_1d:

            wav_tensor = wav_tensor.unsqueeze(0)



        label_word = self.semi_dict[self.wav_list[idx].split('/')[-1]]

        if label_word == "unknown":

            label = 10

        elif label_word == 'silence':

            label = 11

        else:

            label = self.label_words_dict[label_word]



        return {'spec': wav_tensor, 'id': self.wav_list[idx], 'label': label}
import torch

import numpy as np

from torch.utils.data import Dataset

import librosa

from glob import glob

import random

from skimage.transform import resize

import pandas as pd

from random import sample



SR=16000



class SpeechDataset(Dataset):

    def __init__(self, mode, label_words_dict, wav_list, add_noise, preprocess_fun, preprocess_param = {}, sr=SR, resize_shape=None, is_1d=False):

        """Args:

                mode: train or evaluate or test

                label_words_dict: a dict of words for labels

                wav_list: a list of wav file paths

                add_noise: boolean. if background noise should be added

                preprocess_fun: function to load/process wav file

                preprocess_param: params for preprocess_fun

                sr: default 16000

                resize_shape: None. only for 2d cnn.

                is_1d: boolean. if it is going to be 1d cnn or 2d cnn

        """

        self.mode = mode

        self.label_words_dict = label_words_dict

        self.wav_list = wav_list

        self.add_noise = add_noise

        self.sr = sr

        self.n_silence = int(len(wav_list) * 0.09)

        self.preprocess_fun = preprocess_fun

        self.preprocess_param = preprocess_param



        # read all background noise here

        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob("../train/_background_noise_/*.wav")]

        self.resize_shape = resize_shape

        self.is_1d = is_1d



    def get_one_noise(self):

        """generates one single noise clip"""

        selected_noise = self.background_noises[random.randint(0, len(self.background_noises) - 1)]

        # only takes out 16000

        start_idx = random.randint(0, len(selected_noise) - 1 - self.sr)

        return selected_noise[start_idx:(start_idx + self.sr)]



    def get_mix_noises(self, num_noise=1, max_ratio=0.1):

        result = np.zeros(self.sr)

        for _ in range(num_noise):

            result += random.random() * max_ratio * self.get_one_noise()

        return result / num_noise if num_noise > 0 else result



    def get_one_word_wav(self, idx):

        wav = librosa.load(self.wav_list[idx], sr=self.sr)[0]

        if len(wav) < self.sr:

            wav = np.pad(wav, (0, self.sr - len(wav)), 'constant')

        return wav[:self.sr]



    def get_silent_wav(self, num_noise=1, max_ratio=0.5):

        return self.get_mix_noises(num_noise=num_noise, max_ratio=max_ratio)



    def timeshift(self, wav, ms=100):

        shift = (self.sr * ms) // 1000

        shift = random.randint(-shift, shift)

        a = -min(0, shift)

        b = max(0, shift)

        data = np.pad(wav, (a, b), "constant")

        return data[:len(data) - a] if a else data[b:]



    def get_noisy_wav(self, idx):

        scale = random.uniform(0.75, 1.25)

        num_noise = random.choice([1, 2])

        max_ratio = random.choice([0.1, 0.5, 1, 1.5])

        mix_noise_proba = random.choice([0.1, 0.3])

        shift_range = random.randint(80, 120)

        one_word_wav = self.get_one_word_wav(idx)

        if random.random() < mix_noise_proba:

            return scale * (self.timeshift(one_word_wav, shift_range) + self.get_mix_noises(

                num_noise, max_ratio))

        else:

            return one_word_wav



    def __len__(self):

        if self.mode == 'validation':

            return len(self.wav_list)

        else:

            return len(self.wav_list) + self.n_silence



    def __getitem__(self, idx):

        """reads one sample"""

        if idx < len(self.wav_list):

            wav_numpy = self.preprocess_fun(

                self.get_one_word_wav(idx) if self.mode != 'train' else self.get_noisy_wav(idx),

                **self.preprocess_param)

            if self.resize_shape:

                wav_numpy = resize(wav_numpy, (self.resize_shape, self.resize_shape), preserve_range=True)

            wav_tensor = torch.from_numpy(wav_numpy).float()

            if not self.is_1d:

                wav_tensor = wav_tensor.unsqueeze(0)

                

            label = self.label_words_dict[self.wav_list[idx].split("/")[-2]] if self.wav_list[idx].split(

                "/")[-2] in self.label_words_dict else len(self.label_words_dict)

            if self.mode == 'validation':

                return {'spec': wav_tensor, 'id': self.wav_list[idx],'label': label}



            label = self.label_words_dict[self.wav_list[idx].split("/")[-2]] if self.wav_list[idx].split(

                "/")[-2] in self.label_words_dict else len(self.label_words_dict)



            return {'spec': wav_tensor, 'id': self.wav_list[idx], 'label': label}



        else:

            """generates silence here"""

            wav_numpy = self.preprocess_fun(self.get_silent_wav(

                num_noise=random.choice([0, 1, 2, 3]),

                max_ratio=random.choice([x / 10. for x in range(20)])), **self.preprocess_param)

            if self.resize_shape:

                wav_numpy = resize(wav_numpy, (self.resize_shape, self.resize_shape), preserve_range=True)

            wav_tensor = torch.from_numpy(wav_numpy).float()

            if not self.is_1d:

                wav_tensor = wav_tensor.unsqueeze(0)

            return {'spec': wav_tensor, 'id': 'silence', 'label': len(self.label_words_dict) + 1}





def get_label_dict():

    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

    label_to_int = dict(zip(words, range(len(words))))

    int_to_label = dict(zip(range(len(words)), words))

    int_to_label.update({len(words): 'unknown', len(words) + 1: 'silence'})

    return label_to_int, int_to_label





def get_wav_list(words, unknown_ratio=0.2):

    full_train_list = glob("../train/*/*.wav")

    full_validation_list = glob("../validation/*.wav")



    # sample full train list

    sampled_train_list = []

    for w in full_train_list:

        l = w.split("/")[-2]

        if l not in words:

            if random.random() < unknown_ratio:

                sampled_train_list.append(w)

        else:

            sampled_train_list.append(w)



    return sampled_train_list, full_validation_list





def get_sub_list(num, sub_path):

    lst = []

    df = pd.read_csv(sub_path)

    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

    each_num = int(num * 0.085)

    for w in words:

        tmp = df['fname'][df['label'] == w].sample(each_num).tolist()

        lst += ["../validation/" + x for x in tmp]

    return lst





def get_semi_list(words, sub_path, unknown_ratio=0.2, test_ratio=0.2):

    train_list, _ = get_wav_list(words=words, unknown_ratio=unknown_ratio)

    test_list = get_sub_list(num=int(len(train_list) * test_ratio), sub_path=sub_path)

    lst = train_list + test_list

    return sample(lst, len(lst))





def preprocess_mfcc(wave):



    spectrogram = librosa.feature.melspectrogram(wave, sr=SR, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000)

    idx = [spectrogram > 0]

    spectrogram[idx] = np.log(spectrogram[idx])



    dct_filters = librosa.filters.dct(n_filters=40, n_input=40)

    mfcc = [np.matmul(dct_filters, x) for x in np.split(spectrogram, spectrogram.shape[1], axis=1)]

    mfcc = np.hstack(mfcc)

    mfcc = mfcc.astype(np.float32)

    return mfcc





def preprocess_mel(data, n_mels=40, normalization=False):

    spectrogram = librosa.feature.melspectrogram(data, sr=SR, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)

    spectrogram = librosa.power_to_db(spectrogram)

    spectrogram = spectrogram.astype(np.float32)

    if normalization:

        spectrogram = spectrogram.spectrogram()

        spectrogram -= spectrogram

    return spectrogram





def preprocess_wav(wav, normalization=True):

    data = wav.reshape(1, -1)

    if normalization:

        mean = data.mean()

        data -= mean

    return data
"""

model trainer

"""

from torch.autograd import Variable

from torch.utils.data import DataLoader

import torch

from time import time

from torch.nn import Softmax

import numpy as np

import pandas as pd

import os

from random import choice





def train_model(model_class, preprocess_fun, is_1d, reshape_size, BATCH_SIZE, epochs, CODER, preprocess_param={},

                bagging_num=1, semi_train_path=None, pretrained=None, pretraining=False, MGPU=False):

    """

    :param model_class: model class. e.g. vgg, resnet, senet

    :param preprocess_fun: preprocess function. e.g. mel, mfcc, raw wave

    :param is_1d: boolean. True for conv1d models and false for conv2d

    :param reshape_size: int. only for conv2d, reshape the image size

    :param BATCH_SIZE: batch size.

    :param epochs: number of epochs

    :param CODER: string for saving and loading model/files

    :param preprocess_param: parameters for preprocessing function

    :param bagging_num: number of training per model, aka bagging models

    :param semi_train_path: path to semi supervised learning file.

    :param pretrained: path to pretrained model

    :param pretraining: boolean. if this is pretraining

    :param MGPU: whether using multiple gpus

    """



    def get_model(model=model_class, m=MGPU, pretrained=pretrained):

        mdl = torch.nn.DataParallel(model()) if m else model()

        if not pretrained:

            return mdl

        else:

            print("load pretrained model here...")

            mdl.load_state_dict(torch.load(pretrained))

            if 'vgg' in pretrained:

                fixed_layers = list(mdl.features)

                for l in fixed_layers:

                    for p in l.parameters():

                        p.requires_grad = False

            return mdl



    label_to_int, int_to_label = get_label_dict()

    for b in range(bagging_num):

        print("training model # ", b)



        loss_fn = torch.nn.CrossEntropyLoss()



        speechmodel = get_model()

        speechmodel = speechmodel.cuda()



        total_correct = 0

        num_labels = 0

        start_time = time()

        

        training_loss = []

        validation_loss = []



        for e in range(epochs):

            

            total_correct_train = 0

            num_labels_train = 0

        

            total_correct_valid = 0

            num_labels_valid = 0

            

            print("training epoch ", e)

            learning_rate = 0.01 if e < 10 else 0.001

            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, speechmodel.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.00001)

            speechmodel.train()

            if semi_train_path:

                train_list = get_semi_list(words=label_to_int.keys(), sub_path=semi_train_path,

                                           test_ratio=choice([0.2, 0.25, 0.3, 0.35]))

                print("semi training list length: ", len(train_list))

            else:

                train_list, _ = get_wav_list(words=label_to_int.keys())



            if pretraining:

                traindataset = PreDataset(label_words_dict=label_to_int,

                                          add_noise=True, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,

                                          resize_shape=reshape_size, is_1d=is_1d)

            else:

                traindataset = SpeechDataset(mode='train', label_words_dict=label_to_int, wav_list=train_list,

                                             add_noise=True, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,

                                             resize_shape=reshape_size, is_1d=is_1d)

            trainloader = DataLoader(traindataset, BATCH_SIZE, shuffle=True)

            

            _, valid_list = get_wav_list(words=label_to_int.keys())

            valid_dataset = SpeechDataset(mode='validation', label_words_dict=label_to_int, wav_list=valid_list,

                                    add_noise=False, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,

                                    resize_shape=reshape_size, is_1d=is_1d)

            valid_loader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False)            

            

            for batch_idx, batch_data in enumerate(trainloader):

                spec = batch_data['spec']

                label = batch_data['label']

                spec, label = Variable(spec.cuda()), Variable(label.cuda())

                y_pred = speechmodel(spec)

                _, pred_labels = torch.max(y_pred.data, 1)

                correct = (pred_labels == label.data).sum()

                loss = loss_fn(y_pred, label)



                total_correct_train += correct

                num_labels_train += len(label)



                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            #print("training loss:", 100. * total_correct / num_labels, time()-start_time)

            print("training loss:", 100. * total_correct_train.item() / num_labels_train, time()-start_time)

            training_loss.append(100. * total_correct_train.item() / num_labels_train)

            

            

            speechmodel.eval()

            for batch_idx, batch_data in enumerate(valid_loader):

                spec = batch_data['spec']

                label = batch_data['label']

                spec, label = Variable(spec.cuda()), Variable(label.cuda())

                y_pred = speechmodel(spec)

                _, pred_labels = torch.max(y_pred.data, 1)

                correct = (pred_labels == label.data).sum()

                #loss = loss_fn(y_pred, label)



                total_correct_valid += correct

                num_labels_valid += len(label)

                #print(correct,len(label))



                #optimizer.zero_grad()

                #loss.backward()

                #optimizer.step()

            #print(total_correct_valid.item(),num_labels_valid)

            print("valid loss:", 100. * total_correct_valid.item() / num_labels_valid, time()-start_time)

            validation_loss.append(100. * total_correct_valid.item() / num_labels_valid)

            

            

            

        # save model

        create_directory("model")

        torch.save(speechmodel.state_dict(), "model/model_%s_%s.pth" % (CODER, b))



        

        # save the training and validation error

        error = pd.DataFrame()

        error["train"] = training_loss

        error["validation"] = validation_loss

        create_directory("sub")

        error.to_csv("sub/%s.csv" % CODER, index=False)

        

        

            



def create_directory(dir):

    if not os.path.exists(dir):

        os.makedirs(dir)

#list_2d = [('mel', preprocess_mel), ('mfcc', preprocess_mfcc)]

list_2d = [('mfcc', preprocess_mfcc)]



BAGGING_NUM=1



def train_and_predict(cfg_dict, preprocess_list):

    for p, preprocess_fun in preprocess_list:

        cfg = cfg_dict.copy()

        cfg['preprocess_fun'] = preprocess_fun

        cfg['CODER'] += '_%s' %p

        cfg['bagging_num'] = BAGGING_NUM

        print("training ", cfg['CODER'])

        train_model(**cfg)
KERNEL_SIZE=3

PADDING=1





class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=KERNEL_SIZE, stride=stride, padding=PADDING, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)



        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(planes)

            )



        # SE layers

        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear

        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)



    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))



        # Squeeze

        w = F.avg_pool2d(out, out.size(2))

        w = F.relu(self.fc1(w))

        w = F.sigmoid(self.fc2(w))

        # Excitation

        out = out * w



        out += self.shortcut(x)

        out = F.relu(out)

        return out





class PreActBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):

        super(PreActBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=KERNEL_SIZE, stride=stride, padding=PADDING, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING, bias=False)



        if stride != 1 or in_planes != planes:

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

            )



        # SE layers

        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)

        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)



    def forward(self, x):

        out = F.relu(self.bn1(x))

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x

        out = self.conv1(out)

        out = self.conv2(F.relu(self.bn2(out)))



        # Squeeze

        w = F.avg_pool2d(out, out.size(2))

        w = F.relu(self.fc1(w))

        w = F.sigmoid(self.fc2(w))

        # Excitation

        out = out * w



        out += shortcut

        return out





class SENet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=12):

        super(SENet, self).__init__()

        self.in_planes = 64



        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * 4 * 4, num_classes)



    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)

        layers = []

        for stride in strides:

            layers.append(block(self.in_planes, planes, stride))

            self.in_planes = planes

        return nn.Sequential(*layers)



    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return out





def SeModel():

    return SENet(PreActBlock, [2,2,2,2])
se_config = {

    'model_class': SeModel,

    'is_1d': False,

    'reshape_size': 128,

    'BATCH_SIZE': 16,

    'epochs': 2,

    'CODER': 'senet'

}



print("train senet..........")

train_and_predict(se_config, list_2d)
!ls sub/
op = pd.read_csv("sub/senet_mfcc.csv")
op.shape