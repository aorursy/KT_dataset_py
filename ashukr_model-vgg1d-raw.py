import os

from os.path import isdir, join

import shutil

from tqdm import tqdm

# TODO: create train and test dataset



f = open('../input/train/testing_list.txt', 'r')

test_list = f.readlines()

f.close()



audio_path = "../input/train/audio"



test_path_list = []

for n in test_list:

    test_path_list.append(n[:-1])



test_path = "../test/"

os.makedirs(test_path)

i=0

for file in test_path_list:

    #print((audio_path+"/"+ file))

    shutil.copy((audio_path+"/"+ file), "../test/"+str(i)+".wav")

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





print("creating test and train dataset...")



for direct in tqdm(dirs):

    for f in os.listdir(join(audio_path, direct)):

        if f.endswith('.wav'):

            # print(str(direct)+"/"+f)

            if (str(direct)+"/"+f) not in test_path_list:

                

            

                shutil.copy(str(audio_path)+"/"+str(direct)+"/"+f, '../train/'+str(direct))



print("number of test points {}".format(i))

print("counting the train dataset...")

train_recordings = []

for direct in tqdm(dirs):

    waves = [f for f in os.listdir(join(train_path, direct)) if f.endswith('.wav')]

    for wave in waves:

        train_recordings.append(str(direct)+"/"+wave)

print("total number of files in train is {}".format(len(train_recordings)))
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

        if self.mode == 'test':

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

            if self.mode == 'test':

                return {'spec': wav_tensor, 'id': self.wav_list[idx]}



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

    full_test_list = glob("../test/*.wav")



    # sample full train list

    sampled_train_list = []

    for w in full_train_list:

        l = w.split("/")[-2]

        if l not in words:

            if random.random() < unknown_ratio:

                sampled_train_list.append(w)

        else:

            sampled_train_list.append(w)



    return sampled_train_list, full_test_list





def get_sub_list(num, sub_path):

    lst = []

    df = pd.read_csv(sub_path)

    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']

    each_num = int(num * 0.085)

    for w in words:

        tmp = df['fname'][df['label'] == w].sample(each_num).tolist()

        lst += ["../test/" + x for x in tmp]

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



        for e in range(epochs):

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

            for batch_idx, batch_data in enumerate(trainloader):

                spec = batch_data['spec']

                label = batch_data['label']

                spec, label = Variable(spec.cuda()), Variable(label.cuda())

                y_pred = speechmodel(spec)

                _, pred_labels = torch.max(y_pred.data, 1)

                correct = (pred_labels == label.data).sum()

                loss = loss_fn(y_pred, label)



                total_correct += correct

                num_labels += len(label)



                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            print("training loss:", 100. * total_correct / num_labels, time()-start_time)



        # save model

        create_directory("model")

        torch.save(speechmodel.state_dict(), "model/model_%s_%s.pth" % (CODER, b))



    if not pretraining:

        print("doing prediction...")

        softmax = Softmax()



        trained_models = ["model/model_%s_%s.pth" % (CODER, b) for b in range(bagging_num)]



        # prediction

        _, test_list = get_wav_list(words=label_to_int.keys())

        testdataset = SpeechDataset(mode='test', label_words_dict=label_to_int, wav_list=test_list,

                                    add_noise=False, preprocess_fun=preprocess_fun, preprocess_param=preprocess_param,

                                    resize_shape=reshape_size, is_1d=is_1d)

        testloader = DataLoader(testdataset, BATCH_SIZE, shuffle=False)



        for e, m in enumerate(trained_models):

            print("predicting ", m)

            speechmodel = get_model(m=MGPU)

            speechmodel.load_state_dict(torch.load(m))

            speechmodel = speechmodel.cuda()

            speechmodel.eval()



            test_fnames, test_labels = [], []

            pred_scores = []

            # do prediction and make a submission file

            for batch_idx, batch_data in enumerate(testloader):

                spec = Variable(batch_data['spec'].cuda())

                fname = batch_data['id']

                y_pred = softmax(speechmodel(spec))

                pred_scores.append(y_pred.data.cpu().numpy())

                test_fnames += fname



            if e == 0:

                final_pred = np.vstack(pred_scores)

                final_test_fnames = test_fnames

            else:

                final_pred += np.vstack(pred_scores)

                assert final_test_fnames == test_fnames



        final_pred /= len(trained_models)

        final_labels = [int_to_label[x] for x in np.argmax(final_pred, 1)]



        test_fnames = [x.split("/")[-1] for x in final_test_fnames]



        labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']

        pred_scores = pd.DataFrame(np.vstack(final_pred), columns=labels)

        pred_scores['fname'] = test_fnames



        create_directory("pred_scores")

        pred_scores.to_csv("pred_scores/%s.csv" % CODER, index=False)



        create_directory("sub")

        pd.DataFrame({'fname': test_fnames,

                      'label': final_labels}).to_csv("sub/%s.csv" % CODER, index=False)





def create_directory(dir):

    if not os.path.exists(dir):

        os.makedirs(dir)

list_2d = [('mel', preprocess_mel)]

BAGGING_NUM=1



def train_and_predict(cfg_dict, preprocess_list):

    for p, preprocess_fun in preprocess_list:

        cfg = cfg_dict.copy()

        cfg['preprocess_fun'] = preprocess_fun

        cfg['CODER'] += '_%s' %p

        cfg['bagging_num'] = BAGGING_NUM

        print("training ", cfg['CODER'])

        train_model(**cfg)
import torch

import torch.nn.functional as F

import torch.nn as nn





class VGG1D(torch.nn.Module):

    def __init__(self, features):

        super(VGG1D, self).__init__()

        self.features = features

        self.classifier = nn.Sequential(

            nn.Linear(256 * 2, 1024),

            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.Linear(1024, 1024),

            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.Linear(1024, 12),

        )



    def forward(self, x):

        x = self.features(x)

        max_pooled = F.adaptive_max_pool1d(x, 1)

        avg_pooled = F.adaptive_avg_pool1d(x, 1)

        x = torch.cat([max_pooled, avg_pooled], dim=1)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x





class VGGMEL(torch.nn.Module):

    def __init__(self, features):

        super(VGGMEL, self).__init__()

        self.features = features

        self.classifier = nn.Sequential(

            nn.Linear(256*6, 1024),

            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.Linear(1024, 1024),

            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.Linear(1024, 12),

        )



    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x





def make_layers(cfg, batch_norm=True, mk=2, ms=2, lk=5,ls=1,lp=2, in_c=1):

    layers = []

    in_channels = in_c

    for v in cfg:

        if v == 'M':

            layers += [nn.MaxPool1d(kernel_size=mk, stride=ms)]

        else:

            conv1d = nn.Conv1d(in_channels, v, kernel_size=lk, padding=lp, stride=ls)

            if batch_norm:

                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]

            else:

                layers += [conv1d, nn.ReLU(inplace=True)]

            in_channels = v

    return nn.Sequential(*layers)





def vgg1d():

    return VGG1D(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 'M', 512, 'M', 1024, 'M', 1024, 'M', 512, 'M', 256, 'M']))





def vggmel():

    return VGGMEL(make_layers([64, 64, 'M', 512, 512,'M', 1024, 1024, 'M', 512, 256, 'M'], lk=3, lp=1, in_c=40))

vgg1d_config = {

    'model_class': vgg1d,

    'is_1d': True,

    'reshape_size': None,

    'BATCH_SIZE': 32,

    'epochs': 100,

    'CODER': 'vgg1d'

}



print("train vgg1d on raw features..........")

train_and_predict(vgg1d_config, [('raw', preprocess_wav)])
!ls sub/
op = pd.read_csv("sub/vgg1d_raw.csv")
op.shape