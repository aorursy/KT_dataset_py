# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



print('Files in this dataset:')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
! pip install dl-cliche torchsummary pytorch-lightning
# public modules

from dlcliche.notebook import *

from dlcliche.utils import (

    sys, random, Path, np, plt, EasyDict,

    ensure_folder, deterministic_everything,

)

from argparse import Namespace



########################################################################

# setup STD I/O

########################################################################

"""

Standard output is logged in "baseline.log".

"""

import logging



logging.basicConfig(level=logging.DEBUG, filename="baseline.log")

logger = logging.getLogger(' ')

handler = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)

logger.addHandler(handler)
# https://github.com/daisukelab/dcase2020_task2_variants/blob/master/pytorch_common.py

import torch

from torch import nn

import torch.nn.functional as F

import torchsummary

import torch

import pytorch_lightning as pl

import random

from dlcliche.utils import *





def load_weights(model, weight_file):

    model.load_state_dict(torch.load(weight_file))





def summary(device, model, input_size=(1, 640)):

    torchsummary.summary(model.to(device), input_size=input_size)





def summarize_weights(model):

    summary = pd.DataFrame()

    for k, p in model.state_dict().items():

        p = p.cpu().numpy()

        df = pd.Series(p.ravel()).describe()

        summary.loc[k, 'mean'] = df['mean']

        summary.loc[k, 'std'] = df['std']

        summary.loc[k, 'min'] = df['min']

        summary.loc[k, 'max'] = df['max']

    return summary





def show_some_predictions(dl, model, start_index, n_samples, image=False):

    shape = (-1, 64, 64) if image else (-1, 640)

    x, y = next(iter(dl))

    with torch.no_grad():

        yhat = model(x)

    x = x.cpu().numpy().reshape(shape)

    yhat = yhat.cpu().numpy().reshape(shape)

    print(x.shape, yhat.shape)

    for sample_idx in range(start_index, start_index + n_samples):

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        if image:

            axs[0].imshow(x[sample_idx])

            axs[1].imshow(yhat[sample_idx])

        else:

            axs[0].plot(x[sample_idx])

            axs[1].plot(yhat[sample_idx])





def normalize_0to1(X):

    # Normalize to range from [-90, 24] to [0, 1] based on dataset quick stat check.

    X = (X + 90.) / (24. + 90.)

    X = np.clip(X, 0., 1.)

    return X





class ToTensor1ch(object):

    """PyTorch basic transform to convert np array to torch.Tensor.

    Args:

        array: (dim,) or (batch, dims) feature array.

    """

    def __init__(self, device=None, image=False):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.non_batch_shape_len = 2 if image else 1



    def __call__(self, array):

        # (dims)

        if len(array.shape) == self.non_batch_shape_len:

            return torch.Tensor(array).unsqueeze(0).to(self.device)

        # (batch, dims)

        return torch.Tensor(array).unsqueeze(1).to(self.device)



    def __repr__(self):

        return 'to_tensor_1d'





########################################################################

# PyTorch utilities

########################################################################



class Task2Dataset(torch.utils.data.Dataset):

    """PyTorch dataset class for task2. Caching to a file supported.

    Args:

        n_mels, frames, n_fft, hop_length, power, transform: Audio conversion settings.

        normalize: Normalize data value range from [-90, 24] to [0, 1] for VAE, False by default.

        cache_to: Cache filename or None by default, use this for your iterative development.

    """



    def __init__(self, files, n_mels, frames, n_fft, hop_length, power, transform,

                 normalize=False, cache_to=None):

        self.transform = transform

        self.files = files

        self.n_mels, self.frames, self.n_fft = n_mels, frames, n_fft

        self.hop_length, self.power = hop_length, power

        # load cache or convert all the data for the first time

        if cache_to is not None and Path(cache_to).exists():

            logger.info(f'Loading cached {Path(cache_to).name}')

            self.X = np.load(cache_to)

        else:

            self.X = com.list_to_vector_array(self.files,

                             n_mels=self.n_mels,

                             frames=self.frames,

                             n_fft=self.n_fft,

                             hop_length=self.hop_length,

                             power=self.power)

            if cache_to is not None:

                np.save(cache_to, self.X)



        if normalize:

            self.X = normalize_0to1(self.X)



    def __len__(self):

        return len(self.X)



    def __getitem__(self, index):

        x = self.X[index]

        x = self.transform(x)

        return x, x





class Task2Lightning(pl.LightningModule):

    """Task2 PyTorch Lightning class, for training only."""



    def __init__(self, device, model, params, files, normalize=False):

        super().__init__()

        self.device = device

        self.params = params

        self.normalize = normalize

        self.model = model

        self.mseloss = torch.nn.MSELoss()

        # split data files

        if files is not None:

            n_val = int(params.fit.validation_split * len(files))

            self.val_files = random.sample(files, n_val)

            self.train_files = [f for f in files if f not in self.val_files]



    def forward(self, x):

        return self.model(x)



    def training_step(self, batch, batch_nb):

        x, y = batch

        y_hat = self.forward(x)

        loss = self.mseloss(y_hat, y)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}



    def validation_step(self, batch, batch_nb):

        x, y = batch

        y_hat = self.forward(x)

        return {'val_loss': self.mseloss(y_hat, y)}



    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}



    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=self.params.fit.lr,

                                betas=(self.params.fit.b1, self.params.fit.b2),

                                weight_decay=self.params.fit.weight_decay)



    def _get_dl(self, for_what):

        files = self.train_files if for_what == 'train' else self.val_files

        cache_file = f'{self.params.model_directory}/__cache_{str(files[0]).split("/")[-3]}_{for_what}.npy'

        ds = Task2Dataset(files,

                          n_mels=self.params.feature.n_mels,

                          frames=self.params.feature.frames,

                          n_fft=self.params.feature.n_fft,

                          hop_length=self.params.feature.hop_length,

                          power=self.params.feature.power,

                          transform=com.ToTensor1ch(device=self.device),

                          normalize=self.normalize,

                          cache_to=cache_file)

        return torch.utils.data.DataLoader(ds, batch_size=self.params.fit.batch_size,

                          shuffle=(self.params.fit.shuffle if for_what == 'train' else False))



    def train_dataloader(self):

        return self._get_dl('train')



    def val_dataloader(self):

        return self._get_dl('val')
# https://github.com/daisukelab/dcase2020_task2_variants/blob/master/image_common.py



import librosa





def get_log_mel_spectrogram(filename,

                            n_mels=64,

                            n_fft=1024,

                            hop_length=512,

                            power=2.0):

    wav, sampling_rate = com.file_load(filename)

    mel_spectrogram = librosa.feature.melspectrogram(y=wav,

                                                     sr=sampling_rate,

                                                     n_fft=n_fft,

                                                     hop_length=hop_length,

                                                     n_mels=n_mels,

                                                     power=power)

    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    return log_mel_spectrogram





def file_to_vector_array_2d(file_name,

                         n_mels=64,

                         steps=20,

                         n_fft=1024,

                         hop_length=512,

                         power=2.0):

    """convert file_name to a 2d vector array.



    file_name : str

        target .wav file

    return : np.array( np.array( float ) )

        vector array

        * dataset.shape = (dataset_size, n_mels, n_mels)

    """

    # 02 generate melspectrogram using librosa

    y, sr = com.file_load(file_name)

    mel_spectrogram = librosa.feature.melspectrogram(y=y,

                                                     sr=sr,

                                                     n_fft=n_fft,

                                                     hop_length=hop_length,

                                                     n_mels=n_mels,

                                                     power=power)



    # 03 convert melspectrogram to log mel energy

    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)



    # 04 calculate total vector size

    vector_array_size = (log_mel_spectrogram.shape[1] - n_mels + 1) // steps



    # 06 generate feature vectors by concatenating multiframes

    vector_array = np.zeros((vector_array_size, n_mels, n_mels))

    for t in range(vector_array_size):

        vector_array[t] = log_mel_spectrogram[:, t*steps:t*steps+n_mels]



    return vector_array





class Task2ImageDataset(torch.utils.data.Dataset):

    """Task 2 dataset to handle samples as 1 channel image.



    Unlike other dataset, set `preprocessed_file` as preprocessed dataset filename.

    For every output, this dataset class crop square image from this original data.



    Number of total samples is `n_sampling` times number of _live data_.

    _Live data_ is all the original data by default, and filtered by splitting functions.



    - For training use, set `random=True`. This will yield randomly cropped square

      image (64x64 for example) from the original sample (64x431 for 10s sample).

    - For validation use, set `random=False`. Image will be cropped from fixed position.



    Augmentation can be flexibly applied to either the output `x` only or `y` only, or both `x` and `y`.

    `aug_x` and `aug_y` control this behavor.



    Data split for training/validation can be done by using:

        `get_index_by_pct()`: generate list of training index.

        `train_split(train_index)`: set live data as original samples listed on `train_index`.

        `val_split(train_index)`: set live data as original samples NOT listed on `train_index`.

    Yields:

        x: square image expected to be used as source.

        y: square image expected to be used as reference for evaluating reconstructed image by training model.

    """



    def __init__(self, preprocessed_file, n_sampling=10, transform=None, augment_tfm=None,

                 normalize=True, random=True, aug_x=True, aug_y=False, debug=False):

        self.n_sampling = n_sampling

        self.transform, self.augment_tfm = transform, augment_tfm

        self.random, self.aug_x, self.aug_y = random, aug_x, aug_y



        self.X = np.load(preprocessed_file)

        if normalize:

            self.X = normalize_0to1(self.X)



        if debug:

            from dlcliche.utils import display

            from dlcliche.math import np_describe

            display(np_describe(self.X[0].cpu().numpy()))



        self.orgX = self.X

  

    def get_index_by_pct(self, split_pct=0.1):

        n = len(self.orgX)

        return random.sample(range(n), k=(n - int(n * split_pct)))



    def train_split(self, train_index):

        self.train_index = train_index

        self.X = self.orgX[train_index]

    

    def val_split(self, train_index):

        n = len(self.orgX)

        self.val_index = [i for i in range(n) if i not in train_index]

        self.X = self.orgX[self.val_index]



    def __len__(self):

        return len(self.X) * self.n_sampling



    def __getitem__(self, index):

        file_index = index // self.n_sampling

        part_index = index % self.n_sampling

        x = self.X[file_index]

        dim, length = x.shape



        # crop square part of sample

        if self.random:

            # random crop

            start = random.randint(0, length - dim)

        else:

            # crop with fixed position

            start = (length // self.n_sampling) * part_index

        start = min(start, length - dim)

        x = x[:, start:start+dim]



        # augmentation transform

        y = x

        if self.augment_tfm is not None:

            tfm_x = self.augment_tfm(x)

            if self.aug_x: x = tfm_x

            if self.aug_y: y = tfm_x



        # transform (convert to tensor here)

        if self.transform is not None:

            x = self.transform(x)

            y = self.transform(y)

        return x, y
config_yaml = '''

dev_directory : /kaggle/input/dc2020task2prep

eval_directory : /kaggle/input/dc2020task2prep

model_directory: ./model

result_directory: ./result

result_file: result.csv

# target: ['ToyConveyor']  #  set this when you want to test for specific target only.



max_fpr : 0.1



feature:

  n_mels: 64

  frames : 5

  n_fft: 1024

  hop_length: 512

  power: 2.0



fit:

  lr: 0.001

  b1: 0.9

  b2: 0.999

  weight_decay: 0.0

  epochs : 100

  batch_size : 256

  shuffle : True

  validation_split : 0.1

  verbose : 1

'''



import yaml

from easydict import EasyDict

params = EasyDict(yaml.safe_load(config_yaml))



params
import torch

from torch import nn

import torch.nn.functional as F

import torchsummary

import torch

import pytorch_lightning as pl

import random





class CNNAutoEncoder(nn.Module):

    """

    Thanks to http://dl-kento.hatenablog.com/entry/2018/02/22/200811#Convolutional-AutoEncoder

    """



    def  __init__(self, z_dim=40):

        super().__init__()



        # define the network

        # encoder

        self.conv1 = nn.Sequential(nn.ZeroPad2d((1,2,1,2)),

                              nn.Conv2d(1, 32, kernel_size=5, stride=2),

                              nn.ReLU())

        self.conv2 = nn.Sequential(nn.ZeroPad2d((1,2,1,2)),

                              nn.Conv2d(32, 64, kernel_size=5, stride=2),

                              nn.ReLU(), nn.Dropout(0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),

                              nn.ReLU(), nn.Dropout(0.3))

        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),

                              nn.ReLU(), nn.Dropout(0.3))

        self.fc1 = nn.Conv2d(256, z_dim, kernel_size=3)



        # decoder

        self.fc2 = nn.Sequential(nn.ConvTranspose2d(z_dim, 256, kernel_size=3),

                            nn.ReLU(), nn.Dropout(0.3))

        self.conv4d = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0),

                               nn.ReLU(), nn.Dropout(0.3))

        self.conv3d = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),

                               nn.ReLU(), nn.Dropout(0.2))

        self.conv2d = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),

                               nn.ReLU())

        self.conv1d = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2)



    def forward(self, x):

        encoded = self.fc1(self.conv4(self.conv3(self.conv2(self.conv1(x)))))



        decoded = self.fc2(encoded)

        decoded = self.conv4d(decoded)

        decoded = self.conv3d(decoded)

        decoded = self.conv2d(decoded)[:,:,1:-1,1:-1]

        decoded = self.conv1d(decoded)[:,:,0:-1,0:-1]

        decoded = nn.Sigmoid()(decoded)



        return decoded
# create working directory

ensure_folder(params.model_directory)



# test targets

if 'target' in params:

    types = params.target

else:

    types = sorted(set([d.name.split('-')[1] for d in Path(params.dev_directory).glob('*.npy')]))



# dataset

data_files = sorted(Path(params.dev_directory).glob('dc2020t2l1*.npy'))

data = {t:[f for f in data_files if t in str(f)] for t in types}



# fix random seeds

deterministic_everything(2020, pytorch=True)



# PyTorch device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



data, types
# This will show you visualization examples of the dataset when augmentation is applied

if False: # test dataset

    from albumentations import (

        HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomCrop,

        Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,

        IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,

        IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,

        ISONoise, Cutout

    )



    class MyTfm():

        def __init__(self):

            self.album_tfm = Compose([

                Cutout(num_holes=8, max_h_size=6, max_w_size=6, p=0.6),

                ShiftScaleRotate(shift_limit=0.3, scale_limit=0.1, rotate_limit=0, p=.5),

            ], p=1)

        def __call__(self, x):        

            data = {"image": x}

            augmented = self.album_tfm(**data)

            x = augmented["image"]

            return x



    trn_ds = Task2ImageDataset(data[types[0]][1], augment_tfm=MyTfm(), aug_y=True, random=True)

    train_index = trn_ds.get_index_by_pct()

    trn_ds.train_split(train_index)

    fig, ax = plt.subplots(1, 2)

    one = trn_ds[0]; ax[0].imshow(one[0]); ax[1].imshow(one[1]);
class Task2ImageLightning(Task2Lightning):

    """Task2 PyTorch Lightning class, for training only."""



    def __init__(self, device, model, params, preprocessed_file, normalize=True):

        super().__init__(device, model, params, files=None, normalize=normalize)

        self.device = device

        self.params = params

        self.normalize = normalize

        self.model = model

        self.mseloss = torch.nn.MSELoss()



        to_tensor = ToTensor1ch(device=self.device, image=True)

        self.trn_ds = Task2ImageDataset(preprocessed_file, transform=to_tensor,

                                      normalize=normalize)

        self.val_ds = Task2ImageDataset(preprocessed_file, transform=to_tensor,

                                      normalize=normalize, random=False)

        train_index = self.trn_ds.get_index_by_pct()

        self.trn_ds.train_split(train_index)

        self.val_ds.val_split(train_index)



    def _get_dl(self, for_what):

        ds = self.trn_ds if for_what == 'train' else self.val_ds

        return torch.utils.data.DataLoader(ds, batch_size=self.params.fit.batch_size,

                          shuffle=(self.params.fit.shuffle if for_what == 'train' else False))





# train models



for target in types:

    print(f'==== Start training [{target}] with {torch.cuda.device_count()} GPU(s). ====')



    try:

        del model, task2, trainer

    except:

        pass # first time, nothing to delete

    model = CNNAutoEncoder().to(device)

    task2 = Task2ImageLightning(device, model, params, preprocessed_file=data[target][1])

    summary(device, model, input_size=task2.train_dataloader().dataset[0][0].shape)

    trainer = pl.Trainer(max_epochs=params.fit.epochs, gpus=torch.cuda.device_count())

    trainer.fit(task2)



    model_file = f'{params.model_directory}/model_{target}.pth'

    torch.save(task2.model.state_dict(), model_file)

    print(f'saved {model_file}.\n')
show_some_predictions(task2.train_dataloader(), task2.model, 0, 3, image=True)
show_some_predictions(task2.val_dataloader(), task2.model, 0, 3, image=True)
# Getting test file information in a data frame



! wget https://raw.githubusercontent.com/daisukelab/dcase2020_task2_variants/master/file_info.csv



df = pd.read_csv('file_info.csv')

df.file = df.file.map(lambda f: str(f).replace('/data/task2/dev', '/kaggle/input/dc2020task2'))

df['id'] = df.file.map(lambda f: '_'.join(f.split('_')[1:3]))

df['class'] = df.file.map(lambda f: f.split('_')[0].split('/')[-1])

types = df.type.unique()
import glob

import re

import csv

import itertools

from tqdm import tqdm

from sklearn import metrics





deterministic_everything(2022, pytorch=True)





def save_csv(save_file_path,

             save_data):

    with open(save_file_path, "w", newline="") as f:

        writer = csv.writer(f, lineterminator='\n')

        writer.writerows(save_data)





class Task2ImageDatasetForTest(Task2ImageDataset):

    def get_test_batch_x(self, file_index, n_mels=params.feature.n_mels, steps=20):

        log_mel_spectrogram = self.X[file_index]

        vector_array_size = (log_mel_spectrogram.shape[1] - n_mels + 1) // steps

        vector_array = np.zeros((vector_array_size, n_mels, n_mels))

        for t in range(vector_array_size):

            vector_array[t] = log_mel_spectrogram[:, t*steps:t*steps+n_mels]

        return vector_array





def get_machine_id_list_for_test(df, target):

    return sorted(set(df[df['type'] == target].id.values))





def test_file_list_generator(df,

                             target,

                             id_name,

                             prefix_normal="normal",

                             prefix_anomaly="anomaly"):

    """

    target : str

        target machine type

    id_name : str

        id of wav file in <<test_dir_name>> directory

    prefix_normal : str (default="normal")

        normal directory name

    prefix_anomaly : str (default="anomaly")

        anomaly directory name



    return :

        test_files : list [ str ]

            file list for test

        test_labels : list [ boolean ]

            label info. list for test

            * normal/anomaly = 0/1

    """

    logger.info("target_id : {}".format(target+"_"+id_name))

    df = df[df['type'] == target][df.split == 'test'].reset_index()

    df = df[df['id'] == id_name]



    indexes = df.index.values

    files = df.file.values

    labels = df['class'].map(lambda c: 1 if c == prefix_anomaly else 0)

    logger.info("# of test_files : {num}".format(num=len(files)))

    if len(files) == 0:

        logger.exception("no_wav_file!!")

    print("\n========================================")



    return indexes, files, labels

########################################################################





########################################################################

# main 01_test.py

########################################################################

def do_test():

    # check mode

    # "development": mode == True

    # "evaluation": mode == False

    mode =True



    # make output result directory

    os.makedirs(params.result_directory, exist_ok=True)



    # initialize lines in csv for AUC and pAUC

    csv_lines = []



    # PyTorch version specific...

    to_tensor = ToTensor1ch(image=True)



    # loop of the base directory

    for idx, machine_type in enumerate(types):

        print("\n===========================")

        print(f"[{idx}/{len(types)}] {machine_type}")



        print("============== MODEL LOAD ==============")

        # set model path

        model_file = "{model}/model_{machine_type}.pth".format(model=params.model_directory,

                                                               machine_type=machine_type)



        # load model file

        if not os.path.exists(model_file):

            logger.error("{} model not found ".format(machine_type))

            sys.exit(-1)

        logger.info("loading model: {}".format(model_file))

        model = CNNAutoEncoder().to(device)

        load_weights(model, model_file)

        summary(device, model, input_size=(1, params.feature.n_mels, params.feature.n_mels))

        model.eval()



        if mode:

            # results by type

            csv_lines.append([machine_type])

            csv_lines.append(["id", "AUC", "pAUC"])

            performance = []



        machine_id_list = get_machine_id_list_for_test(df, machine_type)



        for id_str in machine_id_list:

            # load test file

            indexes, test_files, y_true = test_file_list_generator(df, machine_type, id_str)

            print(f'test preprocesssed file: {data[machine_type][0]}')

            ds = Task2ImageDatasetForTest(data[machine_type][0], random=False, normalize=True)



            # setup anomaly score file path

            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(

                                                                                     result=params.result_directory,

                                                                                     machine_type=machine_type,

                                                                                     id_str=id_str)

            anomaly_score_list = []



            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")

            y_pred = [0. for k in test_files]

            for batch_idx, (file_idx, file_path) in tqdm(enumerate(zip(indexes, test_files)), total=len(test_files)):

                x = ds.get_test_batch_x(file_idx)

                with torch.no_grad():

                    yhat = model(to_tensor(x)).cpu().detach().numpy().reshape(x.shape)

                    errors = np.mean(np.square(x - yhat), axis=1)

                    if batch_idx in [0, 500]:

                        for i in range(2):

                            fig, axs = plt.subplots(1, 2)

                            axs[0].imshow(x[i])

                            axs[1].imshow(yhat[i])

                            plt.show()

                y_pred[batch_idx] = np.mean(errors)

                anomaly_score_list.append([os.path.basename(file_path), y_pred[batch_idx]])



            # save anomaly score

            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)

            logger.info("anomaly score result ->  {}".format(anomaly_score_csv))



            if mode:

                # append AUC and pAUC to lists

                auc = metrics.roc_auc_score(y_true, y_pred)

                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=params.max_fpr)

                csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])

                performance.append([auc, p_auc])

                logger.info("AUC : {}".format(auc))

                logger.info("pAUC : {}".format(p_auc))



            print("\n============ END OF TEST FOR A MACHINE ID ============")



        if mode:

            # calculate averages for AUCs and pAUCs

            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)

            csv_lines.append(["Average"] + list(averaged_performance))

            csv_lines.append([])



    result_path = "{result}/{file_name}".format(result=params.result_directory, file_name=params.result_file)

    logger.info("AUC and pAUC results -> {}".format(result_path))

    save_csv(save_file_path=result_path, save_data=csv_lines)

    return csv_lines



csv_lines = do_test()
def print_csv_lines(csv_lines):

    for l in csv_lines:

        print('\t\t'.join([(a if type(a) == str else f'{a:.6f}') for a in l]))

        

print_csv_lines(csv_lines)