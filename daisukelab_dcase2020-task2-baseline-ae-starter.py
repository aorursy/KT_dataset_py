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
#   common.py



########################################################################

# import python-library

########################################################################

# default

import glob

import argparse

import sys

import os

from pathlib import Path

import pandas as pd



# additional

import numpy as np

import librosa

import librosa.core

import librosa.feature

from tqdm.auto import tqdm

import torch

########################################################################





########################################################################

# file I/O

########################################################################

# wav file Input

def file_load(wav_name, mono=False):

    """

    load .wav file.



    wav_name : str

        target .wav file

    sampling_rate : int

        audio file sampling_rate

    mono : boolean

        When load a multi channels file and this param True, the returned data will be merged for mono data



    return : np.array( float )

    """

    try:

        return librosa.load(wav_name, sr=None, mono=mono)

    except:

        logger.error("file_broken or not exists!! : {}".format(wav_name))





########################################################################





########################################################################

# feature extractor

########################################################################

def file_to_vector_array(file_name,

                         n_mels=64,

                         frames=5,

                         n_fft=1024,

                         hop_length=512,

                         power=2.0):

    """

    convert file_name to a vector array.



    file_name : str

        target .wav file



    return : np.array( np.array( float ) )

        vector array

        * dataset.shape = (dataset_size, feature_vector_length)

    """

    # 01 calculate the number of dimensions

    dims = n_mels * frames



    # 02 generate melspectrogram using librosa

    y, sr = file_load(file_name)

    mel_spectrogram = librosa.feature.melspectrogram(y=y,

                                                     sr=sr,

                                                     n_fft=n_fft,

                                                     hop_length=hop_length,

                                                     n_mels=n_mels,

                                                     power=power)



    # 03 convert melspectrogram to log mel energy

    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)



    # 04 calculate total vector size

    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1



    # 05 skip too short clips

    if vector_array_size < 1:

        return np.empty((0, dims))



    # 06 generate feature vectors by concatenating multiframes

    vector_array = np.zeros((vector_array_size, dims))

    for t in range(frames):

        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T



    return vector_array





# load dataset

def select_dirs(param, mode):

    """

    param : dict

        baseline.yaml data



    return :

        if active type the development :

            dirs :  list [ str ]

                load base directory list of dev_data

        if active type the evaluation :

            dirs : list [ str ]

                load base directory list of eval_data

    """

    if mode:

        logger.info("load_directory <- development")

        dir_path = os.path.abspath("{base}/*".format(base=param["dev_directory"]))

        dirs = sorted(glob.glob(dir_path))

    else:

        logger.info("load_directory <- evaluation")

        dir_path = os.path.abspath("{base}/*".format(base=param["eval_directory"]))

        dirs = sorted(glob.glob(dir_path))

    dirs = [d for d in dirs if os.path.isdir(d)]



    if 'target' in param:

        def is_one_of_in(substrs, full_str):

            for s in substrs:

                if s in full_str: return True

            return False

        dirs = [d for d in dirs if is_one_of_in(param["target"], str(d))]



    return dirs



########################################################################





def list_to_vector_array(file_list,

                         msg="calc...",

                         n_mels=64,

                         frames=5,

                         n_fft=1024,

                         hop_length=512,

                         power=2.0):

    """

    convert the file_list to a vector array.

    file_to_vector_array() is iterated, and the output vector array is concatenated.



    file_list : list [ str ]

        .wav filename list of dataset

    msg : str ( default = "calc..." )

        description for tqdm.

        this parameter will be input into "desc" param at tqdm.



    return : np.array( np.array( float ) )

        vector array for training (this function is not used for test.)

        * dataset.shape = (number of feature vectors, dimensions of feature vectors)

    """

    # calculate the number of dimensions

    dims = n_mels * frames



    # iterate file_to_vector_array()

    for idx in tqdm(range(len(file_list)), desc=msg):

        vector_array = file_to_vector_array(file_list[idx],

                                            n_mels=n_mels,

                                            frames=frames,

                                            n_fft=n_fft,

                                            hop_length=hop_length,

                                            power=power)

        if idx == 0:

            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)

            logger.info((f'Creating data for {len(file_list)} files: size={dataset.shape[0]}'

                         f', shape={dataset.shape[1:]}'))

        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array



    return dataset





def file_list_generator(target_dir,

                        dir_name="train",

                        ext="wav"):

    """

    target_dir : str

        base directory path of the dev_data or eval_data

    dir_name : str (default="train")

        directory name containing training data

    ext : str (default="wav")

        file extension of audio files



    return :

        train_files : list [ str ]

            file list for training

    """

    logger.info("target_dir : {}".format('/'.join(str(target_dir).split('/')[-2:])))



    # generate training list

    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))

    files = sorted(glob.glob(training_list_path))

    if len(files) == 0:

        logger.exception(f"{training_list_path} -> no_wav_file!!")



    logger.info("# of training samples : {num}".format(num=len(files)))

    return files

########################################################################
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

            self.X = list_to_vector_array(self.files,

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

                          transform=ToTensor1ch(device=self.device),

                          normalize=self.normalize,

                          cache_to=None) ### Unfortunately, cache_file becomes too big to store on Kaggle notebook...

        return torch.utils.data.DataLoader(ds, batch_size=self.params.fit.batch_size,

                          shuffle=(self.params.fit.shuffle if for_what == 'train' else False))



    def train_dataloader(self):

        return self._get_dl('train')



    def val_dataloader(self):

        return self._get_dl('val')
config_yaml = '''

dev_directory : /kaggle/input/dc2020task2

eval_directory : /kaggle/input/dc2020task2

model_directory: ./model

result_directory: ./result

result_file: result.csv

target: ['ToyCar']  #  set this when you want to test for specific target only.



max_fpr : 0.1



feature:

  n_mels: 128

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

  batch_size : 512

  shuffle : True

  validation_split : 0.1

  verbose : 1

'''



import yaml

from easydict import EasyDict

params = EasyDict(yaml.safe_load(config_yaml))
import torch

from torch import nn

import torch.nn.functional as F

import torch





class _LinearUnit(torch.nn.Module):

    """For use in Task2Baseline model."""

    def __init__(self, in_dim, out_dim):

        super().__init__()

        self.lin = torch.nn.Linear(in_dim, out_dim)

        self.bn = torch.nn.BatchNorm1d(out_dim)



    def forward(self, x):

        return torch.relu(self.bn(self.lin(x.view(x.size(0), -1))))





class Task2Baseline(torch.nn.Module):

    """PyTorch version of the baseline model."""

    def __init__(self):

        super().__init__()

        self.unit1 = _LinearUnit(640, 128)

        self.unit2 = _LinearUnit(128, 128)

        self.unit3 = _LinearUnit(128, 128)

        self.unit4 = _LinearUnit(128, 128)

        self.unit5 = _LinearUnit(128, 8)

        self.unit6 = _LinearUnit(8, 128)

        self.unit7 = _LinearUnit(128, 128)

        self.unit8 = _LinearUnit(128, 128)

        self.unit9 = _LinearUnit(128, 128)

        self.output = torch.nn.Linear(128, 640)



    def forward(self, x):

        shape = x.shape

        x = self.unit1(x.view(x.size(0), -1))

        x = self.unit2(x)

        x = self.unit3(x)

        x = self.unit4(x)

        x = self.unit5(x)

        x = self.unit6(x)

        x = self.unit7(x)

        x = self.unit8(x)

        x = self.unit9(x)

        return self.output(x).view(shape)
# create working directory

ensure_folder(params.model_directory)



# test directories

dirs = select_dirs(param=params, mode='development')



# fix random seeds

deterministic_everything(2020, pytorch=True)



# PyTorch device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# train models



for target_dir in dirs:

    target = str(target_dir).split('/')[-1]

    print(f'==== Start training [{target}] with {torch.cuda.device_count()} GPU(s). ====')



    files = file_list_generator(target_dir)



    model = Task2Baseline().to(device)

    if target == 'ToyCar': summary(device, model)

    task2 = Task2Lightning(device, model, params, files)

    trainer = pl.Trainer(max_epochs=params.fit.epochs, gpus=torch.cuda.device_count())

    trainer.fit(task2)

    

    model_file = f'{params.model_directory}/model_{target}.pth'

    torch.save(task2.model.state_dict(), model_file)

    print(f'saved {model_file}.\n')

    

    del model, task2, trainer
import glob

import re

import csv

import itertools

from tqdm import tqdm

from sklearn import metrics





deterministic_everything(2022, pytorch=True)





########################################################################

# def

########################################################################

def save_csv(save_file_path,

             save_data):

    with open(save_file_path, "w", newline="") as f:

        writer = csv.writer(f, lineterminator='\n')

        writer.writerows(save_data)





def get_machine_id_list_for_test(target_dir,

                                 dir_name="test",

                                 ext="wav"):

    """

    target_dir : str

        base directory path of "dev_data" or "eval_data"

    test_dir_name : str (default="test")

        directory containing test data

    ext : str (default="wav)

        file extension of audio files



    return :

        machine_id_list : list [ str ]

            list of machine IDs extracted from the names of test files

    """

    # create test files

    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))

    file_paths = sorted(glob.glob(dir_path))

    # extract id

    machine_id_list = sorted(list(set(itertools.chain.from_iterable(

        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))

    return machine_id_list





def test_file_list_generator(target_dir,

                             id_name,

                             dir_name="test",

                             prefix_normal="normal",

                             prefix_anomaly="anomaly",

                             ext="wav"):

    """

    target_dir : str

        base directory path of the dev_data or eval_data

    id_name : str

        id of wav file in <<test_dir_name>> directory

    dir_name : str (default="test")

        directory containing test data

    prefix_normal : str (default="normal")

        normal directory name

    prefix_anomaly : str (default="anomaly")

        anomaly directory name

    ext : str (default="wav")

        file extension of audio files



    return :

        if the mode is "development":

            test_files : list [ str ]

                file list for test

            test_labels : list [ boolean ]

                label info. list for test

                * normal/anomaly = 0/1

        if the mode is "evaluation":

            test_files : list [ str ]

                file list for test

    """

    logger.info("target_dir : {}".format(Path(target_dir+"_"+id_name).name))



    # development

    if mode:

        normal_files = sorted(

            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,

                                                                                 dir_name=dir_name,

                                                                                 prefix_normal=prefix_normal,

                                                                                 id_name=id_name,

                                                                                 ext=ext)))

        normal_labels = np.zeros(len(normal_files))

        anomaly_files = sorted(

            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,

                                                                                  dir_name=dir_name,

                                                                                  prefix_anomaly=prefix_anomaly,

                                                                                  id_name=id_name,

                                                                                  ext=ext)))

        anomaly_labels = np.ones(len(anomaly_files))

        files = np.concatenate((normal_files, anomaly_files), axis=0)

        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)

        logger.info("test_file  num : {num}".format(num=len(files)))

        if len(files) == 0:

            logger.exception("no_wav_file!!")

        print("\n========================================")



    # evaluation

    else:

        files = sorted(

            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,

                                                                  dir_name=dir_name,

                                                                  id_name=id_name,

                                                                  ext=ext)))

        labels = None

        logger.info("test_file  num : {num}".format(num=len(files)))

        if len(files) == 0:

            logger.exception("no_wav_file!!")

        print("\n=========================================")



    return files, labels

########################################################################





########################################################################

# main 01_test.py

########################################################################

if True:

    # check mode

    # "development": mode == True

    # "evaluation": mode == False

    mode = True



    # make output result directory

    os.makedirs(params.result_directory, exist_ok=True)



    # initialize lines in csv for AUC and pAUC

    csv_lines = []



    # PyTorch version specific...

    to_tensor = ToTensor1ch()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    # loop of the base directory

    for idx, target_dir in enumerate(dirs):

        print("\n===========================")

        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        machine_type = os.path.split(target_dir)[1]



        print("============== MODEL LOAD ==============")

        # set model path

        model_file = "{model}/model_{machine_type}.pth".format(model=params.model_directory,

                                                               machine_type=machine_type)



        # load model file

        if not os.path.exists(model_file):

            logger.error("{} model not found ".format(machine_type))

            sys.exit(-1)

        model = Task2Baseline().to(device)

        load_weights(model, model_file)

        summary(device, model)

        model.eval()



        if mode:

            # results by type

            csv_lines.append([machine_type])

            csv_lines.append(["id", "AUC", "pAUC"])

            performance = []



        machine_id_list = get_machine_id_list_for_test(target_dir)



        for id_str in machine_id_list:

            # load test file

            test_files, y_true = test_file_list_generator(target_dir, id_str)



            # setup anomaly score file path

            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(

                                                                                     result=params.result_directory,

                                                                                     machine_type=machine_type,

                                                                                     id_str=id_str)

            anomaly_score_list = []



            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")

            y_pred = [0. for k in test_files]

            for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):

                x = file_to_vector_array(file_path,

                                         n_mels=params.feature.n_mels,

                                         frames=params.feature.frames,

                                         n_fft=params.feature.n_fft,

                                         hop_length=params.feature.hop_length,

                                         power=params.feature.power)

                with torch.no_grad():

                    yhat = model(to_tensor(x)).cpu().detach().numpy().reshape(x.shape)

                    errors = np.mean(np.square(x - yhat), axis=1)

                    if file_idx in [0, 500]:

                        for i in range(2):

                            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

                            axs[0].plot(x[i])

                            axs[1].plot(yhat[i])

                            plt.show()

                y_pred[file_idx] = np.mean(errors)

                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])



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



    if mode:

        # output results

        result_path = "{result}/{file_name}".format(result=params.result_directory, file_name=params.result_file)

        logger.info("AUC and pAUC results -> {}".format(result_path))

        save_csv(save_file_path=result_path, save_data=csv_lines)
def print_csv_lines(csv_lines):

    for l in csv_lines:

        print('\t\t'.join([(a if type(a) == str else f'{a:.6f}') for a in l]))

        

print_csv_lines(csv_lines)