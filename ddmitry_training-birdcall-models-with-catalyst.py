import os

import sys

import random

from pathlib import Path

from typing import List, Tuple, Callable



import numpy as np

import pandas as pd

import cv2

import librosa

import audioread

import soundfile as sf





import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Sampler, Subset

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score, average_precision_score

from sklearn.exceptions import UndefinedMetricWarning

import warnings





warnings.filterwarnings(action="ignore", category=UserWarning)

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

warnings.filterwarnings(action="ignore", category=RuntimeWarning)





sys.path.append("../input/catalyst-git/catalyst")

sys.path.append("../input/efficientnetpytorch")





from catalyst import dl

from efficientnet_pytorch import EfficientNet
ROOT = Path.cwd().parent

INPUT_ROOT = ROOT / "input"

RAW_DATA = INPUT_ROOT / "birdsong-recognition"

TRAIN_AUDIO_DIR = RAW_DATA / "train_audio"

TRAIN_RESAMPLED_AUDIO_DIRS = [

    INPUT_ROOT / "birdsong-resampled-train-audio-{:0>2}".format(i)

    for i in range(5)

]



train = pd.read_csv(TRAIN_RESAMPLED_AUDIO_DIRS[0] / "train_mod.csv")

print("Train shapes:", train.shape)



# global constans

PERIOD = 5 # seconds to crop

SAMPLE_RATE = 32_000

FOLD_IDX = 0

BATCH_SIZE = 48
tmp_list = []

for audio_d in TRAIN_RESAMPLED_AUDIO_DIRS:

    if not audio_d.exists():

        continue

    for ebird_d in audio_d.iterdir():

        if ebird_d.is_file():

            continue

        for wav_f in ebird_d.iterdir():

            tmp_list.append([ebird_d.name, wav_f.name, wav_f.as_posix()])

            

train_wav_path_exist = pd.DataFrame(tmp_list, columns=["ebird_code", "resampled_filename", "file_path"])



del tmp_list



train_all = pd.merge(train, train_wav_path_exist, on=["ebird_code", "resampled_filename"], how="inner")



print(train.shape)

print(train_wav_path_exist.shape)

print(train_all.shape)
fold_method = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)



train_all["fold"] = -1

for fold_id, (train_index, val_index) in enumerate(fold_method.split(train_all, train_all["ebird_code"])):

    train_all.iloc[val_index, -1] = fold_id



train_all["fold"].value_counts()
BIRD_TO_CODE = {

    "aldfly": 0,

    "ameavo": 1,

    "amebit": 2,

    "amecro": 3,

    "amegfi": 4,

    "amekes": 5,

    "amepip": 6,

    "amered": 7,

    "amerob": 8,

    "amewig": 9,

    "amewoo": 10,

    "amtspa": 11,

    "annhum": 12,

    "astfly": 13,

    "baisan": 14,

    "baleag": 15,

    "balori": 16,

    "banswa": 17,

    "barswa": 18,

    "bawwar": 19,

    "belkin1": 20,

    "belspa2": 21,

    "bewwre": 22,

    "bkbcuc": 23,

    "bkbmag1": 24,

    "bkbwar": 25,

    "bkcchi": 26,

    "bkchum": 27,

    "bkhgro": 28,

    "bkpwar": 29,

    "bktspa": 30,

    "blkpho": 31,

    "blugrb1": 32,

    "blujay": 33,

    "bnhcow": 34,

    "boboli": 35,

    "bongul": 36,

    "brdowl": 37,

    "brebla": 38,

    "brespa": 39,

    "brncre": 40,

    "brnthr": 41,

    "brthum": 42,

    "brwhaw": 43,

    "btbwar": 44,

    "btnwar": 45,

    "btywar": 46,

    "buffle": 47,

    "buggna": 48,

    "buhvir": 49,

    "bulori": 50,

    "bushti": 51,

    "buwtea": 52,

    "buwwar": 53,

    "cacwre": 54,

    "calgul": 55,

    "calqua": 56,

    "camwar": 57,

    "cangoo": 58,

    "canwar": 59,

    "canwre": 60,

    "carwre": 61,

    "casfin": 62,

    "caster1": 63,

    "casvir": 64,

    "cedwax": 65,

    "chispa": 66,

    "chiswi": 67,

    "chswar": 68,

    "chukar": 69,

    "clanut": 70,

    "cliswa": 71,

    "comgol": 72,

    "comgra": 73,

    "comloo": 74,

    "commer": 75,

    "comnig": 76,

    "comrav": 77,

    "comred": 78,

    "comter": 79,

    "comyel": 80,

    "coohaw": 81,

    "coshum": 82,

    "cowscj1": 83,

    "daejun": 84,

    "doccor": 85,

    "dowwoo": 86,

    "dusfly": 87,

    "eargre": 88,

    "easblu": 89,

    "easkin": 90,

    "easmea": 91,

    "easpho": 92,

    "eastow": 93,

    "eawpew": 94,

    "eucdov": 95,

    "eursta": 96,

    "evegro": 97,

    "fiespa": 98,

    "fiscro": 99,

    "foxspa": 100,

    "gadwal": 101,

    "gcrfin": 102,

    "gnttow": 103,

    "gnwtea": 104,

    "gockin": 105,

    "gocspa": 106,

    "goleag": 107,

    "grbher3": 108,

    "grcfly": 109,

    "greegr": 110,

    "greroa": 111,

    "greyel": 112,

    "grhowl": 113,

    "grnher": 114,

    "grtgra": 115,

    "grycat": 116,

    "gryfly": 117,

    "haiwoo": 118,

    "hamfly": 119,

    "hergul": 120,

    "herthr": 121,

    "hoomer": 122,

    "hoowar": 123,

    "horgre": 124,

    "horlar": 125,

    "houfin": 126,

    "houspa": 127,

    "houwre": 128,

    "indbun": 129,

    "juntit1": 130,

    "killde": 131,

    "labwoo": 132,

    "larspa": 133,

    "lazbun": 134,

    "leabit": 135,

    "leafly": 136,

    "leasan": 137,

    "lecthr": 138,

    "lesgol": 139,

    "lesnig": 140,

    "lesyel": 141,

    "lewwoo": 142,

    "linspa": 143,

    "lobcur": 144,

    "lobdow": 145,

    "logshr": 146,

    "lotduc": 147,

    "louwat": 148,

    "macwar": 149,

    "magwar": 150,

    "mallar3": 151,

    "marwre": 152,

    "merlin": 153,

    "moublu": 154,

    "mouchi": 155,

    "moudov": 156,

    "norcar": 157,

    "norfli": 158,

    "norhar2": 159,

    "normoc": 160,

    "norpar": 161,

    "norpin": 162,

    "norsho": 163,

    "norwat": 164,

    "nrwswa": 165,

    "nutwoo": 166,

    "olsfly": 167,

    "orcwar": 168,

    "osprey": 169,

    "ovenbi1": 170,

    "palwar": 171,

    "pasfly": 172,

    "pecsan": 173,

    "perfal": 174,

    "phaino": 175,

    "pibgre": 176,

    "pilwoo": 177,

    "pingro": 178,

    "pinjay": 179,

    "pinsis": 180,

    "pinwar": 181,

    "plsvir": 182,

    "prawar": 183,

    "purfin": 184,

    "pygnut": 185,

    "rebmer": 186,

    "rebnut": 187,

    "rebsap": 188,

    "rebwoo": 189,

    "redcro": 190,

    "redhea": 191,

    "reevir1": 192,

    "renpha": 193,

    "reshaw": 194,

    "rethaw": 195,

    "rewbla": 196,

    "ribgul": 197,

    "rinduc": 198,

    "robgro": 199,

    "rocpig": 200,

    "rocwre": 201,

    "rthhum": 202,

    "ruckin": 203,

    "rudduc": 204,

    "rufgro": 205,

    "rufhum": 206,

    "rusbla": 207,

    "sagspa1": 208,

    "sagthr": 209,

    "savspa": 210,

    "saypho": 211,

    "scatan": 212,

    "scoori": 213,

    "semplo": 214,

    "semsan": 215,

    "sheowl": 216,

    "shshaw": 217,

    "snobun": 218,

    "snogoo": 219,

    "solsan": 220,

    "sonspa": 221,

    "sora": 222,

    "sposan": 223,

    "spotow": 224,

    "stejay": 225,

    "swahaw": 226,

    "swaspa": 227,

    "swathr": 228,

    "treswa": 229,

    "truswa": 230,

    "tuftit": 231,

    "tunswa": 232,

    "veery": 233,

    "vesspa": 234,

    "vigswa": 235,

    "warvir": 236,

    "wesblu": 237,

    "wesgre": 238,

    "weskin": 239,

    "wesmea": 240,

    "wessan": 241,

    "westan": 242,

    "wewpew": 243,

    "whbnut": 244,

    "whcspa": 245,

    "whfibi": 246,

    "whtspa": 247,

    "whtswi": 248,

    "wilfly": 249,

    "wilsni1": 250,

    "wiltur": 251,

    "winwre3": 252,

    "wlswar": 253,

    "wooduc": 254,

    "wooscj2": 255,

    "woothr": 256,

    "y00475": 257,

    "yebfly": 258,

    "yebsap": 259,

    "yehbla": 260,

    "yelwar": 261,

    "yerwar": 262,

    "yetvir": 263,

}

CODE_TO_BIRD = {v: k for k, v in BIRD_TO_CODE.items()}

NUM_BIRDS = len(BIRD_TO_CODE)
def mono_to_color(

    spect: np.ndarray,

    mean: float = None,

    std: float = None,

    norm_max: float = None,

    norm_min: float = None,

    eps: float = 1e-6,

) -> np.ndarray:

    """Convert single channel spectrogram to image like.



    Args:

        spect (np.ndarray): spectrogram

        mean ([type], optional): mean to use for scaling.

            If `None` then `spect` will be used for computations of mean value.

            Default is `None`.

        std ([type], optional): standart deviation to use for scaling.

            If `None` then `spect` will be used for computations of std value.

            Default is `None`.

        norm_max ([type], optional): maximum value to use in `spect`,

            values higher than this value will be replaced with this value.

            If `None` then maximum value will be took from `spect`.

            Default is `None`.

        norm_min ([type], optional): minimum value to use in `spect`,

            values lower than this value will be replaced with this value.

            If `None` then minimum value will be took from `spect`.

            Default is `None`.

        eps ([type], optional): difference level beetween min and max value in

            normalized values to use for replacing with `norm_min`/`norm_max` values.

            Default is `1e-6`.



    Returns:

        image like spectrogram.

    """

    spect = np.stack([spect, spect, spect], axis=-1)



    # standardize

    mean = mean or spect.mean()

    spect = spect - mean



    std = std or spect.std()

    spect_std = spect / (std + eps)



    _min, _max = spect_std.min(), spect_std.max()

    norm_max = norm_max or _max

    norm_min = norm_min or _min



    if (_max - _min) > eps:

        # normalize to [0, 255]

        res = spect_std

        res[res < norm_min] = norm_min

        res[res > norm_max] = norm_max

        res = 255 * (res - norm_min) / (norm_max - norm_min)

        res = res.astype(np.uint8)

    else:

        res = np.zeros_like(spect_std, dtype=np.uint8)

    return res
def pad_waveform(waveform: np.ndarray, effective_length: int) -> np.ndarray:

    """Pad waveform (if length is less than :arg:`effective_length`)

    with some random part of a waveform.



    Args:

        waveform (np.ndarray): raw waveform of a signal

        effective_length (int): length of a segment to crop



    Returns:

        np.ndarray: padded waveform

    """

    len_y = len(waveform)

    if len_y < effective_length:

        new_y = np.zeros(effective_length, dtype=waveform.dtype)

        start = np.random.randint(effective_length - len_y)

        new_y[start : start + len_y] = waveform

        waveform = new_y

    return waveform.astype(np.float32)
def random_crop(waveform: np.ndarray, effective_length: int) -> np.ndarray:

    """Perform random crop from waveform.



    Args:

        waveform (np.ndarray): raw waveform of a signal

        effective_length (int): length of a segment to crop



    Returns:

        np.ndarray: cropped part of a signal

    """

    len_y = len(waveform)

    if len_y < effective_length:

        waveform = pad_waveform(waveform, effective_length)

    elif len_y > effective_length:

        start = np.random.randint(len_y - effective_length)

        waveform = waveform[start : start + effective_length].astype(np.float32)

    else:

        waveform = waveform.astype(np.float32)

    return waveform
class SpectrogramDataset(Dataset):

    def __init__(

        self,

        file_list: List[List[str]],

        img_size: int = 224,

        waveform_transforms: Callable = None,

        spectrogram_transforms: Callable = None,

        melspectrogram_parameters: dict = {},

    ):

        self.file_list = file_list  # list of list: [file_path, ebird_code]

        self.img_size = img_size

        self.waveform_transforms = waveform_transforms

        self.spectrogram_transforms = spectrogram_transforms

        self.melspectrogram_parameters = melspectrogram_parameters



    def __len__(self):

        return len(self.file_list)



    @staticmethod

    def bird_code_to_label(bird: str) -> np.ndarray:

        labels = np.zeros(NUM_BIRDS, dtype="f")

        labels[BIRD_TO_CODE[bird]] = 1

        return labels



    def __getitem__(self, idx: int):

        wav_path, ebird_code = self.file_list[idx]



        waveform, sr = sf.read(wav_path)

        labels = self.bird_code_to_label(ebird_code)



        if self.waveform_transforms:

            waveform = self.waveform_transforms(waveform)

            

        waveform = random_crop(waveform, sr * PERIOD)



        melspec = librosa.feature.melspectrogram(

            waveform, sr=sr, **self.melspectrogram_parameters

        )

        melspec = librosa.power_to_db(melspec).astype(np.float32)



        if self.spectrogram_transforms:

            melspec = self.spectrogram_transforms(melspec)

        else:

            pass



        image = mono_to_color(melspec)

        height, width, _ = image.shape

        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))

        image = np.moveaxis(image, 2, 0)

        image = (image / 255.0).astype(np.float32)



        return image, labels
class MultilabelBalancedSampler(Sampler):

    def __init__(self, targets, classes_num):



        self.targets = targets

        self.classes_num = classes_num



        self.samples_num_per_class = np.sum(self.targets, axis=0)

        self.max_num = np.max(self.samples_num_per_class)



        self.indexes_per_class = []

        # Training indexes of all sound classes. E.g.:

        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]

        for k in range(self.classes_num):

            self.indexes_per_class.append(np.where(self.targets[:, k] == 1)[0])



        self.length = self.classes_num * self.max_num



    def __iter__(self):

        all_indexs = []



        for k in range(self.classes_num):

            if len(self.indexes_per_class[k]) == self.max_num:

                all_indexs.append(self.indexes_per_class[k])

            else:

                gap = self.max_num - len(self.indexes_per_class[k])

                random_choice = np.random.choice(

                    self.indexes_per_class[k], int(gap), replace=True

                )

                all_indexs.append(

                    np.array(list(random_choice) + list(self.indexes_per_class[k]))

                )



        l = np.stack(all_indexs).T

        l = l.reshape(-1)

        random.shuffle(l)

        return iter(l)



    def __len__(self):

        return int(self.length)
class MulticlassEfficientNet(nn.Module):

    """Multiclass efficientnet with pretrains"""



    def __init__(self, pretrain: str, n_classes: int = 1):

        """

        Args:

            pretrain (str): one of:

                - 'efficientnet-b0'

                - 'efficientnet-b1'

                - 'efficientnet-b2'

                - 'efficientnet-b3'

                - 'efficientnet-b4'

                - 'efficientnet-b5'

                - 'efficientnet-b6'

                - 'efficientnet-b7'

            n_classes (int, optional): number of classes to use,

                default is `1`

        """

        super().__init__()

        self.backbone = EfficientNet.from_name(pretrain)

        self.backbone._fc = nn.Linear(self.backbone._fc.in_features, n_classes)



    def forward(self, batch):

        return self.backbone(batch)
def sigmoid(x):

    return 1 / (1 + np.exp(-x))





def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=0)
class F1AverageMetric(dl.Callback):

    def __init__(

        self,

        metric_name: str = "f1",

        input_key: str = "targets",

        output_key: str = "logits",

        threshold: float = 0.5,

        average: str = "micro",

        activation: str = None,

    ):

        super().__init__(dl.CallbackOrder.Metric)

        self.name = metric_name

        self.inp = input_key

        self.outp = output_key

        self.target_container = None

        self.preds_container = None

        self.threshold = threshold

        self.average = average

        if activation == "sigmoid":

            self.activation = sigmoid

        elif activation == "softmax":

            self.activation = softmax

        else:

            self.activation = lambda item: item



    def on_loader_start(self, state: dl.IRunner) -> None:

        self.target_container = []

        self.preds_container = []

        

    def on_batch_end(self, state: dl.IRunner) -> None:

        # collect scores

        target = state.input[self.inp].detach().cpu().numpy()

        self.target_container.append(target)



        pred = state.output[self.outp].detach().cpu().numpy()

        self.preds_container.append(pred)

        

    def on_loader_end(self, state: dl.IRunner) -> None:

        y_pred = np.concatenate(self.preds_container, axis=0)

        y_pred = self.activation(y_pred)

        y_pred = np.where(y_pred > self.threshold, 1, 0)

        

        y_true = np.concatenate(self.target_container, axis=0)

        score = f1_score(y_true, y_pred, average=self.average)

        

        state.loader_metrics[self.name] = score

        # free memory

        self.target_container = None

        self.preds_container = None
train_df = train_all[train_all["fold"] != FOLD_IDX][["file_path", "ebird_code"]]

valid_df = train_all[train_all["fold"] == FOLD_IDX][["file_path", "ebird_code"]]



train_targets = np.zeros((len(train_df), NUM_BIRDS), dtype="f")

for idx, ebird_code in enumerate(train_df["ebird_code"].values):

    train_targets[idx, BIRD_TO_CODE[ebird_code]] = 1



trainset = SpectrogramDataset(

    train_df.values.tolist(),

    img_size=224,

)

# trainset = Subset(trainset, list(range(300)))

print("Train records:", len(trainset))



validset = SpectrogramDataset(

    valid_df.values.tolist(),

    img_size=224,

)

# validset = Subset(validset, list(range(300)))

print("Valid records:", len(validset))





def valid_worker_init_fn(*args, **kwargs):

    random.seed(42)

    np.random.seed(42)





loaders = {

    "train": DataLoader(

        trainset, batch_size=BATCH_SIZE, num_workers=2,

#         sampler=MultilabelBalancedSampler(train_targets, NUM_BIRDS)

    ),

    "valid": DataLoader(

        validset, batch_size=BATCH_SIZE * 2, num_workers=2,

        worker_init_fn=valid_worker_init_fn

    ),

}

print("Batches in train:", len(loaders["train"]))

print("Batches in valid:", len(loaders["valid"]))
# experiment definitions

logdir = "."

num_epochs = 10

model = MulticlassEfficientNet("efficientnet-b1", NUM_BIRDS)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

criterion = nn.BCEWithLogitsLoss()

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)



# training

runner = dl.SupervisedRunner()

runner.train(

    logdir=logdir,

    model=model,

    num_epochs=num_epochs,

    criterion=criterion,

    optimizer=optimizer,

    scheduler=scheduler,

    loaders=loaders,

    callbacks=[

        F1AverageMetric(activation="sigmoid"),

        dl.CheckRunCallback(num_batch_steps=5, num_epoch_steps=num_epochs) # REMOVE THIS FOR FULL TRAINING

    ],

    main_metric="f1",

    minimize_metric=False,

    verbose=True,

)