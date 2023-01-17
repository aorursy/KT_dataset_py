import torchaudio

import matplotlib.pyplot as plt



waveform, sample_rate = torchaudio.load("../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3")



plt.plot(waveform.t().numpy())

plt.xlabel("time")

plt.ylabel("signal");
mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=2**11, f_max=8000)(waveform)

mel_specgram= torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_specgram)



plt.figure()

plt.title("exsample mel Spectrogram")

plt.imshow(mel_specgram[0].detach().numpy()[::-1], cmap='magma',aspect=5);

plt.xlabel("time")

plt.ylabel("mel scale");
import logging

import os

import random

import time

import warnings

from typing import Optional

from fastprogress import progress_bar

from contextlib import contextmanager



import torch

import torchaudio

import numpy as np

import pandas as pd

from tqdm import tqdm

from pathlib import Path



from sklearn.model_selection import train_test_split



import torch.nn as nn

import torch.nn.functional as F

from torch.optim import Adam

from torch.distributions import Uniform

from torch.utils.data import DataLoader, Dataset



from torchaudio.transforms import Spectrogram, MelSpectrogram

from torchaudio.transforms import TimeStretch, AmplitudeToDB, ComplexNorm, Resample

from torchaudio.transforms import FrequencyMasking, TimeMasking
def set_seed(seed: int = 42):

    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)  # type: ignore

    torch.backends.cudnn.deterministic = True  # type: ignore

    torch.backends.cudnn.benchmark = True  # type: ignore

    

    

def get_logger(out_file=None):

    logger = logging.getLogger()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    logger.handlers = []

    logger.setLevel(logging.INFO)



    handler = logging.StreamHandler()

    handler.setFormatter(formatter)

    handler.setLevel(logging.INFO)

    logger.addHandler(handler)



    if out_file is not None:

        fh = logging.FileHandler(out_file)

        fh.setFormatter(formatter)

        fh.setLevel(logging.INFO)

        logger.addHandler(fh)

    logger.info("logger set up")

    return logger



@contextmanager

def timer(name: str, logger: Optional[logging.Logger] = None):

    t0 = time.time()

    msg = f"[{name}] start"

    if logger is None:

        print(msg)

    else:

        logger.info(msg)

    yield



    msg = f"[{name}] done in {time.time() - t0:.2f} s"

    if logger is None:

        print(msg)

    else:

        logger.info(msg)



logger = get_logger("main.log");

set_seed(1213);
TARGET_SR = 32000

TEST = Path("../input/birdsong-recognition/test_audio").exists()



if TEST:

    DATA_DIR = Path("../input/birdsong-recognition/")

else:

    # dataset created by @shonenkov, thanks!

    DATA_DIR = Path("../input/birdcall-check/")





test = pd.read_csv(DATA_DIR / "test.csv")

test_audio = DATA_DIR / "test_audio"



MODE_DIR = Path("../input/birdcallfirstmodelcnn/")
#sub = pd.read_csv("../input/birdsong-recognition/sample_submission.csv")

#sub.to_csv("submission.csv", index=False)  # this will be overwritten if everything goes well
BIRD_CODE = {

    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,

    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,

    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,

    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,

    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,

    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,

    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,

    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,

    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,

    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,

    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,

    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,

    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,

    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,

    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,

    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,

    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,

    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,

    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,

    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,

    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,

    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,

    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,

    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,

    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,

    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,

    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,

    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,

    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,

    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,

    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,

    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,

    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,

    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,

    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,

    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,

    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,

    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,

    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,

    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,

    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,

    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,

    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,

    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,

    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,

    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,

    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,

    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,

    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,

    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,

    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,

    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,

    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263

}



INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}
class RondomStretchMelSpectrogram(nn.Module):

    def __init__(self, sample_rate, n_fft, top_db, max_perc):

        super().__init__()

        self.time_stretch = TimeStretch(hop_length=None, n_freq=n_fft//2+1)

        self.stft = Spectrogram(n_fft=n_fft, power=None)

        self.com_norm = ComplexNorm(power=2.)

        self.mel_specgram = MelSpectrogram(sample_rate, n_fft=n_fft, f_max=8000)

        self.AtoDB= AmplitudeToDB(top_db=top_db)

    

    def forward(self, x):

        x = self.stft(x)

        x = self.com_norm(x)

        x = self.mel_specgram.mel_scale(x)

        x = self.AtoDB(x)



        return x
class cnn_audio(nn.Module):

    def __init__(self, 

                 output_class=264,

                 d_size=256,

                 sample_rate=32000, 

                 n_fft=2**11, 

                 top_db=80,

                 max_perc=0.4):

        

        super().__init__()

        self.mel = RondomStretchMelSpectrogram(sample_rate, n_fft, top_db, max_perc)



        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=(1, 1))

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(0.1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout = nn.Dropout(0.1)



        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1))

        self.bn2 = nn.BatchNorm2d(128)

        self.relu2 = nn.ReLU(0.1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout2 = nn.Dropout(0.1)

        

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1))

        self.bn3 = nn.BatchNorm2d(256)

        self.relu3 = nn.ReLU(0.1)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.dropout3 = nn.Dropout(0.1)

        

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=(1, 1))

        self.bn4 = nn.BatchNorm2d(512)

        self.relu4 = nn.ReLU(0.1)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.dropout4 = nn.Dropout(0.1)

        

        self.lstm = nn.LSTM(6, 512, 2, batch_first=True)

        self.dropout_lstm = nn.Dropout(0.3)

        self.bn_lstm = nn.BatchNorm1d(512)

        

        self.output = nn.Linear(512, output_class)

    

    def forward(self, x):

        x = self.mel(x)

        #x = self.norm_db(x)

        

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x = self.dropout(x)

        

        x = self.conv2(x)

        x = self.bn2(x)

        x = self.relu2(x)

        x = self.maxpool2(x)

        x = self.dropout2(x)

        

        x = self.conv3(x)

        x = self.bn3(x)

        x = self.relu3(x)

        x = self.maxpool3(x)

        x = self.dropout3(x)

        

        x = self.conv4(x)

        x = self.bn4(x)

        x = self.relu4(x)

        x = self.maxpool4(x)

        x = self.dropout4(x)

        

        x, _ = self.lstm(x.view(x.size(0), 512, 6), None)

        x = self.dropout_lstm(x[:, -1, :])

        x = self.bn_lstm(x)

        

        x = x.view(-1, 512)

        x = self.output(x)

        

        return x
class TestDataset(Dataset):

    def __init__(self, df: pd.DataFrame, clip):

        self.df = df

        self.clip = clip

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx: int):

        SR = 32000

        sample = self.df.loc[idx, :]

        site = sample.site

        row_id = sample.row_id

        

        if site == "site_3":

            len_y = self.clip.size()[1]

            start = 0

            end = SR * 5

            waveforms = []

            while len_y > start:

                waveform = self.clip[:, start:end]

                if waveform.size(1) != (SR * 5):

                    break

                start = end

                end = end + SR * 5

                

                waveforms.append(waveform.numpy())

            waveforms = torch.tensor(waveforms)

            return waveforms, row_id, site

        else:

            end_seconds = int(sample.seconds)

            start_seconds = int(end_seconds - 5)

            

            start_index = SR * start_seconds

            end_index = SR * end_seconds

            

            waveform = self.clip[:, start_index:end_index]



            return waveform, row_id, site
def predicter(test_df: pd.DataFrame, 

                        clip: np.ndarray, 

                        model,

                        threshold=0.5):

    

    dataset = TestDataset(df=test_df, 

                          clip=clip)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    model = model.to(device)

    model.eval()

    prediction_dict = {}

    for waveform, row_id, site in progress_bar(loader):

        site = site[0]

        row_id = row_id[0]

        if site in {"site_1", "site_2"}:

            waveform = waveform.to(device)



            with torch.no_grad():

                prediction = model(waveform)

                prediction = prediction.detach().cpu().numpy().reshape(-1)

                

            event = prediction> threshold

            labels = np.argwhere(event).reshape(-1).tolist()

                

        else:

            # to avoid prediction on large batch

            waveform = waveform.squeeze(0)

            batch_size = 16

            whole_size = waveform.size()[0]

            if whole_size % batch_size == 0:

                n_iter = whole_size // batch_size

            else:

                n_iter = whole_size // batch_size + 1

                

            all_events = set()

            for batch_i in range(n_iter):

                batch = waveform[batch_i * batch_size:(batch_i + 1) * batch_size, :, :]



                batch = batch.to(device)

                with torch.no_grad():

                    prediction = model(batch)

                    proba = prediction.detach().cpu().numpy()

                    

                global g

                g = proba

                

                events = proba >= threshold

                for i in range(len(events)):

                    event = events[i, :]

                    labels = np.argwhere(event).reshape(-1).tolist()

                    for label in labels:

                        all_events.add(label)

                        

            labels = list(all_events)

        if len(labels) == 0:

            prediction_dict[row_id] = "nocall"

        else:

            labels_str_list = list(map(lambda x: INV_BIRD_CODE[x], labels))

            label_string = " ".join(labels_str_list)

            prediction_dict[row_id] = label_string

    return prediction_dict

def prediction(test_df: pd.DataFrame,

               test_audio: Path,

               model,

               threshold=0.5):

    

    unique_audio_id = test_df.audio_id.unique()



    prediction_dfs = []

    for audio_id in unique_audio_id:

        clip, _ = torchaudio.load(test_audio / (audio_id + ".mp3"), normalization=True)

        test_df_for_audio_id = test_df.query(

            f"audio_id == '{audio_id}'").reset_index(drop=True)

        with timer(f"Prediction on {audio_id}", logger):

            prediction_dict = predicter(test_df_for_audio_id,

                                                  clip=clip[0].unsqueeze(0),

                                                  model=model,

                                                  threshold=threshold)

            

        row_id = list(prediction_dict.keys())

        birds = list(prediction_dict.values())

        prediction_df = pd.DataFrame({

            "row_id": row_id,

            "birds": birds

        })

        prediction_dfs.append(prediction_df)

    

    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)

    return prediction_df

model = cnn_audio()

checkpoint = torch.load(MODE_DIR / "crnn_2o.model")

model.load_state_dict(checkpoint['model_state_dict'])



submission = prediction(test_df=test,

                        test_audio=test_audio,

                        model=model,

                        threshold=0.0)

submission.to_csv("submission.csv", index=False)
submission