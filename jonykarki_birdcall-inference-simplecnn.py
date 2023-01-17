import numpy as np

import os

import pandas as pd

import librosa

import librosa.display

import matplotlib.pyplot as plt

import random



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader



import warnings

warnings.filterwarnings("ignore")



from tqdm.auto import tqdm

from pathlib import Path
DATA_DIR = "/kaggle/input/birdsong-recognition"

INPUT_DIR = "/kaggle/input/"

SR = 32000

BATCH_SIZE = 16

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
class SimpleCNNModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)

        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)

        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.maxpool4 = nn.MaxPool2d(2)



        self.fc1 = nn.Linear(in_features=64*5*17, out_features=400)

        self.out = nn.Linear(in_features=400, out_features=264)



    def forward(self, t):

        out = self.maxpool1(self.conv1(t))

        out = F.relu(out)



        out = self.maxpool2(self.conv2(out))

        out = F.relu(out)

        

        out = self.maxpool3(self.conv3(out))

        out = F.relu(out)

        

        out = self.maxpool4(self.conv4(out))

        out = F.relu(out)



        out = out.reshape(-1, 64*5*17)

        out = self.fc1(out)

        return self.out(out)
model = SimpleCNNModel()

checkpoint = torch.load("/kaggle/input/simplecnnmodel/best_model.pt")

model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)

model.to(DEVICE)

# model.load_state_dict(torch.load("/kaggle/input/birdcallcnnck/checkpoint.pt"))

# model.to(DEVICE)

model.eval()
TEST = Path("/kaggle/input/birdsong-recognition/test_audio").exists()



if TEST:

    TEST_DF_DIR = "/kaggle/input/birdsong-recognition/"

else:

    TEST_DF_DIR = "/kaggle/input/birdcall-check/"

    

test_df = pd.read_csv(f"{TEST_DF_DIR}test.csv")

test_audio = TEST_DF_DIR + "test_audio/"

test_df.sample(5)
def get_melspectrogram(y):

    mel_spec = librosa.feature.melspectrogram(y=y, sr=SR)

    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec
class BirdSoundDatasetTest(Dataset):

    def __init__(self, df, clip):

        self.df = df

        self.clip = clip

    

    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        sample = self.df.iloc[idx]

        site = sample.site

        row_id = sample.row_id

        

        if site == "site_3":

            y = self.clip.astype(np.float32)

            len_y = len(y)

            start = 0

            end = SR * 5

            images = []

            while len_y > start:

                y_batch = y[start:end].astype(np.float32)

                if len(y_batch) != (SR * 5):

                    break

                start = end

                end = end + SR * 5

                

                image = get_melspectrogram(y_batch)

                image = np.resize(image, (128, 313))

                images.append(image)

            return images, row_id, site

        else:

            end_seconds = int(sample.seconds)

            start_seconds = int(end_seconds - 5)

            

            start_index = SR * start_seconds

            end_index = SR * end_seconds

            

            y = self.clip[start_index:end_index].astype(np.float32)



            image = get_melspectrogram(y)

            image = np.resize(image, (128, 313))



            return image, row_id, site
prediction_dict = {}

prediction_threshold = 0.6



for audio_id in test_df.audio_id.unique():

    y, sr = librosa.load(test_audio + (audio_id + ".mp3"),

                        sr=SR)

    new_test_df = test_df.query(

        f"audio_id == '{audio_id}'"

    ).reset_index(drop=True)



    dataset = BirdSoundDatasetTest(new_test_df, y)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)



    model.eval()



    for image, row_id, site in loader:

        site = site[0]

        row_id = row_id[0]

        if site in {"site_1", "site_2"}:

            image = image.to(DEVICE).unsqueeze(dim=0)



            with torch.no_grad():

                prediction = model(image)

                

                prediction = F.softmax(prediction, dim=1).cpu().numpy()

                

                events = prediction >= prediction_threshold

                labels = np.argwhere(events).tolist()

                

                if len(labels) == 0:

                    prediction_dict[row_id] = "nocall"

                else:

                    labels = [x[1] for x in labels]



                    birds = set()

                    for bird in labels:

                        birds.add(INV_BIRD_CODE[bird])

                    prediction_dict[row_id] = " ".join(birds)

        else:

            birds = set()

            for img in image:

                img = img.to(DEVICE).unsqueeze(dim=0)

                with torch.no_grad():

                    prediction = model(img)

                    prediction = F.softmax(prediction, dim=1).cpu().numpy()

                    

                    events = prediction >= prediction_threshold

                    labels = np.argwhere(events).tolist()

                    

                    if len(labels) == 0:

                        pass

                    else:

                        labels = [x[1] for x in labels]

                        

                        for bird in labels:

                            birds.add(INV_BIRD_CODE[bird])

            if len(birds) == 0:

                prediction_dict[row_id] = "nocall"

            else:

                prediction_dict[row_id] = " ".join(birds)
row_id = list(prediction_dict.keys())

birds = list(prediction_dict.values())



prediction_df = pd.DataFrame({

    "row_id": row_id,

    "birds": birds

})
prediction_df.tail(25)
prediction_df.to_csv("submission.csv", index=False)