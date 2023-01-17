import os

import sys

import pathlib 

import time

import warnings

import multiprocessing

from timeit import default_timer as timer



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import cv2

import librosa



import tensorflow.keras as keras
start_timer = timer()

warnings.filterwarnings('ignore')
IMAGE_SIZE = 224

SR  = 12000 #sampling rate

LEN = 5 # 5 sec window

prob_thr = 0.045

MELSPECTRUM = {

    'n_mels': 128,

    'fmin'  :  20,

    'fmax'  : 16000,

}
model_tar = "../input/birdsong/best_model.tar"

model_tar = "../input/birdsong/mel_reg_0.68512_best_model.ckp/"
!tar -xvf $model_tar
model = keras.models.load_model(model_tar)

model.summary()
print('\tcpu_count = %d' % multiprocessing.cpu_count())

print('\tram = %d MB' % (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') /(1024.**3)))

print('')
# https://www.kaggle.com/shonenkov/sample-submission-using-custom-check

# https://www.kaggle.com/c/birdsong-recognition/discussion/159993



if 0: #local

    ADD_DIR    = os.path.dirname(__file__)

    checkpoint = '/root/share1/kaggle/2020/birdsong/result/reference/resnet50/best_model.pth'

    TEST_CSV   = '/root/share1/kaggle/2020/birdsong/data/other/sample_test/test.csv'

    TEST_AUDIO_DIR = '/root/share1/kaggle/2020/birdsong/data/other/sample_test/audio'



if 1:# kaggle

   

    ADD_DIR  = '../input/bird00'

    TEST_AUDIO_DIR = '../input/birdsong-recognition/test_audio'

    TEST_CSV = '../input/birdsong-recognition/test.csv'



    if not os.path.exists('../input/birdsong-recognition/test_audio'):

        print(TEST_AUDIO_DIR, "not exist",)

        TEST_AUDIO_DIR = '../input/birdcall-check/test_audio'

        TEST_CSV = '../input/birdcall-check/test.csv'



#-------------------------

sys.path.append(ADD_DIR)

print('sys.path.append(ADD_DIR) OK!')
NAME_TO_LABEL = {

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

LABEL_TO_NAME = {v: k for k, v in NAME_TO_LABEL.items()}

def make_batch(wave, second):

    Xs = []

    for s in second:

        t0 = (s-5)*SR

        t1 = s*SR

        x = wave[t0:t1]

        Xs.append(x)

    X = np.expand_dims(np.stack(Xs), 2)

    return X
def time_to_str(t, mode='min'):

    if mode=='min':

        t  = int(t)/60

        hr = t//60

        min = t%60

        return '%2d hr %02d min'%(hr,min)



    elif mode=='sec':

        t   = int(t)

        min = t//60

        sec = t%60

        return '%2d min %02d sec'%(min,sec)



    else:

        raise NotImplementedError



#------------------------------

def melspec_norm_value(m):

    eps = 1e-6

    mean = m.mean()

    std  = m.std()

    m = (m-mean) / (std + eps)

    min, max = m.min(), m.max()

    if (max - min) > eps:

        m = (m - min) / (max - min)

    else:

        m = np.zeros_like(m)

    return m





def melspec_norm_size(m):

    height, width = m.shape

    m = cv2.resize(m, dsize=(int(width * IMAGE_SIZE / height), IMAGE_SIZE))

    return m



def calc_mel_spectrum(wave_arr):

    m = librosa.feature.melspectrogram(wave_arr, sr=SR, **MELSPECTRUM)

    m = librosa.power_to_db(m)

    m = m.astype(np.float32)

    m = melspec_norm_value(m)

    m = melspec_norm_size(m)

    return m
df_submit = pd.DataFrame(columns=('row_id','birds'))

df_test = pd.read_csv(TEST_CSV)



probabilities = []

for audio_id in df_test.audio_id.unique():

    df = df_test[df_test.audio_id == audio_id].reset_index(drop=True).sort_values('seconds')

    wave, _ = librosa.load( TEST_AUDIO_DIR + '/%s.mp3'%audio_id, sr=SR, mono=True)

    wave = wave.astype(np.float32)



    L = len(wave)

    site = df.site.values[0]

    if site == 'site_3':

        second = (np.arange(L//(SR*LEN))+1)*5

    else:

        second = df.seconds.values.astype(np.int32)



    print(audio_id, site, time_to_str(timer() - start_timer, 'min'))

    print('\tlen = %0.2f, num_sec = %d '%(L/SR, len(second)), second[:5], '...',)

    X_test = make_batch(wave, second)

    print("")

    

    probability = []

    L = len(X_test)

    batch_size = 16



    for m in np.array_split(X_test, int(np.ceil(L/batch_size))):

        #print('\tmelspec:', m.shape, '%0.2f mb'%(m.nbytes/1024/1024))

        m=np.apply_along_axis(calc_mel_spectrum, arr=np.squeeze(m,2), axis=1)

        

        #print('\tmelspec:', m.shape, '%0.2f mb'%(m.nbytes/1024/1024))



        p = model.predict(m)

        probability.append(p)



    probability = np.concatenate(probability)

    probabilities.append(probability)

    

    predict = probability>=prob_thr #0.65

    if site == 'site_3':

        predict = predict.max(0, keepdims=True)

        print('\tpredict site 3:', predict.shape)



    if audio_id=='41e6fe6504a34bf6846938ba78d13df1' or audio_id=='07ab324c602e4afab65ddbcc746c31b5': #debug

        print(probability.reshape(-1)[:50], '\n')



    for b,row_id in enumerate(df.row_id.values):

        bird = np.where(predict[b])[0]

        if len(bird)==0:

            bird = 'nocall'

        else:

            bird = list(map(lambda i: LABEL_TO_NAME[i], bird))

            bird = ' '.join(bird)



        df_submit = df_submit.append({'row_id': row_id, 'birds': bird}, ignore_index=True)

    print('')

    

    #break
#debuging



hist = np.histogram(np.concatenate(probabilities).reshape(-1), bins=1240)

hist[0][-60:], hist[1][-60:]
#debuging



plt.hist(np.concatenate(probabilities).reshape(-1), bins=200);

plt.yscale("log")

plt.vlines([prob_thr],0,100)
#debuging



X_test.shape
#debuging



model.predict(m).shape
df_submit.to_csv('submission.csv', index=False)

print('submission.csv')

print(df_submit)
df_submit.birds.value_counts()
# df = pd.read_csv("../input/birdsong-recognition/example_test_audio_metadata.csv") # for validation

# df = pd.read_csv("../input/birdsong-recognition/example_test_audio_summary.csv") # for validation

df = pd.read_csv("../input/birdcall-check/test.csv") # for validation
df