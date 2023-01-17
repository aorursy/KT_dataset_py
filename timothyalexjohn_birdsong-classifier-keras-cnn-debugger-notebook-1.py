import tensorflow.keras as tk

import librosa         # Audio Manipulation Library

import librosa.display

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import cv2
row_id = []



def create_spectrogram(df, start_time, duration, mode):

        

    if mode== 'exa_test':                  

        file = '_'.join(df[0].split('_')[:-1])

        if file=='BLKFR-10-CPL_20190611_093000':

            filepath = (exa_test_dir +file +'.pt540.mp3')

        if file=='ORANGE-7-CAP_20190606_093000':

            filepath = (exa_test_dir +file +'.pt623.mp3')

    

    try:

        fig = plt.figure(figsize=[2.7,2.7])

        filename = filepath.split('/')[-1].split('.')[0]

        clip, sample_rate = librosa.load(filepath, sr=None, offset=start_time, duration=duration)

        S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)

        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))



        if mode== 'exa_test':

            if not os.path.exists(exa_test_filesave_dir):

                os.makedirs(exa_test_filesave_dir)

            plt.savefig((exa_test_filesave_dir + df[0] +'.jpg'), bbox_inches='tight',pad_inches=0, facecolor='black')

            

            row_id.append(df[0])

            

        fig.clear()          

        plt.close(fig)

        plt.close()

        plt.close('all')     # These Lines are Very Important!! If not given, Server will run out of allocated Memory

        plt.cla()

        fig.clf()

        plt.clf()

        plt.close()

    

    except:

        print("found a broken Audio File")    
# Getting Rid of Un-Wanted Warnings when Loading Audio Files

import warnings



warnings.filterwarnings("ignore")
# Trying on example_test_audio



exa_test_dir = '../input/birdsong-recognition/example_test_audio/'

exa_test_csv_dir = '../input/birdsong-recognition/example_test_audio_summary.csv'

exa_test_df = pd.read_csv(exa_test_csv_dir)



exa_test_filesave_dir = '/kaggle/working/' 

 

for row in exa_test_df.values:

        start_time = row[3] - 5

        create_spectrogram(row, start_time, duration= 5, mode= 'exa_test')
model = tk.models.load_model('../input/xception-birdcall-model')    # use your CNN model
def predict(path):

    img = cv2.imread(str(path))

    img = cv2.resize(img, (150,150))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    img = np.reshape(img,(1,150,150,3))

    return model.predict(img)



target = []



for img in row_id:

    target.append(predict(exa_test_filesave_dir +img +'.jpg'))

print(target[0].shape)
# DICTIONARY EXTRACTED FROM  "train_generator.class_indices"  from Part_2



bird_dict = {'nocall': 264, 'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4, 'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9, 'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14, 'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19, 'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24, 'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29, 'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34, 'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39, 'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44, 'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49, 'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54, 'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59, 'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64, 'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69, 'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74, 'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79, 'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84, 'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89, 'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94, 'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99, 'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104, 'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109, 'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114, 'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119, 'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124, 'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129, 'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134, 'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139, 'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144, 'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149, 'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154, 'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159, 'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164, 'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169, 'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174, 'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179, 'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184, 'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189, 'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194, 'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199, 'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204, 'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209, 'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214, 'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219, 'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224, 'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229, 'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234, 'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239, 'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244, 'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249, 'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254, 'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259, 'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263}



rev_bird_dict = {value : key for (key, value) in bird_dict.items()}





pred_class = []

for row in target:

    if row.max()< 0.45:

        pred_class.append('nocall')

    else:

        pred_class.append(rev_bird_dict[np.argmax(row, axis=1)[0]])



predict_df = pd.DataFrame(row_id)

predict_df.columns = ['row_id']

predict_df['birds'] = pred_class
predict_df.to_csv('submission.csv', index=False)

print(predict_df)