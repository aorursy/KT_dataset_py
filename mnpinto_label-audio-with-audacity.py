import librosa

import librosa.display

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

import numpy as np

import pandas as pd
def txt2df(file):

    "Read txt file and returns labels dataframe"

    with open(file) as f:

        labels = f.read()

        labels = [l.split('\t') for l in labels.split('\n')][:-1]

        labels = [l for o in labels for l in o]

        labels = [l for l in labels]

        labels = [l for l in labels if l != '\\']

        labels = np.array(labels).reshape(-1, 5)

        df = pd.DataFrame({'start': labels[:,0].astype(float), 'end': labels[:,1].astype(float), 

                           'fmin': labels[:,3].astype(float), 'fmax': labels[:,4].astype(float),

                           'name': labels[:,2].astype(str)})

    return df



def rectangle(df_row, frequency):

    start, end, fmin, fmax = [df_row[o] for o in ['start', 'end','fmin', 'fmax']]

    if fmin == -1: fmin = 0

    if fmax == -1: fmax = frequency.max()

    xy = (start, fmin)

    width = end - start

    height = fmax - fmin

    return xy, width, height
# Spectrogram

data, sr = librosa.load('/kaggle/input/sample.mp3')

spectrogram = librosa.feature.melspectrogram(y=data, sr=sr)

spectrogram = librosa.power_to_db(spectrogram)



# Labels

labels = txt2df('/kaggle/input/sample_labels.txt')
labels.head()
fig, ax = plt.subplots(figsize=(10,4), dpi=120)

librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel', ax=ax);

for i in range(len(labels)):

    ax.add_patch(Rectangle(*rectangle(labels.iloc[i], ax.axis()[2:]), fc ='none', ec ='r', lw = 4))