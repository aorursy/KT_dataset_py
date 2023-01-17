import os

import warnings



import numpy as np



from scipy.io.wavfile import write



import pandas as pd



from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt



import seaborn as sns



import IPython.display as ipd



import librosa



from tqdm import tqdm
df_train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')



df_train.head()

df_train['ebird_code'].nunique()
#read in dataframe of 5s samples from mp3 files 30s or less

#id is 'filename_x' where x*5 is the start time of the audio sample within the mp3 

df_samples = pd.read_pickle('/kaggle/input/birdcall-samples/train_audio_samples.pkl')



print(len(df_samples.iloc[0]['sample']))

df_samples.head()
df_samples['bird'].nunique()
features = []



#get top most occuring birds

top_birds = df_samples['bird'].value_counts()[:5]



df_samples = df_samples[df_samples['bird'].isin(top_birds.index)]



for sample in tqdm(df_samples['sample']):

    #for each sample populate a row of features

    row = {'mean': sample.mean(),

           'max': sample.max(),

           'min': sample.min(),

           'spectral centroid': librosa.feature.spectral_centroid(sample).mean(),

           'spectral bandwidth': librosa.feature.spectral_bandwidth(sample).mean(),

           'spectral rolloff': librosa.feature.spectral_rolloff(sample).mean()}

    

    #'mfcc': librosa.feature.mfcc(sample).mean()

    #'zero crossing rate': librosa.feature.zero_crossing_rate(sample).mean()

    features.append(row)

    

X = pd.DataFrame(features)

sample_id = df_samples['id'].reset_index(drop = True)

y = df_samples['bird'].reset_index(drop = True)

df = pd.concat([X,y,sample_id], axis = 1)



df.head()
from sklearn.cluster import KMeans



for bird in top_birds.index:

    

    df_bird = pd.DataFrame(df[df['bird']==bird])

    

    #try to cluster birdcalls/background noise

    kmeans = KMeans(n_clusters=2)

    kmeans.fit(df_bird.drop(['bird','id'], axis = 1))



    df_bird['labels'] = kmeans.labels_

    

    g = sns.pairplot(df_bird, hue = 'labels', height = 2)

    g.fig.suptitle(bird, y=1.08)

    perc = sum(df_bird['labels'])/len(df_bird['labels'])

    print(str(perc)+':'+str(1-perc))
print(df_bird[df_bird['labels'] == 0][['bird','id']])
print(df_bird[df_bird['labels'] == 1][['bird','id']])