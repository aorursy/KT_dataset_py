import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

import scipy.signal as signal

import timeit as t
import os
print(os.listdir("../input"))
import time
from tqdm import tqdm, tqdm_notebook
# Loading the data
df = pd.read_csv('../input/PLAsTiCC-2018/training_set.csv')
df.name = 'Training Set'
df_meta = pd.read_csv('../input/PLAsTiCC-2018/training_set_metadata.csv')
df_meta.name = 'Training Metadata Set'
def drawRandomCharts(signalBool = 0, f = np.linspace(1, 100, 100)):
    plt.figure(figsize=(20,20))
    for i,target in enumerate(df_meta['target'].unique()):
        rnd = np.random.randint(0, 20)
        object_id = df_meta[df_meta['target'] == target].iloc[rnd].object_id
        plt.subplot(4, 4, i+1)
        for passband in df['passband'].unique():
            passbandData = df[(df['object_id'] == object_id) & (df['passband'] == passband)]
            plt.title('Target: '+ str(df_meta.loc[df_meta['object_id']==object_id, 'target'].iloc[0])
                     + '\n Object: ' + str(int(object_id)))
            if (signalBool):
                plt.plot(signal.lombscargle(passbandData['mjd'], passbandData['flux'], f, normalize=True))
            else:
                plt.plot('mjd', 'flux', data=passbandData )
                plt.scatter('mjd', 'flux', data=passbandData )

drawRandomCharts(0)
drawRandomCharts(0)
drawRandomCharts(1)
drawRandomCharts(1, np.linspace(1,10,1000))
def drawPassbands(object_id, signalBool = 0, f = np.linspace(1, 100, 100)):
    plt.figure(figsize=(18,20))
    for i,passband in enumerate(df['passband'].unique()):
        passbandData = df[(df['object_id'] == object_id) & (df['passband'] == passband)].copy()
        passbandData['markerSize'] = (passbandData['detected'] + 0.15) * 20
        plt.subplot(6, 1, i+1)
        plt.title('Target: '+ str(df_meta.loc[df_meta['object_id']==object_id, 'target'].iloc[0])
                 + '\n Object: ' + str(int(object_id)))
        if (signalBool):
            plt.suptitle('\n\nLomb-Scargle frequencies', fontsize=18)
            plt.plot(signal.lombscargle(passbandData['mjd'], passbandData['flux'], f, normalize=True)) 
        else:
            plt.suptitle('\n\nRaw signals', fontsize=18)
            plt.plot('mjd', 'flux', data=passbandData, alpha=0.3 )
            plt.scatter('mjd', 'flux', 'markerSize' , data=passbandData )
drawPassbands(713,0)
drawPassbands(713,1)
def starInspector(object_id, f = np.linspace(1, 100, 200)):
    plt.figure(figsize=(18,20), num=1)
    plt.suptitle('Target: '+ str(df_meta.loc[df_meta['object_id']==object_id, 'target'].iloc[0])
                 + '\n Object: ' + str(int(object_id))
                 + '\n ddf = ' + str(df_meta.loc[df_meta['object_id']==object_id, 'ddf'].iloc[0])
                 + '\n specz = ' + str(df_meta.loc[df_meta['object_id']==object_id, 'hostgal_specz'].iloc[0])
                 + '\n photoz = ' + str(df_meta.loc[df_meta['object_id']==object_id, 'hostgal_photoz'].iloc[0])
                 + '\n distance = ' + str(df_meta.loc[df_meta['object_id']==object_id, 'distmod'].iloc[0])
                 + '\n mwebv = ' + str(df_meta.loc[df_meta['object_id']==object_id, 'mwebv'].iloc[0])
                 + '\n freq = ' + str(f.min()) + ' - ' + str(f.max()) + ' --- ' + str(f.shape)
                 , fontsize=16)
    initTime = t.timeit()
    for i,passband in enumerate(df['passband'].unique()):
        passbandData = df[(df['object_id'] == object_id) & (df['passband'] == passband)].copy()
        passbandData['markerSize'] = (passbandData['detected'] + 0.15) * 20
        plt.subplot(6, 2, 2*(i+1)-1)
        plt.title('Passband: ' + str(i) + ' (' + str(passbandData.shape[0]) + ') ')
        plt.plot('mjd', 'flux', data=passbandData, alpha=0.3 )
        plt.scatter('mjd', 'flux', 'markerSize' , data=passbandData )

        plt.subplot(6, 2, 2*(i+1))
        plt.plot(signal.lombscargle(passbandData['mjd'], passbandData['flux'], f, normalize=True))    
starInspector(713)
starInspector(713, np.linspace(0.1,10,1000))
