import pandas as pd

import numpy as np

import math

import os



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 

from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score



from skimage import io

from skimage import feature



import matplotlib.pyplot as plt



import cv2
#LOAD CSV DATASET

dataset_ex = pd.read_csv('../input/dataskripisii/dataindo2.csv')

dataset_in = pd.read_csv('../input/dataskripisii/datainternasional2.csv')

print (len(dataset_ex))

print (dataset_ex)

print (dataset_in)
df_ex = pd.DataFrame(dataset_ex)

df_in = pd.DataFrame(dataset_in)



for i in range(0,df_ex.shape[0]-2):

    df_ex.loc[df_ex.index[i+2],'SMA_3'] = np.round(((df_ex.iloc[i,1]+ df_ex.iloc[i+1,1] +df_ex.iloc[i+2,1])/3),1)

    

for i in range(0,df_in.shape[0]-2):

    df_in.loc[df_in.index[i+2],'SMA_3'] = np.round(((df_in.iloc[i,1]+ df_in.iloc[i+1,1] +df_in.iloc[i+2,1])/3),1)

    

print(df_ex.head())

df_ex['pandas_SMA_3'] = df_ex.iloc[:,1].rolling(window=3).mean()
print(df_in.head())

df_in['pandas_SMA_3'] = df_in.iloc[:,1].rolling(window=3).mean()


df_in['pandas_SMA_3'] = df_in.iloc[:,1].rolling(window=3).mean()

df_ex['pandas_SMA_3'] = df_ex.iloc[:,1].rolling(window=3).mean()
print(df_in.head())

print(df_ex.head())
for i in range(0,df_ex.shape[0]-3):

    df_ex.loc[df_ex.index[i+3],'SMA_4'] = np.round(((df_ex.iloc[i,1]+ df_ex.iloc[i+1,1] +df_ex.iloc[i+2,1]+df_ex.iloc[i+3,1])/4),1)



for i in range(0,df_in.shape[0]-3):

    df_in.loc[df_in.index[i+3],'SMA_4'] = np.round(((df_in.iloc[i,1]+ df_in.iloc[i+1,1] +df_in.iloc[i+2,1]+df_in.iloc[i+3,1])/4),1)
df_in['pandas_SMA_4'] = df_in.iloc[:,1].rolling(window=4).mean()

df_ex['pandas_SMA_4'] = df_ex.iloc[:,1].rolling(window=4).mean()

print(df_in.head())

print(df_ex.head())
import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=[15,10])

plt.grid(True)

plt.plot(df_ex['Jumlah Positif'],label='data')

plt.plot(df_ex['SMA_3'],label='SMA 3 Months')

plt.plot(df_ex['SMA_4'],label='SMA 4 Months')

plt.legend(loc=2)