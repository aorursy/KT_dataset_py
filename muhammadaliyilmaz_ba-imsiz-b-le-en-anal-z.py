# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

from sklearn.decomposition import FastICA

from scipy.io.wavfile import read,write

import matplotlib.pylab as plt

import pandas as pd

import seaborn as sns
Fs,s1=read(r"../input/zeki-mren-ses/yildiz.wav")

Fs,s2=read(r"../input/zeki-mren-ses/gz.wav")
pd.DataFrame(s1).tail()#Şarkımızın biri dataframe şeklinde gösterirsek
pd.DataFrame(s2).tail()#s2 veri setimizde diğer bir ses dosyamızdır
s1=s1[:427235,:]
plt.plot(s1,color="red")

plt.plot(s2,color="green")
S=np.c_[s1,s2]#İki farklı şarkımızı matrissel olarak birleştiriyoruz.

A=np.array([[0.5,1,0.3,0.7],[0.3,0.4,0.8,0.7],[0.6,0.4,0.2,0.1],[0.6,0.4,0.3,0.5]])

X=np.dot(S,A)
pd.DataFrame(X).head()
plt.plot(X)
x1,x2=X[:,:2],X[:,2:4]

x1=x1 / np.max(np.abs(x1)) *25+25

x2=x2 / np.max(np.abs(x2)) *25+25

ica=FastICA()

S_=ica.fit(X).transform(X)
pd.DataFrame(S_).tail()#Yaptığımız bağımsız bileşen analizi sonucu oluşan ham veri setimiz
k1,k2=S_[:,:2],S_[:,2:4]

k1=k1 / np.max(np.abs(k1)) *25+25

k2=k2 / np.max(np.abs(k2)) *25+25
plt.plot(k1,color="red")

plt.plot(k2,color="blue")