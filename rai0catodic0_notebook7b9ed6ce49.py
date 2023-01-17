# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import librosa

from librosa import feature



import soundfile as sf

import seaborn as sns

import matplotlib.pyplot as plt

import IPython.display as ipd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
jazz ,sr_jazz = sf.read("/kaggle/input/jazzrock-classification/ontiva.com_-Sunny-320k.wav")

rock, sr_rock = sf.read("/kaggle/input/jazzrock-classification/ontiva.com LED ZEPPELIN - Rock And Roll-320k.wav")

fusion , sr_fusion = sf.read("/kaggle/input/jazzrock-classification/the-aristocrats-get-it-like-that-hq.wav")

fusion = np.asarray(fusion.T[0])

jazz = np.asarray(jazz.T[0])

rock = np.asarray(rock.T[0])

plt.plot(jazz)

plt.show()

plt.plot(rock)

plt.show()
ipd.Audio(data=jazz[:sr_jazz*10],rate=sr_jazz)
ipd.Audio(data=rock[:sr_rock*10],rate=sr_rock)
rms_jazz = librosa.feature.rms(y=jazz[:sr_jazz*10])

rms_rock = librosa.feature.rms(y=rock[:sr_rock*10])

plt.plot(rms_jazz[0])

plt.plot(rms_rock[0])

plt.show()

#Rock laranja

#jazz azul
spec_flatness_jazz = librosa.feature.spectral_flatness(y=jazz[:sr_jazz*10])

spec_flatness_rock = librosa.feature.spectral_flatness(y=rock[:sr_rock*10])

plt.plot(spec_flatness_jazz[0])

plt.plot(spec_flatness_rock[0])

plt.show()

#Rock laranja

#jazz azul
teste1 = librosa.feature.spectral_centroid(jazz[:sr_jazz*10],sr_jazz)

plt.plot(teste1[0])

teste2 = librosa.feature.spectral_centroid(rock[:sr_rock*10],sr_rock)

plt.plot(teste2[0]) 

plt.show()

#Rock laranja

#jazz azul
def get_features_for_one_music(musica,sr):



    features = {librosa.feature.spectral_centroid:True,librosa.feature.rms:False,librosa.feature.spectral_flatness:False,librosa.feature.mfcc:True}

    feature_array = np.zeros(24).reshape(1,24)

    x = np.array([])

    for feat in features.keys():

        if features[feat]:

            f= feat(musica,sr=sr)

        else:

            f = feat(musica)

        f_delta=feature.delta(f)

        f_2delta =feature.delta(f,order=2)

        x = np.hstack([x,np.array([f.mean(), np.std(f), f_delta.mean(), np.std(f_delta), f_2delta.mean(), np.std(f_2delta)])])

    x = np.hstack([x])

    feature_array = np.vstack([feature_array,x])        

    return pd.DataFrame(data=feature_array).drop(0)
df_jazz = get_features_for_one_music(jazz,sr_jazz)
X = df_jazz
import joblib

scaler = joblib.load("/kaggle/input/jazzrock-classification/scaler(1).jbl")

X = scaler.transform(X)
modelo = joblib.load("/kaggle/input/jazzrock-classification/modelo(1).jbl")
modelo.predict(X)[0]
df_rock = get_features_for_one_music(rock,sr_rock)

df_rock.head()

X_rock = scaler.transform(df_rock)
modelo.predict(X_rock)[0]
df_fusion = get_features_for_one_music(fusion,sr_fusion)

X_fusion = scaler.transform(df_fusion)
modelo.predict(X_fusion)[0]
pca = joblib.load("/kaggle/input/jazzrock-classification/pca.jbl")
pca_jazz = pca.transform(X)

pca_rock = pca.transform(X_rock)

pca_fusion = pca.transform(X_fusion)

df = pd.read_csv("/kaggle/input/jazzrock-classification/jazz-rock.csv")

X,Y  = df.iloc[:,1:-1],df.iloc[:,-1]

X = scaler.transform(X)

pca_X=pca.transform(X)
df
generos = ["jazz","rock","fusion"]

jazzx = pca_jazz[:,0][0]

jazzy = pca_jazz[:,1][0]

rockx = pca_rock[:,0][0]

rocky = pca_rock[:,1][0]

fusionx = pca_fusion[:,0][0]

fusiony = pca_fusion[:,1][0]



x = [jazzx,rockx,fusionx]

y = [jazzy,rocky,fusiony]





sns.scatterplot(x=x,y=y,hue=generos)



sns.scatterplot(x=pca_X[:,0],y=pca_X[:,1],hue=Y)
