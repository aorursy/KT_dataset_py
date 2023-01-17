# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import soundfile as sf

import librosa

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/trabalho-01-classificao-musical/file_dataset.csv")

df.head()
def get_soundfile(file, genre):

    return sf.read(f"/kaggle/input/gtzan-genre-collection/genres/{genre}/{file}")

## Exemplo

sound, samplerate = get_soundfile("metal.00091.au","metal")

sound, samplerate
sound=[]

samplerate=[]



for i in range(len(df["file"])): 

    x, y = get_soundfile(df["file"][i] , df["genre"][i])

    sound.append(x)

    samplerate.append(y)

rms=[]

centroid=[]

flatness=[]

tempogram=[]

tonnetz=[]



for i in range(len(sound)):

    rms.append(librosa.feature.rms(y=sound[i]))

    centroid.append(librosa.feature.spectral_centroid(y=sound[i]))

    flatness.append(librosa.feature.spectral_flatness(y=sound[i]))

    tempogram.append(librosa.feature.tempogram(y=sound[i]))

    tonnetz.append(librosa.feature.tonnetz(y=sound[i]))
for i in range(len(sound)):

    rms[i] = np.array(rms[i])

    centroid[i] = np.array(centroid[i])

    flatness[i] = np.array(flatness[i])

    tempogram[i] = np.array(tempogram[i])

    tonnetz[i] = np.array(tonnetz[i])
fts_easy = np.zeros((1,11))

fts_medium = np.zeros((1,11))

fts_hard = np.zeros((1,11))

for i in range(len(sound)):

    if(df["genre"][i] == "metal" or df["genre"][i] == "reggae"):

        f= np.hstack([centroid[i].mean(), centroid[i].std(), rms[i].mean(), rms[i].std(), flatness[i].mean(), flatness[i].std(), tempogram[i].mean(), tempogram[i].std(), tonnetz[i].mean(), tonnetz[i].std(), df["genre"][i]])

        fts_easy = np.vstack([fts_easy,f])



    elif(df["genre"][i] == "pop" or df["genre"][i] == "disco"):

        f= np.hstack([centroid[i].mean(), centroid[i].std(), rms[i].mean(), rms[i].std(), flatness[i].mean(), flatness[i].std(), tempogram[i].mean(), tempogram[i].std(), tonnetz[i].mean(), tonnetz[i].std(), df["genre"][i]])

        fts_medium = np.vstack([fts_medium,f])



    elif(df["genre"][i] == "rock" or df["genre"][i] == "country"):

        f= np.hstack([centroid[i].mean(), centroid[i].std(), rms[i].mean(), rms[i].std(), flatness[i].mean(), flatness[i].std(), tempogram[i].mean(), tempogram[i].std(), tonnetz[i].mean(), tonnetz[i].std(), df["genre"][i]])

        fts_hard = np.vstack([fts_hard,f])



fts_easy = fts_easy[1:,:]

fts_medium = fts_medium[1:,:]

fts_hard = fts_hard[1:,:]
Df_easy = pd.DataFrame(data=fts_easy)

Df_easy.head()
Df_medium = pd.DataFrame(data=fts_medium)

Df_medium.head()
Df_hard = pd.DataFrame(data=fts_hard)

Df_hard.head()
x, y = Df_hard.iloc[:,:-1], Df_hard.iloc[:,-1]

y, labels = pd.factorize(y)

labels
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state = 67)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns



train_scores = np.zeros(0)

test_scores = np.zeros(0)

random_forests = []



for i in range(1,10):

    random_forest = RandomForestClassifier(n_estimators=i,max_depth=10,random_state=67)

    random_forest.fit(x_train,y_train)

    train_scores = np.hstack([train_scores, random_forest.score(x_train,y_train)])

    test_scores = np.hstack([test_scores, random_forest.score(x_test,y_test)])

    random_forests.append(random_forest)

    

ax = sns.lineplot(x=range(1,10),y=train_scores,label="train score")

sns.lineplot(x=range(1,10),y=test_scores,label="test score")

plt.xlabel('Número de Árvores')

plt.ylabel('Taxa de Acertos')

plt.legend()

plt.show()
random_forest = RandomForestClassifier(n_estimators=8,max_depth=10,random_state=67)

random_forest.fit(x_train,y_train)

random_forest.score(x_test,y_test)
from sklearn.neighbors import KNeighborsClassifier



train_scores = np.zeros(0)

test_scores = np.zeros(0)

knns = []

for i in range(1,10):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    train_scores = np.hstack([train_scores, knn.score(x_train,y_train)])

    test_scores = np.hstack([test_scores, knn.score(x_test,y_test)])

    knns.append(knn)

plt.plot(range(1,10),train_scores,label="train score")

plt.plot(range(1,10),test_scores,label="test score")

plt.xlabel('Número de Vizinhos')

plt.ylabel('Taxa de Acertos')

plt.legend()

plt.show()
knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(x_train,y_train)

knn.score(x_test,y_test)