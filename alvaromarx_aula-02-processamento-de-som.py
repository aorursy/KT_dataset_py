# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa
import librosa.display
import soundfile as sf
import seaborn as sns
import matplotlib.pyplot as plt
import IPython.display as ipd
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        break
# Any results you write to the current directory are saved as output.



















from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

model = SVC(random_state=42)
print("treinando modelo...")
model.fit(X_scaled, y)

genres = ['hiphop','classical']
_fts = np.zeros((1,3))
print("carregando dados de teste...")
for genre in genres:
    path = os.path.join("/kaggle/input/gtzan-genre-collection/genres/",genre)
    for sound in os.listdir(path):
        soundfile = os.path.join(path, sound)
        s, sr = sf.read(soundfile)
        f = np.hstack([f1(y=s).mean(), f2(y=s).mean(), genres.index(genre)])
        _fts = np.vstack([_fts, f])
_fts = _fts[1:,:]
print("tratando dados de teste...")
X_test, y_test = _fts[:,:-1], _fts[:,-1]
X_test = scaler.transform(X_test)
print("medindo acurácia...")
print(f"acurácia: {100*model.score(X_test, y_test)}%")
y_pred = model.predict(X_test)
sns.scatterplot(X_test[:,0],X_test[:,1],hue=[genres[i] for i in y_test.astype("int")])
plt.title("Dados atuais")
plt.xlabel(f1.__name__)
plt.ylabel(f2.__name__)
plt.show()
sns.scatterplot(X_test[:,0],X_test[:,1],hue=[genres[i] for i in y_pred.astype("int")])
plt.title("Dados previstos pelo modelo")
plt.xlabel(f1.__name__)
plt.ylabel(f2.__name__)
plt.show()