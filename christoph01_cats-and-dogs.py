

import librosa

import librosa.display

import matplotlib.pyplot as plt

import IPython

import pandas as pd

import numpy as np

import seaborn as sns

import os

from sklearn import preprocessing

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

import warnings



print(os.listdir("../input"))



path = "../input/cats_dogs/"



#s -> signal, sr -> sampling rate

s, sr = librosa.load(path + "cat_3.wav")

print("Sampling Rate: " + str(sr))

print("Length of Signal: " + str(len(s)))

print("Duration in seconds: " + str(len(s)/sr))
plt.figure(figsize=(10, 5))

librosa.display.waveplot(s, sr=sr)

plt.show()
IPython.display.Audio(s, rate=sr)
s, sr = librosa.load(path + "dog_barking_4.wav")

print("Sampling Rate: " + str(sr))

print("Length of Signal: " + str(len(s)))

print("Duration in seconds: " + str(len(s)/sr))
plt.figure(figsize=(10, 5))

librosa.display.waveplot(s, sr=sr)

plt.show()
IPython.display.Audio(s, rate=sr)
file_names = os.listdir(path)



labels = []

mfccs = []

s_lengths = []



for name in file_names:

    #load signal

    s, sr = librosa.load(path + name)

    

    #store signal length -> sampling rate * time is sec.

    s_lengths.append(len(s))

    

    #extract mfccs taken the mean and store them.

    mfcc = np.mean(librosa.feature.mfcc(s, sr=sr, n_mfcc=20), axis=1)

    mfccs.append(mfcc)

    

    #extract label from the file names

    tokens = name.split("_")

    labels.append(tokens[0])
plt.boxplot([x / sr for x in s_lengths])

plt.xlabel('Signal length')

plt.show()
features = pd.DataFrame(mfccs)

features["size"] = s_lengths



scaler = preprocessing.StandardScaler()



data = pd.DataFrame(scaler.fit_transform(features))



data["y"] = labels

print(data.shape)

data.head(10)
sns.catplot(x="y", kind="count", data=data)

plt.show()
#encode labels cat -> 0, dog -> 1

data.y = data.y.apply(lambda x: 0 if x == "cat" else 1)



data = shuffle(data)

data.head(10)
X_train, X_test, y_train, y_test = train_test_split(data.drop(["y"], axis=1), data.y, test_size=0.20, random_state=0)
#ignore warnings

warnings.filterwarnings('ignore')





lr = LogisticRegression(solver='lbfgs', class_weight={0:1, 1:1.2})

params = {'C': [0.01, 0.1, 0.5, 1, 2, 3, 5, 10]}

clf = GridSearchCV(lr, params, cv=10)

clf.fit(X_train, y_train)



#print best parameter

print("Best hyperparameter: " + str(clf.best_params_))



#plot training accuracy

cv_acc = clf.cv_results_['mean_test_score']

plt.plot([0.01, 0.1, 0.5, 1, 2, 3, 5, 10], cv_acc, "-o")

plt.ylabel('Accuracy')

plt.xlabel('C')

plt.show()



#apply best settings on test set

pred = clf.predict(X_test)

print(pd.DataFrame(confusion_matrix(y_test, pred)))

print("Test Accuracy: " + str(clf.score(X_test, y_test)))