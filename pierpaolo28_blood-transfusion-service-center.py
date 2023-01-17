# https://www.openml.org/d/1464

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import np_utils

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import pickle

import os
print(os.listdir("../input"))
df = pd.read_csv("../input/blood-transfusion-service-center.csv")

df.head()
df = df.dropna(how='all')
df["Class"].value_counts()
from sklearn.utils import resample



df_majority = df[df.Class==2]

df_minority = df[df.Class==1]



# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=178,    # to match majority class

                                 random_state=42) # reproducible results



# Combine majority class with upsampled minority class

df = pd.concat([df_majority, df_minority_upsampled])



# Display new class counts

df.Class.value_counts()
df["Class"].value_counts()
X = df.drop(['Class'], axis=1).values

#X = StandardScaler().fit_transform(X)

Y = df['Class']
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)
trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train,Y_Train)

predictionforest = trainedforest.predict(X_Test)

trainedforest.score(X_Train, Y_Train)
trainedforest = RandomForestClassifier(n_estimators=700).fit(X,Y)
# Saving model to disk

pickle.dump(trainedforest, open('model.pkl','wb'))
# Loading model to compare the results

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,  430, 10350,  86]]))
p = model.predict(X_Test)

#print(X_Test)

print(list(p).count(1))

print(list(p).count(2))