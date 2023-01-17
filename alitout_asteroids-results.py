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
import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/asteroid-dataset/dataset.csv')
df.head()
df.drop(['id', 'spkid', 'full_name', 'pdes', 'prefix', 'name'], axis = 1, inplace = True)
ne = pd.get_dummies(df['neo'], drop_first = True, columns = ['neo'])

pha = pd.get_dummies(df['pha'], drop_first = True, columns = ['Y'])
ne = ne.rename(columns = {'Y': 'neo'})
df = pd.concat([df.drop('neo', axis = 1), ne], axis = 1)
df = pd.concat([df.drop('pha', axis = 1), pha], axis = 1)
df.head()
df.isnull().sum()/958524 #Diameter, albedo, and diamter_sigma are missing way too many of their values so I'll just get rid of them
df.drop(['diameter', 'albedo', 'diameter_sigma', 'orbit_id', 'equinox'], axis = 1, inplace = True)
df.info()
df[df['Y'] == 1].isnull().sum() #Just want to see if any of the null values are when the asteroid is 

#hazardous since I don't want to drop any of those values because we have so little of them
#Since there is such a small amount of missing data points relative to how many asteroids are non-hazardous I'll just drop the null values

df.dropna(inplace = True)
df.isnull().sum()
#I honestly can't tell if class if useful or is just another id so I'm just gonna keep it in case

classes = pd.get_dummies(df['class'], drop_first = True)

classes.head()
df = pd.concat([df.drop('class', axis = 1), classes], axis = 1)

df.head()
df.info()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop(['neo' ,'Y', 'APO', 'AST', 'ATE', 'CEN', 'IEO', 'IMB', 'MBA', 'MCA', 'OMB', 'TJN', 'TNO'], axis = 1))
scaled_df = scaler.transform(df.drop(['neo' ,'Y', 'APO', 'AST', 'ATE', 'CEN', 'IEO', 'IMB', 'MBA', 'MCA', 'OMB', 'TJN', 'TNO'], axis = 1))
new_df = pd.DataFrame(scaled_df, columns = df.columns[:-13])

new_df = pd.concat([new_df , df[['neo' ,'Y', 'APO', 'AST', 'ATE', 'CEN', 'IEO', 'IMB', 'MBA', 'MCA', 'OMB', 'TJN', 'TNO']]], axis = 1)

new_df.info()
new_df.head()
from sklearn.linear_model import LogisticRegression

new_df.dropna(inplace = True)

lg = LogisticRegression(max_iter = 1000)

X = new_df.drop('Y', axis = 1)

y = new_df['Y']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lg.fit(X_train, y_train)
y_lg_pred = lg.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_lg_pred))

print('\n')

print(classification_report(y_test,y_lg_pred ))              #The score looks good because it got most of them right in all, but

print(lg.score(X_test, y_test))                              #really sucks at predicting when an asteroid will actually hit
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, y_train)
y_rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, y_rfc_pred))

print('\n')

print(classification_report(y_test,y_rfc_pred ))         #Okay so less false negatives but more false positives which is better

print(rfc.score(X_test, y_test))                         #because I want to now when an asteroid would actually hit earth
from sklearn.preprocessing import MinMaxScaler

X_n = df.drop('Y', axis = 1).values

y_n = df['Y'].values

neural_scale = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X_n, y_n, test_size=0.3, random_state=101)

X_train = neural_scale.fit_transform(X_train)

X_test = neural_scale.transform(X_test)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout
model = Sequential()



model.add(Dense(45, activation = 'relu'))

model.add(Dropout(rate = 0.3))



model.add(Dense(28, activation = 'relu'))

model.add(Dropout(rate = 0.3))



model.add(Dense(15, activation = 'relu'))

model.add(Dropout(rate = 0.3))



model.add(Dense(1, activation = 'sigmoid'))



model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)
model.fit(X_train, y_train, batch_size = 256, epochs = 40, validation_data = (X_test, y_test), callbacks = [early_stop] )
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
y_neu_pred = model.predict_classes(X_test)
print(confusion_matrix(y_test, y_neu_pred))

print('\n')

print(classification_report(y_test,y_neu_pred ))         #Best predictor by far on the set