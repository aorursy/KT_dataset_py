

import pandas as pd

import numpy as np

import keras.utils as ku

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout
df = pd.read_csv('../input/winecsv/Wine.csv')



df.isnull().sum()

df.columns = [  'name'

                 ,'alcohol'

             	,'malicAcid'

             	,'ash'

            	,'ashalcalinity'

             	,'magnesium'

            	,'totalPhenols'

             	,'flavanoids'

             	,'nonFlavanoidPhenols'

             	,'proanthocyanins'

            	,'colorIntensity'

             	,'hue'

             	,'od280_od315'

             	,'proline'

                ]
df
import seaborn as sns

correlations = df[df.columns].corr(method='pearson')

sns.heatmap(correlations, cmap="YlGnBu", annot = True)
import heapq



print('Absolute overall correlations')

print('-' * 30)

correlations_abs_sum = correlations[correlations.columns].abs().sum()

print(correlations_abs_sum, '\n')



print('Weakest correlations')

print('-' * 30)

print(correlations_abs_sum.nsmallest(3))
df = df.drop(columns=['ash','magnesium', 'colorIntensity'], axis =1)
#Selecting dependent and independent variables

y = df.iloc[: ,0 ].values

X = df.iloc[:, 1:15].values

X[1]
df_hotencoded = pd.get_dummies(y)

df_hotencoded.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, df_hotencoded, test_size = 0.20)

y_test.head()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
import tensorflow as tf
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 32, input_dim = 10, activation='relu'))

ann.add(tf.keras.layers.Dense(units = 64, activation='relu'))



ann.add(tf.keras.layers.Dense(units = 128, activation='relu'))

ann.add(tf.keras.layers.Dense(units = 3, activation='softmax'))
ann.compile(optimizer= 'adam',  loss = 'categorical_crossentropy', metrics= ['accuracy'])
ann.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32 , epochs=100 )
score = ann.evaluate(X_test, y_test)

score[1]*100