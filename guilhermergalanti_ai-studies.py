# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/fake_reg.csv')
sns.pairplot(df)
from sklearn.model_selection import train_test_split
X = df[['feature1', 'feature2']].values # became an array

y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train) # colocando valores em escala

X_test = scaler.transform(X_test)
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

model = Sequential()
# camadas de neuronios

model.add(Dense(4, activation = 'relu')) 

model.add(Dense(4, activation = 'relu'))

model.add(Dense(4, activation = 'relu'))



model.add(Dense(1))



model.compile(optimizer='rmsprop',loss='mse') # regressão linear
model.fit(X_train,y_train.values,epochs=300) # lembrar que colocar o .values na Series
loss = pd.DataFrame(model.history.history)
loss.plot(legend = False)

plt.title('Loss')

plt.xlabel('Epochs')
# Score do treino e do teste

test_score = model.evaluate(X_test, y_test.values, verbose = 0)

train_score = model.evaluate(X_train, y_train.values, verbose = 0)

print(f'Treino: {train_score}; Teste: {test_score}; RMS')
test_predictions = model.predict(X_test) # formato incorreto

test_predictions = pd.Series(test_predictions.reshape(300,))

pred_df = pd.DataFrame(y_test.values,columns=['Test Y']) 

pred_df = pd.concat([pred_df,test_predictions],axis=1)# juntando tudo



pred_df.columns = ['Valores de Y','Predições do modelo']
sns.scatterplot(x='Valores de Y',y='Predições do modelo',data=pred_df)
# criando uma coluna erro

pred_df['Erro'] = pred_df['Valores de Y'] - pred_df['Predições do modelo']

sns.distplot(pred_df['Erro'], bins = 30)
from sklearn.metrics import mean_absolute_error,mean_squared_error

mae = mean_absolute_error(pred_df['Valores de Y'],pred_df['Predições do modelo'])

mse = mean_squared_error(pred_df['Valores de Y'],pred_df['Predições do modelo'])

print(f'MAE: {mae}; MSE: {mse}')
model.save('meu_modelo.h5') # Agora pode ser usado mais tarde