# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import keras

from sklearn.model_selection import train_test_split

import operator

import sklearn

from keras.layers import Input, Embedding, LSTM, Dense

from keras.models import Model

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from sklearn.preprocessing import OneHotEncoder

from keras.utils.vis_utils import plot_model

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

Train = pd.read_csv('../input/milan2020/train.csv')

Test = pd.read_csv('../input/milan2020/test.csv')
TrainX = Train.drop(['Revenue', 'ID'], axis=1)

Test = Test.drop(['ID'], axis=1)

TrainY = Train['Revenue']

Columns = list(TrainX.columns)

print (Columns)

TrainX
TrainY.describe()

TrainY_Unique = np.unique(TrainY.values)

y_count = {}



for i in TrainY_Unique:

	y_count[i] = 0

for j in TrainY:

	y_count[j] += 1

	

print (y_count)

plt_ = sns.barplot(list(y_count.keys()), list(y_count.values()))

plt_.set_xticklabels(plt_.get_xticklabels(), rotation=90)

plt.show()
Column_Unique = {}

for c in Columns:

    x = np.unique(TrainX[c].values)

    print (c,x)

    Column_Unique[c] = x.shape[0]



Sorted_Unique = {k: v for k, v in sorted(Column_Unique.items(), key=lambda item: item[1])}

print (Sorted_Unique)



Need_Encoding = ['Month','OperatingSystems','Browser','Region','VisitorType','SpecialDay']

Categorical = ['Month','OperatingSystems','Browser','Region','VisitorType','SpecialDay','Weekend']
X_train_type1 = TrainX.drop(Categorical, axis=1).to_numpy()

s1 = X_train_type1.shape[1]

s = X_train_type1.shape[0]

X_train_type2 = np.zeros((s,1))

X_train_type2[:,0] = TrainX['Weekend'].values

Y_train = TrainY.to_numpy()



for i in Need_Encoding:

    index = TrainX.columns.get_loc(i)

    onehotencoder = OneHotEncoder()

    x = TrainX[[i]]

    y = onehotencoder.fit_transform(x).toarray()

    s = y.shape[0]

    if i == 'Month':

        r = np.zeros((s,2))

        y = np.append(y,r,axis=1)

        X_train_type2 = np.concatenate((X_train_type2,y),axis=1)

    else:

        X_train_type2 = np.concatenate((X_train_type2,y),axis=1)

        

X_train = np.concatenate((X_train_type1,X_train_type2),axis=1)
X_t, X_v, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2,stratify=Y_train)

X_train_type1 = X_t[:,0:s1]

X_train_type2 = X_t[:,s1:]

X_val_type1 = X_v[:,0:s1]

X_val_type2 = X_v[:,s1:]
print (X_train_type1.shape)

print (X_train_type2.shape)

print (X_val_type1.shape)

print (X_val_type2.shape)
keras.backend.clear_session()



Categorical_Input = Input(shape=(52,), dtype='float32', name='main_input')

x = Dense(128,activation='sigmoid')(Categorical_Input)

Categorical_Partial_Output = Dense(128,activation='sigmoid')(Categorical_Input)

#Categorical_Output = Dense(1, activation='sigmoid', name='aux_output')(Categorical_Partial_Output)



Numerical_Input = Input(shape=(10,), name='aux_input')

x = keras.layers.concatenate([Categorical_Partial_Output, Numerical_Input])



x = Dense(128,activation='relu')(x)

x = Dense(32,activation='relu')(x)

Output = Dense(1,activation='sigmoid',name="main_output")(x)



model = Model(inputs=[Categorical_Input, Numerical_Input], outputs=[Output])
model.summary()

plot_model(model, to_file='Model.png', show_shapes=True)
start = time.time()

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model.fit({'main_input': X_train_type2, 'aux_input': X_train_type1},

          {'main_output': Y_train}, epochs=30, batch_size=16)

end = time.time()



print ("Total Time Taken = ",end-start)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



y_pred = model.predict({'main_input': X_val_type2, 'aux_input': X_val_type1}).ravel()

fpr, tpr, thresholds = roc_curve(Y_val, y_pred)

auc = roc_auc_score(Y_val, y_pred)

print('AUC: %.3f' % auc)



print (np.sum(np.round(y_pred)))
X_train_type1_normalise = sklearn.preprocessing.normalize(X_train_type1)

X_train_type2_normalise = sklearn.preprocessing.normalize(X_train_type2)

X_val_type1_normalise = sklearn.preprocessing.normalize(X_val_type1)

X_val_type2_normalise = sklearn.preprocessing.normalize(X_val_type2)
keras.backend.clear_session()



Categorical_Input = Input(shape=(52,), dtype='float32', name='main_input')

x = Dense(128,activation='sigmoid')(Categorical_Input)

Categorical_Partial_Output = Dense(128,activation='sigmoid')(Categorical_Input)

#Categorical_Output = Dense(1, activation='sigmoid', name='aux_output')(Categorical_Partial_Output)



Numerical_Input = Input(shape=(10,), name='aux_input')

x = keras.layers.concatenate([Categorical_Partial_Output, Numerical_Input])



x = Dense(128,activation='relu')(x)

x = Dense(32,activation='relu')(x)

Output = Dense(1,activation='sigmoid',name="main_output")(x)



model_normalise = Model(inputs=[Categorical_Input, Numerical_Input], outputs=[Output])



start = time.time()

model_normalise.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model_normalise.fit({'main_input': X_train_type2_normalise, 'aux_input': X_train_type1_normalise},

          {'main_output': Y_train}, epochs=30, batch_size=32)

end = time.time()



print ("Total Time Taken = ",end-start)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



y_pred = model.predict({'main_input': X_val_type2_normalise, 'aux_input': X_val_type1_normalise}).ravel()

fpr, tpr, thresholds = roc_curve(Y_val, y_pred)

auc = roc_auc_score(Y_val, y_pred)

print('AUC: %.3f' % auc)
Unique_Encoded = {}

for i in Need_Encoding:

    Unique_Encoded[i] = Sorted_Unique[i]



X_test_type1 = Test.drop(Categorical, axis=1).to_numpy()

s1 = X_test_type1.shape[1]

s = X_test_type1.shape[0]



X_test_type2 = np.zeros((s,1))

X_test_type2[:,0] = Test['Weekend'].values



for col in Need_Encoding:

    c = Unique_Encoded[col]

    if col == "Month":

        c = c+2

    Data = np.zeros((s,c))

    v = Test[col].to_numpy()

    i = 0

    for j in v:

        if col != 'SpecialDay':

            Data[i][j] = 1

            i = i+1

        else:

            Data[i][int(5*j)] = 1

            i = i+1    

        

    X_test_type2 = np.concatenate((X_test_type2,Data),axis=1)



print (np.sum(X_test_type2))

print (np.sum(X_test_type1))





X_test_type2_normalise = sklearn.preprocessing.normalize(X_test_type2)

X_test_type1_normalise = sklearn.preprocessing.normalize(X_test_type1)
predictions = model.predict({'main_input': X_test_type2, 'aux_input': X_test_type1})

predictions = predictions.flatten()

rows = predictions.shape[0]

print (np.sum(predictions))

print (predictions.shape)
output_test_data = pd.DataFrame() 

output_test_data['HasRevenue'] = predictions

output_test_data['Id'] = list(np.arange(0,rows))

submission = output_test_data[['Id','HasRevenue']]

submission.to_csv("submission.csv", index=False)

submission.tail()
submission.describe()
predictions_normalise = model_normalise.predict({'main_input': X_test_type2_normalise, 'aux_input': X_test_type1_normalise})

predictions_normalise = np.round(predictions_normalise).flatten()

rows = predictions.shape[0]

print (np.sum(predictions_normalise))

print (predictions_normalise.shape)
output_test_data_normalise = pd.DataFrame() 

output_test_data_normalise['HasRevenue'] = predictions_normalise

output_test_data_normalise['Id'] = list(np.arange(0,rows))

submission_normalise = output_test_data_normalise[['Id','HasRevenue']]

submission_normalise.to_csv("submission_normalise.csv", index=False)

submission_normalise.tail()
submission_normalise.describe()