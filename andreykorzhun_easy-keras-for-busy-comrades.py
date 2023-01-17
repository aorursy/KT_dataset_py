# Import required libraries

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import sklearn

import tensorflow as tf



# Import necessary modules

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.metrics import classification_report, f1_score

from sklearn.metrics import mean_squared_error

from math import sqrt



# Keras specific

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import to_categorical 
TRAIN_FILEPATH = '../input/credit-default/train.csv'

TEST_FILEPATH = '../input/credit-default/test.csv'



train_df = pd.read_csv(TRAIN_FILEPATH)

test_df = pd.read_csv(TEST_FILEPATH)



target = 'Credit Default'

train_df[target].value_counts()
def clean_df(df):

    ann_inc_median = df['Annual Income'].median()

    cred_score_median = df['Credit Score'].median()



    max_open_cred_max = df['Maximum Open Credit'].max()

    curr_loan_max = df.loc[df['Current Loan Amount'] < 1 * 10**8, 'Current Loan Amount'].max()



    df['Annual Income'] = df['Annual Income'].fillna(ann_inc_median)

    df['Years in current job'] = df['Years in current job'].fillna('< 1 year')

    df = df.drop(columns=['Months since last delinquent'])

    df['Bankruptcies'] = df['Bankruptcies'].fillna(0)

    df['Credit Score'] = df['Credit Score'].fillna(cred_score_median)

    df.loc[df['Annual Income'] > 4 * 10**6, 'Annual Income'] = ann_inc_median

    df.loc[df['Maximum Open Credit'] > max_open_cred_max, 'Maximum Open Credit'] = max_open_cred_max

    df.loc[df['Current Loan Amount'] == 1 * 10**8, 'Current Loan Amount'] = curr_loan_max

    df.loc[df['Credit Score'] >= 3000, 'Credit Score'] //= 10

    

    return df
# Немного чистим

train_df = clean_df(train_df)

test_df = clean_df(test_df)



# Преобразуем категории в отдельные признаки

train_df = pd.get_dummies(train_df, drop_first=True)

test_df = pd.get_dummies(test_df, drop_first=True)

train_df.drop('Purpose_renewable energy', axis=1, inplace = True)
train_df.info()
X = train_df.drop(target, axis=1)

y = train_df[target]



# Scaler

scaler = MinMaxScaler()



X = scaler.fit_transform(X)

test_df = scaler.fit_transform(test_df)



X.shape, test_df.shape
# from collections import Counter

from imblearn.over_sampling import ADASYN

# print('Original dataset shape %s' % Counter(y_train))

ada = ADASYN(random_state=42)

X, y = ada.fit_resample(X, y)

# print('Resampled dataset shape %s' % Counter(y_train))
# one hot encode outputs

y = to_categorical(y)



count_classes = y.shape[1]

print(count_classes)
# Задаём слои

model = Sequential()

model.add(Dense(64, activation='relu', input_dim=38))

model.add(Dense(256, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(2, activation='softmax'))



# Compile the model

model.compile(optimizer='adam', 

              loss='binary_crossentropy', 

              metrics=[tf.keras.metrics.BinaryAccuracy()])
# Обучени модели

model.fit(X, y, epochs=100)



# Предсказание

pred_test = model.predict(test_df)

y_pred_test = np.rint(pred_test[:,1])



# Выгрузка

submit = pd.read_csv('../input/credit-default/sample_submission.csv')

submit['Credit Default'] = y_pred_test.astype('int8')

submit.to_csv('predictions.csv', index=False)