import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import re

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



# Store passenger ID for easy access

PassengerId = test['PassengerId']



train.head(3)
full_data = [train, test]



# Some features of my own that I have added in

# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)



# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)



# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLs in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLs in the Fare column and create a new feature CategoricalFare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search('([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""
# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis=1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

test = test.drop(drop_elements, axis=1)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Rearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y_train = train['Survived'].ravel()

# y_train_onehot = pd.get_dummies(train['Survived']).values

train = train.drop(['Survived'], axis=1)

x_train = train.values # Creates an array of the train data

x_test = test.values # Creats an array of the test data
""" Keras 패키지에서 사용할 기능들 호출 """

# import Keras libs

import keras.backend as K

from keras import optimizers

from keras.layers import Conv1D, SpatialDropout1D

from keras.layers import Activation, Lambda

from keras.layers import Convolution2D, Dense, Bidirectional, GRU, Flatten

from keras.models import Input, Model

from keras.regularizers import l2

from keras.layers import Reshape, Conv2D, MaxPooling1D, MaxPooling2D, BatchNormalization, GlobalMaxPooling1D, GlobalMaxPooling2D, AveragePooling1D, AveragePooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D

import keras.layers
""" 실험 모델 정의 """

def MLP1(n_features=11, num_classes=1):

    input_layer = Input(name='input_layer', shape=(11,))

    

    x = Dense(units=50, activation='relu')(input_layer)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=num_classes)(x)

    x = Activation('sigmoid')(x)



    output_layer = x

    return Model(input_layer, output_layer)





def MLP2(n_features=11, num_classes=1):

    input_layer = Input(name='input_layer', shape=(11,))



    x = Dense(units=50, activation='relu')(input_layer)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=50, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=num_classes)(x)

    x = Activation('sigmoid')(x)



    output_layer = x

    return Model(input_layer, output_layer)





def MLP3(n_features=11, num_classes=1):

    input_layer = Input(name='input_layer', shape=(11,))



    x = Dense(units=50, activation='relu')(input_layer)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=50, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=50, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=num_classes)(x)

    x = Activation('sigmoid')(x)



    output_layer = x

    return Model(input_layer, output_layer)





def MLP4(n_features=11, num_classes=1):

    input_layer = Input(name='input_layer', shape=(11,))



    x = Dense(units=50, activation='relu')(input_layer)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=50, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=50, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=50, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=num_classes)(x)

    x = Activation('sigmoid')(x)



    output_layer = x

    return Model(input_layer, output_layer)





def MLP5(n_features=11, num_classes=1):

    input_layer = Input(name='input_layer', shape=(11,))



    x = Dense(units=50, activation='relu')(input_layer)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=50, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=50, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=50, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=50, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=num_classes)(x)

    x = Activation('sigmoid')(x)



    output_layer = x

    return Model(input_layer, output_layer)





def cascade_MLP(n_features=11, num_classes=1):

    input_layer = Input(name='input_layer', shape=(11,))



    x = Dense(units=22, activation='relu')(input_layer)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=44, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=88, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=44, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=22, activation='relu')(x)

    x = keras.layers.Dropout(rate=0.2)(x)



    x = Dense(units=num_classes)(x)

    x = Activation('sigmoid')(x)



    output_layer = x

    return Model(input_layer, output_layer)







def MLP_deep(n_features=11, layers=5, dropout=0.3, num_classes=1, output_activation='sigmoid'):

    input_layer = Input(name='input_layer', shape=(11,))

    x = input_layer

    

    # deep_MLP

    for i in range(layers):

        x = Dense(units=128, activation='relu')(x)

        x = keras.layers.Dropout(rate=dropout)(x)



    # output

    x = Dense(units=num_classes)(x)

    if output_activation == 'sigmoid':

        x = Activation('sigmoid')(x)

    elif output_activation == 'softmax':

        x = Activation('softmax')(x)



    output_layer = x

    return Model(input_layer, output_layer)



def remove_output_activation(model):

    return Model(model.input, model._layers[-2].output)
""" 학습 환경 설정을 위한 사전 학습 테스트 """

# MLP5를 활용하여 적정 epoch을 찾아 봅시다.

model = MLP5()

model.compile(loss='binary_crossentropy',

              optimizer='Adam',

              metrics=['binary_accuracy'])

model.summary()



callback_list = [

    # best model save

    keras.callbacks.ModelCheckpoint('best_model', monitor='val_loss', verbose=1, save_best_only=True)

]



# 모델 훈련

model.fit(x=x_train, y=y_train, verbose=1, validation_split=0.2, epochs=2000, shuffle=True, callbacks=callback_list)

# 넉넉한 epoch으로 훈련하되

# ㄴ validation set으로 오버피팅 되는 지점 찾기

# ㄴ best model callback으로 적정 epoch 찾기

# ㄴㄴ [0.33840@217, 0.31960@137]
""" 단일 모델의 학습과 test set에 대한 예측 결과 생성 """

# %debug

models = [MLP1(), MLP2(), MLP3(), MLP4(), MLP5(), cascade_MLP()] # 훈련 시킬 단일 모델들

idx = 1

preds_list = []   # 모델 앙상블을 위한 예측 결과 저장

callbacks = []

for model in models:

    print('=============== model {} ==============='.format(idx))

    # 학습 방식 설정

    model.compile(loss='binary_crossentropy',

                  optimizer='Adam',

                  metrics=['binary_accuracy'])

    

    # train single model

    model.fit(x=x_train, y=y_train, verbose=1, epochs=100, shuffle=True)

    # ㄴ 137 epoch을 마지노선이라고 생각하고 설정

    

    # 단일 모델 예측 결과 생성

    preds = model.predict(x_test).flatten()

#     preds = np.argmax(preds, axis=1)

#     print(preds.shape)

    preds = np.array(preds >= 0.5, dtype=int)

    

    # Generate Submission File 

    Submission = pd.DataFrame({ 'PassengerId': PassengerId, 'Survived': preds })

    Submission.to_csv("MLP_e100_{}.csv".format(idx), index=False)

    

    # Ensemble을 위한 단일 모델 예측 값 저장

    preds_list.append(preds)

    idx += 1



preds_list = np.array(preds_list).T
""" 단일 모델들의 예측 결과 앙상블 """

print(preds_list.shape)

major_voted = []

for i in range(len(preds_list)):

    major_voted.append(np.argmax(np.bincount(preds_list[i])))

    

Submission = pd.DataFrame({ 'PassengerId': PassengerId, 'Survived': major_voted })

Submission.to_csv("MLP_e100_major_voted.csv", index=False)