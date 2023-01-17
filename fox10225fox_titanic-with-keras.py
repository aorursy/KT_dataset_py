import numpy as np

import pandas as pd



import keras

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.core import Flatten, Dense, Activation, Dropout
DATA_HOME_DIR = "../input/"

row_data = pd.read_csv(DATA_HOME_DIR + 'train.csv', index_col=0)

test_data = pd.read_csv(DATA_HOME_DIR + 'test.csv', index_col=0)
test_ind = test_data.index



data= row_data[['Survived','Pclass','Sex','Age','SibSp','Parch']].dropna()

tdata = test_data[['Pclass','Sex','Age','SibSp','Parch']]
Pclass = pd.get_dummies(data['Pclass'])

Pclass.columns=['1st','2nd','3rd']

Pclass = Pclass.drop('1st',axis=1)



tPclass = pd.get_dummies(tdata['Pclass'])

tPclass.columns=['1st','2nd','3rd']

tPclass = tPclass.drop('1st',axis=1)



Sex = pd.get_dummies(data['Sex'])

Sex = Sex.drop('male',axis=1)



tSex = pd.get_dummies(tdata['Sex'])

tSex = tSex.drop('male',axis=1)



def male_female_child(passenger):

    age,sex = passenger

    if age < 16:

        return 'child'

    else:

        return sex



data['person'] = data[['Age','Sex']].apply(male_female_child,axis=1)

Age_cat = pd.get_dummies(data['person']).drop('child',axis=1)

tdata['person'] = tdata[['Age','Sex']].apply(male_female_child,axis=1)

tAge_cat = pd.get_dummies(tdata['person']).drop('child',axis=1)



data['Alone'] = data.Parch + data.SibSp

data['Alone'].loc[data['Alone'] >0] = 0

data['Alone'].loc[data['Alone'] == 0] = 1



tdata['Alone'] = tdata.Parch + tdata.SibSp

tdata['Alone'].loc[tdata['Alone'] >0] = 0

tdata['Alone'].loc[tdata['Alone'] == 0] = 1



Data_tmp = data[['Survived', 'Alone']]



Merge_data = pd.merge(Data_tmp,Pclass,right_index=True,left_index=True)

Merge_data = pd.merge(Merge_data,Sex,right_index=True,left_index=True)

Merge_data = pd.merge(Merge_data,Age_cat,right_index=True,left_index=True)

y = Merge_data['Survived'].values

x = Merge_data.drop('Survived',axis=1).values



Data_tmp = tdata[['Alone']]



Merge_data = pd.merge(Data_tmp,tPclass,right_index=True,left_index=True)

Merge_data = pd.merge(Merge_data,tSex,right_index=True,left_index=True)

Merge_data = pd.merge(Merge_data,tAge_cat,right_index=True,left_index=True)

tx = Merge_data



(x.shape, y.shape, tx.shape)
# create model

model = Sequential()

model.add(Dense(64, input_shape=(6,)))



for i in range(0, 8):

    model.add(Dense(units=64))

    model.add(Activation('relu'))



model.add(Dense(units=1))

model.add(Activation('linear'))



model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.fit(x, y, epochs=300)
p_survived = model.predict_classes(tx.values)
submission = pd.DataFrame()

submission['PassengerId'] = test_ind

submission['Survived'] = p_survived
submission.to_csv('submission.csv', index=False)