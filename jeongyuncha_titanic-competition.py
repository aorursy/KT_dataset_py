# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#to make Reproducibility 

import numpy as np

np.random.seed(123)
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping

#get data

data=pd.read_csv('/kaggle/input/titanic/train.csv')



#drop unncessary columns

data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1,inplace=True)



#change Sex col to 0 and 1(male=0, female=1)

data=data.replace('male',0)

data=data.replace('female',1)



#impute nan Age to mean of Age

tempdata_age=data['Age'].mean()

data['Age'].fillna(tempdata_age,inplace=True)



#make x_train, y_train and split data to train data & test data

x_train=data[['Pclass','Sex','Age']]

y_train=data[['Survived']]

x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.2)



#make model with early stopping(to avoid overfitting)

model=Sequential()

model.add(Dense(32,input_dim=len(x_train.columns),activation='relu'))

model.add(Dense(16,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss',mode='min',patience=5,verbose=1)

hist=model.fit(x_train,y_train,validation_split=0.2,batch_size=25,epochs=50,callbacks=[early_stopping])





#show loss and val_loss and evaluate model by test data

plt.plot(hist.history['loss'],label='loss')

plt.plot(hist.history['val_loss'],label='val_loss')

plt.legend(loc='upper right')

plt.show()

model.evaluate(x_test,y_test)
#Get answer

test=pd.read_csv('/kaggle/input/titanic/test.csv')

test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1,inplace=True)



#change Sex col to 0 and 1(male=0, female=1)

test=test.replace('male',0)

test=test.replace('female',1)



#impute nan Age to mean of Age

temp_age=test['Age'].mean()

test['Age'].fillna(temp_age,inplace=True)



#predict(model results are probability. So if result is over 0.5 then we can think result is 'survived')

final_x_test=test[['Pclass','Sex','Age']]

predict=model.predict(final_x_test)

for i in range(len(predict)):

        if predict[i] >= 0.5:

                predict[i]=1

        else:

            predict[i]=0
#make answer file

predict = pd.DataFrame(predict)

predict.rename(columns={0:'Survived'},inplace=True)

test2=pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submission=pd.concat([test2['PassengerId'],predict],axis=1)

gender_submission.to_csv('gender_submission.csv',index=False)