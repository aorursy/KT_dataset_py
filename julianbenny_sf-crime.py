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
sub = pd.read_csv("/kaggle/input/sf-crime/sampleSubmission.csv.zip")

sub
train_data = pd.read_csv("/kaggle/input/sf-crime/train.csv.zip")
train_data.shape
train_data.head(10)
test_data = pd.read_csv("/kaggle/input/sf-crime/test.csv.zip")
test_data.shape
test_data.head(10)
train_data.drop(["Descript","Resolution"],axis=1,inplace=True)
train_data['Category'].value_counts().shape
train_data.drop("Address",axis=1,inplace=True)
test_data.drop("Address",axis=1,inplace=True)
train_data["Time_of_day"] = np.zeros(len(train_data))

test_data["Time_of_day"] = np.zeros(len(test_data))
train_data["Dates"][0][-8:-6]

train_data["Time_of_day"].iloc[10]
for i in range(len(train_data)):

    if int(train_data["Dates"][i][-8:-6])>=21 or int(train_data["Dates"][i][-8:-6])<=6:

        train_data["Time_of_day"].iloc[i] = 1
for i in range(len(test_data)):

    if int(test_data["Dates"][i][-8:-6])>=21 or int(test_data["Dates"][i][-8:-6])<=6:

        test_data["Time_of_day"].iloc[i] = 1
from datetime import date

import holidays
usa_holidays = holidays.US()
pd.Timestamp(train_data["Dates"][0]).date() in usa_holidays
train_data["Holiday"] = np.zeros(len(train_data))

test_data["Holiday"] = np.zeros(len(test_data))
for i in range(len(train_data)):

    if (pd.Timestamp(train_data["Dates"][i]).date() in usa_holidays):

        train_data["Holiday"].iloc[i] = 1
for i in range(len(test_data)):

    if (pd.Timestamp(test_data["Dates"][i]).date() in usa_holidays):

        test_data["Holiday"].iloc[i] = 1
train_data.drop("Dates",axis=1,inplace=True)

test_data.drop("Dates",axis=1,inplace=True)
categorical = train_data.drop("Category",axis=1).columns[train_data.drop("Category",axis=1).dtypes == 'object']

categorical
for i in categorical:

    dummies = pd.get_dummies(train_data[i])

    train_data = pd.concat([train_data,dummies],axis=1)

    train_data.drop(i,axis=1,inplace=True)  
categorical_test = test_data.columns[test_data.dtypes == 'object']

categorical_test
for i in categorical_test:

    dummies = pd.get_dummies(test_data[i])

    test_data = pd.concat([test_data,dummies],axis=1)

    test_data.drop(i,axis=1,inplace=True)  
# train_data.drop(['X',"Y"],axis=1,inplace=True)

# test_data.drop(['X',"Y"],axis=1,inplace=True)
X = train_data.drop("Category",axis=1)

y = train_data["Category"]
Y = pd.get_dummies(y)
Y.shape
X.shape[1]
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

    

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
from keras.models import Sequential

from keras.layers import Dense,Activation
model = Sequential()



model.add(Dense(30,input_shape=(21,)))

model.add(Activation('relu'))



model.add(Dense(30))

model.add(Activation("relu"))



model.add(Dense(30))

model.add(Activation("relu"))



model.add(Dense(30))

model.add(Activation("relu"))



model.add(Dense(30))

model.add(Activation("relu"))



model.add(Dense(39))

model.add(Activation("softmax"))



model.summary()
model.compile(optimizer='adam',

             loss = "categorical_crossentropy",

             metrics=['accuracy'])
model.fit(X,Y,

         batch_size=256,

         epochs = 20,

         verbose = 2,

         validation_data=(X_train,Y_train))
preds_vals = model.predict(test_data.drop("Id",axis=1))
preds_vals
preds = pd.DataFrame(data=preds_vals,columns=Y.columns)

preds
new_df = pd.DataFrame(np.where(preds.T == preds.T.max(), 1, 0),index=preds.columns).T

new_df
new_df['Id'] = test_data["Id"]
cols = list(new_df.columns)

cols = [cols[-1]] + cols[:-1]

new_df = new_df[cols]
new_df
new_df.to_csv('../working/submission.csv', index=False)