import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing

import re
import os
#print(os.listdir("../input"))

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("train shape:", train_df.shape)
print("test shape:", test_df.shape)
train_df.head(10)
passenger_id = test_df['PassengerId'] # keep it for the end
train_df['NameLen'] = train_df['Name'].apply(len)
test_df['NameLen'] = test_df['Name'].apply(len)

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

train_df['Title'] = train_df['Name'].apply(get_title)
test_df['Title'] = test_df['Name'].apply(get_title)
mycols = ['PassengerId','Name','Cabin','Ticket']
train2_df = train_df.drop(mycols, axis = 1)
test2_df = test_df.drop(mycols, axis = 1)
train2_df.head(10)
# list title feature in descending order
train2_df.Title.value_counts()
test2_df.head(10)
train2_df.isnull().sum()
#train2_df.fillna(value=0, inplace=True)
#test2_df.fillna(value=0, inplace=True)

# fill Nan on Embarked field with the most common value
train2_df['Embarked'].fillna('S', inplace=True)
test2_df['Embarked'].fillna('S', inplace=True)

# concat the train and test dataframe (even if the field "survived" is missing in the 2nd one)
full_data = pd.concat([train2_df,test2_df], axis=0, ignore_index=True)

# "fare" in test has 1 value missing: fill with the mean
full_data['Fare'] = pd.to_numeric(full_data['Fare'], errors='coerce')
test2_df['Fare'].fillna(full_data['Fare'].mean(), inplace=True)

age_avg = full_data['Age'].mean()
age_std = full_data['Age'].std()

train2_df['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
test2_df['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
# convert Sex in 0=male, 1=female
sex_conv = {'male' : 0, 'female' : 1}
train2_df['Sex'] = train2_df['Sex'].map(sex_conv)
test2_df['Sex'] = test2_df['Sex'].map(sex_conv)

# we convert Embarked in numeric too
emb_conv = {'S':1, 'C':2,'Q':3}
train2_df['Embarked'] = train2_df['Embarked'].map(emb_conv)
test2_df['Embarked'] = test2_df['Embarked'].map(emb_conv)
# Check if there's any other NaN

train2_df.isnull().sum()
test2_df.isnull().sum()
# add a new features that sum siblings and parch
train2_df['FamilySize'] = train2_df['SibSp'] + train2_df['Parch'] + 1
test2_df['FamilySize'] = test2_df['SibSp'] + test2_df['Parch'] + 1

train2_df['IsAlone'] = np.where(train2_df['FamilySize']==1, 1, 0)
test2_df['IsAlone'] = np.where(test2_df['FamilySize']==1, 1, 0)

# drop useless columns
mycols = ['SibSp','Parch']
train2_df = train2_df.drop(mycols, axis = 1)
test2_df = test2_df.drop(mycols, axis = 1)

#Encoding title feature
from sklearn.preprocessing import LabelEncoder

title_lab_enc = LabelEncoder()
title_lab_enc.fit(pd.concat([train2_df["Title"],test2_df["Title"]]))
train2_df["TitleEnc"] = title_lab_enc.transform(train2_df["Title"])
test2_df["TitleEnc"] = title_lab_enc.transform(test2_df["Title"])

train2_df = train2_df.drop(['Title'], axis = 1)
test2_df = test2_df.drop(['Title'], axis = 1)
train2_df.head(10)
train2_df.info()
# Normalize values train

train3_df = train2_df

scaler = preprocessing.MinMaxScaler() 
scaled_values = scaler.fit_transform(train3_df) 
train3_df.loc[:,:] = scaled_values
train3_df.describe()
# Normalize values test

test3_df = test2_df

scaler = preprocessing.MinMaxScaler() 
scaled_values = scaler.fit_transform(test3_df) 
test3_df.loc[:,:] = scaled_values
test3_df.describe()
train3_df.head(10)
from keras import models
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(9,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])
x_arr = train3_df.iloc[:,1:].values
y_arr = train3_df.iloc[:,0].values

x_train = x_arr[100:] # training set
y_train = y_arr[100:]

x_val = x_arr[:100] # validation set
y_val = y_arr[:100]

x_test = test3_df.values # test set
history = model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=10,
                    validation_data=(x_val, y_val))
print("train acc: ", history.history['acc'][-1])
print("val acc: ", history.history['val_acc'][-1])
import matplotlib.pyplot as plt

history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
y_predict = model.predict(x_test)
y_predict = y_predict.reshape(418,)
temp = pd.DataFrame(pd.DataFrame({
        "PassengerId": passenger_id,
        "Survived": np.round(y_predict,0).astype(int)
        }))
temp.to_csv("../working/submission6.csv", index = False)