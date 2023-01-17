import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Reading the data
df=pd.read_csv('../input/train.csv')
dft=pd.read_csv('../input/test.csv')
# Let's see what all data is there in our training dataset
df.head()
# And the data in our testing dataset
dft.head()
df=df.fillna(0)
dft=dft.fillna(0)
df=df.replace(['male','female'],[0,1])
dft=dft.replace(['male','female'],[0,1])
df=df.drop(columns=['PassengerId'])
out_targets=dft['PassengerId'].values
cols=[i for i in df.describe()]
colt=[i for i in dft.describe()]
df_reduced=df[cols]
dft_red=dft[colt[1:]]
df_reduced.head()
dft_red.head()
train_labels = df_reduced['Survived'].values
df_reduced=df_reduced.drop(columns=['Survived'])
train_data=df_reduced.values
test_data=dft_red.values
print(train_data.shape, train_labels.shape)
print(test_data.shape)
mean=train_data.mean(axis=0)
std=train_data.std(axis=0)
train_data-=mean
train_data/=std

meant=test_data.mean(axis=0)
stdt=test_data.std(axis=0)
test_data-=mean
test_data/=std
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_data, train_labels, test_size=0.20, shuffle=True)
print(x_train.shape, x_test.shape)
index=200
x_val=x_train[:index]
y_val=y_train[:index]
partial_x_train = x_train[index:]
partial_y_train=y_train[index:]
from matplotlib import pyplot as plt
import numpy as np
%matplotlib inline
import seaborn as sns
sns.set_style('dark')
model=models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(6,)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(1, activation='sigmoid'))
sgd = optimizers.SGD(lr=0.001, decay=1e-6 ,momentum=0.9)
# adm = optimizers.Adam(lr=0.1, decay=1e-6)
model.compile(optimizer=sgd,
             loss='binary_crossentropy',
             metrics=['accuracy'])
history = model.fit(partial_x_train, partial_y_train,
                   epochs=500,
                   batch_size=32,
                   validation_data=(x_val,y_val),
                   verbose=0)
hist = history.history
acc=hist['acc']
a=200
b=len(acc)
val_loss=hist['val_loss'][a:b]
loss=hist['loss'][a:b]
val_acc=hist['val_acc']
epc = range(1,(b-a)+1)

plt.figure(figsize=(15,4))
plt.clf()
plt.subplot(1,2,1)
plt.plot(epc, loss, 'r', label='Training_loss')
plt.plot(epc, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

epc=range(1,len(acc)+1)
plt.subplot(1,2,2)
plt.plot(epc, acc, 'r', label='Training_acc')
plt.plot(epc, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()


plt.show()
model=models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(6,)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.001, decay=1e-6 ,momentum=0.9)
# adm = optimizers.Adam(lr=0.1, decay=1e-6)
model.compile(optimizer=sgd,
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.fit(train_data, train_labels,
                   epochs=500,
                   batch_size=32,
                   verbose=0)
results = model.evaluate(x_test, y_test)
results
predictions=model.predict(test_data)
# predictions
pred=[1 if predictions[i]>0.25 else 0 for i in range(len(test_data))]
pred
res=pd.DataFrame()
pd.read_csv('../input/gender_submission.csv').head()
res['PassengerId']=out_targets
res['Survived']=pred
res.head()
res.to_csv('Submission.csv', index=False)
