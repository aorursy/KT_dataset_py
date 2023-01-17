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
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
%matplotlib inline
train = pd.read_csv('../input/rn-2019-diabetes/diabetes_train.csv')
test = pd.read_csv('../input/rn-2019-diabetes/diabetes_test.csv')

print(f'Training shape: {train.shape}')
print(f'Testing shape: {test.shape}')
train.head()
train.tail()
test.head()
# Names of columns
for col in train.columns:
    print(col)
train.describe()
# Making copy of train and dropping the column 'id'
train_copy = train.copy(deep=True)
train.drop('id',inplace=True,axis=1)
#Computing the correlation of attributes
correlation = train.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(7,5))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation, vmax=.3, center=0,cmap=cmap,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
y = train['Outcome']
train.drop('Outcome',inplace=True,axis=1)
xtrain, xval, ytrain, yval = train_test_split(train,y,random_state=1,train_size=0.7)
print(f'xtrain :{xtrain.shape}')
print(f'ytrain :{ytrain.shape}')
print(f'xval :{xval.shape}')
print(f'yval :{yval.shape}')
from tensorflow.keras.layers import Dense,Dropout

model = tf.keras.models.Sequential([
    Dense(64,input_shape=[8],activation='relu'),
    Dense(128,activation='relu'),
    Dropout(0.3),
    Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
history = model.fit(xtrain,ytrain,epochs=1000,validation_data=(xval,yval),verbose=0)
f, axes = plt.subplots(2,1,figsize=(15,10))
epochs = [i for i in range(len(history.history['accuracy']))]
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
sns.lineplot(x=epochs,y=acc,ax=axes[0])
sns.lineplot(x=epochs,y=val_acc,ax=axes[1])
ids = test['id']
test.drop('id',inplace=True,axis=1)
preds = model.predict_classes(xtest)
preds = np.squeeze(preds)
submission_data = list(zip(ids,preds))
sub = pd.DataFrame(submission_data,columns=['id','Outcome'])
sub.to_csv('diabetes_submission_1.csv')