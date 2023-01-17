# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/land classification challenge/socialcops_challenge"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/land classification challenge/socialcops_challenge/land_train.csv')
test = pd.read_csv('../input/land classification challenge/socialcops_challenge/land_test.csv')
print(dataset.columns)
print(test.columns)
dataset.isna().sum()
test.isna().sum()
dataset['target'].hist()
print(dataset[['target' , 'X1']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'X1']].groupby(['target']).mean())
sns.catplot(x='target', y='X1',  kind='bar', data=dataset)
print(dataset[['target' , 'X2']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'X2']].groupby(['target']).mean())
sns.catplot(x='target', y='X2',  kind='bar', data=dataset)
print(dataset[['target' , 'X3']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'X3']].groupby(['target']).mean())
sns.catplot(x='target', y='X3',  kind='bar', data=dataset)
print(dataset[['target' , 'X4']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'X4']].groupby(['target']).mean())
sns.catplot(x='target', y='X4',  kind='bar', data=dataset)
print(dataset[['target' , 'X5']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'X5']].groupby(['target']).mean())
sns.catplot(x='target', y='X5',  kind='bar', data=dataset)
print(dataset[['target' , 'X6']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'X6']].groupby(['target']).mean())
sns.catplot(x='target', y='X6',  kind='bar', data=dataset)
print(dataset[['target' , 'I1']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'I1']].groupby(['target']).mean())
sns.catplot(x='target', y='I1',  kind='bar', data=dataset)
print(dataset[['target' , 'I2']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'I2']].groupby(['target']).mean())
sns.catplot(x='target', y='I2',  kind='bar', data=dataset)
print(dataset[['target' , 'I3']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'I3']].groupby(['target']).mean())
sns.catplot(x='target', y='I3',  kind='bar', data=dataset)
print(dataset[['target' , 'I4']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'I4']].groupby(['target']).mean())
sns.catplot(x='target', y='I4',  kind='bar', data=dataset)
print(dataset[['target' , 'I5']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'I5']].groupby(['target']).mean())
sns.catplot(x='target', y='I5',  kind='bar', data=dataset)
print(dataset[['target' , 'I6']].groupby(['target']).mean())
plt.plot(dataset[['target' , 'I6']].groupby(['target']).mean())
sns.catplot(x='target', y='I6',  kind='bar', data=dataset)
sns.heatmap(dataset.corr(), annot=True, fmt = ".2f", cmap = "coolwarm")
print(dataset.min())
print(dataset.max())
dataset.iloc[: , 6:-1] = dataset.iloc[: , 6:-1] + 5
test.iloc[: , 6:-1] = test.iloc[: , 6:] + 5
from sklearn.utils import shuffle
dataset = shuffle(dataset)
dataset = shuffle(dataset).reset_index()
dataset.drop('index' , axis = 1 , inplace = True)
dataset
print(dataset.min())
print(dataset.max())
y_train = dataset.iloc[: , -1:]
dataset.drop('target' , axis =1 , inplace = True)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state= 2)
x_train , y = sm.fit_sample (dataset , y_train)
pd.DataFrame(y).hist()
x_train = pd.DataFrame(x_train , columns= ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6'])
from sklearn.decomposition import PCA
pca_x = PCA()
x_train.iloc[: ,:6] = pca_x.fit_transform(x_train.iloc[: ,:6])
test.iloc[: , :6] = pca_x.transform(test.iloc[: , :6])
pca_x.explained_variance_ratio_
pca_i = PCA()
x_train.iloc[: ,6:] = pca_i.fit_transform(x_train.iloc[: ,6:])
test.iloc[: , 6:] = pca_x.transform(test.iloc[: , 6:])
pca_i.explained_variance_ratio_
x_train.columns
x_train.drop([ 'I5' , 'I6'] , axis =1 ,inplace =True)
test.drop(['I5' , 'I6'] , axis =1 ,inplace =True)
x_train
from sklearn.preprocessing import MinMaxScaler

normalizer_x = MinMaxScaler()  #for normalizing x1-x6
normalizer_i = MinMaxScaler()  #for normalizing i1-i6
test.shape
x_train.iloc[: , :6] = normalizer_x.fit_transform(x_train.iloc[: , :6])
x_train.iloc[: , 4:] = normalizer_i.fit_transform(x_train.iloc[: , 4:])

test.iloc[: , :6] = normalizer_x.fit_transform(test.iloc[: , :6])
test.iloc[: , 4:] = normalizer_i.fit_transform(test.iloc[: , 4:])

print(x_train.min())
print(x_train.max())
y = y.reshape((len(y) , 1))
#now as our dependant variable is categorical. therefore preprocessing it
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y)
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Adding the output layer
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train ,y, batch_size = 100, epochs = 25 , validation_split = 0.1)

# storing the results
y_pred_1 = np.argmax( classifier.predict(test) , axis = 1)
y_pred_1 = y_pred_1 + 1
y_pred_1 = pd.DataFrame(y_pred_1)
y_pred_1.hist()
final = pd.read_csv('../input/land classification challenge/socialcops_challenge/land_test.csv')
final['target'] = np.array(y_pred_1)
final
final['target'].value_counts()
final.to_csv('result_2.csv' , index=False)
