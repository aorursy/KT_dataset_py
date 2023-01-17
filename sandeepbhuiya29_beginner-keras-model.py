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
dataset = pd.read_csv("/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv")

dataset.head(5)
dataset.info()
dataset.isnull().any().value_counts()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.figure(figsize=(8,8))

sns.pairplot(dataset[['amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFraud']], hue='isFraud')
dataset = dataset.drop(['isFlaggedFraud'],axis=1)

dataset.columns
print("No. of fraud transactions: {}, No. of non-fraud transactions: {}".format((dataset.isFraud == 1).sum(),(dataset.isFraud == 0).sum()))
dataset = dataset.drop(['nameOrig','nameDest'], axis=1)


X = dataset.iloc[:,:-2].values

y = dataset.iloc[:, -2].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler
label_encoder_x= LabelEncoder()

X[:, 3] = label_encoder_x.fit_transform(X[:,3])

X[:, 5] = label_encoder_x.fit_transform(X[:,5])
one_hot_enc = OneHotEncoder()



one_hot_enc.fit(X[:,1].reshape(-1,1))

one_hot_enc.categories_

X_2 = X[:,0].shape

X = np.concatenate((X[:,0].reshape(-1,1),one_hot_enc.transform(X[:,1].reshape(-1,1)).toarray(), X[:,2:]), axis=1)
standard_sc = StandardScaler()

X = standard_sc.fit_transform(X)
from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
from keras.models import Sequential

from keras.layers import Dense, Dropout

import tensorflow as tf

import keras
input_dim = X_train.shape[1]

nb_classes = y_train.shape[0]



model = Sequential()

model.add(Dense(units = 64, input_dim=input_dim, activation = "relu"))

model.add(Dense(units = 128, activation = "relu"))

model.add(Dense(units = 256, activation = "relu"))

model.add(Dense(units = 512, activation = "relu"))

model.add(Dense(units = 256, activation = "relu"))

model.add(Dense(units = 128, activation = "relu"))

model.add(Dense(units = 64, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(units = 32, activation = "relu"))

model.add(Dense(units = 1, activation = "sigmoid"))



model.add(Dense(units = 128, activation = "relu"))

model.add(Dense(units = 256, activation = "relu"))

model.add(Dense(units = 512, activation = "relu"))

model.add(Dense(units = 256, activation = "relu"))

model.add(Dense(units = 128, activation = "relu"))

model.add(Dense(units = 64, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(units = 32, activation = "relu"))

model.add(Dense(units = 1, activation = "sigmoid"))

model.compile(optimizer = "adam", 

              loss = "binary_crossentropy", 

              metrics = ["accuracy"])

model.summary()
model.fit(X_train, y_train, batch_size = 512, epochs = 5)
score = model.evaluate(X_test, y_test)

print(score)