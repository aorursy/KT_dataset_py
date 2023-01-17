# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from keras.models import Sequential

from keras.layers import Dense, Activation

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
seed=7

np.random.seed(seed)

digits=pd.read_csv('../input/train.csv')

testSet=pd.read_csv('../input/test.csv')

digits=np.array(digits)

testSet=np.array(testSet)

label=digits[:,0]

#define training set and testset

trainSet=digits[:,1:(digits.shape[1]+1)]
std_scale=preprocessing.StandardScaler().fit(trainSet)

trainSet_std=std_scale.transform(trainSet)

testSet_std=std_scale.transform(testSet)

#one hot encoding scheme for output label

numclasses=10

numlabels=label.shape[0]

indexOffset=np.arange(numlabels)*numclasses

one_hot_vector=np.zeros((numlabels,numclasses))

one_hot_vector.flat[indexOffset+label.ravel()]=1

label_one_hot=one_hot_vector

label_one_hot.astype(np.uint8)
model=Sequential()

model.add(Dense(390,input_dim=784,init='uniform',activation='relu'))

model.add(Dense(10,init='uniform'))

model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])



model.fit(trainSet_std,label_one_hot,nb_epoch=20,batch_size=10)



eval=model.evaluate(trainSet_std,label_one_hot)

print("%s: %.2f%% " % (model.metrics_names[1], eval[1]*100))

prediction=model.predict_classes(testSet_std)

print(prediction)

print("------------------------------------------------------")