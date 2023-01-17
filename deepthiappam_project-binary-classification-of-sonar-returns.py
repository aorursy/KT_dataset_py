# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
file = r'../input/sonar.all-data.csv'
import numpy as np

import pandas as pd



df = pd.read_csv(file)

df.head(10)
seed =10

np.random.seed = seed



df.shape
dataset = df.values
X = dataset[:,1:60].astype(float)

y = dataset[:,60]
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(y)

encoded_y = encoder.transform(y)
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier



def my_model():

    model = Sequential()

    model.add(Dense(60,input_dim=59,init = 'normal',activation='relu'))

    model.add(Dense(1,init='normal',activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model



model = KerasClassifier(build_fn=my_model,nb_epoch=200,batch_size=10,verbose=0)

from sklearn.model_selection import cross_val_score,StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

kfold = StratifiedKFold(n_splits=10,shuffle =True,random_state=seed)

results = cross_val_score(model,X,encoded_y,cv=kfold)

print("Accuracy:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))
import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.pipeline import Pipeline



def my_model():

    model = Sequential()

    model.add(Dense(60,input_dim=59,init = 'normal',activation='relu'))

    model.add(Dense(1,init='normal',activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

estimators=[]

estimators.append(('standardize',StandardScaler()))

estimators.append(('mlp',KerasClassifier(build_fn=my_model,nb_epoch=100,batch_size=5,verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10,shuffle =True,random_state=seed)

results = cross_val_score(pipeline,X,encoded_y,cv=kfold)

print("Accuracy:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))
import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.pipeline import Pipeline



def small_model():

    model = Sequential()

    model.add(Dense(30,input_dim=59,init = 'normal',activation='relu'))

    model.add(Dense(1,init='normal',activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

estimators=[]

estimators.append(('standardize',StandardScaler()))

estimators.append(('mlp',KerasClassifier(build_fn=small_model,nb_epoch=100,batch_size=5,verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10,shuffle =True,random_state=seed)

results = cross_val_score(pipeline,X,encoded_y,cv=kfold)

print("Smaller:%.2f%% (%.2f%%)"%(results.mean()*100,results.std()*100))
def large_model():

    model = Sequential()

    model.add(Dense(60,input_dim=59,init = 'normal',activation='relu'))

    model.add(Dense(30,init='normal',activation='relu'))

    model.add(Dense(1,init='normal',activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

estimators=[]

estimators.append(('standardize',StandardScaler()))

estimators.append(('mlp',KerasClassifier(build_fn=large_model,nb_epoch=100,batch_size=5,verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10,shuffle =True,random_state=seed)

results = cross_val_score(pipeline,X,encoded_y,cv=kfold)

print("Larger:%.2f%% (%.2f%%)"%(results.mean()*100,results.std()*100))