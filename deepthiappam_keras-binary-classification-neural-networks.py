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
file = r'../input/pima-indians-diabetes.csv'

from keras.models import Sequential

from keras.layers import Dense

#dense for fully connected network

import numpy as np



seed = 10

np.random.seed(seed)
dataset = np.loadtxt(file,delimiter=',')

X = dataset[:,0:8]

Y = dataset[:,8]
# (#neurons,(initial_weights),activation)



model = Sequential()



model.add(Dense(12,input_dim=8,init='uniform',activation='relu'))

model.add(Dense(8,init='uniform',activation='relu'))

model.add(Dense(1,init='uniform',activation='sigmoid'))
model.compile(loss = 'binary_crossentropy',

             optimizer ='adam',metrics=['accuracy'])
model.fit(X,Y,nb_epoch=150,batch_size=10)
scores = model.evaluate(X,Y)

scores
model.fit(X,Y,validation_split=0.33,nb_epoch=150,batch_size=10)
from sklearn.model_selection import StratifiedKFold



kfold = StratifiedKFold(n_splits=10,shuffle=True,

                        random_state=seed)

cvscores = []

for train, test in kfold.split(X, Y):

    model = Sequential()



    model.add(Dense(12,input_dim=8,init='uniform',activation='relu'))

    model.add(Dense(8,init='uniform',activation='relu'))

    model.add(Dense(1,init='uniform',activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy',

             optimizer ='adam',metrics=['accuracy'])

    model.fit(X[train],Y[train],nb_epoch=150,

              batch_size=10,verbose=0)

    

    scores = model.evaluate(X[test],Y[test],verbose=0)

    print("%s : %.2f"%(model.metrics_names[1],

                      scores[1]*100))

    cvscores.append(scores[1]*100)

    

print(np.mean(cvscores),np.std(cvscores))
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score



def my_model():

    model = Sequential()



    model.add(Dense(12,input_dim=8,init='uniform',activation='relu'))

    model.add(Dense(8,init='uniform',activation='relu'))

    model.add(Dense(1,init='uniform',activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy',

             optimizer ='adam',metrics=['accuracy'])

    

    return model



model = KerasClassifier(build_fn = my_model,

                       nb_epoch=150, batch_size=10,verbose=0)



kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)



results = cross_val_score(model,X,Y,cv=kfold)



print(results.mean())
from sklearn.model_selection import GridSearchCV



def my_model(optimizer ='rmsprop',init='glorot_uniforn'):

    model = Sequential()



    model.add(Dense(12,input_dim=8,init=init,activation='relu'))

    model.add(Dense(8,init=init,activation='relu'))

    model.add(Dense(1,init=init,activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy',

             optimizer =optimizer,metrics=['accuracy'])

    return model



model = KerasClassifier(build_fn=my_model,verbose=0)



optimizers = ['rmsprop','adam']

init = ['glorot_uniform','normal','uniform']

epochs= np.array([50,100,150])

batches = np.array([5,10,20])



param_grid = dict(optimizer=optimizers,nb_epoch=epochs,

                 batch_size=batches,init=init)

grid = GridSearchCV(estimator=model,param_grid=param_grid)



grid_result = grid.fit(X,Y)

grid_result.best_score_
print("Best:%f using %s"%(grid_result.best_score_,grid_result.best_params_))