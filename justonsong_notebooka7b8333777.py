import numpy as np

import pandas as pd

from sklearn.metrics import log_loss,accuracy_score

from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

print('read and preprocess train data')

train=pd.read_csv('../input/train.csv')

x_data=train.drop(['id','species'],axis=1).values

scaler=StandardScaler().fit(x_data)

x_data=scaler.transform(x_data)

le=LabelEncoder().fit(train['species'])

y_labels=le.transform(train['species'])

print('read and preprocess test data')

test=pd.read_csv('../input/test.csv')

test_ids=test.pop('id')

x_test=test.values

scaler=StandardScaler().fit(x_test)

x_test=scaler.transform(x_test)

print('split the data into train and valid set')

sss=StratifiedShuffleSplit(test_size=0.1)

for train_index,valid_index in sss.split(x_data,y_labels):

    x_train,x_valid=x_data[train_index],x_data[valid_index]

    y_train,y_valid=y_labels[train_index],y_labels[valid_index]

print(x_train.shape)

print('deep learning using keras')

y_train_dummy=np_utils.to_categorical(y_train)

def base_model():

    model=Sequential()

    model.add(Dense(99,input_dim=192,init='normal',activation='linear'))

    model.add(Dropout(0.5))

    model.add(Dense(99,input_dim=99,init='normal',activation='relu'))	

    model.add(Dropout(0.2))

    model.add(Dense(99,init='normal',activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy','categorical_crossentropy'])

    return model

model=KerasClassifier(build_fn=base_model,verbose=1)

param_grid={'batch_size':[100],'nb_epoch':[1800]}

grid=GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1)

grid_results=grid.fit(x_train,y_train_dummy)

print('best score:',grid_results.best_score_)

print('best params:',grid_results.best_params_)

means = grid_results.cv_results_['mean_test_score']

stds = grid_results.cv_results_['std_test_score']

params = grid_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("Mean:%f (std:%f) with: %r" % (mean, stdev, param))

print(grid_results.cv_results_)

predictions= grid_results.best_estimator_.predict_proba(x_valid)

print('logloss of valid data:',log_loss(y_valid,predictions))

predictions=grid_results.best_estimator_.predict(x_valid)

print('Accuracy of valid data:',accuracy_score(y_valid,predictions))