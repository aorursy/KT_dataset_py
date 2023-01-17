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

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense

from sklearn.metrics import confusion_matrix

import warnings

warnings.filterwarnings("ignore")

from keras.wrappers.scikit_learn import KerasClassifier

from keras.optimizers import Adam

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve,accuracy_score, classification_report

#Import data

HDNames= ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

Data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv' )
Data.head()
Data.info()

#categorical variables: sex, cp -chest pain,  fbs, restecg, exang, slope, ca, thal

#numerical: age, testbps, chol, thalach, oldpeak
Data.isnull().any()
#ca and thal have ? in data values, drop them? In original data there were 5 records with ?, maps to category 4 below

Data['ca'].value_counts()
#drop data of category 4

dropData = Data.loc[Data['ca'] == 4]

print(dropData)

DataNew = Data.drop(dropData.index)
#in original there were 2 '?' maps to category of 0 below

Data['thal'].value_counts()
dropData = DataNew.loc[DataNew['thal'] == 0]

print(dropData)

DataNew = DataNew.drop(dropData.index)
#original data had ca and thal as of type object, conver to numeric

DataNew = DataNew.apply(pd.to_numeric)

DataNew.dtypes

DataNew.info()
#separate into X and y 

X=DataNew.drop('target', axis=1)

y = DataNew["target"]
#target data value counts 

#diagnosis of heart disease (angiographic disease status)

#-- Value 0: < 50% diameter narrowing

#-- Value 1: > 50% diameter narrowing 

y.value_counts()
#separate to train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)





#Scale data using StandardScaler

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
#Define model with 2 hidden layers which takes the input_dim of 13 features and nodes of 20 with relu activation

def create_model():

    model = Sequential()

    model.add(Dense(12, input_dim=X.shape[1], kernel_initializer='normal', activation="relu"))

    model.add(Dense(8,kernel_initializer='normal',  activation="relu"))

    model.add(Dense(1, kernel_initializer='normal', activation="sigmoid"))

    # Compile model

    #adam = Adam(lr=0.001)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
#do cross validation on the model defined above

# create model

model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=10, verbose=0)

# evaluate using 10-fold cross validation

kfold = StratifiedKFold(n_splits=10, shuffle=True)

results = cross_val_score(model, X_train_scaled, y_train, cv=kfold)

print("Cross validation results", results.mean())
#fit model and predict

model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_test_scaled)
#Model evaluation report

#confusion matrix

print("Confusion matrix: ")

print(confusion_matrix(y_test, y_pred))



#precision=TP/(TP+FP)

print("Precision score: " , precision_score(y_test, y_pred))



#recall=TP/(TP+FN)

print("Recall score: ", recall_score(y_test, y_pred))



#F1 score=TP/(TP+ (FP+FN/2)

print("F1 score: ", f1_score(y_test, y_pred))



print("Accuracy Score: ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
#Create model passing inputs to try grid search

# Function to create model, required for KerasClassifier

def create_model(optimizer='rmsprop', init='glorot_uniform'):                   

    # create model

    model = Sequential()

    model.add(Dense(12, input_dim=X.shape[1], kernel_initializer=init, activation='relu')) 

    model.add(Dense(8, kernel_initializer=init, activation='relu'))

    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

    return model
# grid search epochs, batch size and optimizer

# create model

model = KerasClassifier(build_fn=create_model, verbose=0)

optimizers = ['rmsprop', 'adam']

inits = ['glorot_uniform', 'normal', 'uniform']

epochs = [50, 100, 150]

batches = [5, 10, 20]

param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)

kFolds = StratifiedKFold(n_splits=10, shuffle=True)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kFolds)

grid_result = grid.fit(X_train_scaled, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))

print("Best parameters:\n{}".format(grid.best_params_))

print("Test score: {:.2f}".format(grid.score(X_test_scaled, y_test)))
y_pred = grid.best_estimator_.predict(X_test_scaled)
#Model evaluation report

#confusion matrix

print("Confusion matrix: ")

print(confusion_matrix(y_test, y_pred))



#precision=TP/(TP+FP)

print("Precision score: " , precision_score(y_test, y_pred))



#recall=TP/(TP+FN)

print("Recall score: ", recall_score(y_test, y_pred))



#F1 score=TP/(TP+ (FP+FN/2)

print("F1 score: ", f1_score(y_test, y_pred))



print("Accuracy Score: ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
#Final estimation using best params: {'batch_size': 20, 'epochs': 100, 'init': 'uniform', 'optimizer': 'rmsprop'}

#Define model with 2 hidden layers which takes the input_dim of 13 features and nodes of 20 with relu activation

def create_finalmodel():

    model = Sequential()

    model.add(Dense(12, input_dim=X.shape[1], kernel_initializer='uniform', activation="relu"))

    model.add(Dense(8,kernel_initializer='uniform',  activation="relu"))

    model.add(Dense(1, kernel_initializer='uniform', activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model



# create model

model = KerasClassifier(build_fn=create_finalmodel, epochs=100, batch_size=10, verbose=0)

#fit model and predict

model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_test_scaled)
#Model evaluation report

#confusion matrix

print("Confusion matrix: ")

print(confusion_matrix(y_test, y_pred))



#precision=TP/(TP+FP)

print("Precision score: " , precision_score(y_test, y_pred))



#recall=TP/(TP+FN)

print("Recall score: ", recall_score(y_test, y_pred))



#F1 score=TP/(TP+ (FP+FN/2)

print("F1 score: ", f1_score(y_test, y_pred))



print("Accuracy Score: ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))