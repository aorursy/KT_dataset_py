# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import *
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train_with_no_id = pd.read_csv('../input/train.csv')
df_train_with_no_id=df_train_with_no_id.drop(['id'],1)

#x = df_train_with_no_id.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#df_train_with_no_id_pre = pandas.DataFrame(x_scaled)
#df_train_with_no_id_pre.columns = df_train_with_no_id.columns
#df_train_with_no_id_pre=df_train_with_no_id_pre.drop(['diagnosis'],1)
#print(df_train_with_no_id['diagnosis'])
#df_train_with_no_id_pre.insert(0, 'diagnosis', df_train_with_no_id['diagnosis'])
#print(df_train_with_no_id_pre)
X = np.array(df_train_with_no_id.drop(['diagnosis'],1))
y = np.array(df_train_with_no_id['diagnosis'])
X.shape

model = linear_model.LogisticRegression()
model.fit(X,y)

predictions = model.predict(X)
model.score(X,y)

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
name='Logistic Regression'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
datos_submit = pd.read_csv('../input/dataForSubmission.csv')
ids=datos_submit['id']
#ids
datos_submit=datos_submit.drop(['id'],1)
#datos_submit_pre = datos_submit
#x = datos_submit_pre.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#datos_submit_pre = pandas.DataFrame(x_scaled)
#datos_submit_pre.columns = datos_submit.columns

#print(datos_submit_pre)
predictions=model.predict(datos_submit)
predictions
resultados=DataFrame({'Id': ids, 'Predicted': predictions})
resultados
resultados.to_csv('resultados_1.csv',  index = False)

