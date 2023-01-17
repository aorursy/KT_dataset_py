# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/ee-769-assignment1/train.csv")

data1 = pd.read_csv("../input/ee-769-assignment1/test.csv")

data2 = pd.read_csv("../input/ee-769-assignment1/sample_submission.csv")

print (data1)

data.head()

data.shape

data.columns.to_series().groupby(data.dtypes).groups

data1.columns.to_series().groupby(data1.dtypes).groups

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()

print (data.shape)

data.head()
le_count =0;

for col in data.columns[1:]:

    if data[col].dtype == "object":

        if len(list(data[col].unique()))<= 2:

            le.fit(data[col])

            data[col] = le.transform(data[col])

            le_count += 1

print('{} columns were label encoded.'.format(le_count))

le_count =0;

for col in data1.columns[1:]:

    if data1[col].dtype == "object":

        if len(list(data1[col].unique()))<= 2:

            le.fit(data1[col])

            data1[col] = le.transform(data1[col])

            le_count += 1

print('{} columns were label encoded.'.format(le_count))
data = pd.get_dummies(data, drop_first=True)

data1 = pd.get_dummies(data1, drop_first=True)
print(data.shape)

data.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,5))

data_col = list(data.columns)

data_col.remove('Attrition')

for col in data_col:

    data[col] = data[col].astype(float)

    data[col] =scaler.fit_transform(data[[col]])

data['Attrition'] = pd.to_numeric(data['Attrition'],downcast='float')

data.head()



data1_col = list(data1.columns)

for col in data1_col:

    data1[col] = data1[col].astype(float)

    data1[col] =scaler.fit_transform(data1[[col]])
temp = data

temp = temp.drop(columns=['Attrition'])

target = data['Attrition']

x_train = temp

y_train = target

x_test =data1

temp1 =data2

print (x_train.shape)

print (x_test.shape)

print (y_train.shape)
##param_grid = {'C': np.arange(1e-03, 2, 0.01)}

##log_gs = GridSearchCV(LogisticRegression(solver='liblinear', 

  ##                                       class_weight='balanced',

    ##                                     random_state=7),

      ##                iid =True,

        ##              return_train_source = True,

          ##            param_grid = param_grid,

            ##          scoring ='roc_auc',

              ##        cv = 10)

##log_grig =log_gs.fit(x_train, y_train)

##log_opt = log_grid.best_estimator_

##results =log_gs.cv_results_

##print ('best params'+ str(log_grid.best_estimator_))

##print ('best params' + str (log_grid.best_params_))

##print ('best score' + log_gs.best_score_)

       
from sklearn.datasets import make_classification 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

log_reg =LogisticRegression()

log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)
print(y_pred)

temp1['Attrition'] =y_pred

temp1['Attrition'] = pd.to_numeric(temp1['Attrition'],downcast='integer')
temp1.to_csv('outputfile.csv',index=False)

print (temp1)

temp2 =data2

from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(x_train, y_train)

y_tree = model.predict(x_test)

print(y_tree)

temp2['Attrition'] =y_tree

temp2['Attrition'] = pd.to_numeric(temp2['Attrition'],downcast='integer')

temp2.to_csv('outputtree.csv',index=False)

temp3 =data2

from sklearn import svm

model = svm.SVC()

model.fit(x_train, y_train)

y_svm = model.predict(x_test)

print(y_svm)

temp3['Attrition'] =y_svm

temp3['Attrition'] = pd.to_numeric(temp3['Attrition'],downcast='integer')

temp3.to_csv('outputsvm.csv',index=False)