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
#reading the files

train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")

print(train.columns)
a = (train[train.columns[15:-1]]==1).sum() 

b = (test[test.columns[15:]]==1).sum() 

print(pd.concat([a.rename('train'),b.rename('test')], axis=1))
#Seems like cols Soil_Type7 and Soil_Type15 can be dropped without much affecting accuracy

c = (train[train.columns[11:15]]==1).sum() 

d = (test[test.columns[11:15]]==1).sum() 

print(pd.concat([c.rename('train'),d.rename('test')], axis=1))
#The distribution of Wilderness Area appears to be ok.

#dropping Soil_Type7 and Soil_Type15

train = train.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)

test_id = test['Id']

test = test.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)
#prepare data for training the model

X = train.drop(['Cover_Type'], axis = 1)

y = train.Cover_Type
#reducing Soil_Type cols to single col 

X = X.iloc[:, :14].join(X.iloc[:, 14:].dot(range(1,39)).to_frame('Soil_Type1'))

test = test.iloc[:, :14].join(test.iloc[:, 14:].dot(range(1,39)).to_frame('Soil_Type1'))

print(X.columns)
#reducing Wilderness_Area to single col 

X = X.iloc[:,:10].join(X.iloc[:,10:-1].dot(range(1,5)).to_frame('Wilderness_Area1')).join(X.iloc[:,-1])

test = test.iloc[:,:10].join(test.iloc[:,10:-1].dot(range(1,5)).to_frame('Wilderness_Area1')).join(test.iloc[:,-1])

print(X.columns)
#horizontal and vertical distance to hydrology can be easily combined

cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']

X['Distance_to_hydrology'] = X[cols].apply(np.linalg.norm, axis=1)

X = X.drop(cols, axis = 1)

test['Distance_to_hydrology'] = test[cols].apply(np.linalg.norm, axis=1)

test = test.drop(cols, axis = 1)
#another shot in the dark - convert Hillshade like colour tuples to grayscale

cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']

weights = pd.Series([0.299, 0.587, 0.114], index=cols)

X['Hillshade'] = (X[cols]*weights).sum(1)

X = X.drop(cols, axis = 1)

test['Hillshade'] = (test[cols]*weights).sum(1)

test = test.drop(cols, axis=1)
from sklearn.model_selection import train_test_split

#split data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
#preparing model

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier()

params_rf = {'n_estimators': [10, 50, 75]}

rf_gs = GridSearchCV(model, params_rf, cv=8)

rf_gs.fit(X_train,y_train)
from sklearn.metrics import mean_absolute_error

#get the error rate

val_predictions = rf_gs.predict(X_val)

val_mae = mean_absolute_error(y_val,val_predictions)

print('Fifth try mae with RFClassifier: ', val_mae)
test_preds = rf_gs.predict(test)

output = pd.DataFrame({'Id': test_id, 'Cover_Type': test_preds.astype(int)})

output.to_csv('submission.csv', index=False)