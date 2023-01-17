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
import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
#Import Dataset

mush_data = pd.read_csv(os.path.join(dirname, filename))

mush_data.head()
#Data Summary

mush_data.describe()
#Check missing values

mush_data.isna().sum()
#Perform Label encoding

from collections import defaultdict

d = defaultdict(LabelEncoder)

mush_encoded = mush_data.apply(lambda x:d[x.name].fit_transform(x))
#set the seed

import random

random.seed(1111)
#Select independant Feature Variables  

features = mush_encoded.columns[mush_encoded.columns != 'class']



#Split data into train and test dataset

train_data, test_data = train_test_split(mush_encoded, test_size = 0.3)
#Define grid parameters

param_grid ={

    'n_estimators':[10,50,100,200,300,400,500],

    'max_features':['auto','sqrt']

}



rf_model = RandomForestClassifier()



grid_model = GridSearchCV(rf_model, param_grid=param_grid , cv=5)

model = grid_model.fit(train_data[features], train_data['class'])
pred = model.predict(test_data[features])

model_accuracy = accuracy_score(test_data['class'], pred)

print(model_accuracy)
#Determine Important Variables in the model

feature_importance = pd.DataFrame(model.best_estimator_.feature_importances_, index = train_data[features].columns, columns=['importance']).sort_values('importance', ascending=False)

feature_importance
#Important variable 

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

sns.barplot(x=feature_importance['importance'], y=feature_importance.index)