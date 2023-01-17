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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import Imputer

from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
import pandas as pd

#oasis_cross_sectional = pd.read_csv("../input/mri-and-alzheimers/oasis_cross-sectional.csv")

oasis_longitudinal = pd.read_csv("../input/mri-and-alzheimers/oasis_longitudinal.csv")
oasis_longitudinal.describe()
oasis_longitudinal.columns
oasis_longitudinal.head(5)
oasis_longitudinal.isnull().sum()
import missingno as msno

msno.matrix(X)
msno.matrix(oasis_longitudinal)
oasis_longitudinal['SES'].fillna(oasis_longitudinal['SES'].median(), inplace = True)

oasis_longitudinal['MMSE'].fillna(oasis_longitudinal['MMSE'].median(), inplace = True)
msno.matrix(oasis_longitudinal)
y = oasis_longitudinal.CDR           
Feature = ['MRI ID', 'Group', 'Visit', 'MR Delay', 'M/F', 'Hand',

       'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']

XX = oasis_longitudinal[Feature]

X = pd.get_dummies(XX)

from sklearn.model_selection import train_test_split 

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 0)

    
train = X_train.copy()

test = X_val.copy ()
my_imputer = SimpleImputer()

train_X_imputed = pd.DataFrame(my_imputer.fit_transform(train))

test_X_imputed = pd.DataFrame(my_imputer.fit_transform(test))

# Simple example for beginers how to adress ""

import numpy as np

from sklearn                        import metrics, svm

from sklearn.linear_model           import LogisticRegression

from sklearn import preprocessing

from sklearn import utils





lab_enc = preprocessing.LabelEncoder() #label encodint

training_scores_encoded = lab_enc.fit_transform(y_train) #change target for label encoding

print(training_scores_encoded)

print(utils.multiclass.type_of_target(y_train))

print(utils.multiclass.type_of_target(y_train.astype('int')))  #make y_trian a int

print(utils.multiclass.type_of_target(training_scores_encoded)) 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from xgboost import XGBRegressor



#model2 = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=1)

model = GradientBoostingClassifier(random_state=1)

#model3 = DecisionTreeClassifier(max_depth=3, random_state=1)

#model=SGDClassifier(random_state=1)

#model=ExtraTreesClassifier(random_state=1)

#model = XGBRegressor()

# Define the models

model_1 = RandomForestClassifier(n_estimators=50, random_state=0)

model_2 = RandomForestClassifier(n_estimators=100, random_state=0)

model_3 = RandomForestClassifier(n_estimators=200, min_samples_split=20, random_state=0)

model_4 = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=1)

model.fit(train_X_imputed, training_scores_encoded)

prediction = model.predict(test_X_imputed)

print("Random Forest Results, MAE: %f" %(mean_absolute_error(y_val, prediction)))

print('model accuracy score',model.score(X_val,prediction))
