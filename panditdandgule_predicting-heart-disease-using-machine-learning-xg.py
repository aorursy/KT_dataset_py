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
from fastai.imports import *



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestClassifier

 

from sklearn import metrics

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd



%load_ext autoreload

%autoreload 2

%matplotlib inline

pd.options.mode.chained_assignment=None
import pandas as pd

df = pd.read_csv("../input/heartcsv/Heart.csv",index_col=0)
df.head()

df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
df['sex'][df['sex']==0]='female'

df['sex'][df['sex']==1]='male'
df['sex']
df['chest_pain_type'][df['chest_pain_type'] == 1] = 'typical angina'

df['chest_pain_type'][df['chest_pain_type'] == 2] = 'atypical angina'

df['chest_pain_type'][df['chest_pain_type'] == 3] = 'non-anginal pain'

df['chest_pain_type'][df['chest_pain_type'] == 4] = 'asymptomatic'
df['chest_pain_type'].tail()
df['fasting_blood_sugar'][df['fasting_blood_sugar']==0]='lower than 120mg/ml'

df['fasting_blood_sugar'][df['fasting_blood_sugar']==1]='greater than 120mg/ml'
df['fasting_blood_sugar']
df['rest_ecg'][df['rest_ecg']==0]='normal'

df['rest_ecg'][df['rest_ecg']==1]='ST-T wave abnormality'

df['rest_ecg'][df['rest_ecg']==2]='left ventiricular hypertrophy'
df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no'

df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes'
df['st_slope'][df['st_slope'] == 1] = 'upsloping'

df['st_slope'][df['st_slope'] == 2] = 'flat'

df['st_slope'][df['st_slope'] == 3] = 'downsloping'



df['thalassemia'][df['thalassemia'] == 1] = 'normal'

df['thalassemia'][df['thalassemia'] == 2] = 'fixed defect'

df['thalassemia'][df['thalassemia'] == 3] = 'reversable defect'
df.head()
def missing_data_ratio(df):

    all_data_na=(df.isnull().sum()/len(df))*100

    all_data_na=all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)[:30]

    missing_data=pd.DataFrame({'Missing Ratio':all_data_na})

    return missing_data
import warnings

with warnings.catch_warnings():

    warnings.filterwarnings('ignore',category=DeprecationWarning)

    import imp
import pandas_profiling
profile=pandas_profiling.ProfileReport(df)
missing_data_ratio(df)
profile
df.columns
df.chest_pain_type = df.chest_pain_type.astype("category")

df.exercise_induced_angina = df.exercise_induced_angina.astype("category")

df.fasting_blood_sugar = df.fasting_blood_sugar.astype("category")

df.rest_ecg = df.rest_ecg.astype("category")

df.sex = df.sex.astype("category")

df.st_slope = df.st_slope.astype("category")

df.thalassemia = df.thalassemia.astype("category")
df=pd.get_dummies(df,drop_first=True)
from sklearn.model_selection import RandomizedSearchCV
rf_param_grid = {

                 'max_depth' : [4, 6, 8,10],

                 'n_estimators': range(1,30),

                 'max_features': ['sqrt', 'auto', 'log2'],

                 'min_samples_split': [2, 3, 10,20],

                 'min_samples_leaf': [1, 3, 10,18],

                 'bootstrap': [True, False],

                 

                 }
m = RandomForestClassifier()
m_r = RandomizedSearchCV(param_distributions=rf_param_grid, 

                                    estimator = m, scoring = "accuracy", 

                                    verbose = 0, n_iter = 100, cv = 5)
m_r.best_score_

m_r.best_params_