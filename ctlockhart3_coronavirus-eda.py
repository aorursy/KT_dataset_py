import numpy as np 

import pandas as pd 

import seaborn as sns

from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import train_test_split







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
patient = pd.read_csv('/kaggle/input/coronavirusdataset/patient.csv')

route = pd.read_csv('/kaggle/input/coronavirusdataset/route.csv')

time = pd.read_csv('/kaggle/input/coronavirusdataset/time.csv')

trend = pd.read_csv('/kaggle/input/coronavirusdataset/trend.csv')
count = len(patient.index)

print('Count(*):', count)





patient.head()
def exploreField(col_name, df, pk):

    if col_name in df.columns and col_name != pk:

        print('======== ' + str(col_name) + ' ========')

        print('count(*) group by: ' + str(df.groupby([col_name]).size()))

        print('% null: ' + str(df[col_name].isnull().sum() / len(df.index)))

        print('# of distinct values:' + str(len(df[col_name].unique())))





for col in patient.columns:

    exploreField(col, patient, pk = 'id')

    if patient[col].isnull().sum() / len(patient.index) > 0.75:

        patient = patient.drop(labels = col, axis = 1)

    print('\n')
patient.head()