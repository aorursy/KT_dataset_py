import pandas as pd 

import numpy as np

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('dark')

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

print('Setup complete')
df = pd.read_csv('..//input//covid19-patient-precondition-dataset//covid.csv', index_col='id')

df.head()
df.info()
df.corr()
sns.heatmap(df.corr())
ELD = np.zeros_like(df['diabetes'].values, dtype='int32')



for col in df.columns[9:19]:

    uniques = df[col].unique()

    uniques = np.sort(uniques)

    ELD += df[col].replace(uniques[1:], 0).values

df['ELD_indx'] = ELD
sns.heatmap(df.drop(df.columns[9:19], axis=1).corr())
df = df.drop(df.columns[9:19], axis=1)
from datetime import datetime

def convert_date(day, first_day="01-01-2020", sep='-'):

    d1 = first_day.replace('-', sep)

    fmt = f'%d{sep}%m{sep}%Y'

    d1 = datetime.strptime(d1, fmt)

    d2 = datetime.strptime(day, fmt)

    delta = d2 - d1

    return delta.days
df['date_died'] = df['date_died'].replace('9999-99-99', 0)

df['day_died'] = df['date_died'].apply(lambda date: np.NaN if date == 0 else convert_date(date))



df['entry_date'] = df['entry_date'].replace('9999-99-99', 0)

df['entry_day'] = df['entry_date'].apply(lambda date: np.NaN if date == 0 else convert_date(date))



df['date_symptoms'] = df['date_symptoms'].replace('9999-99-99', 0)

df['day_symptoms'] = df['date_symptoms'].apply(lambda date: np.NaN if date == 0 else convert_date(date))
df['died'] = df['date_died'].apply(lambda x: 0 if x == 0 else 1)
df.head()