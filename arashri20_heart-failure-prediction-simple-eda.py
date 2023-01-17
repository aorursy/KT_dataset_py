# all import statements

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt 

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



%matplotlib inline

pd.set_option('display.max_rows', 100)

sns.set(color_codes=True)
heartDF = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

heartDF.head()
heartDF.shape
heartDF.info()
heartDF.describe()
heartDF.isnull().sum()
heartDF.head()
heartDF.DEATH_EVENT = np.where(heartDF.DEATH_EVENT==1, 'Death', 'Alive')

heartDF.sex = np.where(heartDF.sex==1, 'Male', 'Female')

heartDF.smoking = np.where(heartDF.smoking==1, True, False)

heartDF.head()
numerical_columns = list(heartDF.select_dtypes(include=np.number).columns)

for feature in numerical_columns:

    fig, ax = plt.subplots()

    heartDF.plot(kind='box', y=feature, ax=ax)

plt.show()
heartDF = heartDF.loc[(heartDF['serum_creatinine'] < 3) &

                      (heartDF['platelets'] < 600000) &

                      (heartDF['creatinine_phosphokinase'] < 1500) &

                      (heartDF['ejection_fraction'] < 70)]
for feature in numerical_columns:

    fig, ax = plt.subplots()

    heartDF.plot(kind='box', y=feature, ax=ax)

plt.show()
heartDF.shape
for feature in numerical_columns:

    fig = px.histogram(heartDF, x=feature, color='DEATH_EVENT')

    fig.show()
for feature in numerical_columns:

    ax = sns.violinplot(y=heartDF[feature], x=heartDF['sex'],

                        hue=heartDF['DEATH_EVENT'], split=True)

    plt.show()