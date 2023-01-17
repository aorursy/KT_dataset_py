import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #visualization



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/so-survey-2017/survey_results_public.csv')

schema=pd.read_csv('/kaggle/input/so-survey-2017/survey_results_schema.csv')

#display the first five rows of the dataset

df.head()
#shape of the dataset

df.shape
#display the first five rows of the schema dataset

schema.head()
#shape of the schema dataset

schema.shape
#List of all the coloumns with missing values

print(df.columns[df.isnull().any()].to_list()[:10])

print('Number of columns with missing values:',len(df.columns[df.isnull().any()].to_list()))
#columns with more than 75% missing values

print(df.columns[df.isnull().mean()>0.75].to_list())

print('Number of columns with more the 75% missing values:',len(df.columns[df.isnull().mean()>0.75].to_list()))
#unique professionals

print(df["Professional"].unique().tolist())

print('There are {} unique Professional in the provided dataset'.format(len(df['Professional'].unique().tolist())))
professional_stats=df['Professional'].value_counts()

(professional_stats/df.shape[0]).plot(kind='bar')

plt.title("What kind of developer are you?")

#Formal Education

print(df['FormalEducation'].unique().tolist())
Formaledu_stats=df['FormalEducation'].value_counts()

(Formaledu_stats/df.shape[0]).plot(kind='bar')

plt.title("Formal Education")
