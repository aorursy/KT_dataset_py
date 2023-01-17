# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_patient=pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')

df_patient.head()
df_patient.columns
df_patient['disease'].value_counts()
df_patient['sex'].isnull().sum()
df_patient['sex'].value_counts()
df_patient["sex"].value_counts().plot.pie(explode=[0.01,0.01],autopct='%.1f%%')
df_patient['country'].value_counts()
sns.countplot("country",data=df_patient)
df_patient=df_patient.drop(['patient_id'],axis=1)
df_patient.columns
df_patient["infection_case"].value_counts()
df_patient["infection_case"].value_counts().plot.pie(autopct='%.1f%%')
sns.countplot("infection_case",data=df_patient)

df_patient['infection_order'].value_counts()
df_patient.columns
df_patient['birth_year']
df_patient['age']
df_patient["age"].value_counts().plot.pie(autopct='%.1f%%')
df_patient.columns #I'm just lazy to scroll up.
df_patient=df_patient.drop(['global_num'],axis=1)
#df_patient=df_patient.drop(['birth_year'],axis=1)

df_patient=df_patient.drop(['contact_number'],axis=1)
df_patient['symptom_onset_date'].isnull().sum()

df_patient.info()
df_patient=df_patient.drop(['symptom_onset_date'],axis=1)
df_patient.columns
df_patient['state'].value_counts