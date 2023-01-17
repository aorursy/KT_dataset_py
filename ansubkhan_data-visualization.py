# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.express as px

import pandas as pd



#reading the csv file



df = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

df.head()
df.describe()

#age



fig = px.histogram (df,"age", nbins=20, title='Patients Age Distribution', width=700)

fig.show()
#anaemia



ds = df['anaemia'].value_counts().reset_index()

ds.columns = ['anaemia', 'count']



fig = px.pie(ds,values='count', names="anaemia", title='Anaemia Pie Chart', width=600, height=500)

fig.show()
#creatinine_phosphokinase



fig = px.histogram (df,"creatinine_phosphokinase", nbins=10, title='Creatinine Phosphokinase', width=700)

fig.show()
#diabetes



ds = df["diabetes"].value_counts().reset_index()

ds.columns = ["diabetes",'count']



fig = px.pie(

    ds,

    values='count',

    names="diabetes",

    title="Diabetic Pie Chart", 

    width=600,

    height=500

)

fig.show()


#ejection_fraction



fig = px.histogram (df,"ejection_fraction", nbins=15, title='Ejection Fraction', width=700)

fig.show()
#high_blood_pressure



ds = df["high_blood_pressure"].value_counts().reset_index()

ds.columns = ["high_blood_pressure",'count']



fig = px.pie(

    ds,

    values='count',

    names="high_blood_pressure",

    title="Blood Pressure Pie Chart", 

    width=600,

    height=500

)

fig.show()
#platelets



fig = px.histogram (df,"platelets", nbins=40, title='Platelets', width=700)

fig.show()
#serum_creatinine



fig = px.histogram (df,"serum_creatinine", nbins=40, title='Serum Creatinine', width=700)

fig.show()
#serum_sodium



fig = px.histogram (df,"serum_sodium", nbins=25, title='Serum Sodium', width=700)

fig.show()
#sex



ds = df["sex"].value_counts().reset_index()

ds.columns = ["sex",'count']



fig = px.pie(

    ds,

    values='count',

    names="sex",

    title="Sex", 

    width=600,

    height=500

)

fig.show()
#death_event



ds = df['DEATH_EVENT'].value_counts().reset_index()

ds.columns = ['DEATH_EVENT', 'count']

fig = px.pie(

    ds, 

    values='count', 

    names="DEATH_EVENT", 

    title='DEATH_EVENT bar chart', 

    width=600, 

    height=500

)

fig.show()