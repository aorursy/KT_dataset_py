import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as px

import plotly.graph_objs as go 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
# Importing the data with pandas

df = pd.read_csv("../input/StudentsPerformance.csv")
df.head()
df.info()
df.describe()
df.isnull().values.any()
px.violin(df, y="math score", x="gender", color="gender", box=True)
px.violin(df, y="reading score", x="gender", color="gender", box=True)
px.violin(df, y="writing score", x="gender", color="gender", box=True)
np.sum(df.gender =='female')
df['avg_grade'] = 0

df['avg_grade'] = df.apply(lambda x : (df['math score'] + df['writing score'] + df['reading score'])/3, axis = 0)

df.head()
px.violin(df, y="avg_grade", x="gender", color="gender", box=True, labels={'avg_grade': 'Average of the 3 scores', 'gender': 'Student gender'}, title='Average of the 3 scores for female and male students')
print('female', np.average(df.avg_grade[df.gender == 'female']), 'vs male', np.average(df.avg_grade[df.gender == 'male']))
fig = px.scatter(df, y="avg_grade", x="race/ethnicity", color="race/ethnicity", labels={'avg_grade': 'Average of the 3 scores', 'race/ethnicity': 'Race/Ethnicity of the student'})



fig.update(layout = dict(showlegend = False))
fig = px.violin(df, y="race/ethnicity", x="avg_grade", color="race/ethnicity", orientation = 'h', 

                labels={'avg_grade': 'Average of the 3 scores', 'race/ethnicity': 'Race/Ethnicity of the student'})



fig.update(layout = dict(showlegend = False))
df["race/ethnicity"].value_counts()
print('group A', np.average(df.avg_grade[df["race/ethnicity"] == 'group A']), '\nvs group B', 

      np.average(df.avg_grade[df["race/ethnicity"] == 'group B']), '\nvs group C',

     np.average(df.avg_grade[df["race/ethnicity"] == 'group C']), '\nvs group D',

     np.average(df.avg_grade[df["race/ethnicity"] == 'group D']), '\nvs group E',

     np.average(df.avg_grade[df["race/ethnicity"] == 'group E']))
fig = px.violin(df, y="parental level of education", x="avg_grade", color="parental level of education", 

          orientation = 'h', labels={'avg_grade': 'Average of the 3 scores', 'parental level of education': 'Parental level of education'})



fig.update(layout = dict(showlegend = False))
df["parental level of education"].value_counts()
px.violin(df, y="avg_grade", x="test preparation course",box = True,  color="test preparation course", orientation = 'v', labels={'avg_grade': 'Average of the 3 scores', 'test preparation course': 'Preparation Course'})
df["test preparation course"].value_counts()
px.violin(df, y="avg_grade", x="lunch", color = "lunch", box = True, orientation = 'v', labels={'avg_grade': 'Average of the 3 scores', 'lunch': 'Lunch'})
df["lunch"].value_counts()
px.violin(df, y="avg_grade", x="lunch", color="test preparation course", box = True, orientation = 'v', labels={'avg_grade': 'Average of the 3 scores', 'lunch': 'Lunch'})