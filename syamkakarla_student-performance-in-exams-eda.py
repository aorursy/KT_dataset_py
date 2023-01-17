 #!/usr/bin/env python -W ignore::DeprecationWarning



# Data Handling 

import pandas as pd

import numpy as np

from itertools import combinations



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from IPython.display import HTML

plt.rcParams['figure.figsize'] = (14, 8)

sns.set_style('whitegrid')



pwd
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.shape
df.info()
df.describe()
df.reset_index()

df.head(10)
(df.head(20)

 .style

 .hide_index()

 .bar(color='#70A1D7', vmin=0, subset=['math score'])

 .bar(color='#FF6F61', vmin=0, subset=['reading score'])

 .bar(color='mediumspringgreen', vmin=0, subset=['writing score'])

 .set_caption(''))


for attribute in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:

  f, ax = plt.subplots(1,2)

  data = df[attribute].value_counts().sort_index()

  bar = sns.barplot(x = data.index, y = data, ax = ax[0], palette="Set2",)

  for item in bar.get_xticklabels():

    item.set_rotation(45)

  ax[1].pie(data.values.tolist() , labels= [i.title() for i in data.index.tolist()], autopct='%1.1f%%',shadow=True, startangle=90);

  plt.show()


for lab, col in zip(['math score', 'reading score', 'writing score'], ['tomato', 'mediumspringgreen', 'blue']):

  sns.distplot(df[lab], label=lab.title(), color = col, ).set(xlabel=lab.title(), ylabel='Count')

  plt.show()



for attr, col in zip(list(combinations(['math score', 'reading score', 'writing score'], 2)), ['#77DF79', '#82B3FF', '#F47C7C']):

  sns.jointplot(df[attr[0]], df[attr[1]], color = col)

  plt.show()


df.groupby('parental level of education')['math score', 'reading score', 'writing score'].mean().plot(kind = 'bar');



cond_plot = sns.FacetGrid(data=df, col='parental level of education', hue='gender', col_wrap=3, height = 5)

cond_plot.map(sns.scatterplot, 'reading score', 'writing score' );
df.groupby('race/ethnicity')['test preparation course'].value_counts().plot(kind = 'bar', colormap='Set2')

plt.ylabel('Count');