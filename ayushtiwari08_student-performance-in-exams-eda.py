



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

import plotly.io as pio

pio.templates.default = "ggplot2"



pwd
data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

data 
data.shape
print(data.info())

print(data.isna().sum())
data.describe()
data.reset_index()

data.head(10)
(data.head(100).style.set_properties(**{'background-color': '#F2F21E',

                           'color': 'black',

                           'border-color': 'white'})

 .bar(color='#C70039', vmin=0, subset=['math score'])

 .bar(color='#1790BF', vmin=0, subset=['reading score'])

 .bar(color='#DA2C85', vmin=0, subset=['writing score'])

 .set_caption('Data Visualization in Pandas'))


px.pie(data,"gender",title='Number Of Male & Females In Data')

px.pie(data,"race/ethnicity",color="race/ethnicity",title="Percent Of Students Inside Each Groups")
px.pie(data,"parental level of education",title="Level Of Education")
px.pie(data,"test preparation course",color="test preparation course",title="How many Completed The Test Preparation Code")


sns.pairplot(data, hue ="gender", palette ='coolwarm')
d=data["math score"]

sns.distplot(d,kde=False, rug=False,color="Blue")
d=data["reading score"]

sns.distplot(d,kde=False, rug=False,color="lawngreen")
d=data["writing score"]

sns.distplot(d,kde=False, rug=False,color="gray")
data.groupby('parental level of education')['math score', 'reading score', 'writing score'].mean().plot(kind = 'bar')
cplot = sns.FacetGrid(data, col='parental level of education', hue='gender', col_wrap=3, height = 5)

cplot.map(sns.scatterplot, 'reading score', 'writing score' );



x=data.groupby('race/ethnicity')['test preparation course'].value_counts().plot(kind = 'bar', colormap='Set2')

plt.ylabel('Count')

plt.figure(figsize=(20,8))

plt.subplot(1, 3, 1)

sns.barplot(x='test preparation course',y='math score',data=data,hue='gender',palette="YlGn")

plt.title('MATH SCORES')

plt.subplot(1, 3, 2)

sns.barplot(x='test preparation course',y='reading score',data=data,hue='gender',palette="YlGnBu")

plt.title('READING SCORES')

plt.subplot(1, 3, 3)

sns.barplot(x='test preparation course',y='writing score',data=data,hue='gender',palette="YlOrRd")

plt.title('WRITING SCORES')

plt.show()
plt.figure(figsize=(20,8))

plt.subplot(1, 3, 1)

sns.barplot(x='test preparation course',y='math score',data=data,hue='lunch',palette="Spectral")

plt.title('MATH SCORES')

plt.subplot(1, 3, 2)

sns.barplot(x='test preparation course',y='reading score',data=data,hue='lunch',palette="Spectral")

plt.title('READING SCORES')

plt.subplot(1, 3, 3)

sns.barplot(x='test preparation course',y='writing score',data=data,hue='lunch',palette="Spectral")

plt.title('WRITING SCORES')

plt.show()
plt.figure(figsize=(20,8))

plt.subplot(1, 3, 1)

plt.title('MATH SCORES')

sns.barplot(x='race/ethnicity',y='math score',data=data,hue='gender',palette='Paired')

plt.subplot(1, 3, 2)

plt.title('READING SCORES')

sns.barplot(x='race/ethnicity',y='reading score',data=data,hue='gender',palette='Paired')

plt.subplot(1, 3, 3)

plt.title('WRITING SCORES')

sns.barplot(x='race/ethnicity',y='writing score',data=data,hue='gender',palette='Paired')

plt.show()
fig = px.scatter(data, x='writing score', y='reading score', color='parental level of education', opacity=0.5)

fig.show()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

data['parental level of education'] = LE.fit_transform(data['parental level of education'])

fig = px.scatter(data, x='writing score', y='reading score', 

                 color='parental level of education',

                 size='parental level of education',

                )

fig.show()
sns.jointplot(data=data, x='writing score', y='reading score', kind='reg', color='skyblue')

sns.jointplot(data=data, x='writing score', y='reading score', kind='hex', color='gold')

sns.jointplot(data=data, x='writing score', y='reading score', kind='kde', color='forestgreen' )

plt.show()
male_mean = data[["math score", "reading score", "writing score"]].mean()

female_mean = data[["math score", "reading score", "writing score"]].mean()

mean_scores_by_gender = pd.concat([male_mean, female_mean], axis = 1, names = ["test", "lol"])

mean_scores_by_gender.columns = ["Male Mean", "Female Mean"] 

display(mean_scores_by_gender)
# Results based on the lunch type

dataset_lunch = data[["lunch", "math score", "reading score", "writing score"]].copy()

dataset_lunch = dataset_lunch.groupby(by = ["lunch"]).mean()

# Display the table and the heatmap

display(dataset_lunch)

fig5, ax10 = plt.subplots(figsize=(12.8, 6))

sns.heatmap(dataset_lunch,linewidths=.1, ax=ax10)
dataset_preparation = data[["test preparation course", "math score", "reading score", "writing score"]].copy()

dataset_preparation = dataset_preparation.groupby(by = ["test preparation course"]).mean()

display(dataset_preparation)