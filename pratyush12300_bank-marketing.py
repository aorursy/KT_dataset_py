import numpy as np

from plotly import tools

import chart_studio.plotly as py



import plotly.figure_factory as ff

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



#This is a work in progress file .I add to it slowly after every few day



import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data=pd.read_csv('../input/bank-marketing/bank-full.csv')

data.head()
f, ax = plt.subplots(1,2, figsize=(16,8))



colors = ["#FA5858", "#64FE2E"]

labels ="Did not Open Term Suscriptions", "Opened Term Suscriptions"



plt.suptitle('Information on Term Suscriptions', fontsize=20)



data["y"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 

                                             labels=labels, fontsize=12, startangle=25)

ax[0].set_ylabel('% of Condition of Loans', fontsize=14)

palette = ["#64FE2E", "#FA5858"]

sns.barplot(x="education", y="balance", hue="y", data=data, palette=palette, estimator=lambda x: len(x) / len(data) * 100)

ax[1].set(ylabel="(%)")

ax[1].set_xticklabels(data["education"].unique(), rotation=0, rotation_mode="anchor")

plt.show()

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

data.hist(bins=20, figsize=(14,10), color='#E14906')

plt.show()

data.describe()
#Value Counts of items in the age variable 

data.age.value_counts()
plt.figure(figsize = (20, 10))

sns.distplot(data.age)

#Categorizing the age variable

bins=[18,20,25,30,35,42,50,60,70,80,90,100] 

labels=['18 to 20','20 to 25','25 to 30','30 to 35','35 to 42','42 to 50','50 to 60','60 to 70','70 to 80','80 to 90','90 to 100']

data['age group']=pd.cut(data.age,bins=bins,labels=labels,include_lowest=True,right=False)

data['age group'].value_counts()
pd.crosstab(data['age group'],data.y)

#Job Variable

plt.figure(figsize = (15, 10))

chart=sns.countplot(data.job)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

chart
pd.crosstab(data['job'],data.y)

plt.figure(figsize = (12, 9))

sns.countplot(data.marital)
plt.figure(figsize = (9, 6))

sns.countplot(data.education)
