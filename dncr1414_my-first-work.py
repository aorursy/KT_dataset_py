# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/StudentsPerformance.csv')

data.head()
data.info()
data.plot()
data.describe()
f,ax = plt.subplots(figsize=(6, 6))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns
data.rename(columns={'math score':'mathScore',

                    'reading score':'readingScore',

                    'writing score':'writingScore'},inplace=True)
data.columns
data.plot(kind='scatter' ,x='mathScore', y='readingScore',alpha = 0.5,color = 'red')

plt.xlabel('mathScore')              

plt.ylabel('readingScore')

plt.title('Math and reading Score')      
data.plot(kind='scatter' ,x='mathScore', y='writingScore',alpha = 0.5,color = 'green')

plt.xlabel('mathScore')              

plt.ylabel('writingScore')

plt.title('Math and Writing Score')   
data.plot(kind='scatter' ,x='readingScore', y='writingScore',alpha = 0.5,color = 'blue')

plt.xlabel('readingScore')              

plt.ylabel('writingScore')

plt.title('Read and Writing Score')   
best=data['mathScore']>85



data[best]
data.shape

data.boxplot(column="mathScore", by ="gender")
#data_new=data.head()

#data_new

#melted = pd.melt(frame=data_new,id_vars = ' ', value_vars= ['mathScore','writingScore'])

#melted
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True)

conc_data_row
data1 = data['writingScore'].head()

data2= data['readingScore'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
assert  data['mathScore'].notnull().all()

assert data['readingScore'].notnull().all()
assert data['writingScore'].notnull().all()
data1=data.loc[:,["mathScore","readingScore","writingScore"]]

data1.plot()
data1.plot(subplots=True)

plt.show()
data1.plot(kind = "scatter",x="writingScore",y = "readingScore")

plt.show()
data1.plot(kind = "hist",y = "mathScore",bins = 50,range= (0,100),normed = True)
data["totalScore"] = data.mathScore + data.readingScore + data.writingScore

data.head()
parent_come_college=(data['parental level of education']=="some college")

limit=data['totalScore']>254 



#data[limit&parent_come_college]



count=parent_come_college.value_counts() #226

succes=(limit&parent_come_college).value_counts() #26

parent_bachelor_degree=(data['parental level of education']=="bachelor's degree")

limit=data['totalScore']>254 



data[limit&parent_bachelor_degree]



count=parent_bachelor_degree.value_counts()  #118

succes=(limit&parent_bachelor_degree).value_counts()  #22
parent_master_degree=(data['parental level of education']=="master's degree")

limit=data['totalScore']>254 



data[limit&parent_master_degree]



count=parent_master_degree.value_counts() #59

succes=(limit&parent_master_degree).value_counts() #16



parent_associate_degree=(data['parental level of education']=="associate's degree")

limit=data['totalScore']>254 



data[limit&parent_associate_degree]



count=parent_associate_degree.value_counts() #222

succes=(limit&parent_associate_degree).value_counts() #34



parent_high_school=(data['parental level of education']=="high school")

limit=data['totalScore']>254 



data[limit&parent_high_school]



parent_high_school_size=parent_high_school.value_counts() #196

total=(limit&parent_high_school).value_counts() #7

import plotly.plotly as py

from plotly.offline import init_notebook_mode ,iplot 

init_notebook_mode(connected=True)

import plotly.graph_objs as go
"""trace1=go.Bar(

x=parent_high_school_size,

y=['parental level of education'], 	

name="high school",

marker=dict(color='rgb(100,100,75,0.8)'),

text=['parental level of education']

)



trace2=go.Bar(

    x=total,

    y=['parental level of education'],

    name="high school",

    marker=dict(color='rgb(45,125,180,0.7)'),

    text=['parental level of education']

)



data=[trace1,trace2]



layout=go.Layout(barmode="group")

fig=go.Figure(data=data,layout=layout)

iplot(fig)"""
data.head()
course=(data['test preparation course']=="none").value_counts() #358

course

course_succes=((data['test preparation course']=="none")&limit).value_counts() #63

course_succes
(data['lunch']=='standard').value_counts() #645

((data['lunch']=='standard')&limit).value_counts() #101
(data['lunch']=='free/reduced').value_counts() #355

((data['lunch']=='free/reduced')&limit).value_counts() #15