# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')



data.head()
import pandas_profiling 



data.profile_report(style={'full_width':True})
# ratio of male to female having heart conditions

data1 = data

data1['sex']=data1['sex'].replace(to_replace =0,value='female')

data1['sex']=data1['sex'].replace(to_replace =1,value='male')



import plotly.graph_objects as go



fig = go.Figure(data=[

    go.Bar(name='1', x=data1['sex'], y=data1.target[data1['target']==1]),

    go.Bar(name='2', x=data1['sex'], y=data1.target[data1['target']==0])

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()



# since the data set has no target = 0 values hence only showing target=1 for male and female



# finding heartbeat ranges in men/women causing angia(exang).



data_hb_exang = data[['thalach','exang','sex','age']]
# extracting the maximum heart rate achieved data along in a different data frame

exang_men_women = data_hb_exang.loc[(data_hb_exang['exang']==1 )]



exang_men_women['sex']=exang_men_women['sex'].replace(to_replace =0,value='female')

exang_men_women['sex']=exang_men_women['sex'].replace(to_replace =1,value='male')

exang_men_women.head()
# getting age wise data of patiets having mx heart rate> 100

exang_men_women_over_100 = exang_men_women.loc[(exang_men_women['thalach']>100)]

exang_men_women_over_100.head()
# plotting male to female heart rate values

import plotly.express as px



fig = px.violin(exang_men_women, x="sex", y="thalach",color="sex",points="all",box=True,hover_data=exang_men_women.columns)



fig.show()
#Age wise plot of max heart rate values(>100)

import plotly.express as px

fig = px.histogram(exang_men_women_over_100, x="age", y="thalach", color="sex",marginal="violin")

fig.show()
#Age wise barplot of max heart rate values(>100)

import plotly.express as px



fig = px.box(exang_men_women_over_100, x="age", y="thalach",color="sex")



fig.show()
# finding male to female distribution of fasting blood suger

import plotly.express as px

fig = px.bar(data, x="sex", y="fbs")

fig.show()
# Age wise cholestoral level distribution in male and female

import plotly.express as px

fig = px.histogram(data1, x="age", y="chol", color="sex",marginal="violin")

fig.show()
# Thalium stress result data for male and female



import plotly.graph_objects as go



fig = go.Figure(data=[

    go.Bar(name='normal', x=data1['sex'], y=data1.target[data1['thal']==1]),

    go.Bar(name='fixed defect', x=data1['sex'], y=data1.target[data1['thal']==2]),

    go.Bar(name='reversible defect', x=data1['sex'], y=data1.target[data1['thal']==3])

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
# plotting the distribution of chol value and max_heart_rate achieved



import plotly.express as px



fig = px.density_contour(data1, x="thalach", y="chol", color="sex", marginal_x="violin", marginal_y="histogram")

fig.show()
data1.columns

data2 = data1.drop(['target'],axis=1)

data2.columns