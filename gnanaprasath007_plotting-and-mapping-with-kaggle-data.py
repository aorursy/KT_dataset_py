# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#importing dataset

data=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv', encoding='ISO-8859-1')

data.head(10)

#importing lib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# basic count plot for Gender count



plt.figure(figsize=(12,8))

plot=sns.countplot(x='GenderSelect',data=data)

plot.set_xticklabels(plot.get_xticklabels(), fontsize=10)

plot.set_xticklabels(plot.get_xticklabels(), rotation=55)

plt.title("Gender count")
# basic violin plot. It will show Gender vs Age with the split on Coder or not



plt.figure(figsize=(12,6))

violin=sns.violinplot(x='GenderSelect',y="Age",data=data,hue='CodeWriter',split=True,width=0.8)

violin.set_xticklabels(violin.get_xticklabels(), fontsize=10)

violin.set_xticklabels(violin.get_xticklabels(), rotation=55)

plt.title("Gender vs Age plot")
#importing the dataset for choropleth. Dataset contains country and country codes

country_data=pd.read_csv("../input/country-code/Country code.csv")

country_data.head()

#changing the name of country to make country_data sync with data

country_data.at[42,'index']="People 's Republic of China"

country_data.at[108,'index']='South Korea'
#get the count of participants from the country

count=data['Country'].value_counts()

# reset index help to move the index value to columns

count=count.reset_index()

#merge the datasets

count=pd.merge(count,country_data, on='index',how='outer')

count.columns=['Country','No of particiants','Index_code','Country_code']
#importing plotly to plot choropleth

import plotly.plotly as py

import plotly.graph_objs as go



from plotly.offline import download_plotlyjs,plot,iplot,init_notebook_mode

init_notebook_mode(connected=True)
#creating data to input for choropleth

data_map=dict(type='choropleth',locations=count['Country_code'],

          colorscale='Portland',z=count['No of particiants'],text=count['Country'],

          colorbar={'title':'No of participants'})



layout=dict(title= "Kaggle survey User", geo=dict(showframe=False,projection={'type':'Mercator'}))
#plot the map

choromap=go.Figure(data=[data_map],layout=layout)

iplot(choromap)



#getting the popular tools that kaggle users going to learn

ML_data=data['MLToolNextYearSelect'].value_counts()

ML_data=ML_data.reset_index()

ML_data.columns=['Tools','Number of people interested']
#plotting the figure

plt.figure(figsize=(15,20))

sns.barplot(x='Number of people interested',y='Tools',data=ML_data)

plt.title("Tools people Interested")