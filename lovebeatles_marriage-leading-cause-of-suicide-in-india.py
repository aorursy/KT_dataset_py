#python libraries  



import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Reading the data 

data=pd.read_csv('../input/Suicides in India 2001-2012.csv')



#first ten rows of the data

data.head(10)



#description of the data

data.describe()



#To check if there is any null vlaue in the data

data.isnull().any()



# Data preprocessing and modification



data.drop(data.index[209476:210412], inplace=True)   #deleting some rows having redundant values 

data.count()







#Barplot showing the average number of suicides by different causes mentioned in the data 



sns.set_style('whitegrid')

sns.barplot(x='Type_code',y='Total',data=data,color='orange').set_title('No of suicides by diiferent causes')

plt.xticks(rotation=20)

plt.ylabel('Average number of suicides')

plt.show()



data.groupby('Year').mean()['Total'].plot(figsize=(12,6),title='Year wise suicides frequency',colormap='cubehelix')

plt.ylabel('Average no of suicides')





#Barplot showing the average no of suicides state wise in decreasing order 



data1=data.groupby(['State','Gender'])['Total'].mean().reset_index().sort_values('Total',ascending=False) 

sns.set_style('white')

plt.subplots(figsize=(8,30))

g = sns.barplot(y=data1['State'],x='Total',data=data1,hue='Gender',palette="Set2").set_title('State wise number of average suicides in India')

plt.xlabel('Average number of suicides')









import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()







a1=data.loc[data.State=='Maharashtra']

a2=data.loc[data.State=='Andhra Pradesh']

a3=data.loc[data.State=='Tamil Nadu']

a4=data.loc[data.State=='Karnataka']

a5=data.loc[data.State=='West Bengal']



a1=a1.append(a2)

a1=a1.append(a3)

a1=a1.append(a4)

a1=a1.append(a5)



trace = go.Heatmap(z=a1['Total'],

                   x=a1['Year'],

                   y=a1['State'])

d=[trace]

figure = dict(data = d)

iplot(figure)



# Barplot showing the major causes of suicide 



data1=data.loc[data.Type_code!='Means_adopted']

data1=data1.groupby(['Type','Type_code','Gender'])['Total'].mean().reset_index().sort_values('Total',ascending=False)

data1.set_index(['Type','Type_code'])

data1=data1.set_index(['Type','Type_code'])

sns.set_style('white')

plt.subplots(figsize=(8,30))

g = sns.barplot(y=data1.index,x='Total',data=data1,palette="Set2",hue='Gender').set_title('Leading causes of suicide in India ')

plt.xlabel('Average number of suicides')

means=data.loc[data.Type_code=='Means_adopted']

s = means.groupby(['Type'])['Total'].mean().reset_index().sort_values('Total',ascending=False)

sns.barplot(x='Type',y='Total',data=s,color='orange')

plt.xticks(rotation=75)


#suicides based on gender 

s = data.groupby(['Gender'])['Total'].mean()

s.plot.pie(colors=['orange', 'grey'],autopct='%.2f', fontsize=20, figsize=(6, 6))

plt.title('Percentage of male and female suicide in india')







# suicides based on the age group

data2=data[data['Age_group']!='0-100+'] #excluding the age group of 0-100+ due to redundancy

sns.barplot(x='Age_group',y='Total',hue='Gender',data=data2,color='grey').set_title('Suicides based on Age group')

plt.ylabel('Average no of suicides ')





agegroup=data[data['Age_group']!='0-100+']



adults=data.loc[data.Age_group=='15-29']

adults1=adults.loc[adults.Type_code!='Means_adopted']

young1=adults1.groupby(['Type'])['Total'].mean().reset_index().sort_values('Total',ascending=False)



plt.subplots(figsize=(8,15))

sns.barplot(x='Total',y='Type',data=young1,palette="Set2").set_title('Causes of suicides of young adults ')

plt.xlabel('Suicide rate ')













child=data.loc[data.Age_group=='0-14']

child1=child.groupby(['State'])['Total'].sum().reset_index().sort_values('Total',ascending=False)

child1[0:5]







child=data.loc[data.Age_group=='0-14']

child1=child.loc[child.Type_code!='Means_adopted']

child1=child1.groupby(['Type'])['Total'].mean().reset_index().sort_values('Total',ascending=False)

plt.subplots(figsize=(8,15))

sns.barplot(x='Total',y='Type',data=child1,palette="Set2")

plt.xlabel('average no of child suicide')
oldage=data.loc[data.Age_group=='60+']

oldage2=oldage.groupby(['State'])['Total'].mean().reset_index().sort_values('Total',ascending=False)

oldage2[0:5]





#barplot of old age showing no of suicides 



oldage1=oldage.loc[oldage.Type_code!='Means_adopted']

oldage2=oldage1.groupby(['Type'])['Total'].mean().reset_index().sort_values('Total',ascending=False)



plt.subplots(figsize=(8,15))

sns.barplot(x='Total',y='Type',data=oldage2,color="orange")

plt.xlabel('average no of suicide')







a=data.loc[data.Age_group=='30-44']

a1=a.loc[a.Type_code!='Means_adopted']

a2=a1.groupby(['Type'])['Total'].mean().reset_index().sort_values('Total',ascending=False)

plt.subplots(figsize=(8,15))

sns.barplot(x='Total',y='Type',data=a2,palette="Set2")

plt.xlabel('average no of suicide')





# Death by Poverty

Poverty=data.loc[data.Type=='Poverty']

Poverty1=Poverty.groupby(['State'])['Total'].mean().reset_index().sort_values('Total',ascending=False)

Poverty2=Poverty1[0:5]









import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()







labels = Poverty2['State']

values = Poverty2['Total']

trace = go.Pie(labels=labels, values=values)

d1=[trace]

figure = dict(data = d1)

iplot(figure)

















data.replace('Unemployment','Unemployed',inplace=True)

unemployment=data.loc[(data.Type=='Unemployed')] 

unemployment1=unemployment.groupby(['State'])['Total'].mean().reset_index().sort_values('Total',ascending=False)

unemployment2=unemployment1[0:5]







labels = unemployment2['State']

values = unemployment2['Total']

trace = go.Pie(labels=labels, values=values)

d2=[trace]

figure = dict(data = d2)

iplot(figure)









abuse=data.loc[(data.Type=='Love Affairs')] 

abuse1=abuse.groupby(['State'])['Total'].mean().reset_index().sort_values('Total',ascending=False)

abuse2=abuse1[0:5]



labels = abuse2['State']

values = abuse2['Total']

trace = go.Pie(labels=labels, values=values)

d3=[trace]

figure = dict(data = d3)

iplot(figure)



















#heatmap showing the realtion between the causes of suicide and year wise suicide frequency 







a1=data.loc[data.State=='Maharashtra']

a2=data.loc[data.State=='Andhra Pradesh']

a3=data.loc[data.State=='Tamil Nadu']

a4=data.loc[data.State=='Karnataka']

a5=data.loc[data.State=='West Bengal']



a1=a1.append(a2)

a1=a1.append(a3)

a1=a1.append(a4)

a1=a1.append(a5)



s = a1.groupby(['Type','Year'])['Total'].mean().reset_index().sort_values('Total',ascending=False)



trace = go.Heatmap(z=s['Total'],

                   x=s['Year'],

                   y=s['Type'])

d=[trace]

figure = dict(data = d)

iplot(figure)
















