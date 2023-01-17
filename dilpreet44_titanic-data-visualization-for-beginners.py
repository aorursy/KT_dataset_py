import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sea

import plotly.express as px

import plotly.graph_objects as go
train_data = pd.read_csv('../input/titanic/train.csv')
train_data.head()
train_data.shape
train_data.isnull().sum()
mean = train_data['Age'].mean()

train_data['Age'] = train_data['Age'].fillna(mean)
train_data = train_data.drop(['Ticket'],axis = 1)
train_data = train_data.drop(['PassengerId'],axis = 1)
sea.heatmap(train_data.corr(),annot = True)
sea.pairplot(train_data)

sea.set(style="ticks", color_codes=True)
sea.countplot(train_data['Survived'])
#plotting the graph sex vs survived

plot_1 = train_data[['Sex','Survived']].groupby(['Sex'],as_index=False).sum().sort_values(by='Survived',ascending=False)

fig = px.bar(plot_1,x = plot_1['Sex'],y = plot_1['Survived'],color = 'Sex',text = 'Survived')

fig_1 = px.pie(plot_1,names = 'Sex',values = 'Survived',color = 'Sex')

fig_1.update_layout(width = 500,height = 500)

fig.update_layout(width = 500,height = 500)

fig_1.update_traces(pull = 0.05)

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig_1.show()

fig.show()
plot_2 = train_data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).sum().sort_values(by='Survived',ascending=False)

fig = px.bar(plot_2,x = 'Pclass',y = 'Survived',color = 'Pclass',text = 'Survived')

fig_1 = px.pie(plot_2,names = 'Pclass',values = 'Survived',color = 'Pclass')

fig.update_layout(width = 500,height = 500)

fig_1.update_layout(width = 500,height = 500)

fig_1.update_traces(pull = 0.05)

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig_1.show()

fig.show()
g = sea.FacetGrid(train_data,col = 'Survived')

g.map(plt.hist,'Age', bins = 20)
g = sea.FacetGrid(train_data,col = 'Survived',row = 'Sex')

g.map(plt.hist,'Age', bins = 20)
plot_4 = train_data[['Embarked','Survived']].groupby(['Embarked'],as_index=False).sum().sort_values(by='Survived',ascending=False)

fig = px.bar(plot_4,x = 'Embarked',y = 'Survived',color = 'Embarked',text = 'Survived')

fig_1 = px.pie(plot_4,names = 'Embarked',values = 'Survived',color = 'Embarked')

fig.update_layout(width = 500,height = 500)

fig_1.update_layout(width = 500,height = 500)

fig_1.update_traces(pull = 0.05)

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig_1.show()

fig.show()
grid2 = sea.FacetGrid(train_data,row='Embarked')

grid2.map(sea.lineplot,'Pclass','Survived','Sex')
freq_port = train_data.Embarked.dropna().mode()[0]

train_data['Embarked'] = train_data['Embarked'].fillna(freq_port)
grid = sea.FacetGrid(train_data, row='Embarked', col='Survived', height=2.2, aspect=1.6)

grid.map(sea.barplot, 'Sex', 'Fare',alpha = 1)
plot_4 = train_data[['SibSp','Survived']].groupby(['SibSp'],as_index=False).sum().sort_values(by='Survived',ascending=False)

fig = px.line(plot_4,x = 'SibSp',y = 'Survived')

fig_1 = px.pie(plot_4,names = 'SibSp',values = 'Survived',color = 'SibSp')

fig.update_layout(width = 500,height = 500)

fig_1.update_layout(width = 500,height = 500)

fig_1.update_traces(pull = 0.05)

#fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig_1.show()

fig.show()