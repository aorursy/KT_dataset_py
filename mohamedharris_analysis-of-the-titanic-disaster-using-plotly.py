import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #simple visualizations

%matplotlib inline

import plotly.offline as py #interactive visualizations

from plotly.offline import init_notebook_mode, iplot #plotting plotly graphs in notebook

import plotly.graph_objs as go #advanced graph objects



import warnings

warnings.filterwarnings('ignore') #ignore warnings



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Reading train and test datasets



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



#Storing the original copies for future use

train_orig = train.copy()

test_orig = test.copy()
#Checking the shape of training and testing datasets



print('Training Data :', train.shape)

print('Testing Data :', test.shape)
#Checking the info of the datasets



print(train.info())

print('-' * 30)

print(test.info())
#Importing missingno library to plot a bar graph of #missing values in each feature



import missingno as msno

msno.bar(train)

plt.show()
msno.bar(test)

plt.show()
#Descriptive statistics of the training dataset



train.describe()
train.head()
init_notebook_mode(connected = True)
#Check the number of passengers survived and not survived



survived = train['Survived'].value_counts()



labels = survived.index

values = survived.values
#Plotting a Pie Graph with the number of passengers survived and not survived



data = go.Pie(labels = labels,

      values = values,

      hole = 0.4,

      name = 'Survival Status',

      marker = dict(colors = ['seagreen', 'blue'],line = dict(width = 1.5, color = 'black')))



layout = go.Layout(title = 'Survival Status', legend = dict(orientation = 'h'))



fig = go.Figure(data = [data], layout = layout)

iplot(fig)
#Splitting the train dataset based on the survival status of the passengers



surv = train.loc[train['Survived'] == 1, :]

n_surv = train.loc[train['Survived'] == 0, :]
#Creating two traces with each one for survival status (0 and 1) and plotting boxplots together



trace1 = go.Box(y = surv['Age'],

             name = 'Survived',

             marker = dict(color = '#63ace0'))



trace2 = go.Box(y = n_surv['Age'],

             name = 'Not Survived',

             marker = dict(color = '#169c4e'))



layout = go.Layout(title = 'Survival by Age Group',

                  xaxis = dict(title = 'Survival Status'),

                  yaxis = dict(title = 'Age'))



fig = go.Figure(data = [trace1, trace2], layout = layout)

iplot(fig)
# Creating two traces of survival status and plotting boxplots with fare as a feature of interest



trace1 = go.Box(y = surv['Fare'],

             name = 'Survived',

             marker = dict(color = '#63ace0'))



trace2 = go.Box(y = n_surv['Fare'],

             name = 'Not Survived',

             marker = dict(color = '#169c4e'))



layout = go.Layout(title = 'Survival by Fare Group',

                  xaxis = dict(title = 'Survival Status'),

                  yaxis = dict(title = 'Fare'))



fig = go.Figure(data = [trace1, trace2], layout = layout)

iplot(fig)
#Importing make_subplots to create graphs as sub plots



from plotly.subplots import make_subplots
# Checking the number of passengers survived based on the gender



surv_gender = surv['Sex'].value_counts()

n_surv_gender = n_surv['Sex'].value_counts()
# Create two Pie graphs for each of the survival status and plotting as sub plots



fig = make_subplots(rows = 1, cols = 2, specs = [[{'type' : 'domain'}, {'type' : 'domain'}]])



fig.add_trace(go.Pie(labels = surv_gender.index,

       values = surv_gender.values,

       name = 'Survived',

       marker = dict(colors = ['seagreen', 'blue'])), 1, 1)



fig.add_trace(go.Pie(labels = n_surv_gender.index,

       values = n_surv_gender.values,

       name = 'Not Survived',

       marker = dict(colors = ['seagreen', 'blue'])), 1, 2)



fig.update_traces(hole = 0.3)



fig.update_layout(title_text = 'Survival by Gender', legend = dict(orientation = 'h'),

                 annotations = [dict(text = 'Survived', showarrow = False, x = 0.19),

                            dict(text = 'Not Survived', showarrow = False, x = 0.82)])



fig.data[0].marker.line.width = 1.5

fig.data[1].marker.line.width = 1.5



fig.data[0].marker.line.color = 'black'

fig.data[1].marker.line.color = 'black'



fig.show()
# Splitting the dataset based on the passenger class



pclass1 = train.loc[train['Pclass'] == 1, :]

pclass2 = train.loc[train['Pclass'] == 2, :]

pclass3 = train.loc[train['Pclass'] == 3, :]
# Replacing the survival status with more meaningful values



pclass1['Survived'].replace({0 : 'Not Survived', 1 : 'Survived'}, inplace = True)

pclass2['Survived'].replace({0 : 'Not Survived', 1 : 'Survived'}, inplace = True)

pclass3['Survived'].replace({0 : 'Not Survived', 1 : 'Survived'}, inplace = True)
#Checking the #passengers in each of the passenger class



sur_pclass1 = pclass1['Survived'].value_counts(normalize = True)

sur_pclass2 = pclass2['Survived'].value_counts(normalize = True)

sur_pclass3 = pclass3['Survived'].value_counts(normalize = True)
# Create subplots and plot pie graphs for each of the passenger classes



fig = make_subplots(rows = 1, cols = 3, specs = [[{'type' : 'domain'}, {'type' : 'domain'}, {'type' : 'domain'}]])



fig.add_trace(go.Pie(labels = sur_pclass1.index,

       values = sur_pclass1.values,

       name = 'Class 1',

       marker = dict(colors = ['seagreen', 'blue'])), 1, 1)



fig.add_trace(go.Pie(labels = sur_pclass2.index,

       values = sur_pclass2.values,

       name = 'Class 2',

       marker = dict(colors = ['seagreen', 'blue'])), 1, 2)



fig.add_trace(go.Pie(labels = sur_pclass3.index,

       values = sur_pclass3.values,

       name = 'Class 3',

       marker = dict(colors = ['seagreen', 'blue'])), 1, 3)



fig.update_traces(hole = 0.3)



fig.update_layout(title_text = 'Survival by Passenger Class', legend = dict(orientation = 'h'),

                 annotations = [dict(text = 'Class 1', showarrow = False, x = 0.12),

                            dict(text = 'Class 2', showarrow = False, x = 0.50),

                               dict(text = 'Class 3', showarrow = False, x = 0.87)])



fig.data[0].marker.line.width = 1.5

fig.data[1].marker.line.width = 1.5

fig.data[2].marker.line.width = 1.5



fig.data[0].marker.line.color = 'black'

fig.data[1].marker.line.color = 'black'

fig.data[2].marker.line.color = 'black'



fig.show()
southampton = train.loc[train['Embarked'] == 'S', :]

queens = train.loc[train['Embarked'] == 'Q', :]

cherbourg = train.loc[train['Embarked'] == 'C', :]
southampton['Survived'].replace({0 : 'Not Survived', 1 : 'Survived'}, inplace = True)

queens['Survived'].replace({0 : 'Not Survived', 1 : 'Survived'}, inplace = True)

cherbourg['Survived'].replace({0 : 'Not Survived', 1 : 'Survived'}, inplace = True)
sur_southampton = southampton['Survived'].value_counts(normalize = True)

sur_queens = queens['Survived'].value_counts(normalize = True)

sur_cherbourg = cherbourg['Survived'].value_counts(normalize = True)
fig = make_subplots(rows = 1, cols = 3, specs = [[{'type' : 'domain'}, {'type' : 'domain'}, {'type' : 'domain'}]])



fig.add_trace(go.Pie(labels = sur_southampton.index,

       values = sur_southampton.values,

       name = 'Southampton',

       marker = dict(colors = ['seagreen', 'blue'])), 1, 1)



fig.add_trace(go.Pie(labels = sur_queens.index,

       values = sur_queens.values,

       name = 'Queenstown',

       marker = dict(colors = ['seagreen', 'blue'])), 1, 2)



fig.add_trace(go.Pie(labels = sur_cherbourg.index,

       values = sur_cherbourg.values,

       name = 'Cherbourg',

       marker = dict(colors = ['seagreen', 'blue'])), 1, 3)



fig.update_traces(hole = 0.3)



fig.update_layout(title_text = 'Survival by Port of Embarkation', legend = dict(orientation = 'h'),

                 annotations = [dict(text = 'Southampton', showarrow = False, x = 0.11),

                            dict(text = 'QueensTown', showarrow = False, x = 0.50),

                               dict(text = 'Cherbourg', showarrow = False, x = 0.88)])



fig.data[0].marker.line.width = 1.5

fig.data[1].marker.line.width = 1.5

fig.data[2].marker.line.width = 1.5



fig.data[0].marker.line.color = 'black'

fig.data[1].marker.line.color = 'black'

fig.data[2].marker.line.color = 'black'



fig.show()
surv_sib = train.loc[train['Survived'] == 1, :]

n_surv_sib = train.loc[train['Survived'] == 0, :]
fig = make_subplots(rows = 1, cols = 2)



fig.add_trace(go.Histogram(x = surv_sib['SibSp'],

            name = 'Survived',

            opacity = 0.8,

            marker = dict(color = 'seagreen')), 1, 1)



fig.add_trace(go.Histogram(x = n_surv_sib['SibSp'],

            name = 'Not Survived',

            opacity = 0.8,

            marker = dict(color = 'blue')), 1, 2)



fig.update_layout(title_text = 'Survival by #Siblings Aboard',

                 legend = dict(orientation = 'h'))



fig.show()
fig = make_subplots(rows = 1, cols = 2)



fig.add_trace(go.Histogram(x = surv_sib['Parch'],

            name = 'Survived',

            opacity = 0.8,

            marker = dict(color = 'seagreen')), 1, 1)



fig.add_trace(go.Histogram(x = n_surv_sib['Parch'],

            name = 'Not Survived',

            opacity = 0.8,

            marker = dict(color = 'blue')), 1, 2)



fig.update_layout(title_text = 'Survival by #Parents/Children Aboard',

                 legend = dict(orientation = 'h'))



fig.show()