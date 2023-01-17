# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
init_notebook_mode(connected=True) 
df_meet = pd.read_csv('../input/meets.csv')
df_open = pd.read_csv('../input/openpowerlifting.csv')
sns.heatmap(df_meet.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_meet.drop('MeetTown',axis=1,inplace=True)
sns.heatmap(df_open.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_open.drop(labels=['Squat4Kg','Bench4Kg','Deadlift4Kg','Age'],axis=1,inplace=True)
df_open.dropna(inplace=True)
df_open.info()
df_open = df_open[(df_open['Place']!='DQ') & (df_open['Place']!='G')]
df_open['Place'] = df_open['Place'].apply(lambda x: int(x))
df_open.info()
sns.countplot(x='Sex', data=df_open)
sns.pairplot(df_open, vars=['BodyweightKg','BestSquatKg','BestBenchKg',
                            'BestDeadliftKg'],hue='Sex',palette='coolwarm')
df_open = df_open[df_open['BestSquatKg']>0]
#sns.pairplot(df_open, 
#             vars=['BodyweightKg','BestSquatKg','BestBenchKg','BestDeadliftKg'],hue='Sex',palette='coolwarm')
#Select meets from the US
df_usa = df_meet[df_meet['MeetCountry']=='USA']
#Organize and counts meets by state
df_state = df_usa.groupby('MeetState').count()
df_state.reset_index(inplace=True)
#Data and layout dictionaries for plotly
data = dict(type='choropleth',
            colorscale = 'Viridis',
            locations = df_state['MeetState'],
            z = df_state['MeetID'],
            locationmode = 'USA-states',
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"Number of Meets"}
            )
layout = dict(title = 'Number of meets by State',
              geo = dict(scope='usa',
                         showlakes = False,
                         lakecolor = 'rgb(85,173,240)')
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
#Creating another DataFrame, that will be used in the predictions
df_prov = pd.DataFrame.copy(df_open)
#Removing object columns that will not be used
df_prov.drop(labels=['MeetID','Name','Division','WeightClassKg','Equipment','Wilks'],axis=1,inplace=True)
#Categorize Sex column
cat_feats = ['Sex']
df_predict = pd.get_dummies(df_prov,columns=cat_feats,drop_first=True)
#Functions that returns 1 if the person is the winner (Place == 1) and 0 otherwise
def change_place(val):
    if val > 1:
        return 0
    else:
        return 1
df_predict['Place']=df_predict['Place'].apply(change_place)
df_predict.head()
#Importing Machine learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
X_train, X_test, y_train, y_test = train_test_split(df_predict.drop('Place',axis=1), df_predict['Place'], 
                                                    test_size=0.3)
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
pred_rfc = rfc.predict(X_test)
print('Classification report for Random Forest:')
print(classification_report(y_test,pred_rfc))

sns.boxplot(x='Place',y='TotalKg',data=df_predict)
