import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
from plotly.offline import iplot, plot, download_plotlyjs, init_notebook_mode
from plotly.graph_objs import graph_objs as go
import cufflinks as cf
init_notebook_mode(connected = True)
cf.go_offline(connected = True)
import os
print(os.listdir("../input/fifa19"))
fifa_data = pd.read_csv('../input/fifa19/data.csv')
fifa_data.columns
fifa_data['Growth'] = fifa_data['Potential'] - fifa_data['Overall']
fifa_data.isnull().sum()[fifa_data.isnull().sum() > 8000]
fifa_data.drop('Loaned From', axis = 1, inplace = True)
fifa_data.drop('Release Clause', axis = 1, inplace = True)
fifa_data.drop(['Club', 'Jersey Number', 'Contract Valid Until', 'Joined'], axis = 1, inplace = True)
fifa_data.drop(['Photo', 'Flag', 'Club Logo', 'Real Face'], axis = 1, inplace = True)
fifa_data.drop(index = fifa_data[fifa_data['Preferred Foot'].isna()].index, inplace = True)
fifa_data.drop(index = fifa_data[fifa_data['Position'].isna()].index, inplace = True)
fifa_data.drop(index = fifa_data[fifa_data['RB'].isna()].index, inplace = True )
def convertValue(value) :
    if value[-1] == 'M' :
        value = value[1:-1]
        value = float(value) * 1000000
        return value
    
    if value[-1] == 'K' :
        value = value[1:-1]
        value = float(value) * 1000
        return value
fifa_data['Wage'] = fifa_data['Wage'].apply(lambda x : convertValue(x))
fifa_data['Value'] = fifa_data['Value'].apply(lambda x : convertValue(x))
fifa_data.select_dtypes(include = object).columns
fifa_data['Body Type'][fifa_data['Body Type'] == 'Messi'] = 'Normal'
fifa_data['Body Type'][fifa_data['Body Type'] == 'C. Ronaldo'] = 'Normal'
fifa_data['Body Type'][fifa_data['Body Type'] == 'Neymar'] = 'Lean'
fifa_data['Body Type'][fifa_data['Body Type'] == 'PLAYER_BODY_TYPE_25'] = 'Normal'
fifa_data['Body Type'][fifa_data['Body Type'] == 'Shaqiri'] = 'Stocky'
fifa_data['Body Type'][fifa_data['Body Type'] == 'Akinfenwa'] = 'Stocky'
def convertWeight(weight) :
    weight = weight[0:3]
    return weight
def convertHeight(height) :
    height = height.split("'")
    height = float(height[0]) * 30.48 + float(height[1]) * 2.54 
    
    return height
fifa_data['Weight'] = fifa_data['Weight'].apply(lambda x : convertWeight(x))
fifa_data['Height'] = fifa_data['Height'].apply(lambda x : convertHeight(x))
def convertPosition(val) :
    
    if val == 'RF' or val == 'ST' or val == 'LW' or val == 'LF' or val == 'RS' or val == 'LS' or val == 'RM' or val == 'LM' or val == 'RW' or val == 'CF' :
        return 'Forward'
    
    elif val == 'GK' :
        return 'GoalKeeper'
    
    elif val == 'RCM' or val == 'LCM' or val == 'LDM' or val == 'CAM' or val == 'CDM' or val == 'LAM' or val == 'RDM' or val == 'CM' or val == 'RAM' :
        return 'MidFielder'
    
    return 'Defender'

fifa_data['Position'] = fifa_data['Position'].apply(lambda x : convertPosition(x))
temp_columns =['LS', 'ST', 'RS', 'LW', 'LF', 'CF',
               'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB',
               'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
def convertRatings(rating) :
    rating = rating.split('+')
    rating = int(rating[0]) + int(rating[1])
    return rating
for column in temp_columns :
    fifa_data[column] = fifa_data[column].apply(lambda x : convertRatings(x))
def getTOP10(feature) :
   return fifa_data.sort_values(by = feature, ascending = False)[['Name', feature]].head(10)
getTOP10('RW')
Player_Count = fifa_data.groupby('Nationality').size().reset_index()
Player_Count.columns = ['Country', 'Count']
data1 = go.Choropleth(locationmode = 'country names', locations = Player_Count['Country'],
                     z =  Player_Count['Count'], colorscale = 'oranges', )
layout1 = go.Layout(title = 'Players Count Per Country') 
graphPlayerCountPerCountry = go.Figure(data = data1, layout = layout1)
graphPlayerCountPerCountry
graphPlayerAge = fifa_data['Age'].iplot(kind = 'histogram', title = 'Player Age Distribution', xTitle = 'Age',
                                        yTitle = 'Count', theme = 'pearl', )
plt.figure(figsize =(30,15))
graphPlayerHeightWeight = sns.boxplot(x = 'Weight', y = 'Height', data = fifa_data, )
data2 = go.Pie(labels = fifa_data['Position'].value_counts().index.values, values = fifa_data['Position'].value_counts().values, 
               hole = 0.3)
layout2 = go.Layout(title = 'Player Position Distribution')
graphPlayerPosition = go.Figure(data = data2, layout = layout2)
graphPlayerPosition
graphPlayerOverall = fifa_data['Overall'].iplot(kind = 'histogram', title = 'Player Overall Distribution', 
                                                xTitle = 'Overall Rating', yTitle ='Count')
data3 = go.Pie(labels = fifa_data['Preferred Foot'].value_counts().index.values, values = fifa_data['Preferred Foot'].value_counts().values, 
               hole = 0.3)
layout3 = go.Layout(title = 'Player Preferred Foot Distribution')
graphPlayerPreferredFoot = go.Figure(data = data3, layout = layout3)
graphPlayerPreferredFoot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
fifa_data['Value'] = fifa_data['Value'].apply(lambda x  : x/1000000)
x = fifa_data[['Potential', 'Value', 'LS', 'ST', 'RS', 'LW',
       'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM',
       'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB',
       'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'Growth']]
y = fifa_data['Overall']
x.fillna(value = 0, inplace = True)
y.fillna(value = 0, inplace = True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)
from sklearn.model_selection import GridSearchCV
paramlist = {'n_jobs' : [0.1 , 1, 10, 100 ]}
gridSearch = GridSearchCV(estimator = LinearRegression(), param_grid = paramlist, verbose = 5)
gridSearch.fit(x_train,y_train)
gridSearch.best_params_
model = LinearRegression(n_jobs = 0.1)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(y_true = y_test, y_pred = predictions)
mean_squared_error(y_true = y_test, y_pred = predictions)
