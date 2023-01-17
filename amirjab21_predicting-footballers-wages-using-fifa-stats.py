import sqlite3

import pandas as pd

import seaborn as sns

import numpy as np

import sklearn as sk

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import pyplot

#from fuzzywuzzy import fuzz

#from fuzzywuzzy import process

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

#import xgboost as xgb

#from xgboost import XGBRegressor



fifa_extra = pd.read_csv('../input/completedataset/CompleteDataset.csv')



#new feature to add --> Potential - Overall instead of potential.



import plotly as py

import plotly.graph_objs as go

import ipywidgets as widgets

import numpy as np

py.offline.init_notebook_mode(connected =True)
FIFAFINAL = pd.DataFrame(data = fifa_extra, columns = ['Name' , 'Age', 'Nationality','Overall', 'Potential', 'Club' ,'Value', 'Wage', 'Special', 'Preferred Positions','Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control',

       'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing',

       'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking',

       'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions',

       'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties',

       'Positioning', 'Reactions', 'Short passing', 'Shot power',

       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',

       'Strength', 'Vision', 'Volleys'])



def turnvalueintonumber(value):

    #turn the wages and values into numbers

    if 'M' in value:

        a = float(value[1:(len(value)-1)])

        b = a*1000

        return b

    elif 'K' in value:

        b = float(value[1:(len(value)-1)])

        return b



FIFAFINAL['Value2'] = FIFAFINAL['Value'].apply(turnvalueintonumber)

FIFAFINAL['Wage2'] = FIFAFINAL['Wage'].apply(turnvalueintonumber)

FIFAFINAL2 = FIFAFINAL.dropna()

FIFAFINAL2 = FIFAFINAL2.drop(columns = ['Value','Wage'])

FIFAFINAL2['1'] = 1



#FIFAFINAL2 IS JUST FIFAFINAL WITHOUT THE NA VALUES.



listprem = ['Bournemouth',

'Arsenal',

'Brighton & Hove Albion',

'Burnley',

'Swansea City',

'Chelsea',

'Crystal Palace',

'Everton',

'Stoke City',

'Huddersfield Town',

'Leicester City',

'Liverpool',

'Manchester City',

'Manchester United',

'Newcastle United',

'Southampton',

'Tottenham Hotspur',

'Watford',

'West Ham United',

'West Bromwich Albion']



FIFAFINAL2['BPL'] = FIFAFINAL2['Club'].apply(lambda x: 1 if x in listprem else 0)
def positions(X):

    if 'GK' in str(X):

        position = 'GK'

    elif 'ST' in str(X):

        position = 'Striker'

    elif 'LW' in str(X):

        position = 'Striker'

    elif 'RW' in str(X):

        position = 'Striker'

    elif 'CM' in str(X):

        position = 'Midfielder'

    elif 'CDM' in str(X):

        position = 'Midfielder'

    elif 'CAM' in str(X):

        position = 'Midfielder'

    elif 'RM' in str(X):

        position = 'Midfielder'

    elif 'LM' in str(X):

        position = 'Midfielder'

    else:

        position = 'Defender'

    return position



FIFAFINAL2['Pos'] = FIFAFINAL2['Preferred Positions'].apply(lambda x: positions(x))
FIFA4 = pd.merge(left = FIFAFINAL2, right= pd.get_dummies(FIFAFINAL2['Pos']),right_index=True,left_index=True,how='inner')
#Lets prepare the dataframes to go into the random forrest classifier.
FIFA6 = FIFA4

def turntonum(numb):

    if '+' in str(numb):

        b = str(numb).split('+')

        x2 = float(b[0]) + float(b[1])

    elif '-' in str(numb):

        b = str(numb).split('-')

        x2 = float(b[0]) - float(b[1])

    else:

        x2 = float(numb)

    return x2
FIFA6["Acceleration"]=FIFA4["Acceleration"].apply(lambda x: turntonum(x))

FIFA6["Agility"]=FIFA4["Agility"].apply(lambda x: turntonum(x))

FIFA6["Balance"]=FIFA4["Balance"].apply(lambda x: turntonum(x))

FIFA6["Ball control"]=FIFA4["Ball control"].apply(lambda x: turntonum(x))

FIFA6["Composure"]=FIFA4["Composure"].apply(lambda x: turntonum(x))

FIFA6["Crossing"]=FIFA4["Crossing"].apply(lambda x: turntonum(x))

FIFA6["Curve"]=FIFA4["Curve"].apply(lambda x: turntonum(x))

FIFA6["Dribbling"]=FIFA4["Dribbling"].apply(lambda x: turntonum(x))

FIFA6["Finishing"]=FIFA4["Finishing"].apply(lambda x: turntonum(x))

FIFA6["Free kick accuracy"]=FIFA4["Free kick accuracy"].apply(lambda x: turntonum(x))

FIFA6["GK diving"]=FIFA4["GK diving"].apply(lambda x: turntonum(x))

FIFA6["GK handling"]=FIFA4["GK handling"].apply(lambda x: turntonum(x))

FIFA6["GK kicking"]=FIFA4["GK kicking"].apply(lambda x: turntonum(x))

FIFA6["GK positioning"]=FIFA4["GK positioning"].apply(lambda x: turntonum(x))

FIFA6["GK reflexes"]=FIFA4["GK reflexes"].apply(lambda x: turntonum(x))

FIFA6["Heading accuracy"]=FIFA4["Heading accuracy"].apply(lambda x: turntonum(x))

FIFA6["Jumping"]=FIFA4["Jumping"].apply(lambda x: turntonum(x))

FIFA6["Long passing"]=FIFA4["Long passing"].apply(lambda x: turntonum(x))

FIFA6["Long shots"]=FIFA4["Long shots"].apply(lambda x: turntonum(x))

FIFA6["Marking"]=FIFA4["Marking"].apply(lambda x: turntonum(x))

FIFA6["Penalties"]=FIFA4["Penalties"].apply(lambda x: turntonum(x))

FIFA6["Positioning"]=FIFA4["Positioning"].apply(lambda x: turntonum(x))

FIFA6["Reactions"]=FIFA4["Reactions"].apply(lambda x: turntonum(x))

FIFA6["Short passing"]=FIFA4["Short passing"].apply(lambda x: turntonum(x))

FIFA6["Shot power"]=FIFA4["Shot power"].apply(lambda x: turntonum(x))

FIFA6["Sliding tackle"]=FIFA4["Sliding tackle"].apply(lambda x: turntonum(x))

FIFA6["Sprint speed"]=FIFA4["Sprint speed"].apply(lambda x: turntonum(x))

FIFA6["Stamina"]=FIFA4["Stamina"].apply(lambda x: turntonum(x))

FIFA6["Standing tackle"]=FIFA4["Standing tackle"].apply(lambda x: turntonum(x))

FIFA6["Strength"]=FIFA4["Strength"].apply(lambda x: turntonum(x))

FIFA6["Vision"]=FIFA4["Vision"].apply(lambda x: turntonum(x))

FIFA6["Volleys"]=FIFA4["Volleys"].apply(lambda x: turntonum(x))

FIFA6["Interceptions"]=FIFA4["Interceptions"].apply(lambda x: turntonum(x))

FIFA6['Aggression'] = FIFA4['Aggression'].apply(lambda x:turntonum(x))
FIFA6['GK'] = FIFA4['GK'].apply(lambda x: float(x))

FIFA6['Midfielder'] = FIFA4['Midfielder'].apply(lambda x: float(x))

FIFA6['Defender'] = FIFA4['Defender'].apply(lambda x: float(x))

FIFA6['Striker'] = FIFA4['Striker'].apply(lambda x: float(x))
FIFA6.columns
X1 = FIFA6.drop(columns = ['Preferred Positions', 'Wage2','Name','Club', 'Nationality','Pos','1', 'BPL'])

Y1 = FIFA6['Wage2']
X_train, X_test, Y_train, Y_test = train_test_split(X1,Y1,test_size=0.1, shuffle = True)
#Cleaning done
model = LinearRegression().fit(X = X_train, y = Y_train)



model2 = RandomForestRegressor().fit(X = X_train, y = Y_train)



print("Linear Regression model 1")

print ("R squared of train set:"  + str(model.score(X=X_train, y = Y_train)))

print ("R squared of test set:" + str(model.score(X=X_test, y=Y_test)))





print("Random forrest model 1")

print ("R squared of train set:"  + str(model2.score(X=X_train, y = Y_train)))

print ("R squared of test set:" + str(model2.score(X=X_test, y=Y_test)))
#listzz = FIFA6['Club'].unique()

#listz = pd.DataFrame(data = listzz)

#listz.to_csv(r'C:\Users\JabarivasalA\Documents\Football prediction\lol2.csv',sep=',')



print ("All the above steps have been run, some magic has been done to these clubs to separate clubs in the top 5 Europe leagues"

        " and it is imported back in below ")
model = RandomForestRegressor()
imported = pd.read_csv('../input/importz/importz.csv')



FIFA7 = pd.merge(left = FIFA6, right = imported, left_on = 'Club', right_on = 'club')
FIFA7.loc[FIFA7['country'].isnull(),'country'] = 0

FIFA7.loc[FIFA7['country'] == 'Beskitas', 'country'] = 0
FIFA8 = pd.merge(left = FIFA7, right = pd.get_dummies(data = FIFA7['country']), left_index=True, right_index = True)
FIFA8.loc[FIFA8['rev2'].isnull(),'rev2'] = 0

X2 = FIFA8.drop(columns = ['Preferred Positions', 'Wage2','Name','Club', 'Nationality','Pos','1','club',0,'country', 'index', 'BPL', 'rev2'])

Y2 = FIFA8['Wage2']

X_train, X_test, Y_train, Y_test = train_test_split(X2,Y2,test_size=0.1, shuffle = True)
model3 = LinearRegression().fit(X = X_train, y = Y_train)
model3.score(X = X_test, y = Y_test)
model3.score(X = X_train, y = Y_train)
#82% accuracy!
model4 = RandomForestRegressor().fit(X = X_train, y = Y_train)



print (model4.score(X = X_test, y = Y_test))

print (model4.score(X = X_train, y = Y_train))
mergz = pd.merge(left = FIFA8, right = pd.DataFrame(model4.predict(X = X2)), left_index = True, right_index=True)
#89% accruacy! I'll take that!
mergz = mergz.rename(columns = {"0_y":"predictedwage"})

#mergz = mergz.drop(columns = "0_x")

mergz2 = mergz[mergz['country']!=0]

#mergz2 only contains players in the top 5 Europe leagues.
mergz['potover'] = mergz['Potential'] - mergz['Overall']

mergz['overpaid'] = mergz['Wage2'] - mergz['predictedwage']



mergz2['potover'] = mergz2['Potential'] - mergz2['Overall']

mergz2['overpaid'] = mergz2['Wage2'] - mergz2['predictedwage']

fig, ax = pyplot.subplots(figsize=(10,10))

plt.scatter(y = mergz['overpaid'], x = mergz['potover'])

plt.ylabel('Overpaid')

plt.xlabel('potential increase')
#Will potential-overall make a difference to our model?
FIFA6['potover'] = FIFA6['Potential']-FIFA6['Overall']
X1 = FIFA6.drop(columns = ['Preferred Positions', 'Wage2','Name','Club', 'Nationality','Pos','1'])

Y1 = FIFA6['Wage2']
X_train, X_test, Y_train, Y_test = train_test_split(X1,Y1,test_size=0.1, shuffle = True)
model8 = RandomForestRegressor()

model8.fit(X= X_train, y = Y_train)
print (model8.score(X= X_train, y = Y_train), model8.score(X=X_test, y=Y_test))
def countryadaptor(x):

    b = 0

    if x == "Spain":

        b = 1

    elif x == "England":

        b = 2

    elif x == "Italy":

        b = 3

    elif x == "Germany":

        b = 4

    elif x == "France":

        b = 5

    return b

        

mergz['Countrynumb'] = mergz['country'].apply(lambda x: countryadaptor(x))

mergz2['Countrynumb'] = mergz2['country'].apply(lambda x: countryadaptor(x))
#so we've looked at wages. Now lets look at what positions are important 
plt.figure(figsize = (20,20))

layout = go.Layout(

    title = 'Wage prediction model'

    , yaxis = dict(title = 'predictedwages'),

    xaxis = dict(

        title = 'actualwages',

        

    ),autosize=False,

    width=1250,

    height=1250

)



trace1 = go.Scatter(showlegend = True, x = mergz['Wage2'], y = mergz['predictedwage'], mode = 'markers', hovertext=mergz['Name'],

                    

                    marker=dict(

        

        color = mergz2['Countrynumb'],

        colorscale = 'Jet',

        

        showscale=True

    )

                   )



fig = go.Figure(data = [trace1], layout = layout)

py.offline.iplot(fig)
plt.figure(figsize = (20,20))

layout = go.Layout(

    title = 'Wage prediction model'

    , yaxis = dict(title = 'overpaidby'),

    xaxis = dict(

        title = 'actualwages',

        

    ),autosize=False,

    width=1250,

    height=1250

)



trace1 = go.Scatter(showlegend = True, x = mergz2['Wage2'], y = mergz2['overpaid'], mode = 'markers', hovertext=mergz2['Name'],

                    

                    marker=dict(

        

        color = mergz2['Countrynumb'],

        colorscale = 'Jet',

        

        showscale=True

    )

                   )



fig = go.Figure(data = [trace1], layout = layout)

py.offline.iplot(fig)
mergz3 = mergz2[(mergz2['Countrynumb'] == 1) | (mergz2['Countrynumb'] == 2)]



def positionnumbering(x):

    b = 0

    if x == 'GK':

        b = 1

    elif x == 'Defender':

        b = 2

    elif x == 'Midfielder':

        b = 3

    elif x == 'Striker':

        b = 4

    return b

mergz['posnumber'] = mergz['Pos'].apply(lambda x: positionnumbering(x))
plt.figure(figsize = (20,20))

layout = go.Layout(

    title = 'Wage prediction model'

    , yaxis = dict(title = 'overpaidby'),

    xaxis = dict(

        title = 'actualwages',

        

    ),autosize=False,

    width=1250,

    height=1250

)



trace1 = go.Scatter(showlegend = True, x = mergz['Wage2'], y = mergz['overpaid'], mode = 'markers', hovertext=mergz['Name'],

                    

                    marker=dict(

        

        color = mergz['posnumber'],

        colorscale = 'RdBu',

        

        showscale=True

    )

                   )



fig = go.Figure(data = [trace1], layout = layout)

py.offline.iplot(fig)

mergz.groupby(by = 'Pos').mean()[['overpaid','Wage2']]
a = mergz.groupby(by = 'Nationality').mean()[['overpaid','Age', 'Wage2']]

a.sort_values(by = 'overpaid')
#lets view the same stat for top 5 leagues.

b = mergz2.groupby(by = 'Nationality').mean()[['overpaid','Age', 'Wage2']]

b.sort_values(by = 'overpaid')
mergz.groupby(by = 'Pos').mean()

#This allows us to see where the labour market is tight. It is tight for strikers which is why they are overpaid. A team with a limited budget should therefore buy midfielders or goalkeepers.")
#lets take out the club factor.
#lets predict total value.
FIFA8['potover'] = FIFA8['Potential'] - FIFA8['Overall']
X3 = FIFA8.drop(columns = ['Preferred Positions','Name','Club', 'Nationality','Pos','1','club',0,'country', 'index', 'BPL', 'rev2','Value2'])

Y3 = FIFA8['Value2']

X_train, X_test, Y_train, Y_test = train_test_split(X3,Y3,test_size=0.1, shuffle = True)

model6 = LinearRegression().fit(X = X_train, y = Y_train)
model6.score(X = X_test, y = Y_test)
predictions = model6.predict(X=X3).reshape(17725,1)

actuals = np.array(Y3).reshape(17725,1)



predarrays = np.concatenate((predictions, actuals), axis = 1)

predarrays



dataframez = pd.DataFrame(data = predarrays)



merger = pd.merge(left = dataframez, right = FIFA8, left_index = True, right_index = True)

merger.drop(columns = 1)

merger = merger.rename(index = str, columns = {'0_x':'predictedvalue'})

merger['posnumber'] = merger['Pos'].apply(lambda x: positionnumbering(x))

merger['Countrynumb'] = merger['country'].apply(lambda x: countryadaptor(x))
merger['predictedvalue'] = merger['predictedvalue'].apply(lambda x: x/1000)

merger['Value2'] = merger['Value2'].apply(lambda x: x/1000)
plt.figure(figsize = (15,15))

layout = go.Layout(

    title = 'Wage prediction model'

    , yaxis = dict(title = 'predictedvalue'),

    xaxis = dict(

        title = 'actualvalue',

        

    ),autosize=False,

    width=1250,

    height=1250

)



trace1 = go.Scatter(showlegend = True, x = merger['Value2'], y = merger['predictedvalue'], mode = 'markers', hovertext=mergz['Name'],

                    

                    marker=dict(

        

        color = merger['posnumber'],

        colorscale = 'RdBu',

        

        showscale=True

    )

                   )



fig = go.Figure(data = [trace1], layout = layout)

py.offline.iplot(fig)

mergz.groupby(by = 'Pos').mean()[['overpaid','Wage2']]
plt.figure(figsize = (15,15))

layout = go.Layout(

    title = 'Wage prediction model'

    , yaxis = dict(title = 'predictedvalue'),

    xaxis = dict(

        title = 'actualvalue',

        

    ),autosize=False,

    width=1250,

    height=1250

)



trace1 = go.Scatter(showlegend = True, x = merger['Value2'], y = merger['predictedvalue'], mode = 'markers', hovertext=mergz['Name'],

                    

                    marker=dict(

        

        color = merger['Countrynumb'],

        colorscale = 'Jet',

        

        showscale=True

    )

                   )



fig = go.Figure(data = [trace1], layout = layout)

py.offline.iplot(fig)

mergz.groupby(by = 'Pos').mean()[['overpaid','Wage2']]