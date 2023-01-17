import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sb



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def print_full(x):

    pd.set_option('display.max_rows', len(x))

    print(x)

    pd.reset_option('display.max_rows')
data = pd.read_csv('/kaggle/input/fifa19/data.csv')
data.head()
model_data = data.copy()
clutter = ['Unnamed: 0','ID','Photo','Flag','Club Logo','Jersey Number','Joined','Special','Loaned From','Body Type', 'Release Clause',

               'Weight','Height','Contract Valid Until','Wage','Value','Name','Club']

model_data.drop(clutter, axis=1, inplace=True)
model_data.info()
model_data.drop(model_data.iloc[:, 11:37], axis=1, inplace=True)
tempwork = model_data["Work Rate"].str.split("/ ", n = 1, expand = True)

model_data["WorkRate1"]= tempwork[0]

model_data["WorkRate2"]= tempwork[1]



model_data.drop('Work Rate', axis=1, inplace = True)
def wage(val):

    try:

        act = float(val[1:-1])

        end = val[-1:]

        

        if end.lower() == 'k':

            act = act*1000

        elif end.lower() == 'm':

            act = act*10000000

    except ValueError:

        act = 0

    return act



data['Value'] = data['Value'].apply(wage)

data['Wage'] = data['Wage'].apply(wage)
def club_details(club_name):

    return data[data['Club'] == club_name].sort_values(by='Potential', ascending=False).head(5)
club_details('Tranmere Rovers')
import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot



init_notebook_mode(connected = True)

import plotly.graph_objs as go




potential = pd.DataFrame(data[['Nationality','Potential']].sort_values(by = 'Potential', ascending=False)).head(100)



trace = [go.Choropleth(

            locationmode = 'country names',

            locations = potential['Nationality'],

            text = potential['Nationality'],

            z = potential['Potential'],

)]



layout = go.Layout(title = 'Country of top 100 players in Fifa 19')



fig = go.Figure(data = trace, layout = layout)

py.iplot(fig)



x = data.groupby(by='Nationality', sort=False)['Overall'].max().head()

x
data[['Name','Wage']].sort_values(by='Wage', ascending=False).head(15)

#This will give Highest of them all.

#data.loc[data['Wage'].idxmax()][1] 
count_age_overall = data.sort_values(by = 'Overall', ascending=False).head(15)[['Name','Overall', 'Age']]

count_age_overall
comparing_aspects = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',

       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',

       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',

       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',

       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']



for i in comparing_aspects:

    print('{0}: {1}'.format(i, data.loc[data[i].idxmax()][1]))
pot = data.sort_values(by='Potential')

diff = pd.DataFrame(data= [pot['Potential']-pot['Overall'], pot['Name']], index=['Potential  left', 'name']).T

diff = diff.sort_values(by = 'Potential  left', ascending=False).head(15)

diff
plt.figure(figsize=(12,10))

sb.distplot(data['Overall'])

plt.title("Overall is normally distributed.")
plt.figure(figsize=(12,10))

sb.jointplot(data['Age'].head(500), data['Potential'].head(500), kind='hex')
sb.jointplot(data['Age'].head(500), data['Overall'].head(500), kind='hex', color='green')
count_age = data.sort_values(by = 'Potential', ascending=False).head(100)[['Name','Potential', 'Age']]

count_age

sb.countplot(count_age['Age'])
count_age_overall = data.sort_values(by = 'Overall', ascending=False).head(100)[['Name','Potential', 'Age']]

sb.countplot(count_age_overall['Age'])
sb.lineplot(x=data['Age'], y=data['Overall'])
age_data = data.copy()

def age_class(x):

    if x<20:

        x = 'Under 20'

    elif x>=20 and x<25:

        x = '21 to 25'

    elif x>=25 and x<30:

        x = '25 to 30'

    elif x>=30 and x<35:

        x = '30 to 30'

    elif x>=35:

        x = 'Over 35'

    return x



age_class = age_data['Age'].apply(age_class)
sb.boxplot(age_class, 'Wage', data = age_data)
plt.figure(figsize=(12,10))

sb.distplot(data['Wage'])
position_remap = {'LWB':'side_def',

                  'LB':'side_def',

                  'RB':'side_def',

                  'RWB':'side_def',



                  'LCB':'cent_def',

                  'CB':'cent_def',

                  'RCB':'cent_def',

                  

                  'LDM':'cent_mid',

                  'CDM':'cent_mid',

                  'RDM':'cent_mid',



                  'LCM':'cent_mid',

                  'CM':'cent_mid',

                  'RCM':'cent_mid',

                  'CAM':'cent_mid',



                  'LW':'side_mid',

                  'LM':'side_mid',

                  'RM':'side_mid',

                  'RW':'side_mid',

                  'LAM':'side_mid',

                  'RAM':'side_mid',



                  'LF':'side_fwd',

                  'LS':'side_fwd', 

                  'RS':'side_fwd',

                  'RF':'side_fwd',

                  'CF':'cent_fwd',

                  'ST':'cent_fwd'

                 }  

data['Position'] = data['Position'].map(position_remap)

position_group = data['Position']

position_group.dropna(axis=0)
plt.figure(figsize=(18,10))

sb.heatmap(data.groupby(position_group).mean()[comparing_aspects], annot=True)
remap_remap = {'side_mid':'mid',

              'side_fwd':'fwd',

              'side_def':'def',

              'cent_def':'def',

              'cent_mid':'mid',

              'cent_fwd':'fwd'}

data['Position'] =  data['Position'].map(remap_remap)

position_group_remap = data['Position']

position_group_remap.dropna(axis=0)
data['Position']
sb.boxenplot(data[data['Position']=='def']['Overall'], data[data['Position']=='def']['Position'])
sb.boxenplot(data[data['Position']=='fwd']['Overall'], data[data['Position']=='fwd']['Position'])
sb.boxenplot(data[data['Position']=='mid']['Overall'], data[data['Position']=='mid']['Position'])
sb.stripplot(x='Position', y="Dribbling", data=data.head(900), jitter=True,palette='Set1',dodge=True)
sb.stripplot(x='WorkRate1', y="Crossing", data=model_data.head(900), jitter=True,palette='Set1',dodge=True)
sb.stripplot(x='WorkRate2', y="Dribbling", data=model_data.head(900), jitter=True,palette='Set1',dodge=True)
sb.lmplot('Crossing', 'Dribbling', data.head(900), col='Position')
sb.countplot(data['Preferred Foot'],hue=data['Position'])
sb.countplot(data['Preferred Foot'],hue=data['Weak Foot'] )
differ = pd.DataFrame(data= [pot['Potential']-pot['Overall'], pot['Name'], pot['Potential'], pot['Overall']], index=['diff', 'name', 'potential', 'overall']).T

differ = differ.sort_values(by = 'diff', ascending=False).head(15)



plt.figure(figsize=(18,6))

plt.bar(differ['name'], differ['potential'])

plt.bar(differ['name'], differ['overall'])

plt.title('15 players with the highest remaining potential left to grow')
plt.figure(figsize=(12,10))

plt.plot(data.groupby('Age').mean()['Potential'])

plt.plot(data.groupby('Age').mean()['Overall'])

plt.title("The potential meets overall somewhere near at 29 years old.")

plt.grid(linestyle='--')

plt.xlabel('Age')

plt.ylabel('Rating')
model_data['Nationality'].unique()
model_data.drop('Nationality', axis=1, inplace=True)
model_data = pd.get_dummies(model_data)
print_full(model_data.isna().sum())
model_data.dropna(inplace=True)
print_full(model_data.isna().sum())
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.metrics import r2_score, mean_absolute_error, auc
X = model_data.drop('Overall', axis=1)

y = model_data.Overall

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
model1 = LinearRegression()

model2 = SVR()

models = [model1, model2]
for model in models:

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("For model {} r2 score is {}\n".format(model, r2_score(y_test, preds)))

    print("For model {} mean absolute error is {}\n".format(model, mean_absolute_error(y_test, preds)))