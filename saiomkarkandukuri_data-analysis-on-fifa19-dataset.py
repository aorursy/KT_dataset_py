import numpy as np 

import pandas as pd 

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import PCA

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)



data = pd.read_csv('../input/data.csv', index_col = 0)
print(data.head())
#data.iloc[:, 53:83].corr()

data.iloc[:, :].corr().loc[:,'Skill Moves']

#data.iloc[:, :].corr()#.loc[:,'Work Rate']
attributes = data.iloc[:, 53:83]



data['Skill Moves'].head()



#Data Preprocessing

attributes = data.iloc[:, 54:83]

attributes['Skill Moves'] = data['Skill Moves']

#workrate = data['Work Rate'].str.get_dummies(sep='/ ')

#attributes = pd.concat([attributes, workrate], axis=1)

df = attributes

attributes = attributes.dropna()

df['Name'] = data['Name']

df = df.dropna()

print(attributes.columns)
scaled = StandardScaler()

X = scaled.fit_transform(attributes)
ball = NearestNeighbors(n_neighbors=16, algorithm='ball_tree').fit(X)
kd_tree = NearestNeighbors(n_neighbors=16, algorithm='kd_tree').fit(X)
indices_ball = ball.kneighbors(X)[1]
indices_kd_tree = kd_tree.kneighbors(X)[1]
def get_index(x):

    return df[df['Name']==x].index.tolist()[0]
def neighbour_ball(player):

    print('15 players similar to', player, ':' '\n')

    index = get_index(player)

    for i in indices_ball[index][1:]:

            print(df.iloc[i]['Name'], '\n')
def neighbour_kd_tree(player):

    print('15 players similar to', player, ':' '\n')

    index = get_index(player)

    for i in indices_kd_tree[index][1:]:

            print(df.iloc[i]['Name'], '\n')
neighbour_ball("Neymar Jr")
neighbour_kd_tree('Neymar Jr')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import eli5

from eli5.sklearn import PermutationImportance

from collections import Counter

import missingno as msno

sns.set_style('darkgrid')

regression_data = pd.read_csv('../input/data.csv', index_col = 0)

full_data = pd.read_csv('../input/data.csv', index_col = 0)
regression_data.info()
regression_data.head()
regression_data.columns
#regression_data.drop(['Flag','Club Logo'],axis=1,inplace=True)

#regression_data.drop(columns=['Photo'])

#msno.bar(regression_data.sample( 18207 ),(28,10),color='blue')

#regression_data.drop(['Loaned From'],axis=1,inplace=True)

#sns.pairplot(regression_data)

#sns.heatmap(df.corr(),cmap = 'Blues', annot=True)
regression_attributes = regression_data.iloc[:, 53:82]

#regression_attributes = regression_attributes.dropna()

#regression_attributes = regression_attributes.iloc[:,:].astype('int64')

regression_attributes['Skill Moves'] = regression_data['Skill Moves']

regression_attributes[['International Reputation', 'Wage', 'Value']] = full_data[['International Reputation', 'Wage', 'Value']]
regression_attributes.info()
regression_attributes.isnull().any()

#Many ways to deal with null values



regression_attributes = regression_attributes.dropna()

#regression_attributes = regression_attributes.fillna(method='ffill')

#regression_attributes = regression_attributes.fillna(regression_attributes.mean())
regression_attributes.isnull().any()
ind = regression_attributes.index

for i in ind:

    if ('M' in regression_attributes.loc[i,'Value']) and ('.' in regression_attributes.loc[i,'Value']):

        regression_attributes.loc[i,'Value'] = regression_attributes.loc[i,'Value'].replace('€','').replace('.','').replace('M','00000')

    else:

        regression_attributes.loc[i,'Value'] = regression_attributes.loc[i,'Value'].replace('€','').replace('K','000').replace('M','000000')

    regression_attributes.loc[i,'Wage'] = regression_attributes.loc[i,'Wage'].replace('€','').replace('K','000')

    

regression_attributes.Value = regression_attributes.Value.astype('int64')

regression_attributes.Wage = regression_attributes.Wage.astype('int64')
#X = regression_attributes[['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',

#       'Dribbling', 'Curve', 'LongPassing', 'BallControl',

#       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',

#       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

#       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

#      'Composure', 'Marking', 'StandingTackle', 'SlidingTackle']]

#       #'Skill Moves']]

    

X =  regression_attributes[['Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'Curve', 'FKAccuracy', 'Acceleration', 

       'SprintSpeed', 'ShotPower', 'BallControl', 'Penalties']]

#y = regression_attributes[['International Reputation']]

y = regression_attributes[['Wage']]

#y = regression_attributes[['Value']]
regression_attributes.corr()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
lm.coef_
lm_pred = lm.predict(X_test)
plt.scatter(y_test,lm_pred)

plt.xlabel('Actual Wage')

plt.ylabel('Predicted Wage')

plt.show
plt.scatter(y_test,lm_pred)

plt.xlabel("Actual International Reputaion")

plt.ylabel("Predicted International Reputation")

plt.show()
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, lm_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, lm_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, lm_pred)))

#print('R2 error', r2_score(y_test,lm_pred))
Potential_data = pd.read_csv('../input/data.csv')
Potential_attributes_x=Potential_data.iloc[:,[3,66,68,71,72,79]]
Potential_attributes_x.isnull().any()
Potential_attributes_x = Potential_attributes_x.fillna(Potential_attributes_x.mean())
Potential_attributes_x.isnull().any()
Potential_attributes_y =Potential_data.iloc[:,8]
X_train,X_test,y_train,y_test=train_test_split(Potential_attributes_x,Potential_attributes_y,test_size=0.3)
lm = LinearRegression()
lm.fit(X_train,y_train)
lm_pred = lm.predict(X_test)
plt.scatter(y_test,lm_pred)

plt.xlabel("Actual Potential")

plt.ylabel("Predicted Potential")

plt.show()
lm.intercept_
lm.coef_
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, lm_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, lm_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, lm_pred)))
Viz_data = pd.read_csv("../input/data.csv")
Viz_data.head()
Viz_data.drop(['Flag','Club Logo','Photo'],axis=1,inplace=True)

#msno.bar(regression_data.sample( 18207 ),(28,10),color='blue')

#regression_data.drop(['Loaned From'],axis=1,inplace=True)

#sns.pairplot(regression_data)

#sns.heatmap(df.corr(),cmap = 'Blues', annot=True)
plt.rcParams['figure.figsize']=(25,16)

heat_map=sns.heatmap(data[['Age', 'Overall', 'Potential', 'Special',

    'Body Type', 'Position',

    'Height', 'Weight', 'Crossing',

    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

    'Marking', 'StandingTackle', 'SlidingTackle']].corr(), annot = True, linewidths=.5, cmap='Blues')
plt.figure(figsize=(15,32))

sns.countplot(y = Viz_data.Nationality,palette="Set2")
plt.figure(figsize=(15,6))

sns.countplot(x="Age",data=Viz_data)  
skills = ['Overall', 'Potential', 'Crossing',

   'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

   'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

   'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

   'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

   'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

   'Marking', 'StandingTackle', 'SlidingTackle']
messi = Viz_data.loc[data['Name'] == 'L. Messi']

messi = pd.DataFrame(messi, columns = skills)

ronaldo = Viz_data.loc[data['Name'] == 'Cristiano Ronaldo']

ronaldo = pd.DataFrame(ronaldo, columns = skills)

Zlatan = Viz_data.loc[data['Name'] == 'Z. Ibrahimović']

Zlatan = pd.DataFrame(Zlatan, columns = skills)

Salah = Viz_data.loc[data['Name'] == 'M. Salah']

Salah = pd.DataFrame(Salah, columns = skills)







plt.figure(figsize = (14,8))

sns.pointplot(data = messi,color = 'blue',alpha = 0.6)

sns.pointplot(data = ronaldo, color = 'red', alpha = 0.6)

sns.pointplot(data = Zlatan, color = 'black', alpha = 0.6)

sns.pointplot(data = Salah, color = 'green', alpha = 0.6)

plt.text(5,55,'Messi',color ='blue',fontsize = 25)

plt.text(5,50,'Ronaldo',color ='red',fontsize = 25)

plt.text(5,60,'Zlatan',color ='black',fontsize = 25)

plt.text(5,65,'Salah',color ='green',fontsize = 25)

plt.xticks(rotation=90)

plt.xlabel('Skills', fontsize=20)

plt.ylabel('Skill value', fontsize=20)

plt.title('Player skill comparision', fontsize = 25)

plt.grid()