# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)



sns.set_style("darkgrid")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

my_data=pd.read_csv('../input/fifa19/data.csv',index_col=0)



#df_test = pd.read_csc("Titanic")

# Any results you write to the current directory are saved as output.

%matplotlib inline
my_data.head()
#my_data.info()
#my_data.isna().count()
# Missing Data.



import missingno as msno

msno.bar(my_data.sample(18207))
#Relationship between 'Age' and 'Potential'.



sns.set_style('darkgrid')



sns.lmplot(x='Age', y="Potential", data=my_data, 

           aspect=2,palette="coolwarm")
#difference between ages.



plt.figure(figsize=(10,8))

sns.countplot(x ='Age',data=my_data, palette='coolwarm')
#'Preferred Fot' left or right.



plt.figure(figsize=(10,8))

sns.countplot(x ='Preferred Foot',data=my_data, palette='BrBG')
#Visualisation all 'Position'.



plt.figure(figsize=(12,8))

sns.set_style('whitegrid')

sns.countplot(x='Position',data=my_data, palette='coolwarm')

#Relation between 'Age' and 'Agility'.



sns.set_style('whitegrid')

g = sns.JointGrid(x='Age', y='Agility',data=my_data, height=8)

g = g.plot_joint(plt.scatter, color="0.8", edgecolor="purple", )

g = g.plot_marginals(sns.distplot, kde=True, color=".5")
plt.figure(figsize=(12,8))

sns.regplot(x = 'Age', y = 'Potential', data = my_data, marker=None)
nat=pd.value_counts(my_data['Nationality'])

nat=nat.head(10)



sns.set_style('darkgrid')

plt.figure(figsize=(12,8))

plt.plot(nat, marker=".", markersize=20, markerfacecolor='red')

plt.ylabel('Nº Players')
import folium





df = pd.DataFrame({

    "lat": [-15.7744227,51.5287352, 52.5069312,41.3948976, -34.6154611, 

           48.8589507,48.8591341,4.6486259,35.5062897, 52.1950964 ],

    

    "long": [-48.0772903, -0.3817841,13.1445471, 2.0787274,-58.5733844,

            2.2770198,2.2770196,-74.2478958,138.6484981,3.0364464  ],

    

    "name": ['Brasil - approximately 800','England  - approximately 1600 ',

             'Germany  - approximately 1200',

             'Spain  - approximately 1050',

             'Argentina  - approximately 950',

             'France  - approximately 900', 'Italy  - approximately 750',

             'Colombia  - approximately 600','Japan  - approximately 450',

             'Netherlands  - approximately 420' ]

})



m = folium.Map(location=None, tiles="Mapbox Bright", zoom_start=10)





for x in range (0,len(df)):

    

    folium.Marker(

        location=[df.iloc[x]['lat'],df.iloc[x]['long']],

        popup =df.iloc[x]['name'],

        tooltip=df.iloc[x]['name']

    ).add_to(m)

    

m
my_data['Wage'] = my_data['Wage'].map(lambda x: x.lstrip('€').rstrip('K'))

my_data['Value'] = my_data['Value'].map(lambda x: x.lstrip('€').rstrip('M').rstrip('K'))



my_data['Wage'].astype('int64')

my_data['Value'].astype('float')





my_data.info()

my_data.drop(['ID', 'Name', 'Photo', 'Nationality', 'Flag', 

       'Club', 'Club Logo',

       'Preferred Foot', 'International Reputation',

       'Work Rate', 'Body Type', 'Real Face', 'Position', 

        'Joined', 'Loaned From', 'Contract Valid Until',

       'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',

       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',

       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing',

       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',

       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',

       'SprintSpeed', 'Reactions', 'Balance',

       'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

       'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause'], axis=1,inplace=True)

my_data.columns
import missingno as msno

msno.matrix(my_data)
my_data['Weak Foot'].fillna(my_data['Weak Foot'].mean(), inplace=True)

my_data['Acceleration'].fillna(my_data['Acceleration'].mean(), inplace=True)

my_data['Agility'].fillna(my_data['Agility'].mean(), inplace=True)

my_data['ShotPower'].fillna(my_data['ShotPower'].mean(), inplace=True)

my_data['Aggression'].fillna(my_data['Aggression'].mean(), inplace=True)

my_data['Dribbling'].fillna(my_data['Dribbling'].mean(), inplace=True)

my_data['Jersey Number'].fillna(my_data['Jersey Number'].mean(), inplace=True)

my_data['Skill Moves'].fillna(my_data['Skill Moves'].mean(), inplace=True)

msno.matrix(my_data)
X = my_data.drop(['Potential'], axis=1)

y = my_data['Potential']
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef_
predict = lm.predict(X_test)
plt.figure(figsize=(12,8))

sns.regplot(y_test,predict, marker='*', color='green' )

plt.xlabel("Potential", fontsize=12 )

plt.ylabel('datas of hability and market value', fontsize=12)
plt.figure(figsize=(12,8))

sns.distplot(y_test-predict)

plt.title("Distribution Plot - Waste")



# if the Waste Distribution is Normal Distribution, so the Linear Regression its correct.

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predict))

print('MSE:', metrics.mean_squared_error(y_test, predict))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))
lm.score(X,y)