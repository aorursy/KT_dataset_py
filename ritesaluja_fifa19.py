# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from sklearn.feature_selection import mutual_info_regression

from sklearn.model_selection import GridSearchCV

import gc

# Any results you write to the current directory are saved as output.
#Welcome to FifaLand

df = pd.read_csv("../input/data.csv")

df.head(5)
#list of columns in Dataset

df.columns
#Data Cleaning & Preparation 

#dropping off the unnecessary columns

dfa = df.drop(['Unnamed: 0','Flag','Photo','Club Logo','Loaned From'],axis=1)

#formatting the fields

dfa['Value_kEuro'] = dfa['Value'].str.replace('€','').str.replace('M','000').str.replace('K','')

dfa['Wage_kEuro'] = dfa['Wage'].str.replace('€','').str.replace('K','')

dfa['ReleaseClaus_kEuro'] = dfa['Release Clause'].str.replace('€','').str.replace('M','000').str.replace('K','')

dfa.drop(['Value','Wage','Release Clause'],axis=1,inplace=True)

#changing datatypes

dfa['Value_kEuro'] = pd.to_numeric(dfa['Value_kEuro'])

dfa['Wage_kEuro'] = pd.to_numeric(dfa['Wage_kEuro'])

dfa['ReleaseClaus_kEuro'] = pd.to_numeric(dfa['ReleaseClaus_kEuro'])

#checking for nulls

dfa[['Value_kEuro','Wage_kEuro']].isnull().sum()
y= dfa['Value_kEuro'].values

x = dfa['Wage_kEuro'].values



fig = plt.figure(figsize=(12,8))

plt.title("How does Market Value relate to Wages (in thousand €)?")

plt.scatter(x,y)

#sns.lmplot(x='Wage_kEuro',y='Value_kEuro',order = order, data=dfa)

#sns.regplot(x='Wage_kEuro',y='Value_kEuro',data=dfa)

plt.xlabel('Wages')

plt.ylabel('Market Value')

plt.show()
footCount = dfa.groupby('Preferred Foot')['ID'].count()

footCount = footCount/footCount.sum()*100



plt.bar(x=['Left','Right'], height=footCount.values)

plt.title("Preferred Foot (%)");
#Get the required details from the dataframe

dfo = dfa[['Position','Preferred Foot']].groupby('Position')['Preferred Foot'].value_counts().unstack()



#Top 5 Left foot 

print("Top 5 Left Foot Positions:")

print(dfo['Left'].sort_values(ascending = False).head(5))



#Top 5 Right foot 

print("\nTop 5 Right Foot Positions:")

print(dfo['Right'].sort_values(ascending = False).head(5))



#plotting

dfo['Left']= dfo['Left']/dfo['Left'].sum()

dfo['Right']= dfo['Right']/dfo['Right'].sum()

fig, ax = plt.subplots(figsize=(15,7));

dfo['Right'].plot(ax=ax);

dfo['Left'].plot(ax=ax);

plt.legend(['Right','Left'])

plt.title("Position vs Foot");
#following CRISP DM methodology to answer this question 

#features chosen

dfv = dfa[['Preferred Foot','Position','Crossing', 'Finishing', 'HeadingAccuracy',

       'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',

       'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',

       'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',

       'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision',

       'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle',

       'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes','Value_kEuro']]



#Several Attributes, Positioning and Prefered Foot are chosen as features to predict the Market Value

# Get one hot encoding of column - position and preferred foot

one_hot = pd.get_dummies(dfv[['Position','Preferred Foot']])

one_hot



# Drop columns as it is now encoded

dfv = dfv.drop(columns = ['Position','Preferred Foot'],axis = 1)



# Join the encoded df

dfv = dfv.join(one_hot)

dfv.head(5)
(dfv.isnull().sum()/dfv.count())*100
dfv.dropna(inplace= True)

dfv.isnull().sum() #no null left
#To predict the "value" based on chosen attributes, defining y and x

y = dfv['Value_kEuro']

X = dfv.drop(['Value_kEuro'],axis=1)



#train test split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
# Create the parameter grid based on the results of random search 

gc.collect()



def modeling_gridsearch(base_estimator, param_grid):

    """

        Purpose - to Grid Search to find best parameters before prediction 

        

        Input:  Parameters Grid and Base Estimator 

        

        Output: Model & Best Parameters 

    """    

    

    # Instantiate the grid search model

    grid_search = GridSearchCV(estimator = base_estimator, param_grid = param_grid, 

                              cv = 3, n_jobs = 1, verbose = 2,scoring='neg_mean_squared_error')





    # Fit the grid search to the data

    grid_search.fit(X_train,y_train)

    print("Best Parameters set: ",grid_search.best_params_)

    

    return grid_search 
#Random Forest Ensemble Model is chosen for Modeling 



rf = RandomForestRegressor() #instantiating base estimator & parameters grid for gridsearch

param_grid = {

        'max_depth': [150,180,200],

        'max_features': [6,7,8,9],

        'min_samples_leaf': [2,3, 4, 5],

        'min_samples_split': [5,6,8, 10],

        'n_estimators': [100, 200, 300]

    }



model = modeling_gridsearch(rf,param_grid)



#prediction

y_pred = model.predict(X_test)

#Evaluation using RSquared 

print("Rsquared: ",r2_score(y_test, y_pred))
gc.collect()



#Mutual info regressor to find the best set of features that effect the Market value of a Player. 

mutual_infos=mutual_info_regression(X.values,y.values ) 



feature_importances = {}

for i,f in enumerate(X.columns):

    feature_importances[f] = mutual_infos[i]    

    

sorted(feature_importances.items(), key=lambda kv: kv[1], reverse=True)
plt.figure(figsize=(16,8))

sns.countplot(x = 'Position',

              data = dfa,

              order = dfa['Position'].value_counts().index,palette=sns.color_palette("Blues_d",n_colors=27));
dfa[['Wage_kEuro','Club']].groupby(['Club'])['Wage_kEuro'].median().sort_values(ascending=False).head(11)
dfa[['ReleaseClaus_kEuro','Name']].sort_values(by='ReleaseClaus_kEuro',ascending=False)['Name'].head(11).reset_index(drop=True)