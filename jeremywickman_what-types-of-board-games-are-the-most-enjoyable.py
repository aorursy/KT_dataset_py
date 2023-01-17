#Import packages to be used

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

matplotlib.style.use('ggplot')

%matplotlib inline



#Load in the data and tell me something about it

import sqlite3

conn = sqlite3.connect('../input/database.sqlite')

query = "SELECT * FROM BoardGames"

df_boardgame_full = pd.read_sql_query(query,conn)
#Let's look at what fields we have available

df_boardgame_full.info()
#Parse down fields to those which describe attributes of the game

df_boardgame = df_boardgame_full.loc[:,

                                     ('game.id',

                                      'game.type',

                                      'details.description',

                                      'details.maxplayers',

                                      'details.maxplaytime',

                                      'details.minage',

                                      'details.minplayers',

                                      'details.minplaytime',

                                      'details.name',

                                      'details.playingtime',

                                      'details.yearpublished',

                                      'attributes.boardgamecategory',

                                      'attributes.boardgamemechanic',

                                      'attributes.boardgamepublisher',

                                      'stats.averageweight',

                                      'stats.average')]
#Let's look at some distributions

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numdf = df_boardgame.select_dtypes(include=numerics)

numdf = numdf.dropna(axis=0, how='any')

numdf_variables = list(numdf)



#Set the number of graphs in the facet chart

graphs = len(numdf_variables)-1



#create a list of positions for the chart

position = []

for i in range(4):

    for j in range(2):

        b = i,j

        position.append(b)



#Create base of subplot chart.. rows x columbs = graphs

fig, axes = plt.subplots(nrows=4, ncols=2, sharey=False, sharex=False, figsize=(12,20))

fig.subplots_adjust(hspace=.5)



#Fill in base with graphs based off of position

for i in range(graphs):

    sns.distplot(numdf[numdf_variables[i]], ax=axes[position[i]], kde=False)
#Calculate outliers for all numeric details

for field in numdf_variables:

    value_mean = df_boardgame[field].median()

    value_q1 = df_boardgame[field].quantile(.25)

    value_q3 = df_boardgame[field].quantile(.75)

    value_qrange = value_q3-value_q1

    lower_outlier = value_q1-(4.5 * value_qrange)

    upper_outlier = value_q3+(4.5 * value_qrange)

    print(field)

    print(len(df_boardgame[df_boardgame[field]>upper_outlier]), 'upper outliers at', upper_outlier)

    print(len(df_boardgame[df_boardgame[field]<lower_outlier]), 'lower outliers at', lower_outlier)

#Correlation Matrix

fig, ax = plt.subplots(figsize=(16,10))

corr = df_boardgame.corr()

corr = (corr)

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values, linewidths=.5, ax=ax)

corr['stats.average'].sort_values(ascending=False)
#Let's make a new dataframe to work on categorizing the boardgame correctly

#New manageable data frame

df_category = df_boardgame.loc[:, ['game.id','attributes.boardgamecategory']]

df_category['attributes.boardgamecategory'] = df_category['attributes.boardgamecategory'].fillna("None")



#Use comma splits to break out individual mechanical types

df_category = df_category['attributes.boardgamecategory'].apply(lambda x: pd.Series(x.split(',')))



#Sum the rows to determine most popular category types

category_counts = df_category.apply(pd.Series.value_counts).fillna(0)

category_counts['Total'] = category_counts.sum(axis=1)



#Let's get the most popular categories of games for our dataset

category_counts = category_counts.sort_values(by='Total', ascending=False)

category_list = category_counts[category_counts['Total']>500].index.tolist()



#Create dummies for all the boardgame categories

df_boardgame['attributes.boardgamecategory'].fillna(0, inplace=True)

for i in category_list:

    df_boardgame.loc[df_boardgame['attributes.boardgamecategory'].str.contains(i) == True ,i] = 1

    df_boardgame.loc[df_boardgame['attributes.boardgamecategory'].str.contains(i) == False ,i] = 0

    df_boardgame[i].fillna(0, inplace=True)
#Create a dataframe that takes boardgame categories and sorts by their overall mean rating

d =[]

for i in category_list:

    score = df_boardgame[df_boardgame[i]==1]['stats.average'].mean()

    d.append({'Avg_Rating': score, 'Game Type': i})

df_categorymean = pd.DataFrame(d).sort_values(by='Avg_Rating', ascending=False)

df_categorymean[:10]
#Create base of subplot chart.. rows x columbs = graphs

fig, (ax1, ax2) = plt.subplots(nrows=2, sharey=True, sharex=False, figsize=(20,12))

fig.subplots_adjust(hspace=.5)

sns.barplot(x="Game Type", y="Avg_Rating", data=df_categorymean[:10], ax=ax1)

sns.barplot(x="Game Type", y="Avg_Rating", data=df_categorymean[-10:], ax=ax2)

ax1.title.set_text('Top Rated Game Categories')

ax2.title.set_text('Lowest Rated Game Categories')
#Drop outliers for details.minage

df_boardgame = df_boardgame[df_boardgame['details.minage']<66]



#Drop "The Ancients" i.e. games that came out before 1900

df_boardgame = df_boardgame[df_boardgame['details.yearpublished']>1900]



'''

I don't like these details.maxplayers and details.maxplaytime, because 0s indicate the opposite

of their direction impact i.e. maxplayers = 0 indicates infinite players

'''
model_data = df_boardgame.loc[:,

                                     (

                                      'game.type',

                                      'details.minage',

                                      'details.minplayers',

                                      'details.yearpublished',

                                      'attributes.boardgamecategory',

                                      'stats.averageweight',

                                      'stats.average')]
#Create dummies for all the boardgame categories on model_data

for i in category_list:

    model_data.loc[model_data['attributes.boardgamecategory'].str.contains(i) == True ,i] = 1

    model_data.loc[model_data['attributes.boardgamecategory'].str.contains(i) == False ,i] = 0

    model_data[i].fillna(0, inplace=True)
#Get dummy variables for boardgame/expansion type

model_data = pd.get_dummies(model_data, columns=['game.type'])
#Get rid of remaining object categories

model_data = model_data.loc[:, model_data.columns != 'attributes.boardgamecategory']
#Create Training / Test splits

from sklearn.model_selection import train_test_split



target_name = 'stats.average'

X = model_data.drop('stats.average', axis=1)

y=model_data[target_name]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=243)
#Let's use the basic OLS regression from sklearn

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)



#What are the features that have the most weight?

ols_coefficients = pd.DataFrame({'feature': X_train.columns, 'importance': lm.coef_})

ols_coefficients.sort_values('importance', ascending=False)[:10]
#Get OLS mean squared error on test dataset 

from sklearn import metrics

ols_y_predict = lm.predict(X_test)

ols_mse = np.sqrt(metrics.mean_squared_error(y_test, ols_y_predict))

ols_mse
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=7, random_state=0)

rf.fit(X_train, y_train)

rf_importances = rf.feature_importances_

feat_names = X_train.columns

rf_result = pd.DataFrame({'feature': feat_names, 'importance': rf_importances})

rf_result.sort_values(by='importance',ascending=False)[:10].plot(x='feature', y='importance', kind='bar')
#Get Random Forest mean squared error on test dataset

rf_y_predict = rf.predict(X_test)

rf_mse = np.sqrt(metrics.mean_squared_error(y_test, rf_y_predict))

rf_mse
from sklearn.metrics import roc_auc_score

print("AUC - ROC : ", roc_auc_score(y_train,rf.oob_score))

#What would be the MSE if we just used the test median/mean

mean_array = [y_train.mean()] * len(X_test)

mean_mse = np.sqrt(metrics.mean_squared_error(y_test, mean_array))

median_array = [y_train.median()] * len(X_test)

median_mse = np.sqrt(metrics.mean_squared_error(y_test, mean_array))
print("Random Forest MSE:", rf_mse)

print("OLS MSE",ols_mse)

print("Median MSE",median_mse)

print("Mean MSE",mean_mse)