# Imports and preparing the dataset

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import scipy



import warnings

warnings.filterwarnings("ignore")



import folium # for the map



# Setting the default style of the plots

sns.set_style('whitegrid')

sns.set_palette('Set2')



# My custom color palette

my_palette = ["#7A92FF", "#FF7AEF", "#B77AFF", "#A9FF7A", "#FFB27A", "#FF7A7A",

             "#7AFEFF", "#D57AFF", "#FFDF7A", "#D3FF7A"]



# Importing the 3 datasets

data_2015 = pd.read_csv("../input/world-happiness/2015.csv")

data_2016 = pd.read_csv("../input/world-happiness/2016.csv")

data_2017 = pd.read_csv("../input/world-happiness/2017.csv")



# First we need to prepare the data for merging the tables together (to form only 1 table)

# Tables have different columns, so first we will keep only the columns we need

data_2015 = data_2015[['Country', 'Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',

                       'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)', 

                       'Dystopia Residual']]

data_2016 = data_2016[['Country', 'Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',

                       'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)', 

                       'Dystopia Residual']]

data_2017 = data_2017[['Country', 'Happiness.Rank', 'Happiness.Score', 'Economy..GDP.per.Capita.', 'Family',

                       'Health..Life.Expectancy.', 'Freedom', 'Generosity', 'Trust..Government.Corruption.', 

                       'Dystopia.Residual']]



# Tables do not have the same column names, so we need to fix that

new_names = ['Country', 'Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',

                       'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)', 

                       'Dystopia Residual']



data_2015.columns = new_names

data_2016.columns = new_names

data_2017.columns = new_names



# Add a new column containing the year of the survey

data_2015['Year'] = 2015

data_2016['Year'] = 2016

data_2017['Year'] = 2017



# Merge the data together

data = pd.concat([data_2015, data_2016, data_2017], axis=0)

data.head(3)
# New data

data_2018 = pd.read_csv("../input/world-happiness/2018.csv")

data_2019 = pd.read_csv("../input/world-happiness/2019.csv")



# Concatenate data

data_2018['Year'] = 2018

data_2019['Year'] = 2019



new_data = pd.concat([data_2018, data_2019], axis=0)



# Switching overall rank column with country/ region

columns_titles = ['Country or region', 'Overall rank', 'Score', 'GDP per capita',

       'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',

       'Perceptions of corruption', 'Year']

new_data = new_data.reindex(columns=columns_titles)



# Renaming old data columns:

old_data = data[['Country', 'Happiness Rank', 'Happiness Score','Economy (GDP per Capita)', 'Family', 

                 'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)', 'Year']]

old_data.columns = ['Country or region', 'Overall rank', 'Score', 'GDP per capita',

       'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',

       'Perceptions of corruption', 'Year']



# Finally, concatenating all data

data = pd.concat([old_data, new_data], axis=0)



data.head(3)
data[data['Perceptions of corruption'].isna()]
data.dropna(axis = 0, inplace = True)
# Double check to see if there are any missing values left

plt.figure(figsize = (16,6))

sns.heatmap(data = data.isna(), cmap = 'Blues')



plt.xticks(fontsize = 13.5);
data.shape



# 10 columns, 781 rows
data.groupby(by='Year')['Score'].describe()
# First we group the data by year and average the factors

grouped = data.groupby(by = 'Year')[['Score', 'GDP per capita',

       'Social support', 'Healthy life expectancy',

       'Freedom to make life choices', 'Generosity',

       'Perceptions of corruption']].mean().reset_index()



# Now we reconstruct the df by using melt() function

grouped = pd.melt(frame = grouped, id_vars='Year', value_vars=['Score', 'GDP per capita',

       'Social support', 'Healthy life expectancy',

       'Freedom to make life choices', 'Generosity',

       'Perceptions of corruption'], var_name='Factor', value_name='Avg_value')



grouped.head()
plt.figure(figsize = (16, 9))



ax = sns.barplot(x = grouped[grouped['Factor'] != 'Score']['Factor'], y = grouped['Avg_value'], 

            palette = my_palette[1:], hue = grouped['Year'])



plt.title("Difference in Factors - Then and Now - ", fontsize = 25)

plt.xlabel("Factor", fontsize = 20)

plt.ylabel("Average Score", fontsize = 20)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.legend(fontsize = 15)



ax.set_xticklabels(['Money','Family', 'Health', 'Freedom', 'Generosity', 'Trust']);
# Average top 5 most happy countries

country_score_avg = data[data['Year']==2019].groupby(by = ['Country or region'])['Score'].mean().reset_index()

table = country_score_avg.sort_values(by = 'Score', ascending = False).head(10)



table
plt.figure(figsize = (16, 9))

sns.barplot(y = table['Country or region'], x = table['Score'], palette = my_palette)



plt.title("Top 10 Happiest Countries in 2019", fontsize = 25)

plt.xlabel("Happiness Score", fontsize = 20)

plt.ylabel("Country", fontsize = 20)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15);
# Average top 5 most "not that happy" countries

table2 = country_score_avg.sort_values(by = 'Score', ascending = True).head(10)



table2
plt.figure(figsize = (16, 9))

sns.barplot(y = table2['Country or region'], x = table2['Score'], palette = my_palette)



plt.title("Top 10 Least Happy Countries in 2019", fontsize = 25)

plt.xlabel("Happiness Score", fontsize = 20)

plt.ylabel("Country", fontsize = 20)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15);
# Checking the distribution for Happiness Score

plt.figure(figsize = (16, 9))



sns.distplot(a = country_score_avg['Score'], bins = 20, kde = True, color = "#A9FF7A")

plt.xlabel('Happiness Score', fontsize = 20)

plt.title('Distribution of Average Happiness Score - 2019 -', fontsize = 25)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.xlim((1.5, 8.9));
## Creating the grouped table

country_factors_avg = data[data['Year'] == 2019].groupby(by = ['Country or region'])[['GDP per capita',

       'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',

       'Perceptions of corruption']].mean().reset_index()



plt.figure(figsize = (16, 9))



sns.kdeplot(data = country_factors_avg['GDP per capita'], color = "#B77AFF", shade = True)

sns.kdeplot(data = country_factors_avg['Social support'], color = "#FD7AFF", shade = True)

sns.kdeplot(data = country_factors_avg['Healthy life expectancy'], color = "#FFB27A", shade = True)

sns.kdeplot(data = country_factors_avg['Freedom to make life choices'], color = "#A9FF7A", shade = True)

sns.kdeplot(data = country_factors_avg['Generosity'], color = "#7AFFD4", shade = True)

sns.kdeplot(data = country_factors_avg['Perceptions of corruption'], color = "#FF7A7A", shade = True)



plt.xlabel('Factors Score', fontsize = 20)

plt.title('Distribution of Average Factors Score - 2019 -', fontsize = 25)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.xlim((-0.5, 2.3))

plt.legend(fontsize = 15);
# Calculating the Pearson Correlation



c1 = scipy.stats.pearsonr(data['Score'], data['GDP per capita'])

c2 = scipy.stats.pearsonr(data['Score'], data['Social support'])

c3 = scipy.stats.pearsonr(data['Score'], data['Healthy life expectancy'])

c4 = scipy.stats.pearsonr(data['Score'], data['Freedom to make life choices'])

c5 = scipy.stats.pearsonr(data['Score'], data['Generosity'])

c6 = scipy.stats.pearsonr(data['Score'], data['Perceptions of corruption'])



print('Happiness Score + GDP: pearson = ', round(c1[0],2), '   pvalue = ', round(c1[1],4))

print('Happiness Score + Family: pearson = ', round(c2[0],2), '   pvalue = ', round(c2[1],4))

print('Happiness Score + Health: pearson = ', round(c3[0],2), '   pvalue = ', round(c3[1],4))

print('Happiness Score + Freedom: pearson = ', round(c4[0],2), '   pvalue = ', round(c4[1],4))

print('Happiness Score + Generosity: pearson = ', round(c5[0],2), '   pvalue = ', round(c5[1],4))

print('Happiness Score + Trust: pearson = ', round(c6[0],2), '   pvalue = ', round(c6[1],4))
# Computing the Correlation Matrix



corr = data.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(16, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(0, 25, as_cmap=True, s = 90, l = 45, n = 5)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.title('What influences our happiness?', fontsize = 25)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15);
# import os

# print(list(os.listdir("../input")))
#json file with the world map

import matplotlib.pyplot as plt

import geopandas as gpd



country_geo = gpd.read_file('../input/worldcountries/world-countries.json')



#import another CSV file that contains country codes

country_codes = pd.read_csv('../input/iso-country-codes-global/wikipedia-iso-country-codes.csv')

country_codes.rename(columns = {'English short name lower case' : 'Country or region'}, inplace = True)



#Merge the 2 files together to create the data to display on the map

data_to_plot = pd.merge(left= country_codes[['Alpha-3 code', 'Country or region']], 

                        right= country_score_avg[['Score', 'Country or region']], 

                        how='inner', on = ['Country or region'])

data_to_plot.drop(labels = 'Country or region', axis = 1, inplace = True)



data_to_plot.head(2)
#Creating the map using Folium Package

my_map = folium.Map(location=[10, 6], zoom_start=1.49)



my_map.choropleth(geo_data=country_geo, data=data_to_plot, 

                  name='choropleth',

                  columns=['Alpha-3 code', 'Score'],

                  key_on='feature.id',

                  fill_color='BuPu', fill_opacity=0.5, line_opacity=0.2,

                  nan_fill_color='white',

                  legend_name='Average Happiness Indicator')



my_map.save('data_to_plot.html')



from IPython.display import HTML

HTML('<iframe src=data_to_plot.html width=850 height=500></iframe>')
# Importing the libraries

from sklearn.model_selection import train_test_split # for data validation



# Models

from sklearn.linear_model import LinearRegression, BayesianRidge, LassoLars

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

from xgboost import XGBRegressor



# Metrics and Grid Search

from sklearn import model_selection, metrics

from sklearn.model_selection import GridSearchCV
# Creating the table

data_model = data.groupby(by= 'Country or region')['Score', 'GDP per capita',

       'Social support', 'Healthy life expectancy',

       'Freedom to make life choices', 'Generosity',

       'Perceptions of corruption'].mean().reset_index()



# Creating the dependent and independent variables

y = data_model['Score']

X = data_model[['GDP per capita',

       'Social support', 'Healthy life expectancy',

       'Freedom to make life choices', 'Generosity',

       'Perceptions of corruption']]



# Splitting the data to avoid under/overfitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# Creating a predefined function to test the models

def modelfit(model):

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = metrics.mean_absolute_error(y_test, preds)

    print('MAE:', round(mae,4))
# Linear Regression



lm = LinearRegression(n_jobs = 10000)

modelfit(lm)
# Random Forest Regressor



rf = RandomForestRegressor(n_jobs = 1000)

modelfit(rf)
# XGBoost

xg = XGBRegressor(learning_rate=0.1, n_estimators=5000)

modelfit(xg)
# Decision Tree

dt = DecisionTreeRegressor()

modelfit(dt)
# Bayesian Linear Model

br = BayesianRidge(n_iter=1000, tol = 0.5)

modelfit(br)
# Lasso Lars

ls = LassoLars()

modelfit(ls)
final_model = BayesianRidge(n_iter = 10, tol = 0.1, alpha_2 = 0.1)

final_model.fit(X_train, y_train)
# How important is each variable into predicting the overall Happiness Score?



import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(estimator=final_model, random_state=1)

perm.fit(X_test, y_test)



eli5.show_weights(estimator= perm, feature_names = X_test.columns.tolist())