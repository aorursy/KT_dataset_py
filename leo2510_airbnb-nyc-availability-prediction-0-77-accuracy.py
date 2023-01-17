# Load necessary library

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor



import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn')



from wordcloud import WordCloud



%matplotlib inline



# set default plot size

plt.rcParams["figure.figsize"] = (15,8)
# Load and preview data 

ab_nyc = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")





ab_nyc.head()
# drop id and name columns

ab_nyc.drop(['id','name','host_id','host_name'],axis=1,inplace = True)

ab_nyc.describe()
# Check each column for nas

ab_nyc.isnull().sum()
# data cleanning

# remove outliers for price and minimun nights column



from scipy import stats



ab_nyc['z_price'] = np.abs(stats.zscore(ab_nyc['price']))

ab_nyc['z_min_nights'] = np.abs(stats.zscore(ab_nyc['minimum_nights']))



# remove z scroe that are greater than 3



ab_nyc_final = ab_nyc[(ab_nyc['z_price'] < 3)]

ab_nyc_final = ab_nyc_final[(ab_nyc_final['price'] > 3)]

ab_nyc_final = ab_nyc_final[(ab_nyc['z_min_nights'] < 3)]



# convert numneric variables into categorical variables



ab_nyc_final['minimum_nights_group'] = 'Others'

ab_nyc_final['minimum_nights_group'][ab_nyc_final['minimum_nights'] == 1] = 'one night'

ab_nyc_final['minimum_nights_group'][ab_nyc_final['minimum_nights'] == 2] = 'two nights'

ab_nyc_final['minimum_nights_group'][ab_nyc_final['minimum_nights'] == 3] = 'three nights'

ab_nyc_final['minimum_nights_group'][ab_nyc_final['minimum_nights'] == 4] = 'four nights'

ab_nyc_final['minimum_nights_group'][ab_nyc_final['minimum_nights'] > 4] = 'five nights or more'

# ab_nyc_final.groupby('minimum_nights_group').size()



ab_nyc_final['calculated_host_listings_count_group'] = 'Others'

ab_nyc_final['calculated_host_listings_count_group'][ab_nyc_final['calculated_host_listings_count'] == 1] = 'one listing'

ab_nyc_final['calculated_host_listings_count_group'][ab_nyc_final['calculated_host_listings_count'] == 2] = 'two listings'

ab_nyc_final['calculated_host_listings_count_group'][ab_nyc_final['calculated_host_listings_count'] > 2] = 'more than two listings'





# ab_nyc_final.groupby('calculated_host_listings_count_group').size()

# remove unused columns

ab_nyc_final.drop(['z_price','z_min_nights','minimum_nights','last_review','neighbourhood',

                   'calculated_host_listings_count','reviews_per_month'],

                  axis = 1,inplace = True)

ab_nyc_final.head()
ab_nyc_final.describe()
ab_nyc_cor = ab_nyc_final.drop(['latitude','longitude'],axis=1).corr()

ab_nyc_cor
# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(ab_nyc_cor, dtype=np.bool))



# Set up the matplotlib figure

fig, ax = plt.subplots(figsize=(15, 8))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(ab_nyc_cor, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot = True)
import plotly.express as px



lat = np.mean(ab_nyc_final['latitude'])

lon = np.mean(ab_nyc_final['longitude'])





fig = px.density_mapbox(ab_nyc_final, lat='latitude', lon='longitude', z='price', radius=2,

                        center=dict(lat = lat, lon = lon), zoom=10,

                        mapbox_style="carto-positron")

fig.show()
ab_nyc_final.groupby(['neighbourhood_group'])['price','number_of_reviews','availability_365'].agg(['count', 'mean','median'])



# ab_nyc_final.groupby(['neighbourhood_group']).agg({

#     'price': ['mean', 'count', 'median'], 

#     'number_of_reviews': ['mean', 'count', 'median'],

#     'availability_365': ['mean', 'count', 'median']

# })
# boxplot of neighbourhood group and price

# entire home/apt have the highest median price over other room types

# manhattan entire home/apt have the highest median price

sns.boxplot(x="neighbourhood_group", y="price",hue = "room_type",data=ab_nyc_final)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# manhattan and brooklyn private room have the lowest availability/highest demand overall

sns.boxplot(x="neighbourhood_group", y="availability_365",hue = "room_type",data=ab_nyc_final)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# four nights minimum are the most popular

sns.boxplot(x="minimum_nights_group", y="availability_365",data=ab_nyc_final,

            order = ['one night','two nights','three nights','four nights','five nights or more'])
sns.boxplot(x="minimum_nights_group", y="price",data=ab_nyc_final,

            order = ['one night','two nights','three nights','four nights','five nights or more'])
sns.pairplot(ab_nyc_final.drop(['latitude','longitude'],axis=1))
ab_nyc_model = ab_nyc_final.drop(['latitude','longitude'],axis = 1)

ab_nyc_model.head()
# Building the model

# first convert categorical variables to dummy variables using one hot encoding



categorical_var = ['neighbourhood_group','room_type','minimum_nights_group','calculated_host_listings_count_group']



# create dummy variables for all the other categorical variables



for variable in categorical_var:

# #     fill missing data

#     recruit[variable].fillna('Missing',inplace=True)

#     create dummy variables for given columns

    dummies = pd.get_dummies(ab_nyc_model[variable],prefix=variable)

#     update data and drop original columns

    ab_nyc_model = pd.concat([ab_nyc_model,dummies],axis=1)

    ab_nyc_model.drop([variable],axis=1,inplace=True)



ab_nyc_model.head()
x = ab_nyc_model.drop(['availability_365'], axis=1)

y = ab_nyc_model['availability_365'].astype(float)



# split train and test dataset

train_x, test_x, train_y, test_y = train_test_split(x,y , test_size=0.3, random_state=42)



print(train_x.shape)

print(train_y.shape)



print(test_x.shape)

print(test_y.shape)
# training using statmodel



linear_model_sm = sm.OLS(train_y,sm.tools.add_constant(train_x).astype(float))

results_sm = linear_model_sm.fit()

print(results_sm.summary())
# using sklearn



linear_model_sk = LinearRegression()  

linear_model_sk.fit(train_x, train_y)

linear_model_sk.score(test_x, test_y)
pred_y = linear_model_sk.predict(test_x)

df = pd.DataFrame({'Actual': test_y, 'Predicted': pred_y})

df.head(30)
# random forest regressor for non-linear regression



rf_regressor = RandomForestRegressor(n_estimators=100,random_state=0)

rf_regressor.fit(train_x,train_y)
rf_regressor.score(train_x,train_y)
feature_importance = pd.Series(rf_regressor.feature_importances_,index=x.columns)

feature_importance = feature_importance.sort_values()

feature_importance.plot(kind='barh')
# parameter tunning

# # of trees trained parameter tunning



results_rf = []

n_estimator_options = [30,50,100,200,500,1000,2000]



for trees in n_estimator_options:

    model = RandomForestRegressor(trees,oob_score=True,n_jobs=-1,random_state=42)

    model.fit(train_x,train_y)

    print(trees," trees")

    score = model.score(train_x,train_y)

    print(score)

    results_rf.append(score)

    print("")



pd.Series(results_rf,n_estimator_options).plot()



# use 500 trees
# max number of features parameter tunning

results_rf = []

max_features_options = ['auto',None,'sqrt','log2',0.9,0.2]



for max_features in max_features_options:

    model = RandomForestRegressor(n_estimators=500,oob_score=True,n_jobs=-1,

                                  random_state=42,max_features=max_features)

    model.fit(train_x,train_y)

    print(max_features," option")

    score = model.score(train_x,train_y)

    print(score)

    results_rf.append(score)

    print("")



pd.Series(results_rf,max_features_options).plot(kind='barh')



# use auto option
# final model using the parameter tuning

rf_regressor = RandomForestRegressor(n_estimators=500,oob_score=True,n_jobs=-1,

                                  random_state=42,max_features='auto')

rf_regressor.fit(train_x,train_y)

rf_regressor.score(train_x,train_y)
pred_y = rf_regressor.predict(test_x)

df = pd.DataFrame({'Actual': test_y, 'Predicted': pred_y})

df.head(30)
fig, ax = plt.subplots()

ax.scatter(test_y, pred_y)

ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()