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

calendar_data = pd.read_csv('../input/calendar.csv', header = 0)

listings_data = pd.read_csv('../input/listings.csv', header = 0)

reviews_data = pd.read_csv('../input/reviews.csv', header = 0)

#Let's see what these puppies contain

calendar_data.info()

calendar_data[:5]
listings_data.info()

listings_data[:5]
reviews_data.info()

reviews_data[:5]
#Cleaning time... let's trim the dataset down from 92 fields into ones that are going to be the most valuable

#DESCRIPTIONS - name, summary, space, description

#HOST - host_response_time, host_response_rate, host_acceptance_rate, host_is_superhost

#HOME DETAILS - property_type, room_type, accomodates, bathrooms, bedrooms, beds, bed_type, amenities

#LISTING ELEMENTS - price, cleaning_fee, extra_people, minimum_nights, maximum_nights, instant_bookable, cancellation policy

#OUTCOME VARIABLES - 'review_scores_rating', 'review_scores_accuracy', review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'



ld = listings_data.loc[:,['name', 'summary', 'space', 'description', 

    'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 

    'neighbourhood_group_cleansed', 'property_type', 'room_type', 

    'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type',

    'amenities', 'price', 'cleaning_fee', 'minimum_nights',

    'maximum_nights', 'instant_bookable',

    'cancellation_policy', 'review_scores_rating', 'review_scores_accuracy',

    'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',

    'review_scores_location', 'review_scores_value']]
#Let's see what we have now

ld.info()
#Get lengths of description elements

ld.loc[:,'name_length'] = ld['name'].str.len()

ld.loc[:,'summary_length'] = ld['summary'].str.len()

ld.loc[:,'space_length'] = ld['space'].str.len()

ld.loc[:,'description_length'] = ld['description'].str.len()

ld.loc[:,'amenities_length'] = ld['amenities'].str.len()

ld.loc[:,'name_length'].fillna(0, inplace=True)

ld.loc[:,'summary_length'].fillna(0, inplace=True)

ld.loc[:,'space_length'].fillna(0, inplace=True)

ld.loc[:,'description_length'].fillna(0, inplace=True)

ld.loc[:,'amenities_length'] = ld['amenities'].str.len()



#Drop original description variables

ld = ld.drop(['name', 'summary', 'space', 'description', 'amenities'], 1)
#Recode host_response_time to integers as it is a spectrum

ld['host_response_time'].replace('within an hour', 1, inplace=True)

ld['host_response_time'].replace('within a few hours', 2, inplace=True)

ld['host_response_time'].replace('within a day', 3, inplace=True)

ld['host_response_time'].replace('a few days or more', 4, inplace=True)



#Impute missing values using forward-fill method

ld['host_response_time'].fillna(method='ffill', inplace=True)
#Convert acceptance/response percentages to numbers

ld['host_acceptance_rate'] = ld['host_acceptance_rate'].replace('%','',regex=True).astype('float64')/100.00

ld['host_response_rate'] = ld['host_response_rate'].replace('%','',regex=True).astype('float64')/100.00



#Impute missing values using forward-fill method

ld['host_response_rate'].fillna(method='ffill', inplace=True)

ld['host_acceptance_rate'].fillna(method='ffill', inplace=True)
#Change f/t values to binary 0/1

ld['host_is_superhost'].replace('t',1, inplace=True)

ld['host_is_superhost'].replace('f',0, inplace=True)

ld['instant_bookable'].replace('t',1, inplace=True)

ld['instant_bookable'].replace('f',0, inplace=True)
#Clean property_type category to 5 categories

def recode(value):

    if value not in ['House', 'Apartment', 'Touwnhouse', 'Condominium']:

        return 'Other'

    return value



ld['property_type'] = ld['property_type'].apply(recode)
#Clean bed_type to binary real bed or other

def recode(value):

    if value not in ['Real Bed']:

        return 'Other'

    return value



ld['bed_type'] = ld['bed_type'].apply(recode)
#Convert cleaning fee and price from strings to numbers

ld['price'] = ld['price'].str.replace('$', '')

ld['price'] = ld['price'].str.replace(',', '').astype('float64')

ld['cleaning_fee'] = ld['cleaning_fee'].str.replace('$', '')

ld['cleaning_fee'] = ld['cleaning_fee'].str.replace(',', '').astype('float64')
#Missing data for cleaning fee indicates a $0 cleaning fee

ld['cleaning_fee'].fillna(0, inplace=True)



#Can't be sure what a missing value for these so we'll fill na

ld['bathrooms'].fillna(method='ffill', inplace=True)

ld['bedrooms'].fillna(method='ffill', inplace=True)

ld['beds'].fillna(method='ffill', inplace=True)

ld['host_is_superhost'].fillna(method='ffill', inplace=True)
#Get rid of all records that don't have our outcome variable

ld = ld[ld['review_scores_rating'].isnull() == 0]
ld.info()
#Let's look at some distributions

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numdf = ld.select_dtypes(include=numerics)

numdf = numdf.dropna(axis=0, how='any')

numdf_variables = list(numdf)



#Set the number of graphs in the facet chart

graphs = len(numdf_variables)-1



#create a list of positions for the chart

position = []

for i in range(8):

    for j in range(3):

        b = i,j

        position.append(b)



#Create base of subplot chart.. rows x columbs = graphs

fig, axes = plt.subplots(nrows=8, ncols=3, sharey=False, sharex=False, figsize=(12,20))

fig.subplots_adjust(hspace=.5)



#Fill in base with graphs based off of position

for i in range(graphs):

    sns.distplot(numdf[numdf_variables[i]], ax=axes[position[i]], kde=False)
#Correlation Matrix

fig, ax = plt.subplots(figsize=(16,10))

corr = ld.corr()

corr = (corr)

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values, linewidths=.5, ax=ax)

corr['review_scores_rating'].sort_values(ascending=False)
#Remove all ratings but our overall review_scores_rating outcome

model_data = ld.loc[:,('host_response_time',

 'host_response_rate',

 'host_acceptance_rate',

 'host_is_superhost',

 'neighbourhood_group_cleansed',

 'property_type',

 'room_type',

 'accommodates',

 'bathrooms',

 'bedrooms',

 'beds',

 'bed_type',

 'price',

 'cleaning_fee',

 'minimum_nights',

 'maximum_nights',

 'instant_bookable',

 'cancellation_policy',

 'review_scores_rating',

 'name_length',

 'summary_length',

 'space_length',

 'description_length',

 'amenities_length')]
#Get dummy variables for our 5 categorical fields

model_data = pd.get_dummies(model_data, columns=['neighbourhood_group_cleansed', 'property_type', 'room_type', 'bed_type', 'cancellation_policy'])
#Create Training / Test splits

from sklearn.model_selection import train_test_split



target_name = 'review_scores_rating'

X = model_data.drop('review_scores_rating', axis=1)

y=model_data[target_name]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=243)
#BASIC OLS REGRESSION... PUT IT ALL IN

Xr = X_train #Set dependent variable

yr = y_train #Target outcome is review_scores_rating

Xr = sm.add_constant(Xr) ## let's add an intercept (beta_0) to our model

ols_model = sm.OLS(yr, Xr).fit() ## sm.OLS(output, input)



# Print out the statistics

ols_model.summary()
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
from sklearn import tree

from sklearn.tree import DecisionTreeRegressor



#Make the decision tree

dtree = tree.DecisionTreeClassifier(

    class_weight="balanced",

    min_weight_fraction_leaf=0.01,)

dtree = dtree.fit(X_train,y_train)



#Look at outputs

importances = dtree.feature_importances_

feat_names = X_train.columns

tree_result = pd.DataFrame({'feature': feat_names, 'importance': importances})

tree_result.sort_values(by='importance',ascending=False)[:10].plot(x='feature', y='importance', kind='bar')
#Get Decision Tree mean squared error on test dataset

dtree_y_predict = dtree.predict(X_test)

dtree_mse = np.sqrt(metrics.mean_squared_error(y_test, dtree_y_predict))

dtree_mse
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=3, random_state=0)

rf.fit(X_train, y_train)

rf_importances = rf.feature_importances_

rf_result = pd.DataFrame({'feature': feat_names, 'importance': rf_importances})

rf_result.sort_values(by='importance',ascending=False)[:10].plot(x='feature', y='importance', kind='bar')
#Get Random Forest mean squared error on test dataset

rf_y_predict = rf.predict(X_test)

rf_mse = np.sqrt(metrics.mean_squared_error(y_test, rf_y_predict))

rf_mse
#What would be the MSE if we just used the test median/mean

mean_array = [y_train.mean()] * len(X_test)

mean_mse = np.sqrt(metrics.mean_squared_error(y_test, mean_array))

median_array = [y_train.median()] * len(X_test)

median_mse = np.sqrt(metrics.mean_squared_error(y_test, mean_array))
print("Random Forest MSE:", rf_mse)

print("Decision Tree MSE:", dtree_mse)

print("OLS MSE",ols_mse)

print("Median MSE",median_mse)

print("Mean MSE",mean_mse)
#Restrict the Random Forest model to the top 10 predictors

rf_10_features = rf_result.sort_values(by='importance', ascending=False)['feature'][:10].tolist()

rf_10_features_model = rf.fit(X_train[rf_10_features], y_train)

rf_10_features_model_predict=rf_10_features_model.predict(X_test[rf_10_features])

rf_10_features_model_mse = np.sqrt(metrics.mean_squared_error(y_test, rf_10_features_model_predict))

print(rf_10_features_model_mse, "Not much better")
# Recursive Feature Elimination

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

# create a base classifier used to evaluate a subset of attributes

model = LogisticRegression()

# create the RFE model and select 3 attributes

rfe = RFE(model, 10)

rfe = rfe.fit(X_train, y_train)

# summarize the selection of the attributes

rfe_result = pd.DataFrame({'feature': feat_names, 'ranking': rfe.ranking_, 'selection': rfe.support_})

rfe_result.sort_values(by='ranking')[:10]
#Does our Random Forest model like these 10 predictors better?

rf_10_features = rfe_result.sort_values(by='ranking')['feature'][:10].tolist()

rf_10_features_model = rf.fit(X_train[rf_10_features], y_train)

rf_10_features_model_predict=rf_10_features_model.predict(X_test[rf_10_features])

rf_10_features_model_mse = np.sqrt(metrics.mean_squared_error(y_test, rf_10_features_model_predict))

print(rf_10_features_model_mse, "Not really")
# Recursive Feature Elimination

# create a base classifier used to evaluate a subset of attributes

model = LinearRegression()

# create the RFE model and select 3 attributes

rfe = RFE(model, 10)

rfe = rfe.fit(X_train, y_train)

# summarize the selection of the attributes

rfe_result = pd.DataFrame({'feature': feat_names, 'ranking': rfe.ranking_, 'selection': rfe.support_})

rfe_result.sort_values(by='ranking')[:10]
#Does our Random Forest model like these 10 predictors better?

rf_10_features = rfe_result.sort_values(by='ranking')['feature'][:10].tolist()

rf_10_features_model = rf.fit(X_train[rf_10_features], y_train)

rf_10_features_model_predict=rf_10_features_model.predict(X_test[rf_10_features])

rf_10_features_model_mse = np.sqrt(metrics.mean_squared_error(y_test, rf_10_features_model_predict))

print(rf_10_features_model_mse, "Not really")