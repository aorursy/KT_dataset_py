# Import all the necessary libraries 



# commonly used libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



# visualization library

import seaborn as sns



# data manipulation utility libraries

import distutils

import datetime

import re



# sklearn libraries

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error



# stats library

from scipy import stats

from scipy.stats import boxcox





# Import the data

df = pd.read_csv('../input/airbnb-amsterdam/listings_details.csv')
#Look at top rows of dataframe

df.head()

#Quick summary of dataframe

df.describe()

#number of rows and columns

df.shape

#sum of NaN values in price column

np.sum(df.price.notnull())
# price column

df.price
# check out all columns with numeric values

num_vars = df.select_dtypes(include=['float', 'int']).columns

num_vars
# check out all columns with categorical values

num_cat = df.select_dtypes(include=['object']).columns



num_cat
#check values in different neighbourhood columns

df[['neighbourhood','neighborhood_overview','neighbourhood_cleansed']]
# see ratio of categorical values 

df.neighbourhood_cleansed.value_counts() / df.shape[0]
# drop columns that are irelevant 

df_clean = df.drop(['id', 'scrape_id', 'thumbnail_url', 'medium_url', 'xl_picture_url',

              'host_id', 'host_total_listings_count', 'neighbourhood_group_cleansed',

              'latitude','longitude', 'calculated_host_listings_count', 

              'listing_url', 'last_scraped', 'name', 'summary', 'space',

              'description', 'experiences_offered', 'neighborhood_overview', 'notes',

              'transit', 'access', 'interaction', 'house_rules', 'picture_url',

              'host_url', 'host_name', 'host_location', 'host_about',

              'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood','host_verifications',

              'street', 'neighbourhood', 'city', 'state',

              'zipcode', 'market', 'smart_location', 'country_code', 'country',

              'weekly_price', 'monthly_price','security_deposit', 'cleaning_fee',

              'extra_people', 'calendar_last_scraped', 'requires_license', 'license',

              'jurisdiction_names','guests_included','host_response_time','host_response_rate',

              'host_acceptance_rate','square_feet'

             ], axis=1)



# drop all NaN values

df_clean = df_clean.dropna()
# use string.replace to get rid of string items in price column

df_clean = df_clean.assign(price=df_clean['price'].str.replace(r'$', ''))

df_clean = df_clean.assign(price=df_clean['price'].str.replace(r',', ''))



#Set price as float type

df_clean['price'] = df_clean['price'].astype(float)
# use lambda and distutils to go from string to boolean expression

df_clean = df_clean.assign(host_is_superhost=df_clean['host_is_superhost'].apply(lambda x: bool(distutils.util.strtobool(x))))

df_clean = df_clean.assign(host_has_profile_pic=df_clean['host_has_profile_pic'].apply(lambda x: bool(distutils.util.strtobool(x))))

df_clean = df_clean.assign(host_identity_verified=df_clean['host_identity_verified'].apply(lambda x: bool(distutils.util.strtobool(x))))

df_clean = df_clean.assign(is_location_exact=df_clean['is_location_exact'].apply(lambda x: bool(distutils.util.strtobool(x))))

df_clean = df_clean.assign(instant_bookable=df_clean['instant_bookable'].apply(lambda x: bool(distutils.util.strtobool(x))))

df_clean = df_clean.assign(is_business_travel_ready=df_clean['is_business_travel_ready'].apply(lambda x: bool(distutils.util.strtobool(x))))

df_clean = df_clean.assign(require_guest_profile_picture=df_clean['require_guest_profile_picture'].apply(lambda x: bool(distutils.util.strtobool(x))))

df_clean = df_clean.assign(require_guest_phone_verification=df_clean['require_guest_phone_verification'].apply(lambda x: bool(distutils.util.strtobool(x))))

# find amenity availability by amenity

df_clean = df_clean.assign(has_tv=df_clean['amenities'].apply(lambda x: x.find('Wifi') != -1))

df_clean = df_clean.assign(has_fireplace=df_clean['amenities'].apply(lambda x: x.find('Indoor fireplace') != -1))

df_clean = df_clean.assign(has_kitchen=df_clean['amenities'].apply(lambda x: x.find('Kitchen') != -1))

df_clean = df_clean.assign(has_family_friendly=df_clean['amenities'].apply(lambda x: x.find('Family/kid friendly') != -1))

df_clean = df_clean.assign(has_host_greeting=df_clean['amenities'].apply(lambda x: x.find('Host greets you') != -1))

df_clean = df_clean.assign(has_24hrs_checkin=df_clean['amenities'].apply(lambda x: x.find('24-hour check-in') != -1))

df_clean = df_clean.assign(has_breakfast=df_clean['amenities'].apply(lambda x: x.find('Breakfast') != -1))

df_clean = df_clean.assign(has_pets=df_clean['amenities'].apply(lambda x: x.find('Pets live on this property') != -1))

df_clean = df_clean.assign(has_dishwasher=df_clean['amenities'].apply(lambda x: x.find('Dishwasher') != -1))

df_clean = df_clean.assign(has_private_entrance=df_clean['amenities'].apply(lambda x: x.find('Private entrance') != -1))

df_clean = df_clean.assign(has_patio_balcony=df_clean['amenities'].apply(lambda x: x.find('Patio or balcony') != -1))

df_clean = df_clean.assign(has_self_checkin=df_clean['amenities'].apply(lambda x: x.find('Self check-in') != -1))

df_clean = df_clean.assign(has_workspace=df_clean['amenities'].apply(lambda x: x.find('Laptop friendly workspace') != -1))

df_clean = df_clean.assign(has_bathtub=df_clean['amenities'].apply(lambda x: x.find('Bathtub') != -1))

df_clean = df_clean.assign(has_longterm=df_clean['amenities'].apply(lambda x: x.find('Long term stays allowed') != -1))

df_clean = df_clean.assign(has_parking=df_clean['amenities'].apply(lambda x: x.find('Free parking on premises') != -1))

df_clean = df_clean.assign(has_garden=df_clean['amenities'].apply(lambda x: x.find('Garden or backyard') != -1))



# drop amenities column

df_clean = df_clean.drop(['amenities'],axis=1)
# create days delta calculation function

day_calc = lambda x: (datetime.date.today() - datetime.datetime.strptime(x, "%Y-%m-%d").date()).days



# apply on host_since column

df_clean = df_clean.assign(host_since=df_clean['host_since'].apply(day_calc))
#drop columns that will not add more value than host since column

df_clean = df_clean.drop(['first_review','last_review'],axis=1)
# max price

df_clean.price.max()
# ratio of occurrences of value by certain column

df_clean.number_of_reviews.value_counts() / df_clean.shape[0]
# ratio of occurrences of value by certain column

df_clean.minimum_nights.value_counts() / df_clean.shape[0]

    
# check boxplot for price

sns.boxplot(x=df_clean['price'])

# create mod_z function (copied from: https://stackoverflow.com/questions/58127935/how-to-calculate-modified-z-score-and-iqr-on-each-column-of-the-dataframe)

def mod_z(col: pd.Series, thresh: float=3.5) -> pd.Series:

    med_col = col.median()

    med_abs_dev = (np.abs(col - med_col)).median()

    mod_z = 0.7413 * ((col - med_col) / med_abs_dev)

    mod_z = mod_z[np.abs(mod_z) < thresh]

    return np.abs(mod_z)



# run mod_z function on dataframe

df_mod_z = df_clean.select_dtypes(include=[np.number]).apply(mod_z)
#Apply above function to price 

df_clean_filtered = df_clean[df_mod_z['price'] >= 0]

df_clean_filtered = df_clean_filtered[df_clean_filtered['price'] > 0]



#check shape

df_clean_filtered.shape
# check summary

df_clean_filtered.describe()
#Cut outliers

df_clean_filtered = df_clean_filtered[df_clean_filtered['host_listings_count'] < 60]

df_clean_filtered = df_clean_filtered[df_clean_filtered['bathrooms'] < 20]

df_clean_filtered = df_clean_filtered[df_clean_filtered['beds'] < 20]

df_clean_filtered = df_clean_filtered[df_clean_filtered['minimum_nights'] < 365]

df_clean_filtered = df_clean_filtered[df_clean_filtered['maximum_nights'] < 2000]



#Check shape

df_clean_filtered.shape
# drop some  because they're too tricky 

df_clean_filtered_drop = df_clean_filtered.drop(['calendar_updated','has_availability'],axis=1)
# create dummy in dataframe function

def create_dummy_df(df, cat_cols, dummy_na):

    '''

    INPUT:

    df - pandas dataframe with categorical variables you want to dummy

    cat_cols - list of strings that are associated with names of the categorical columns

    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not

    

    OUTPUT:

    df - a new dataframe that has the following characteristics:

            1. contains all columns that were not specified as categorical

            2. removes all the original columns in cat_cols

            3. dummy columns for each of the categorical columns in cat_cols

            4. if dummy_na is True - it also contains dummy columns for the NaN values

            5. Use a prefix of the column name with an underscore (_) for separating 

    '''

    for col in  cat_cols:

        try:

            # for each cat add dummy var, drop original column

            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)

        except:

            continue

    return df
#Set categorical columns to be dummied

cat_cols_lst = df_clean_filtered_drop.select_dtypes(include=['object']).columns



#Apply create dummy function

df_model = create_dummy_df(df_clean_filtered_drop, cat_cols_lst, dummy_na=False) #Use your newly created function



#check df

df_model
#plot distribution of Price

plt.figure(figsize=(15,10))

plt.tight_layout()

sns.distplot(df_model['price'])
#Import Normalizer from sklearn

from sklearn.preprocessing import Normalizer 



#Set variables to normalize

norm_vars = ['beds','bedrooms','accommodates']



# initiate normalizer and apply to scaled data array

normalize = Normalizer().fit(df_model[norm_vars])

norm_array = normalize.transform(df_model[norm_vars])



# create a DataFrame from the array

df_model_norm_vars = pd.DataFrame(norm_array, columns = norm_vars, index = df_model.index)



# merge new DataFrame with full dataframe model 

df_model_merged = pd.merge(df_model_norm_vars,df_model.drop(norm_vars,axis=1), right_index=True, left_index=True)



# create combined column for beds, bedrooms and acommodates

df_model_merged['combine_beds_bedrooms_acommodates'] = df_model_merged['beds'] + df_model_merged['bedrooms'] + df['accommodates']



# drop already combined variables

df_model_merged = df_model_merged.drop(['beds','bedrooms','accommodates'],axis=1)
# Histograms of all vars with power transformation BoxCox



# set vars to check

check_vars = ['host_since', 'host_listings_count', 'bathrooms',

       'price', 'minimum_nights', 'maximum_nights',

       'number_of_reviews', 'review_scores_rating',

       'review_scores_communication', 'review_scores_location',

       'review_scores_value','combine_beds_bedrooms_acommodates',

           'reviews_per_month','availability_30', 'availability_60','availability_90', 'availability_365',

            'review_scores_accuracy','review_scores_cleanliness', 'review_scores_checkin']



# For loop on showing separate histograms per item

i = 0

for x in check_vars:

    # set data

    data = df_model_merged[x]

    

    # plot

    plt.figure(i)

    plt.title(x)

    plt.hist(data)

    print(plt.figure(i))

    

    # iterate 

    i = i + 1

    



# list of variables that need to be transformed to fit a normal distribution



to_check_vars = ['price','host_listings_count','bathrooms','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating',

            'review_scores_communication','review_scores_location','review_scores_value','review_scores_accuracy',

             'review_scores_cleanliness','review_scores_checkin',

             'combine_beds_bedrooms_acommodates',

            'reviews_per_month'

           ]





poly_vars = ['availability_30','availability_60','availability_90','availability_365']
# For loop on showing separate histograms per item

i = 0

for x in to_check_vars:

    plt.figure(i)

    plt.title(x)

    

    # power transform

    data = df_model_merged[x] + 1

    data = boxcox(data)

    

    #print the boxcox lambda value

    print(data[1])

    

    #plot the graph

    plt.hist(data)

    print(plt.figure(i))

    

    #increment the counter

    i = i + 1
# variables to keep and boxcox transform 

boxcox_vars = ['price','number_of_reviews','review_scores_rating',

            'review_scores_location','review_scores_value'

              ,'combine_beds_bedrooms_acommodates']





# variables to drop

to_drop = ['host_listings_count','bathrooms','minimum_nights','maximum_nights',

           'review_scores_communication','review_scores_accuracy','review_scores_cleanliness',

           'review_scores_checkin','reviews_per_month']
#drop variables

df_model_merged = df_model_merged.drop(to_drop,axis=1)



#drop poly variables

df_model_merged = df_model_merged.drop(poly_vars,axis=1)

# create boxcox in dataframe function

def create_boxcox_df(df, boxcox_vars):

    '''

    INPUT:

    df - pandas dataframe with categorical variables you want to dummy

    boxcox_vars - list of strings that are associated with selected boxcox columns

    

    OUTPUT:

    df - a new dataframe that has the following characteristics:

            1. contains all columns that were not specified as boxcox columns

            2. removes all the original columns in boxcox_vars

            3. boxcox transforms for each of the boxcox columns in boxcox_vars

            4. returns df

            5. returns list of lambda values for maxlog()

    '''

    #iniate empty list

    lambda_list = list()

    

    #start for loop

    for col in boxcox_vars:

        try:

            # for each var boxcox transform

            data = df[col] +1

            data = boxcox(data)

            

            #lambda list append

            lambda_list.append(data[1])

            

            #create dataframe from array

            df_insert = pd.DataFrame(data[0],columns = [col],index = df_model_merged.index)

            

            #concat dataframes

            df = pd.merge(df_insert,df.drop(col, axis=1), right_index=True, left_index=True)



            

        except:

            continue

    return df, lambda_list
# apply to dataframe

df_model_merged, lambda_list = create_boxcox_df(df_model_merged,boxcox_vars=boxcox_vars)

# histograms of the variables

df_hist = df_model_merged[boxcox_vars]



df_hist.hist()

print(plt.show())
# performing preprocessing part 

from sklearn.preprocessing import StandardScaler



# take all the variables that we want to standardize 

scaler_vars = ['number_of_reviews','review_scores_rating',

            'review_scores_location','review_scores_value'

              ,'combine_beds_bedrooms_acommodates']





# initiate standardscaler and apply to data

sc = StandardScaler()

scaled_array = sc.fit_transform(df_model_merged[scaler_vars])



# create new dataframe with scaled variables

df_model_scaled = pd.DataFrame(scaled_array, columns = scaler_vars, index = df_model.index)



# merge them all back together

df_model_merged = pd.merge(df_model_scaled,df_model_merged.drop(scaler_vars,axis=1), right_index=True, left_index=True)



#print(df_model_merged.describe())
# histograms of the variables

df_hist = df_model_merged[boxcox_vars]



df_hist.hist()

print(plt.show())
# histograms of the variables

df_hist = df_model_merged[boxcox_vars]



df_hist.hist()

plt.show



num_vars = ['host_since','price','number_of_reviews', 'review_scores_rating', 'review_scores_location','review_scores_value']



#df_plot = df_model_merged.select_dtypes(include=[np.number])

df_plot = df_model_merged[num_vars]



matrix = np.triu(df_plot.corr())



plt.figure(figsize=(12, 9))

sns.heatmap(df_plot.corr(), annot=False, mask=matrix, linewidths=.5, fmt='.1f')
# based on above correlation matrix, take out a few columns

df_model_merged = df_model_merged.drop(['review_scores_value'],axis=1)



def fit_linear_mod(df, response_col, test_size=.3, rand_state=42):

    '''

    INPUT:

    df - a dataframe holding all the variables of interest

    response_col - a string holding the name of the column

    test_size - a float between [0,1] about what proportion of data should be in the test dataset

    rand_state - an int that is provided as the random state for splitting the data into training and test 

    

    OUTPUT:

    test_score - float - r2 score on the test data

    train_score - float - r2 score on the test data

    lm_model - model object from sklearn

    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model

    

    This function should:

    1. Split your data into an X matrix and a response vector y

    2. Create training and test sets of data

    3. Instantiate a LinearRegression model with normalized data

    4. Fit your model to the training data

    5. Predict the response for the training data and the test data

    6. Obtain an rsquared value for both the training and test data

    '''



    #Split into explanatory and response variables

    X = df.drop(response_col, axis=1)

    y = df[response_col]



    #Split into train and test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)



    lm_model = LinearRegression(normalize=True) # Instantiate

    lm_model.fit(X_train, y_train) #Fit



    #Predict using your model

    y_test_preds = lm_model.predict(X_test)

    y_train_preds = lm_model.predict(X_train)



    #Score using your model

    test_score = r2_score(y_test, y_test_preds)

    train_score = r2_score(y_train, y_train_preds)



    return test_score, train_score, lm_model, X_train, X_test, y_train, y_test, y_test_preds, y_train_preds


#Test your function with the above dataset

test_score, train_score, lm_model, X_train, X_test, y_train, y_test, y_test_preds, y_train_preds = fit_linear_mod(df_model, 'price')



#Print training and testing score

print("The rsquared on the training data was {}.  The rsquared on the test data was {}.".format(train_score, test_score))



# The coefficients

#print('Coefficients: \n', lm_model.coef_)



# print RMSE

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_preds)))
# after preprocessing



#Test your function with the above dataset

test_score, train_score, lm_model, X_train, X_test, y_train, y_test, y_test_preds, y_train_preds = fit_linear_mod(df_model_merged, 'price')



#Print training and testing score

print("The rsquared on the training data was {}.  The rsquared on the test data was {}.".format(train_score, test_score))



# The coefficients

#print('Coefficients: \n', lm_model.coef_)



# print RMSE

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_preds)))
# Applying PCA function on training 

from sklearn.decomposition import PCA 

  

pca = PCA()

  

X_train = pca.fit_transform(X_train) 

  

explained_variance = pca.explained_variance_ratio_

print(explained_variance)
#import statsmodel library

import statsmodels.api as sm



#set response column and df to use

response_col = 'price'

df_to_use_ols = df_model_merged



# set X matrix and y 

X = df_to_use_ols.drop(response_col, axis=1)

X = sm.add_constant(X)

y = df_to_use_ols[response_col]



# fit and predict

est = sm.OLS(y.astype(float), X.astype(float)).fit()

ypred = est.predict(X)



# evaluate

rmse = np.sqrt(mean_squared_error(y, ypred))

print(rmse)



# show stats summary

est.summary()

from sklearn.ensemble import RandomForestRegressor



regr = RandomForestRegressor(max_depth=2, random_state=0)



response_col = 'price'

df_rfr_to_use = df_model_merged



X = df_rfr_to_use.drop(response_col, axis=1)

y = df_rfr_to_use[response_col]

regr.fit(X, y)

modelPred = regr.predict(X)



print("The R2 score: ",regr.score(X,y))



print("Number of predictions:",len(modelPred))



meanSquaredError=mean_squared_error(y, modelPred)

print("MSE:", meanSquaredError)

rootMeanSquaredError = np.sqrt(meanSquaredError)

print("RMSE:", rootMeanSquaredError)

#https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

# tutorial: https://github.com/h2oai/h2o-tutorials/blob/master/h2o-world-2017/automl/Python/automl_binary_classification_product_backorders.ipynb



import h2o

from h2o.automl import H2OAutoML

h2o.init()



df_h2o_to_use = df_model_merged



# Identify predictors and response

x = df.columns.tolist()

y = "price"

x = x.remove(y)



df_model_h2o = h2o.H2OFrame(df_h2o_to_use)

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)

aml = H2OAutoML(max_models=20, seed=1)

aml.train(x=x, y=y, training_frame=df_model_h2o)



# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)