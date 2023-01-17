#Importing the necessary packages

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import tree

from collections import Counter

from sklearn.svm import SVR

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



train = pd.read_csv("../input/train_cab.csv")

test      = pd.read_csv("../input/test.csv")
# Loading the the train and test datasets associated with the Cab Fare: 

train.head(5)
test.head(5)
#description of our datatests

train.describe()
train['fare_amount'].describe()
# data types of our taining data

train.dtypes
#Converting the fare amount column into numeric data form

train["fare_amount"] = pd.to_numeric(train["fare_amount"],errors='coerce')
## The fare amount column is having some neagative values, Lets Check it

Counter(train['fare_amount']<0)
#Also there are values like more 6 persons in a cab, Lets cross check

Counter(train['passenger_count']>6)
############################ Missing Value Analysis #############################################################



#Create dataframe with missing percentage toc check missing values in our both dataset

def missin_val(df):

    missin_val = pd.DataFrame(df.isnull().sum())

    missin_val = missin_val.reset_index()

    missin_val = missin_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

    missin_val['Missing_percentage'] = (missin_val['Missing_percentage']/len(df))*100

    missin_val = missin_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)

    return(missin_val)



print("The missing value percentage in training data : \n\n",missin_val(train))

print("\n")

print("The missing value percentage in test data : \n\n",missin_val(test))

print("\n")



#Impute the missing values

train["passenger_count"] = train["passenger_count"].fillna(train["passenger_count"].median())

train["fare_amount"] = train["fare_amount"].fillna(train["fare_amount"].median())



#check if any missing  value still exists

print("Is there still any missing value in the training data:\n\n",train.isnull().sum())

print("\n")
########################## Proper Aligning the Dataset ############################################################



#Split our Datetime into individual columns for ease of data processing and modelling

def align_datetime(df):

    df["pickup_datetime"] = df["pickup_datetime"].map(lambda x: str(x)[:-3])

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], format='%Y-%m-%d %H:%M:%S')

    df['year'] = df.pickup_datetime.dt.year

    df['month'] = df.pickup_datetime.dt.month

    df['day'] = df.pickup_datetime.dt.day

    df['weekday'] = df.pickup_datetime.dt.weekday

    df['hour'] = df.pickup_datetime.dt.hour

    return(df["pickup_datetime"].head())

    

align_datetime(train)

align_datetime(test)



print(train.head(5))

print("\n")

print(test.head(5))
#Remove the datetime column

train.drop('pickup_datetime', axis=1, inplace=True)

test.drop('pickup_datetime', axis=1, inplace=True)

#Checking NA in the fresh Dataset

train.isnull().sum()

train=train.fillna(train.mean())

train.isnull().sum()
#Originally, Latitudes range from -90 to 90.

#Originally, Longitudes range from -180 to 180.

#But our data is purely negative Longitudes and purely positive latitudes

#lets align our data in its respective minimum and maximum Longitudes 

#and latitudes values, also removing fare amount those are negative and over valued.



def proper_data(df):

    df = df[((df['pickup_longitude'] > -78) & (df['pickup_longitude'] < -70)) & 

           ((df['dropoff_longitude'] > -78) & (df['dropoff_longitude'] < -70)) & 

           ((df['pickup_latitude'] > 37) & (df['pickup_latitude'] < 45)) & 

           ((df['dropoff_latitude'] > 37) & (df['dropoff_latitude'] < 45)) & 

           ((df['passenger_count'] > 0) & (df['passenger_count'] < 7)) &

           ((df['fare_amount'] >= 2.5) & (df['fare_amount'] < 500))]

    

    return(df)



train = proper_data(train)
test = test[((test['pickup_longitude'] > -78) & (test['pickup_longitude'] < -70)) & 

           ((test['dropoff_longitude'] > -78) & (test['dropoff_longitude'] < -70)) & 

           ((test['pickup_latitude'] > 37) & (test['pickup_latitude'] < 45)) & 

           ((test['dropoff_latitude'] > 37) & (test['dropoff_latitude'] < 45)) & 

           (test['passenger_count'] > 0) ]

#Setting proper data type for each columns

train= train.astype({"fare_amount":float,"pickup_longitude":float,"pickup_latitude":float,"dropoff_longitude":float,"dropoff_latitude":float,"passenger_count":int,"year":int,"month":int ,"day" :int,"weekday":int,"hour":int})

train.dtypes

test = test.astype({"pickup_longitude":float,"pickup_latitude":float,"dropoff_longitude":float,"dropoff_latitude":float,"passenger_count":int,"year":int,"month":int ,"day" :int,"weekday":int,"hour":int})

test.dtypes
######################################### Data Exploration #########################################################################################################



#Histogram Plot of passenger_count Column

plt.figure(figsize=(7,7))

plt.hist(train['passenger_count'],bins=6)

plt.xlabel('No. of Passengers')

plt.ylabel('Frequency')

#Histogram Plot of passenger_count Column

plt.figure(figsize=(7,7))

plt.hist(train['fare_amount'],bins=25)

plt.xlabel('Different amount of Fare')

plt.ylabel('Frequency')
#Histogram Plot of day Column

plt.figure(figsize=(7,7))

plt.hist(train['day'],bins=10)

plt.xlabel('Different Days of the month')

plt.ylabel('Frequency')



#Histogram Plot of weekday Column

plt.figure(figsize=(7,7))

plt.hist(train['weekday'],bins=10)

plt.xlabel('Different Days of the week')

plt.ylabel('Frequency')
#Histogram Plot of hour Column

plt.figure(figsize=(7,7))

plt.hist(train['hour'],bins=10)

plt.xlabel('Different hours of the Day')

plt.ylabel('Frequency')

#Histogram Plot of month Column

plt.figure(figsize=(7,7))

plt.hist(train['month'],bins=10)

plt.xlabel('Different Months of the year')

plt.ylabel('Frequency')



#Histogram Plot of year Column

plt.figure(figsize=(7,7))

plt.hist(train['year'],bins=10)

plt.xlabel('Years')

plt.ylabel('Frequency')
################################################## Bivariate Plots #################################################################################################



cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.scatterplot(x="passenger_count", y="fare_amount", data= train, palette="Set2")

sns.scatterplot(x="month", y="fare_amount", data= train, palette="Set2")

sns.scatterplot(x="weekday", y="fare_amount", data= train, palette="Set2")

sns.scatterplot(x="hour", y="fare_amount", data= train, palette="Set2")
##################################################  Outlier Analysis ###############################################################################################

train.plot(kind='box', subplots=True, layout=(8,3), sharex=False, sharey=False, fontsize=8)

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top= 3,wspace=0.2, hspace=0.2)

plt.show()
##Detect and delete outliers from data

def outliers_analysis(df): 

    for i in df.columns:

        print(i)

        q75, q25 = np.percentile(df.loc[:,i], [75 ,25])

        iqr = q75 - q25



        min = q25 - (iqr*1.5)

        max = q75 + (iqr*1.5)

        print(min)

        print(max)

    

        df = df.drop(df[df.loc[:,i] < min].index)

        df = df.drop(df[df.loc[:,i] > max].index)

        return(df)



train = outliers_analysis(train)

test = outliers_analysis(test)



def eliminate_rows_with_zero_value(df):

    df= df[df!= 0]

    df=df.fillna(df.mean())

    return(df)

    

train = eliminate_rows_with_zero_value(train)

test = eliminate_rows_with_zero_value(test)

## Splitting DataSets######

X_train = train.loc[:,train.columns != 'fare_amount']

y_train = train['fare_amount']
############################ Feature Scaling ##############################

# #Normalisation

def Normalisation(df):

    for i in df.columns:

        df[i] = (df[i] - df[i].min())/(df[i].max() - df[i].min())

        

Normalisation(X_train)

Normalisation(test)
########################### Feature Selection ############################# 

##Correlation analysis

#Correlation plot

def Correlation(df):

    df_corr = df.loc[:,df.columns]

    sns.set()

    plt.figure(figsize=(9, 9))

    corr = df_corr.corr()

    sns.heatmap(corr, annot= True,fmt = " .3f", linewidths = 0.5,

            square=True)

    

Correlation(X_train)

Correlation(test)

######### Dimension Reduction ########

pca = PCA(n_components=10)

pca.fit(X_train)

var= pca.explained_variance_ratio_

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.plot(var1)

plt.show()

#lets reduce our no. of variables to 5 as it explains 100% features of our Data

pca = PCA(n_components=5)

X = pca.fit(X_train).transform(X_train)

test = pca.fit(test).transform(test)
###### Sampling the splits through stratified way ###########

X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.2)

np.savetxt("Training data preprocessed.csv" , X_train, delimiter=",")
###################################### MACHINE LEARNING MODELLING ###############################

  



###### KNN Modelling ########

def train_KNN(n_neigh):

    knn = KNeighborsRegressor(n_neighbors= n_neigh)

    knn_model = knn.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    print('n_neighbours : {}  ----KNN rmse: {}'.format(n_neigh,(sqrt(mean_squared_error(y_test,y_pred)))))



for n_neigh in [10,20,30,40,50,60,70,80,90,100]:

    train_KNN(n_neigh)
KNN_model = KNeighborsRegressor(n_neighbors= 100).fit(X_train , y_train)

KNN_pred_train = KNN_model.predict(X_train)

KNN_pred= KNN_model.predict(X_test)

KNN_pred_test = KNN_model.predict(test)

print("Train Data")

print('n_neighbours : {}  ----KNN rmse: {}'.format(n_neigh,  (sqrt(mean_squared_error(y_train,KNN_pred_train)))))

print("Test Data")

print('n_neighbours : {}  ----KNN rmse: {}'.format(n_neigh,  (sqrt(mean_squared_error(y_test,KNN_pred)))))

KNN_model.score(X_train, y_train)
####### Linear Regression ######

ols = LinearRegression()

ols_model = ols.fit(X_train, y_train)

y_pred_train = ols_model.predict(X_train)

y_pred = ols_model.predict(X_test)

y_pred_test = ols_model.predict(test)

print("Train Data")

print('Ordinary Least Squares rmse: {}'.format(sqrt(mean_squared_error(y_train,y_pred_train))))

print("Test Data")

print('Ordinary Least Squares rmse: {}'.format(sqrt(mean_squared_error(y_test,y_pred))))

ols_model.score(X_train, y_train)
#######Ridge Regression ######

def train_ridge(alpha):

    ridge = Ridge(alpha= alpha)

    ridge_model = ridge.fit(X_train, y_train)

    y_pred = ridge_model.predict(X_test)

    print('alpha : {}  ----Ridge rmse: {}'.format(alpha, (sqrt(mean_squared_error(y_test,y_pred)))))



for alpha in [0.1,0.5, 1.0,3.0,7.0,10.0]:

    train_ridge(alpha)
ridge_model = Ridge(alpha= 10).fit(X_train , y_train)

ridge_pred_train = ridge_model.predict(X_train)

ridge_pred= ridge_model.predict(X_test)

print("Train Data")

print('Ridge rmse: {}'.format(sqrt(mean_squared_error(y_train,ridge_pred_train))))

print("Test Data")

print('Ridge rmse: {}'.format(sqrt(mean_squared_error(y_test,ridge_pred))))

ridge_model.score(X_train, y_train)
####### Lasso Regression ######

def train_lasso(alpha):

    lasso = Lasso(alpha= alpha)

    lasso_model = lasso.fit(X_train, y_train)

    y_pred = lasso_model.predict(X_test)

    print('alpha : {}  ---- Lasso rmse: {}'.format(alpha,(sqrt(mean_squared_error(y_test,y_pred)))))



for alpha in [0.1,0.5, 1.0,3.0,7.0,10.0]:

    train_lasso(alpha)
lasso_model = Lasso(alpha= 10).fit(X_train , y_train)

lasso_pred_train = lasso_model.predict(X_train)

lasso_pred= lasso_model.predict(X_test)

print("Train Data")

print('Lasso rmse: {}'.format(sqrt(mean_squared_error(y_train,lasso_pred_train))))

print("Test Data")

print('Lasso rmse: {}'.format(sqrt(mean_squared_error(y_test,lasso_pred))))

lasso_model.score(X_train, y_train)

###### SVM Modelling ##########

def train_SVR(C, gamma):

    svr = SVR(C= C, gamma = gamma)

    svr_model = svr.fit(X_train, y_train)

    y_pred = svr_model.predict(X_test)

    print('C : {} , gamma : {} ----SVR rmse: {}'.format(C, gamma ,(sqrt(mean_squared_error(y_test,y_pred)))))

    

for C in [1, 10, 100,1000]:

    for gamma in [0.001, 0.0001]:

        train_SVR(C, gamma)

        
svr_model = SVR(C= 1000 , gamma = 0.0001).fit(X_train , y_train)

svr_pred_train = svr_model.predict(X_train)

svr_pred= svr_model.predict(X_test)

print("Train Data")

print('Support Vector Regression rmse: {}'.format(sqrt(mean_squared_error(y_train,svr_pred_train))))

print("Test Data")

print('Support Vector Regression rmse: {}'.format(sqrt(mean_squared_error(y_test,svr_pred))))

svr_model.score(X_train, y_train)

###### DecisionTree Modelling ##########

def DT(depth):

    dt = tree.DecisionTreeRegressor( max_depth = depth)

    dt_model = dt.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)

    print('depth : {} ----  Decision Tree rmse: {}'.format(depth,(sqrt(mean_squared_error(y_test,y_pred)))))

    

for depth in [1,2,5,10,20]:

    DT(depth)
dt_model = tree.DecisionTreeRegressor(max_depth =2).fit(X_train, y_train)

dt_pred_train = dt_model.predict(X_train)

dt_pred= dt_model.predict(X_test)

print("Train Data")

print('Decision Tree rmse: {}'.format(sqrt(mean_squared_error(y_train,dt_pred_train))))

print("Test Data")

print('Decision Tree rmse: {}'.format(sqrt(mean_squared_error(y_test,dt_pred))))

dt_model.score(X_train, y_train)
###### GBR Modelling ##########

def GBR(depth, learning_rate):

    gbr = GradientBoostingRegressor( max_depth = depth, learning_rate =learning_rate)

    gbr_model = gbr.fit(X_train, y_train)

    y_pred = gbr_model.predict(X_test)

    print('depth : {}, learning_rate{}  ---- Gradient Boosting Regression rmse: {}'.format(depth, learning_rate, (sqrt(mean_squared_error(y_test,y_pred)))))  

    

for depth in [1,2,5]:

    for learning_rate in [0.001,0.01,0.1]:

        GBR(depth, learning_rate)

        
gbr_model = GradientBoostingRegressor(max_depth= 1,learning_rate = 0.1).fit(X_train , y_train)

gbr_pred_train = gbr_model.predict(X_train)

gbr_pred= gbr_model.predict(X_test)

print("Train Data")

print('GBDT rmse: {}'.format(sqrt(mean_squared_error(y_train,dt_pred_train))))

print("Test Data")

print('GBDT rmse: {}'.format(sqrt(mean_squared_error(y_test,dt_pred))))

gbr_model.score(X_train, y_train)
###### RandomForest Modelling ##########

def train_RF(n_est, depth):

    rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth, n_jobs=-1)

    rf_model = rf.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    print('depth : {}, n_estimators : {}  ---- Random Forest Regression rmse: {}'.format(depth, n_est, (sqrt(mean_squared_error(y_test,y_pred)))))  



for n_est in [100, 200]:

    for depth in [2, 5, 10 , 20, 30]:

        train_RF(n_est, depth)
rf_model = RandomForestRegressor(max_depth= 5, n_estimators = 100).fit(X_train , y_train)

rf_pred_train = rf_model.predict(X_train)

rf_pred= rf_model.predict(X_test)

print("Train Data")

print('Random Forest rmse: {}'.format(sqrt(mean_squared_error(y_train,rf_pred_train))))

print("Test Data")

print('Random Forest rmse: {}'.format(sqrt(mean_squared_error(y_test,rf_pred))))

rf_model.score(X_train, y_train)

df = pd.DataFrame({"rmse":[4.08,4.09,4.06,4.10,4.20,4.08,4.08,4.06],\

                   "Model" : ['KNN Regression' ,'Ordinary Least Square','Ridge Regression',\

                             'Lasso Regression' , 'Support Vector Regression','Decision Trees', "GBDT", "Random Forest"]})

print(df)
## Ridge Regression and Random Forest gives best result

ridge_pred_test= ridge_model.predict(test)
#Convert our results to submission and then to a csv file

submission= pd.DataFrame()

submission["fare_amount"] = ridge_pred_test
submission.to_csv("submission.csv")