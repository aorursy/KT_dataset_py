# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import datetime

import time 



%matplotlib inline

import warnings

warnings.simplefilter('ignore')
from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.datasets import make_regression

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import RidgeCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
df_train = pd.read_csv('../input/predciting-price-transaction/trainPrice.csv')

sub_sch = pd.read_csv('../input/subway-schoolcsv/subway_school.csv')

df_train = pd.merge(df_train, sub_sch)

df_train1=df_train[df_train['city']==0]

df_train2=df_train[df_train['city']==1]
# Convert

df_train['elapsed_time'] = df_train['transaction_year_month'].map(lambda x: int(x/100)) - df_train['year_of_completion']

df_train['transaction_year'] = df_train['transaction_year_month'].map(lambda x: str(int(x/100)))

df_train['heat_type'] = df_train['heat_type'].astype('category')

df_train['heat_fuel'] = df_train['heat_fuel'].astype('category')

df_train['front_door_structure'] = df_train['front_door_structure'].astype('category')

df_train['heat_type'] = pd.Categorical(df_train['heat_type']).codes

df_train['heat_fuel'] = pd.Categorical(df_train['heat_fuel']).codes

df_train['front_door_structure'] = pd.Categorical(df_train['front_door_structure']).codes

# Drop redundant

df_train = df_train.drop(['year_of_completion', 'apartment_id', 'key', 'transaction_year_month', 'transaction_date', 'address_by_law', 'room_id'], axis=1)

# Imputation

df_train = df_train.dropna(subset=['room_count','bathroom_count'] )

df_train = df_train.dropna(subset=['tallest_building_in_sites','lowest_building_in_sites'])

df_train = df_train.dropna(subset=['heat_type','heat_fuel'])

df_train['total_parking_capacity_in_site'].fillna(df_train['total_parking_capacity_in_site'].mean(),inplace=True)

plt.figure(figsize=(15,18))

sns.heatmap(df_train.corr(), annot=True)
drop_cols1 = ['apartment_building_count_in_sites','bathroom_count','city','elapsed_time']

list(set(df_train.columns) - set(drop_cols1))
# Reference: Course_STAT8017 Tutorial 2 & 5

class Solution(object):

    def convert_impute(self,df_train):

        # Convert

        df_train['elapsed_time'] = df_train['transaction_year_month'].map(lambda x: int(x/100)) - df_train['year_of_completion']

        df_train['transaction_year'] = df_train['transaction_year_month'].map(lambda x: str(int(x/100)))

        df_train['heat_type'] = df_train['heat_type'].astype('category')

        df_train['heat_fuel'] = df_train['heat_fuel'].astype('category')

        df_train['front_door_structure'] = df_train['front_door_structure'].astype('category')

        df_train['heat_type'] = pd.Categorical(df_train['heat_type']).codes

        df_train['heat_fuel'] = pd.Categorical(df_train['heat_fuel']).codes

        df_train['front_door_structure'] = pd.Categorical(df_train['front_door_structure']).codes

        # Drop redundant

        df_train = df_train.drop(['year_of_completion', 'apartment_id', 'key', 'city', 'transaction_year_month', 'transaction_date', 'address_by_law', 'room_id'], axis=1)

        # Imputation

        df_train = df_train.dropna(subset=['room_count','bathroom_count'] )

        df_train = df_train.dropna(subset=['tallest_building_in_sites','lowest_building_in_sites'])

        df_train = df_train.dropna(subset=['heat_type','heat_fuel'])

        df_train['total_parking_capacity_in_site'].fillna(df_train['total_parking_capacity_in_site'].mean(),inplace=True)

        return df_train



    def corr(self,df_train):

        plt.figure(figsize=(15,18))

        sns.heatmap(data.corr(), annot=True)

    

    def drophighcor(self,df_train,variable):

        df_train = df_train.drop(variable, axis=1)

        return df_train

    

    def pretrainx(self,df_train):

        train_X = df_train.drop(['transaction_real_price'], axis = 1)

        train_y = pd.DataFrame(df_train['transaction_real_price'], columns=['transaction_real_price'])

        scaler_x = MinMaxScaler((-1, 1))

        X = scaler_x.fit_transform(train_X)

        y = train_y['transaction_real_price'].map(lambda x: np.log(x))

        return X

        # Source: HKU Course_STAT8017 Tutorial 2

        

    def pretrainy(self,df_train):

        train_X = df_train.drop(['transaction_real_price'], axis = 1)

        train_y = pd.DataFrame(df_train['transaction_real_price'], columns=['transaction_real_price'])

        scaler_x = MinMaxScaler((-1, 1))

        X = scaler_x.fit_transform(train_X)

        y = train_y['transaction_real_price'].map(lambda x: np.log(x))

        return y

        # Source: HKU Course_STAT8017 Tutorial 2

    

    def plotalpha_l(self,X,y):

        # Create an array of alphas and lists to store scores

        alpha_space = np.logspace(-4, 0, 50)

        lasso_scores = []

        lasso_scores_std = []

        lasso = Lasso()

        # Compute scores over range of alphas

        for alpha in alpha_space:

            lasso.alpha = alpha

            lasso_cv_scores = cross_val_score(lasso, X, y, cv=5)

            lasso_scores.append(np.mean(lasso_cv_scores))

            lasso_scores_std.append(np.std(lasso_cv_scores))

        def display_plot(cv_scores, cv_scores_std):

            fig = plt.figure()

            ax = fig.add_subplot(1,1,1)

            ax.plot(alpha_space, cv_scores)

            std_error = cv_scores_std / np.sqrt(10)

            ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)

            ax.set_ylabel('CV Score +/- Std Error')

            ax.set_xlabel('Alpha of lasso')

            ax.axhline(np.max(cv_scores), linestyle='--', color='.5')

            ax.set_xlim([alpha_space[0], alpha_space[-1]])

            ax.set_xscale('log')

            plt.show()

        display_plot(lasso_scores, lasso_scores_std)

        

    def fitlasso(self,X,y,df_train,alpha,threshold):

        #X, y = make_regression()

        train_X = df_train.drop(['transaction_real_price'], axis = 1)

        lasso = Lasso(alpha)

        lasso = lasso.fit(X,y)

        pred_train_lasso = lasso.predict(X)

        mse_train_lasso = mean_squared_error(y, pred_train_lasso)

        ######################## Lasso #########################

        drop_cols1 = list(np.where(abs(lasso.coef_) <= threshold))

        drop_cols1 = df_train.columns[[drop_cols1]]

        print('Parameters of lasso:', lasso.coef_)

        print('Coefficient of determination R^2 of the prediction: ',lasso.score(X, y))

        print('Variables can be removed:', drop_cols1[0])

        print('Variables should be reserved:', list(set(df_train.columns) - set(drop_cols1[0])))

        plt.figure(figsize=(12,8))

        plt.plot(lasso.coef_, label='LASSO')

        plt.xticks(range(len(train_X.columns)),train_X.columns, rotation=80) 

        plt.margins(0.01)

        plt.show()

        return drop_cols1[0]

          

    def plotalpha_r(self,df_train,y):

        train_X = df_train.drop(['transaction_real_price'], axis = 1)

        alpha_space = np.logspace(-4, 0, 50)

        ridge_scores = []

        ridge_scores_std = []

        ridge = Ridge(normalize=True)

        for alpha in alpha_space:

            ridge.alpha = alpha

            ridge_cv_scores = cross_val_score(ridge, train_X, y, cv=5)

            ridge_scores.append(np.mean(ridge_cv_scores))

            ridge_scores_std.append(np.std(ridge_cv_scores))

        def display_plot(cv_scores, cv_scores_std):

            fig = plt.figure()

            ax = fig.add_subplot(1,1,1)

            ax.plot(alpha_space, cv_scores)

            std_error = cv_scores_std / np.sqrt(10)

            ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)

            ax.set_ylabel('CV Score +/- Std Error')

            ax.set_xlabel('Alpha of ridge')

            ax.axhline(np.max(cv_scores), linestyle='--', color='.5')

            ax.set_xlim([alpha_space[0], alpha_space[-1]])

            ax.set_xscale('log')

            plt.show()

        # Display the plot

        display_plot(ridge_scores, ridge_scores_std)

        

    def fitridge(self,X,y,df_train,alpha,threshold):

        #X, y = make_regression()

        train_X = df_train.drop(['transaction_real_price'], axis = 1)

        ridge = Ridge(alpha,normalize=True).fit(train_X, y)

        ######################## Ridge #########################

        drop_cols2 = list(np.where(abs(ridge.coef_) <=threshold ))

        drop_cols2 = df_train.columns[[drop_cols2]]

        print('Parameters of ridge:', ridge.coef_)

        print('Coefficient of determination R^2 of the prediction: ',ridge.score(train_X, y))

        print('From Ridege, variables can be removed:', drop_cols2[0])

        print('Variables should be reserved:', list(set(df_train.columns) - set(drop_cols2[0])))

        return drop_cols2[0]

    

    def fit_random_forest(self,X,y,df_train):

        train_X = df_train.drop(['transaction_real_price'], axis = 1)

        rfr = RandomForestRegressor()

        rfr.fit(train_X, y)

        pred_train_rfr = rfr.predict(train_X)

        Importance = pd.DataFrame({'Features':train_X.columns, 'Importance':rfr.feature_importances_*100})

        ######################## Random Forest #########################

        drop_cols3 = list(Importance.sort_values(by='Importance', axis=0, ascending=False).iloc[-5:,0])

        #drop_cols3 = df_train.columns[[drop_cols3]] 

        Importance = pd.DataFrame({'Importance':rfr.feature_importances_*100}, index=train_X.columns)

        Importance.sort_values(by='Importance', axis=0, ascending=True).plot(kind='barh', color='b', )

        plt.xlabel('Variable Importance')

        plt.gca().legend_ = None

        print('From Random Forest, variables can be removed:', drop_cols3)

        print('Variables should be reserved:', list(set(df_train.columns) - set(drop_cols3)))

        return drop_cols3

    

    def finaldrop(self,df_train,drop_cols2):

        train_X = df_train.drop(['transaction_real_price'], axis = 1)

        train_X = train_X.drop(drop_cols2, axis=1)

        return train_X

    

    def training(self,train_X,y):

        scaler_x = MinMaxScaler((-1, 1))

        X = scaler_x.fit_transform(train_X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        ind_var = list(train_X.columns)

        print('%d Features are selected for training:'%len(ind_var), ind_var)

        # Lasso

        lasso = LassoCV(cv=5)

        lasso.fit(X_train, y_train)

        pred_train_lasso = lasso.predict(X_train)

        pred_test_lasso = lasso.predict(X_test)

        # Ridge

        ridge = RidgeCV(cv=5)

        ridge.fit(X_train, y_train)

        pred_train_ridge = ridge.predict(X_train)

        pred_test_ridge = ridge.predict(X_test)

        # Decision tree

        base = DecisionTreeRegressor()

        grid = GridSearchCV(base, param_grid={'max_depth': (2, 4, 6, 8, 10), 'min_samples_leaf': [5, 10, 15, 20]}, cv=10)

        grid.fit(X_train, y_train)                # 5^4 possible tree

        dtr = grid.best_estimator_

        pred_train_dtr = dtr.predict(X_train)

        pred_test_dtr = dtr.predict(X_test)

        # Random forest

        rfr = RandomForestRegressor(max_features=10)

        rfr.fit(X_train, y_train)

        pred_train_rfr = rfr.predict(X_train)

        pred_test_rfr = rfr.predict(X_test)

        plt.figure(figsize=(12,8))

        plt.plot(rfr.feature_importances_)

        plt.xticks(range(len(train_X.columns)),train_X.columns, rotation=90) 

        plt.show()



        mse_train_lasso = mean_squared_error(y_train, pred_train_lasso)

        mse_test_lasso = mean_squared_error(y_test, pred_test_lasso)



        mse_train_ridge = mean_squared_error(y_train, pred_train_ridge)

        mse_test_ridge = mean_squared_error(y_test, pred_test_ridge)



        mse_train_dtr = mean_squared_error(y_train, pred_train_dtr)

        mse_test_dtr = mean_squared_error(y_test, pred_test_dtr)



        mse_train_rfr = mean_squared_error(y_train, pred_train_rfr)

        mse_test_rfr = mean_squared_error(y_test, pred_test_rfr)

        

        print("LASSO: Training MSE:", round(mse_train_lasso, 5), "Testing MSE:", round(mse_test_lasso, 5))

        print("RIDGE: Training MSE:", round(mse_train_ridge, 5), "Testing MSE:", round(mse_test_ridge, 5))

        print("DTR: Training MSE:", round(mse_train_dtr, 5), "Testing MSE:", round(mse_test_dtr, 5))

        print("RFR: Training MSE:", round(mse_train_rfr, 5), "Testing MSE:", round(mse_test_rfr, 5)) 

        return rfr

    

    def testset(self,df_test):

        df_test['elapsed_time'] = df_test['transaction_year_month'].map(lambda x: int(x/100)) - df_test['year_of_completion']

        df_test['transaction_year'] = df_test['transaction_year_month'].map(lambda x: str(int(x/100)))

        df_test['heat_type'] = df_test['heat_type'].astype('category')

        df_test['heat_fuel'] = df_test['heat_fuel'].astype('category')

        df_test['front_door_structure'] = df_test['front_door_structure'].astype('category')

        df_test['heat_type'] = pd.Categorical(df_test['heat_type']).codes

        df_test['heat_fuel'] = pd.Categorical(df_test['heat_fuel']).codes

        df_test['front_door_structure'] = pd.Categorical(df_test['front_door_structure']).codes

        df_test['total_parking_capacity_in_site'].fillna(df_test['total_parking_capacity_in_site'].mean(),inplace=True)

        df_test['room_count'].fillna(df_test['room_count'].mean(),inplace=True)

        df_test['bathroom_count'].fillna(df_test['bathroom_count'].mean(),inplace=True)

        df_test['tallest_building_in_sites'].fillna(df_test['tallest_building_in_sites'].mean(),inplace=True)

        df_test['lowest_building_in_sites'].fillna(df_test['lowest_building_in_sites'].mean(),inplace=True)

        return df_test

    

    def prediction(self,train_X,df_test,rfr):

        ind_var = list(train_X.columns)

        df_test['room_count'].fillna(df_test['room_count'].mean(),inplace=True)

        df_test['bathroom_count'].fillna(df_test['bathroom_count'].mean(),inplace=True)

        df_test['tallest_building_in_sites'].fillna(df_test['tallest_building_in_sites'].mean(),inplace=True)

        df_test['lowest_building_in_sites'].fillna(df_test['lowest_building_in_sites'].mean(),inplace=True)

        X_test = df_test[ind_var]

        scaler_x = MinMaxScaler((-1, 1))

        X_test = scaler_x.fit_transform(X_test)

        pred_test_rfr = rfr.predict(X_test)

        d = {'key': df_test['key'], 'transaction_real_price': np.exp(pred_test_rfr)}

        submit_prediction = pd.DataFrame(data=d)

        submit_prediction.to_csv('Submission_Price.csv', index=False)

        return submit_prediction
# Original data convert and impute

data=Solution().convert_impute(df_train1)

Solution().corr(data)

# drop high correlation variables

data1=Solution().drophighcor(data,['total_household_count_in_sites','exclusive_use_area'])

# Split X&y and scale X

X=Solution().pretrainx(data1)

y=Solution().pretrainy(data1)
print('------------------------------------ Plot alpha for lasso ------------------------------------')

Solution().plotalpha_l(X,y)

print('-------------------------------------- Lasso Regression --------------------------------------')

Solution().fitlasso(X,y,data1,0.005,0.00001)
print('------------------------------------ Plot alpha for Ridge ------------------------------------')

Solution().plotalpha_r(data1,y)

print('-------------------------------------- Ridge Regression --------------------------------------')

Solution().fitridge(X,y,data1,0.05,0.001)
print('---------------------------------------- Random Forest ---------------------------------------')

drop = Solution().fit_random_forest(X,y,data1)
# Drop variables

data2=Solution().finaldrop(data1,drop)

data2.head()

print('------------------------------------ Comparison of Models ------------------------------------')

regressor1=Solution().training(data2,y)
print('---------------------------------- Prediction for Test Set -----------------------------------')

df_test=pd.read_csv('../input/predciting-price-transaction/testPrice.csv')

df_test = pd.merge(df_test, sub_sch)

df_test1=df_test[df_test['city']==0]

df_test1=Solution().testset(df_test1)

df_test1.head()

Solution().prediction(data2,df_test1,regressor1).head()
# Original data convert and impute

data=Solution().convert_impute(df_train2)

print('------------------------------------- Correlation Map -------------------------------------')

Solution().corr(data)

# drop high correlation variables

data1=Solution().drophighcor(data,['total_household_count_in_sites','exclusive_use_area'])

# Split X&y and scale X

X=Solution().pretrainx(data1)

y=Solution().pretrainy(data1)
print('------------------------------------ Plot alpha for lasso ------------------------------------')

Solution().plotalpha_l(X,y)

print('-------------------------------------- Lasso Regression --------------------------------------')

Solution().fitlasso(X,y,data1,0.005,0.00001)
print('------------------------------------ Plot alpha for ridge ------------------------------------')

Solution().plotalpha_r(data1,y)

print('-------------------------------------- Ridge Regression --------------------------------------')

Solution().fitridge(X,y,data1,0.05,0.001)
print('---------------------------------------- Random Forest ---------------------------------------')

drop = Solution().fit_random_forest(X,y,data1)
data2=Solution().finaldrop(data1,drop)

data2.head()

print('------------------------------------ Comparison of Models ------------------------------------')

regressor2=Solution().training(data2,y)
print('---------------------------------- Prediction for Test Set ------- ---------------------------')

df_test=pd.read_csv('../input/predciting-price-transaction/testPrice.csv')

#df_test=pd.read_csv('C:/Users/jjl/Desktop/testPrice.csv')

df_test = pd.merge(df_test, sub_sch)

df_test2=df_test[df_test['city']==1]

df_test2=Solution().testset(df_test2)

df_test2.head()

Solution().prediction(data2,df_test2,regressor2).head()
from sklearn.cluster import AgglomerativeClustering, KMeans

import scipy.cluster.hierarchy as sch

np.set_printoptions(suppress=True, precision = 3)
def process_for_trend(df_train):

    f = lambda x,y : x/y

    df_train['avg_price'] = df_train[['transaction_real_price','exclusive_use_area']].apply(lambda x: f(*x), axis=1)

    df_train['elapsed_time'] = df_train['transaction_year_month'].map(lambda x: int(x/100)) - df_train['year_of_completion']

    df_train['transaction_year'] = df_train['transaction_year_month'].map(lambda x: str(int(x/100)))

    first_hand = df_train[df_train['elapsed_time'] <= 1]

    first_hand = first_hand.dropna(subset=['room_count','bathroom_count','heat_type','heat_fuel','heat_type','front_door_structure','total_parking_capacity_in_site'] )

    first_hand = first_hand.drop_duplicates(subset=['apartment_id','room_id','floor'],keep='first')

    first_hand['heat_type'] = first_hand['heat_type'].astype('category')

    first_hand['heat_fuel'] = first_hand['heat_fuel'].astype('category')

    first_hand['front_door_structure'] = first_hand['front_door_structure'].astype('category')

    first_hand['heat_type'] = pd.Categorical(first_hand['heat_type']).codes

    first_hand['heat_fuel'] = pd.Categorical(first_hand['heat_fuel']).codes

    first_hand['front_door_structure'] = pd.Categorical(first_hand['front_door_structure']).codes

    return first_hand
def trend(first_hand):

    df_trend = first_hand[['exclusive_use_area','floor', 'total_parking_capacity_in_site', 'room_count','bathroom_count','sub_count','sch_count', 'tallest_building_in_sites','lowest_building_in_sites','avg_price']].groupby(first_hand['transaction_year'])

    annual_trend = df_trend.mean()

    plt.figure(figsize=(12,8))

    #plot the first 5 variables

    for col in annual_trend.columns[:5]:

        #plt.plot(annual_trend[col], label=str(col))

        plt.plot((annual_trend[col]-annual_trend[col].mean())/(annual_trend[col].max()-annual_trend[col].min()), label=str(col))

    plt.legend(loc='upper right')

    plt.ylabel("Scaled Mean")

    plt.show()

    #plot the last 5variables

    plt.figure(figsize=(10,7))

    for col in annual_trend.columns[5:10]:

        plt.plot((annual_trend[col]-annual_trend[col].mean())/(annual_trend[col].max()-annual_trend[col].min()), label=str(col))

    plt.legend(loc='upper right')

    plt.ylabel("Scaled Mean")

    plt.show()
def cluster(first_hand):

    X = first_hand[['total_parking_capacity_in_site', 'floor', 'supply_area','sub_count','sch_count', 'tallest_building_in_sites','lowest_building_in_sites']]

    scaler_x = MinMaxScaler((-1, 1))

    sX = scaler_x.fit_transform(X)

    kmean1 = KMeans(n_clusters=4, init='k-means++', random_state=42)

    kmean1.fit(sX)

    plt.figure(figsize=(10, 8))

    plt.plot(kmean1.cluster_centers_[0], label='Cluster 1')

    plt.plot(kmean1.cluster_centers_[1], label='Cluster 2')

    plt.plot(kmean1.cluster_centers_[2], label='Cluster 3')

    plt.plot(kmean1.cluster_centers_[3], label='Cluster 4')

    plt.legend(loc='upper right')

    plt.ylabel("Mean of Variables")

    plt.xticks(range(len(X.columns)),X.columns, rotation=80) 

    plt.show()
first_hand1 = process_for_trend(df_train1)

first_hand2 = process_for_trend(df_train2)
print("-------------------------------- First-hand House Trend  in Busan ---------------------------------")

trend(first_hand1)
print("-------------------------------- First-hand House Trend in Busan ---------------------------------")

trend(first_hand2)
print("-------------------------------- Cluster Analysis for First-hand House in Busan ---------------------------------")

cluster(first_hand1)
print("-------------------------------- Cluster Analysis for First-hand House in Seoul ---------------------------------")

cluster(first_hand2)