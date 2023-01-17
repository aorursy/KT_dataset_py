# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



movies=pd.read_csv('/kaggle/input/all-us-movies-imdb-from-1972-to-2016/movies_imdb_us_1972_2016_cs.csv')
#checking the data.

movies.info()
movies.isnull().sum()
movies.columns.values
movies.dropna(subset=['genre','votes','cast'],inplace=True)
movies.duplicated(['title','year','director']).sum()

dup=movies[movies.duplicated(['title','year','director'])]

movies.drop_duplicates(subset=['title','year','director'],inplace=True,keep='first')
u=movies[movies['rating'].isna() & movies['gross'].isna()]

movies.drop(u.index, axis=0,inplace=True)

yr=movies[movies['year']>2019]

movies.drop(yr.index, axis=0,inplace=True)
#numerical or categorical

numerical_cols = [col for col in movies.columns if movies[col].dtype != 'object']

categorical_cols = [col for col in movies.columns if movies[col].dtype == 'object']

numerical_cols
categorical_cols
eda_num=movies[numerical_cols].describe()

eda_cat=movies[categorical_cols].describe()
eda_num
eda_cat
#No of unique director?

movies.director.nunique()
#genre frequency?

genre_freq=pd.DataFrame(movies.genre.value_counts())



genre_freq.reset_index(inplace=True)

genre_freq.columns=['genre','freq']

genre_freq.head(10)

movies.reset_index(inplace = True, drop = True)



#lets seperate the movies with no ratings. We will try to predict their ratings later after fitting a model.

movies.isnull().sum()

without_ratings=movies[movies['rating'].isna()]

movies.drop(without_ratings.index, axis=0,inplace=True)
#profiling

import pandas_profiling

movies.profile_report()
#lets count number of movies done by a director.

#Data Analysis



director_name_value_counts = movies.director.value_counts()

director_name_value_counts  = pd.DataFrame(director_name_value_counts).reset_index().rename(columns = {'index': 'Director', 'director':'director_name_value_counts'})



movies=pd.merge(movies,director_name_value_counts,left_on='director',right_on='Director',how='left')

movies = movies.drop(columns = 'director')
movies['main_genre'] = movies.genre.str.split('|').str[0]



#no of unique main genre

movies.main_genre.nunique()
#converting main genre column to numerical

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

movies['main_genre'] = le.fit_transform(movies.main_genre)

#genre count

genre_value_counts = movies.genre.value_counts()

genre_value_counts  = pd.DataFrame(genre_value_counts).reset_index().rename(columns = {'index': 'genre', 'genre':'genre_value_counts'})



movies=pd.merge(movies,genre_value_counts,left_on='genre',right_on='genre',how='left')



movies['actor_1'] = movies.cast.str.split('|').str[0]

movies['actor_2'] = movies.cast.str.split('|').str[1]

movies['actor_3'] = movies.cast.str.split('|').str[2]

movies['actor_4'] = movies.cast.str.split('|').str[3]

movies['Less_than_five_actors']=movies.cast.str.split('|').str[4].isna()
#convering to numerical

movies['Less_than_five_actors'] = le.fit_transform(movies.Less_than_five_actors)





#actors value counts

actor1_name_value_counts = movies.actor_1.value_counts()

actor1_name_value_counts  = pd.DataFrame(actor1_name_value_counts).reset_index().rename(columns = {'index': 'actor', 'actor_1':'actor1_name_value_counts'})



movies=pd.merge(movies,actor1_name_value_counts,left_on='actor_1',right_on='actor',how='left')

movies = movies.drop(columns = ['actor_1','actor'])



actor2_name_value_counts = movies.actor_2.value_counts()

actor2_name_value_counts  = pd.DataFrame(actor2_name_value_counts).reset_index().rename(columns = {'index': 'actor', 'actor_2':'actor2_name_value_counts'})



movies=pd.merge(movies,actor2_name_value_counts,left_on='actor_2',right_on='actor',how='left')

movies = movies.drop(columns = ['actor_2','actor'])



actor3_name_value_counts = movies.actor_3.value_counts()

actor3_name_value_counts  = pd.DataFrame(actor3_name_value_counts).reset_index().rename(columns = {'index': 'actor', 'actor_3':'actor3_name_value_counts'})



movies=pd.merge(movies,actor3_name_value_counts,left_on='actor_3',right_on='actor',how='left')

movies = movies.drop(columns = ['actor_3','actor'])



actor4_name_value_counts = movies.actor_4.value_counts()

actor4_name_value_counts  = pd.DataFrame(actor4_name_value_counts).reset_index().rename(columns = {'index': 'actor', 'actor_4':'actor4_name_value_counts'})



movies=pd.merge(movies,actor4_name_value_counts,left_on='actor_4',right_on='actor',how='left')

movies = movies.drop(columns = ['actor_4','actor'])

movies['certificate'].fillna(movies['certificate'].mode()[0],inplace=True)



movies['certificate'] = le.fit_transform(movies.certificate)
movies = movies.drop(columns = ['genre','Director','cast','title','gross'])

movies['runtime'].fillna(movies['runtime'].mean(),inplace=True)

movies['actor2_name_value_counts'].fillna(0,inplace=True)

movies['actor3_name_value_counts'].fillna(0,inplace=True)

movies['actor4_name_value_counts'].fillna(0,inplace=True)
movies.profile_report()
#modelling



moviesR = movies.copy() #lets keep our original movies for reference. Here moviesR is for Regression model

moviesC = movies.copy() #Here moviesC is for classification model

from sklearn.model_selection import train_test_split

y = moviesR.pop('rating')

X = moviesR

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 42)



X_train.shape, y_train.shape, X_test.shape, y_test.shape



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)



X_test = pd.DataFrame(scaler.transform(X_test.values), columns = X_train.columns, index = X_test.index)

#removing variables with high colinearity

def correlation(movies, threshold):

    col_corr = set() # Set of all the names of deleted columns

    corr_matrix = movies.corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):

                colname = corr_matrix.columns[i] # getting the name of column

                col_corr.add(colname)

                if colname in movies.columns:

                    del movies[colname] # deleting the column from the movies

correlation(X_train,0.90)
#importing the required libraries

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression





# Running RFE with the output number of the variable equal to 15

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 15)            # running RFE

rfe = rfe.fit(X_train, y_train)



list(zip(X_train.columns,rfe.support_,rfe.ranking_))

col_rfe = X_train.columns[rfe.support_]

col_rfe

X_train.columns[~rfe.support_]



#Creating a X_train dataframe with rfe varianles

X_train_rfe = X_train[col_rfe]

#Simple Linear Regression

# Adding a constant variable for using the stats model

import statsmodels.api as sm

X_train_rfe_constant = sm.add_constant(X_train_rfe)

lm = sm.OLS(y_train,X_train_rfe_constant).fit()   # Running the linear model



#Let's see the summary of our linear model

print(lm.summary())



X_test_rfe = X_test[col_rfe]

X_test_rfe_constant = sm.add_constant(X_test_rfe)



y_pred_linear = lm.predict(X_test_rfe_constant)

y_pred_linear.values

y_pred_linear.min(), y_pred_linear.max()



from sklearn.metrics import mean_squared_error

mean_squared_error(y_pred_linear, y_test)



#SVM

from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', gamma=0.1)

svr_lin = SVR(kernel='linear', gamma='auto')

svr_poly = SVR(kernel='poly', gamma='auto', degree=3)



#SVM with RBF

svr_rbf.fit(X_train_rfe, y_train)

y_pred_svm_rbf = svr_rbf.predict(X_test_rfe)



y_pred_svm_rbf

y_pred_svm_rbf.min(), y_pred_svm_rbf.max()

mean_squared_error(y_pred_svm_rbf, y_test)
#SVM linear

svr_lin.fit(X_train_rfe, y_train)

y_pred_svm_lin = svr_lin.predict(X_test_rfe)

mean_squared_error(y_pred_svm_lin, y_test)



data = [['Linear regression', 0.90], ['RBF_SVM', 0.88],['SVM_linear',0.90]] 

MSE_table = pd.DataFrame(data, columns = ['Model', 'mse']) 



#SVM_poly

svr_poly.fit(X_train_rfe, y_train)

y_pred_svm_poly = svr_poly.predict(X_test_rfe)

mean_squared_error(y_pred_svm_poly, y_test)



MSE_table = MSE_table.append({'Model':'SVM_poly', 'mse':1.01}, ignore_index=True)
MSE_table
#ensemble models

#Gradient Boosting with Hyper Parameter Tuning



from sklearn import ensemble

n_trees=200

gradientboost = ensemble.GradientBoostingRegressor(loss='ls',learning_rate=0.04,n_estimators=n_trees,max_depth=4)

gradientboost.fit(X_train_rfe,y_train)



y_pred_gb=gradientboost.predict(X_test_rfe)

error=gradientboost.loss_(y_test,y_pred_gb) ##Loss function== Mean square error

print("MSE:%.3f" % error)

MSE_table = MSE_table.append({'Model':'GradientBoostingRegressor', 'mse':0.605}, ignore_index=True)
from sklearn.model_selection import GridSearchCV



# Create the parameter grid based on the results of random search 

param_grid = {

    'loss' : ['ls'],

    'max_depth' : [3, 4, 5],

    'learning_rate' : [0.01, 0.001],

    'n_estimators': [100, 200, 500]

}

# Create a based model

gb = ensemble.GradientBoostingRegressor()

# Instantiate the grid search model

grid_search_gb = GridSearchCV(estimator = gb, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)

grid_search_gb.fit(X_train_rfe, y_train)

grid_search_gb.best_params_

grid_search_gb_pred = grid_search_gb.predict(X_test_rfe)

mean_squared_error(y_test.values, grid_search_gb_pred)
#Random Forest with Hyper Parameter Tuning

from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators = 500)

rf_regressor.fit(X_train_rfe, y_train)

rf_pred = rf_regressor.predict(X_test_rfe)

mean_squared_error(rf_pred, y_test)



# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [90, 100],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4],

    'min_samples_split': [8, 10],

    'n_estimators': [100,200,300,400, 500,600, 1000]

}

# Create a based model

rf = RandomForestRegressor()

# Instantiate the grid search model

grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



grid_search_rf.fit(X_train_rfe, y_train)

grid_search_rf.best_params_



y_grid_pred_rf = grid_search_rf.predict(X_test_rfe)

mean_squared_error(y_grid_pred_rf, y_test.values)





MSE_table = MSE_table.append({'Model':'RandomForestRegressor', 'mse':0.61}, ignore_index=True)

#XGBoost with Hyperparameter tuning

import xgboost as xgb

xg_model = xgb.XGBRegressor(n_estimators = 500)

xg_model.fit(X_train_rfe, y_train)



results = xg_model.predict(X_test_rfe)

mean_squared_error(results, y_test.values)

xg_model.score(X_train_rfe, y_train)
from sklearn.metrics import r2_score

r2_score(y_test, results)
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [3, 4],

    'learning_rate' : [0.1, 0.01, 0.05],

    'n_estimators' : [100, 500, 1000]

}

# Create a based model

model_xgb= xgb.XGBRegressor()

# Instantiate the grid search model

grid_search_xgb = GridSearchCV(estimator = model_xgb, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)

grid_search_xgb.fit(X_train_rfe, y_train)

grid_search_xgb.best_params_

y_pred_xgb = grid_search_xgb.predict(X_test_rfe)

mean_squared_error(y_test.values, y_pred_xgb)
MSE_table = MSE_table.append({'Model':'XGBoost', 'mse':mean_squared_error(y_test.values, y_pred_xgb)}, ignore_index=True)

feature_importance = grid_search_xgb.best_estimator_.feature_importances_

sorted_importance = np.argsort(feature_importance)

pos = np.arange(len(sorted_importance))

plt.figure(figsize=(12,5))

plt.barh(pos, feature_importance[sorted_importance],align='center')

plt.yticks(pos, X_train_rfe.columns[sorted_importance],fontsize=15)

plt.title('Feature Importance ',fontsize=18)

plt.show()
#Building a Classification Model



y_train_classification = y_train.copy()

y_train_classification = pd.cut(y_train_classification, bins=[1, 3, 6, float('Inf')], labels=['Flop Movie', 'Average Movie', 'Hit Movie'])



y_test_classification = y_test.copy()

y_test_classification = pd.cut(y_test_classification, bins=[1, 3, 6, float('Inf')], labels=['Flop Movie', 'Average Movie', 'Hit Movie'])





X_train_rfe_classification = X_train_rfe.copy()

X_test_rfe_classification = X_test_rfe.copy()

#Logistic Regression

from sklearn.linear_model import LogisticRegression

logit_model = LogisticRegression(solver = 'saga', random_state = 0)

logit_model.fit(X_train_rfe_classification, y_train_classification)



y_logit_pred = logit_model.predict(X_test_rfe_classification)



from sklearn import metrics

count_misclassified = (y_test_classification != y_logit_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test_classification, y_logit_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test_classification, y_logit_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test_classification, y_logit_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test_classification, y_logit_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
#Support Vector Classifier with Linear, Polynomial, RBF



from sklearn.svm import SVC

svc_linear_model = SVC(kernel='linear', C=100, gamma= 'scale', decision_function_shape='ovo', random_state = 42)



svc_linear_model.fit(X_train_rfe_classification, y_train_classification)

y_svc_linear_pred = svc_linear_model.predict(X_test_rfe_classification)



from sklearn import metrics

count_misclassified = (y_test_classification != y_svc_linear_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test_classification, y_svc_linear_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test_classification, y_svc_linear_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test_classification, y_svc_linear_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test_classification, y_svc_linear_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))

from sklearn.svm import SVC

svc_poly_model = SVC(kernel='poly', C=100, gamma= 'scale', degree = 3, decision_function_shape='ovo', random_state = 42)



svc_poly_model.fit(X_train_rfe_classification, y_train_classification)

y_svc_poly_pred = svc_poly_model.predict(X_test_rfe_classification)



from sklearn import metrics

count_misclassified = (y_test_classification != y_svc_poly_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test_classification, y_svc_poly_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test_classification, y_svc_poly_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test_classification, y_svc_poly_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test_classification, y_svc_poly_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
from sklearn.svm import SVC

svc_rbf_model = SVC(kernel='rbf', C=100, gamma= 'scale', decision_function_shape='ovo', random_state = 42)



svc_rbf_model.fit(X_train_rfe_classification, y_train_classification)

y_svc_rbf_pred = svc_rbf_model.predict(X_test_rfe_classification)



from sklearn import metrics

count_misclassified = (y_test_classification != y_svc_rbf_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test_classification, y_svc_rbf_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test_classification, y_svc_rbf_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test_classification, y_svc_rbf_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test_classification, y_svc_rbf_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
#Ensemble Models

#Random Forest Classifier with Hyper Parameter tuning



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [90,100],#list(range(90,100)),

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4],

    'min_samples_split': [8, 10],

    'n_estimators': [100, 500, 1000],

    'random_state' :[0]

}

# Create a based model

rf_model_classification = RandomForestClassifier()

# Instantiate the grid search model

grid_search_rf_model_classificaiton = GridSearchCV(estimator = rf_model_classification, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



grid_search_rf_model_classificaiton.fit(X_train_rfe_classification, y_train_classification)



y_rf_classification_pred = grid_search_rf_model_classificaiton.predict(X_test_rfe_classification)



from sklearn import metrics

count_misclassified = (y_test_classification != y_rf_classification_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test_classification, y_rf_classification_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test_classification, y_rf_classification_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test_classification, y_rf_classification_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test_classification, y_rf_classification_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
#Gradient Boost Classifier with Hyper Parameter Tuning



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [10, 50, 90],

    'max_features': [3],

    'min_samples_leaf': [3],

    'min_samples_split': [8, 10],

    'n_estimators': [100, 500],

    'learning_rate' : [0.01, 0.2],

    'random_state' : [0]

}

# Create a based model

gbc_model_classification = GradientBoostingClassifier()

# Instantiate the grid search model

grid_search_gbc_model_classificaiton = GridSearchCV(estimator = gbc_model_classification, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



grid_search_gbc_model_classificaiton.fit(X_train_rfe_classification, y_train_classification)

y_gbc_model_pred = grid_search_gbc_model_classificaiton.predict(X_test_rfe_classification)



from sklearn import metrics

count_misclassified = (y_test_classification != y_gbc_model_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test_classification, y_gbc_model_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test_classification, y_gbc_model_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test_classification, y_gbc_model_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test_classification, y_gbc_model_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
#XG Boost Classifier with Hyper Parameter Tuning

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

param_grid = {

     'objective' : ['multi:softmax', 'multi:softprob'],

     'n_estimators': [100, 500, 1000],

     'random_state': [0]

}

# Create a based model

xgb_model_classification = XGBClassifier()

# Instantiate the grid search model

grid_search_xgb_model_classificaiton = GridSearchCV(estimator = xgb_model_classification, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



grid_search_xgb_model_classificaiton.fit(X_train_rfe_classification, y_train_classification)



y_xgb_classification_pred = grid_search_xgb_model_classificaiton.predict(X_test_rfe_classification)



from sklearn import metrics

count_misclassified = (y_test_classification != y_xgb_classification_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test_classification, y_xgb_classification_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test_classification, y_xgb_classification_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test_classification, y_xgb_classification_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test_classification, y_xgb_classification_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
#classification model result



data = {'Classification Model' : ['Logistic Regression', 'SVC_Linear', 'SVC_Poly', 'SVC_RBF', 'Random forest', 'Gradient Boosting','XGBoost'],

        'MisClassifications':[596,647,516,523,456,441,430],

        'Accuracy' : [0.68,0.65,0.72,0.72,0.75,0.76,0.77],

        'Precision' : [0.43,0.22,0.46,0.46,0.49,0.49,0.50],

        'Recall' : [0.38,0.33,0.44,0.44,0.47,0.48,0.49],

        'F1-Score' : [0.37,0.26,0.44,0.45,0.47,0.48,0.49]

        }

Accuracy_table = pd.DataFrame(data) 
feature_importance = grid_search_gbc_model_classificaiton.best_estimator_.feature_importances_

sorted_importance = np.argsort(feature_importance)

pos = np.arange(len(sorted_importance))

plt.figure(figsize=(12,5))

plt.barh(pos, feature_importance[sorted_importance],align='center')

plt.yticks(pos, X_train_rfe.columns[sorted_importance],fontsize=15)

plt.title('Feature Importance ',fontsize=18)

plt.show()
Accuracy_table
MSE_table