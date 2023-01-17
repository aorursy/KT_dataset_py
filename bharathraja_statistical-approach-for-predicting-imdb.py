#importing the libraries that we use

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling as pp
#importing the dataset

dataset = pd.read_csv('../input/imdb-5000-movie-dataset/movie_metadata.csv')

dataset.head()
dataset.shape
dataset.columns
dataset.profile_report()
dataset.drop_duplicates(inplace = True)

dataset.shape
numerical_cols = [col for col in dataset.columns if dataset[col].dtype != 'object']

categorical_cols = [col for col in dataset.columns if dataset[col].dtype == 'object']
categorical_cols, numerical_cols
dataset[numerical_cols].describe()
dataset[categorical_cols].describe()
dataset.isnull().sum()
dataset.color.unique()
color_mode = dataset['color'].mode().iloc[0]

dataset.color.fillna(color_mode, inplace = True)

dataset.color.isnull().sum()
dataset.director_name.nunique(), dataset.director_name.isnull().sum()
dataset = dataset.dropna(axis = 0, subset = ['director_name'] )
dataset.num_critic_for_reviews.min(), dataset.num_critic_for_reviews.max(), dataset.num_critic_for_reviews.median()
num_critic_for_reviews_median = dataset['num_critic_for_reviews'].median()

dataset.num_critic_for_reviews.fillna(num_critic_for_reviews_median, inplace = True)

dataset.num_critic_for_reviews.isnull().sum()
dataset.duration.min(), dataset.duration.max(), dataset.duration.median()
duration_median = dataset.duration.median()

dataset.duration.fillna(duration_median, inplace = True)

dataset.duration.isnull().sum()
dataset.director_facebook_likes.min(), dataset.director_facebook_likes.max(), dataset.director_facebook_likes.median(),dataset.director_facebook_likes.mean()
director_facebook_likes_mean = dataset.director_facebook_likes.mean()

dataset.director_facebook_likes.fillna(director_facebook_likes_mean, inplace = True)

dataset.director_facebook_likes.isnull().sum()
dataset.actor_3_facebook_likes.min(), dataset.actor_3_facebook_likes.max(), dataset.actor_3_facebook_likes.median(),dataset.actor_3_facebook_likes.mean()
actor_3_facebook_likes_mean = dataset.actor_3_facebook_likes.mean()

dataset.actor_3_facebook_likes.fillna(actor_3_facebook_likes_mean, inplace = True)

dataset.actor_3_facebook_likes.isnull().sum()
dataset = dataset.dropna(axis = 0, subset = ['actor_2_name'])

dataset.actor_2_name.isnull().sum()
dataset.actor_1_facebook_likes.min(), dataset.actor_1_facebook_likes.max(), dataset.actor_1_facebook_likes.median(),dataset.actor_1_facebook_likes.mean()
actor_1_facebook_likes_mean = dataset.actor_1_facebook_likes.mean()

dataset.actor_1_facebook_likes.fillna(actor_1_facebook_likes_mean, inplace = True)

dataset.actor_1_facebook_likes.isnull().sum()
dataset.gross.describe()
dataset.gross.isnull().sum()
dataset = dataset.dropna(axis = 0, subset = ['gross'])

dataset.gross.isnull().sum()
dataset.shape
dataset.isnull().sum()
dataset = dataset.dropna(axis = 0, subset = ['budget'])

dataset.budget.isnull().sum()
dataset.isnull().sum()
dataset.shape
dataset = dataset.dropna(axis = 0, subset = ['actor_3_name'])

dataset.actor_3_name.isnull().sum()
facenumber_in_poster_median = dataset.facenumber_in_poster.median()

dataset.facenumber_in_poster.fillna(facenumber_in_poster_median, inplace = True)

dataset.facenumber_in_poster.isnull().sum()
dataset.plot_keywords.unique()
dataset.language.unique()
dataset.language.value_counts()
language_mode = dataset.language.mode().iloc[0]

dataset.language.fillna(language_mode, inplace = True)

dataset.language.isnull().sum()
dataset = dataset.dropna(axis = 0, subset = ['plot_keywords'])

dataset.plot_keywords.isnull().sum()
dataset.content_rating.unique()
dataset.content_rating.fillna('Not Rated', inplace = True)
dataset.aspect_ratio.unique()
aspect_ratio_mode = dataset.aspect_ratio.mode().iloc[0]

dataset.aspect_ratio.fillna(aspect_ratio_mode, inplace = True)                                                    
dataset.isnull().sum()
dataset.reset_index(inplace = True, drop = True)
dataset.profile_report()
numerical_cols, categorical_cols
dataset.color.unique(), dataset.color.nunique()
dataset['color'] = dataset.color.map({'Color' : 1 , ' Black and White' : 0})
dataset.director_name.unique(), dataset.director_name.nunique()
director_name_value_counts = dataset.director_name.value_counts()
director_name_value_counts  = pd.DataFrame(director_name_value_counts).reset_index().rename(columns = {'index': 'director_name', 'director_name':'director_name_value_counts'})
dataset = pd.merge(dataset, director_name_value_counts,left_on = 'director_name', right_on = 'director_name', how = 'left')
dataset = dataset.drop(columns = 'director_name')
dataset.actor_2_name.unique(), dataset.actor_2_name.nunique()
actor_2_name_value_counts = dataset.actor_2_name.value_counts()
actor_2_name_value_counts  = pd.DataFrame(actor_2_name_value_counts).reset_index().rename(columns = {'index': 'actor_2_name', 'actor_2_name':'actor_2_name_value_counts'})
dataset = pd.merge(dataset, actor_2_name_value_counts,left_on = 'actor_2_name', right_on = 'actor_2_name', how = 'left')
dataset = dataset.drop(columns = 'actor_2_name')
dataset.genres.unique(), dataset.genres.nunique()
dataset['main_genre'] = dataset.genres.str.split('|').str[0]
dataset.main_genre.unique(), dataset.main_genre.nunique()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

dataset['main_genre'] = le.fit_transform(dataset.main_genre)
genres_value_counts = dataset.genres.value_counts()
genres_value_counts  = pd.DataFrame(genres_value_counts).reset_index().rename(columns = {'index' : 'genres', 'genres' : 'genres_value_counts'})
dataset = pd.merge(dataset, genres_value_counts,left_on = 'genres', right_on = 'genres', how = 'left')
dataset = dataset.drop(columns = 'genres')
dataset.actor_1_name.unique(), dataset.actor_1_name.nunique()
actor_1_name_value_counts = dataset.actor_1_name.value_counts()
actor_1_name_value_counts = pd.DataFrame(actor_1_name_value_counts).reset_index().rename(columns = {'index' : 'actor_1_name', 'actor_1_name' : 'actor_1_name_value_counts'})
dataset = pd.merge(dataset, actor_1_name_value_counts,left_on = 'actor_1_name', right_on = 'actor_1_name', how = 'left')
dataset = dataset.drop(columns = 'actor_1_name')
dataset.movie_title.unique(), dataset.movie_title.nunique()
dataset = dataset.drop(columns = 'movie_title')
dataset.actor_3_name.unique(), dataset.actor_3_name.nunique()
actor_3_name_value_counts = dataset.actor_3_name.value_counts()
actor_3_name_value_counts = pd.DataFrame(actor_3_name_value_counts).reset_index().rename(columns = {'index' : 'actor_3_name', 'actor_3_name' : 'actor_3_name_value_counts'})
dataset= pd.merge(dataset, actor_3_name_value_counts,left_on = 'actor_3_name', right_on = 'actor_3_name', how = 'left')
dataset = dataset.drop(columns = 'actor_3_name')
dataset.plot_keywords.unique(), dataset.plot_keywords.nunique()
dataset['main_plot_keyword'] = dataset.plot_keywords.str.split('|').str[0]
dataset = dataset.drop(columns = 'plot_keywords')
dataset.main_plot_keyword.unique(), dataset.main_plot_keyword.nunique()
main_plot_keyword_value_counts = dataset.main_plot_keyword.value_counts()
main_plot_keyword_value_counts = pd.DataFrame(main_plot_keyword_value_counts).reset_index().rename(columns = {'index' : 'main_plot_keyword', 'main_plot_keyword' : 'main_plot_keyword_value_counts'})
dataset = pd.merge(dataset, main_plot_keyword_value_counts, left_on = 'main_plot_keyword', right_on = 'main_plot_keyword', how = 'left')
dataset = dataset.drop(columns = 'main_plot_keyword')
dataset.movie_imdb_link.unique(), dataset.movie_imdb_link.nunique()
dataset = dataset.drop(columns = 'movie_imdb_link')
dataset.language.unique(), dataset.language.nunique()
from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()

dataset['language'] = le1.fit_transform(dataset.language)
dataset.country.unique(), dataset.country.nunique()
from sklearn.preprocessing import LabelEncoder

le2 = LabelEncoder()

dataset['country'] = le2.fit_transform(dataset.country)
dataset.content_rating.unique(),dataset.content_rating.nunique()
from sklearn.preprocessing import LabelEncoder

le3 = LabelEncoder()

dataset['content_rating'] = le3.fit_transform(dataset.content_rating)
dataset.head().T
dataset.profile_report()
datasetR = dataset.copy() #lets keep our original dataset for reference. Here datasetR is for Regression model

datasetC = dataset.copy() #Here datasetC is for classification model
from sklearn.model_selection import train_test_split

y = datasetR.pop('imdb_score')

X = datasetR

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test.values), columns = X_train.columns, index = X_test.index)
X_train.shape
#removing variables with high colinearity

def correlation(dataset, threshold):

    col_corr = set() # Set of all the names of deleted columns

    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):

                colname = corr_matrix.columns[i] # getting the name of column

                col_corr.add(colname)

                if colname in dataset.columns:

                    del dataset[colname] # deleting the column from the dataset

correlation(X_train,0.90)
X_train.shape
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
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', gamma=0.1)

svr_lin = SVR(kernel='linear', gamma='auto')

svr_poly = SVR(kernel='poly', gamma='auto', degree=3)
svr_rbf.fit(X_train_rfe, y_train)

y_pred_svm_rbf = svr_rbf.predict(X_test_rfe)
y_pred_svm_rbf
y_pred_svm_rbf.min(), y_pred_svm_rbf.max()
mean_squared_error(y_pred_svm_rbf, y_test)
svr_lin.fit(X_train_rfe, y_train)

y_pred_svm_lin = svr_lin.predict(X_test_rfe)
y_pred_svm_lin
y_pred_svm_lin.min(), y_pred_svm_lin.max()
mean_squared_error(y_pred_svm_lin, y_test)
svr_poly.fit(X_train_rfe, y_train)

y_pred_svm_poly = svr_poly.predict(X_test_rfe)
y_pred_svm_poly
y_pred_svm_poly.min(), y_pred_svm_poly.max()
mean_squared_error(y_pred_svm_poly, y_test)
from sklearn import ensemble

n_trees=200

gradientboost = ensemble.GradientBoostingRegressor(loss='ls',learning_rate=0.03,n_estimators=n_trees,max_depth=4)

gradientboost.fit(X_train_rfe,y_train)
y_pred_gb=gradientboost.predict(X_test_rfe)

error=gradientboost.loss_(y_test,y_pred_gb) ##Loss function== Mean square error

print("MSE:%.3f" % error)
mean_squared_error(y_pred_gb, y_test)
y_pred_gb.min(), y_pred_gb.max()
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
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators = 500)

rf_regressor.fit(X_train_rfe, y_train)

rf_pred = rf_regressor.predict(X_test_rfe)
mean_squared_error(rf_pred, y_test)
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [90, 100],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4],

    'min_samples_split': [8, 10],

    'n_estimators': [100, 500, 1000]

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
import xgboost as xgb

xg_model = xgb.XGBRegressor(n_estimators = 500)

xg_model.fit(X_train_rfe, y_train)
results = xg_model.predict(X_test_rfe)
mean_squared_error(results, y_test.values)
xg_model.score(X_train_rfe, y_train)
from sklearn.metrics import r2_score

r2_score(y_test, results)
from sklearn.model_selection import GridSearchCV

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
feature_importance = grid_search_xgb.best_estimator_.feature_importances_

sorted_importance = np.argsort(feature_importance)

pos = np.arange(len(sorted_importance))

plt.figure(figsize=(12,5))

plt.barh(pos, feature_importance[sorted_importance],align='center')

plt.yticks(pos, X_train_rfe.columns[sorted_importance],fontsize=15)

plt.title('Feature Importance ',fontsize=18)

plt.show()
datasetC.head()
y_train_classification = y_train.copy()
y_train_classification = pd.cut(y_train_classification, bins=[1, 3, 6, float('Inf')], labels=['Flop Movie', 'Average Movie', 'Hit Movie'])
y_test_classification = y_test.copy()
y_test_classification = pd.cut(y_test_classification, bins=[1, 3, 6, float('Inf')], labels=['Flop Movie', 'Average Movie', 'Hit Movie'])
X_train_rfe_classification = X_train_rfe.copy()

X_test_rfe_classification = X_test_rfe.copy()
from sklearn.linear_model import LogisticRegression

logit_model = LogisticRegression(solver = 'saga', random_state = 0)

logit_model.fit(X_train_rfe_classification, y_train_classification)
y_logit_pred = logit_model.predict(X_test_rfe_classification)
y_logit_pred
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
from sklearn.svm import SVC

svc_linear_model = SVC(kernel='linear', C=100, gamma= 'scale', decision_function_shape='ovo', random_state = 42)
svc_linear_model.fit(X_train_rfe_classification, y_train_classification)

y_svc_linear_pred = svc_linear_model.predict(X_test_rfe_classification)
y_svc_linear_pred
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
y_svc_poly_pred
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
y_svc_rbf_pred
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
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [90, 100],

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
y_rf_classification_pred
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
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [10, 50, 90],

    'max_features': [3],

    'min_samples_leaf': [3],

    'min_samples_split': [8, 10],

    'n_estimators': [100, 500],

    'learning_rate' : [0.1, 0.2],

    'random_state' : [0]

}

# Create a based model

gbc_model_classification = GradientBoostingClassifier()

# Instantiate the grid search model

grid_search_gbc_model_classificaiton = GridSearchCV(estimator = gbc_model_classification, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)
grid_search_gbc_model_classificaiton.fit(X_train_rfe_classification, y_train_classification)
y_gbc_model_pred = grid_search_gbc_model_classificaiton.predict(X_test_rfe_classification)
y_gbc_model_pred
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
y_xgb_classification_pred
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
feature_importance = grid_search_gbc_model_classificaiton.best_estimator_.feature_importances_

sorted_importance = np.argsort(feature_importance)

pos = np.arange(len(sorted_importance))

plt.figure(figsize=(12,5))

plt.barh(pos, feature_importance[sorted_importance],align='center')

plt.yticks(pos, X_train_rfe.columns[sorted_importance],fontsize=15)

plt.title('Feature Importance ',fontsize=18)

plt.show()