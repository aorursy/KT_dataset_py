import os

import pandas as pd



DATA_PATH = '../input/red-wine-quality-cortez-et-al-2009/'

FILE_NAME = 'winequality-red.csv'

def load_wine_data(data_path=DATA_PATH, file_name=FILE_NAME):

    csv_path = os.path.join(data_path, file_name)

    return pd.read_csv(csv_path)



wines = load_wine_data()
wines.head(10)
wines.info()
wines.describe()
import matplotlib.pyplot as plt



wines.hist(bins=50, figsize=(20,15), color="orange")

plt.show()
from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(wines, test_size=0.2, random_state=42)
wines_train_set = train_set.drop('quality', axis=1)

train_set_labels = train_set['quality'].copy()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer



numerical_pipeline = Pipeline([

        ('std_scaler', StandardScaler()),

    ])



cols = list(wines_train_set)

pipeline = ColumnTransformer([

        ("numerical_attributes", numerical_pipeline, cols),

    ])



prepared_train_set = pipeline.fit_transform(wines_train_set)

prepared_train_set.shape
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.metrics import mean_squared_error

import numpy as np



lin_scores = cross_val_score(lin_reg, prepared_train_set, train_set_labels,

                             scoring="neg_mean_squared_error", cv = 10)

lin_scores_rmse = np.sqrt(-lin_scores)
def display_cv_scores(scores):

    print("Scores:\t", scores)

    print("Mean:\t", scores.mean())

    print("Std:\t", scores.std())



display_cv_scores(lin_scores_rmse)
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'copy_X':[True,False], 'fit_intercept':[True,False], 'normalize':[True,False]}

]



grid_search = GridSearchCV(lin_reg, param_grid, cv=10, scoring='neg_mean_squared_error')

grid_search.fit(prepared_train_set, train_set_labels)
grid_search.best_estimator_
final_model = grid_search.best_estimator_



X_test = test_set.drop('quality', axis=1)

y_test = test_set['quality'].copy()



X_test_prepared = pipeline.transform(X_test)



final_predictions = final_model.predict(X_test_prepared)



final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)



print('RMSE:',final_rmse)
final_predictions[:5]
y_test[:5]
from sklearn.linear_model import Ridge



ridge_reg = Ridge(alpha=1)



#Cross_validation

ridge_scores = cross_val_score(ridge_reg, prepared_train_set, train_set_labels,

                             scoring="neg_mean_squared_error", cv = 10)

ridge_scores_rmse = np.sqrt(-ridge_scores)



print('Cross-validation mean RMSE:', ridge_scores_rmse.mean())



#Grid Search

rr_param_grid = [

    {'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000],

     "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

]



rr_grid_search = GridSearchCV(ridge_reg, rr_param_grid, cv=10, scoring='neg_mean_squared_error')

rr_grid_search.fit(prepared_train_set, train_set_labels)



#Evaluate on Test Set

rr_final_model = rr_grid_search.best_estimator_



rr_final_predictions = rr_final_model.predict(X_test_prepared)



rr_final_mse = mean_squared_error(y_test, rr_final_predictions)

rr_final_rmse = np.sqrt(rr_final_mse)



print('\nRidge Regression RMSE:', rr_final_rmse)
from sklearn.linear_model import Lasso



lasso_reg = Lasso(alpha=0.1)



lasso_scores = cross_val_score(lasso_reg, prepared_train_set, train_set_labels,

                             scoring="neg_mean_squared_error", cv = 10)

lasso_scores_rmse = np.sqrt(-lasso_scores)



print('Cross-validation mean RMSE:', lasso_scores_rmse.mean())



#Grid Search

lr_param_grid = [

    {'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000],

    }

]



lr_grid_search = GridSearchCV(lasso_reg, lr_param_grid, cv=10, scoring='neg_mean_squared_error')

lr_grid_search.fit(prepared_train_set, train_set_labels)



#Evaluate on Test Set

lr_final_model = lr_grid_search.best_estimator_



lr_final_predictions = lr_final_model.predict(X_test_prepared)



lr_final_mse = mean_squared_error(y_test, lr_final_predictions)

lr_final_rmse = np.sqrt(lr_final_mse)



print('\nLasso Regression RMSE:', lr_final_rmse)
from sklearn.linear_model import ElasticNet



en_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)



en_scores = cross_val_score(en_reg, prepared_train_set, train_set_labels,

                             scoring="neg_mean_squared_error", cv = 10)

en_scores_rmse = np.sqrt(-en_scores)



print('Cross-validation mean RMSE:', en_scores_rmse.mean())



#Grid Search

en_param_grid = [

    {'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000],

     'l1_ratio':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    }

]



en_grid_search = GridSearchCV(en_reg, en_param_grid, cv=10, scoring='neg_mean_squared_error')

en_grid_search.fit(prepared_train_set, train_set_labels)



#Evaluate on Test Set

en_final_model = en_grid_search.best_estimator_



en_final_predictions = en_final_model.predict(X_test_prepared)



en_final_mse = mean_squared_error(y_test, en_final_predictions)

en_final_rmse = np.sqrt(en_final_mse)



print('\nElasticNet Regression RMSE:', en_final_rmse)