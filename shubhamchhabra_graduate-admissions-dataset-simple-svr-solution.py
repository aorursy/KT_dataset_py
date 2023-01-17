import pandas as pd

import numpy as np
%matplotlib inline

import matplotlib.pyplot as plt
RAW_DATA_FILE = '../input/Admission_Predict_Ver1.1.csv'
raw_data = pd.read_csv(RAW_DATA_FILE)
raw_data.describe().transpose()
raw_data.columns = [c.replace(' ', '_') for c in raw_data.columns]

raw_data.drop(columns=['Serial_No.'],inplace=True)
raw_data.info()
raw_data.hist(bins=50, figsize=(20,15))

plt.show()
corr_matrix = raw_data.corr()

corr_matrix.Chance_of_Admit_.sort_values(ascending=False)
from pandas.plotting import scatter_matrix



attributes = ['Chance_of_Admit_','GRE_Score', 'TOEFL_Score','University_Rating','SOP','LOR_','CGPA','Research']

scatter_matrix(raw_data[attributes], figsize=(20, 15))
corr_matrix_plot = pd.DataFrame(columns=['University_Rating','GRE_Score','TOEFL_Score','SOP','LOR_','CGPA','Research'])

for uni_rating in range(1,6):

    raw_data_univ_rating = raw_data.loc[raw_data['University_Rating'] == uni_rating]

    corr_matrix_univ_rating = raw_data_univ_rating.corr()

    corr_matrix_univ_rating.drop(columns=['University_Rating'],inplace=True)

    corr_matrix_plot.loc[uni_rating-1] = corr_matrix_univ_rating.Chance_of_Admit_

    corr_matrix_plot.loc[uni_rating-1].at['University_Rating'] = uni_rating



corr_matrix_plot.plot(figsize=(10,4),kind='line',x='University_Rating',y=['GRE_Score','TOEFL_Score','SOP','LOR_','CGPA','Research'])
strat_sampling_bins = [i*0.1 for i in range(0,11)]

raw_data['chance_of_admit_category'] = pd.cut(raw_data.Chance_of_Admit_, bins=strat_sampling_bins, labels=[i for i in range(0,10)])

raw_data.chance_of_admit_category.value_counts()
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(raw_data, raw_data.chance_of_admit_category):

    strat_train_set = raw_data.loc[train_index]

    strat_test_set = raw_data.loc[test_index]
strat_train_set.describe().transpose()
strat_test_set.describe().transpose()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer



data_pipeline = Pipeline([('imputer', Imputer(strategy='median')),

                          ('standardize_data', StandardScaler())

                         ])
strat_train_set_labels = strat_train_set.Chance_of_Admit_.copy()

strat_train_set.drop(columns=['Chance_of_Admit_'],inplace=True)



strat_test_set_labels = strat_test_set.Chance_of_Admit_.copy()

strat_test_set.drop(columns=['Chance_of_Admit_'],inplace=True)
strat_train_set_transformed = data_pipeline.fit_transform(X=strat_train_set)

strat_test_set_transformed = data_pipeline.transform(X=strat_test_set)
from sklearn.svm import SVR

from sklearn.model_selection import RandomizedSearchCV



param_grid = {'kernel':['linear','poly','rbf','sigmoid'],'degree':[i*3 for i in range(1,10)]}

svr = SVR()

random_search = RandomizedSearchCV(svr, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)

random_search.fit(strat_train_set_transformed, strat_train_set_labels)
random_search.best_params_
cv_res = random_search.cv_results_

for mean_score, params in zip(cv_res['mean_test_score'], cv_res['params']):

    print(np.sqrt(-mean_score), params)
from sklearn.metrics import mean_squared_error

final_model = random_search.best_estimator_

final_predictions = final_model.predict(strat_test_set_transformed)

final_mse = mean_squared_error(y_true=strat_test_set_labels, y_pred=final_predictions)

print('mean sq. error: ' + str(final_mse))

print('rmse: ' + str(np.sqrt(final_mse)))