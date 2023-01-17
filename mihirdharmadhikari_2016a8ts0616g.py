# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



from sklearn.impute import SimpleImputer



from sklearn.linear_model import HuberRegressor, Lasso, Ridge, BayesianRidge, RANSACRegressor, LassoCV, ElasticNet

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn import metrics

from sklearn import preprocessing

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



from sklearn.decomposition import PCA



import seaborn as sns

from scipy import stats

from scipy.stats import norm

from scipy.stats import binned_statistic

import warnings

import matplotlib.pyplot as plt  

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Parameters

remove_outliers = False

correlation_reduction = False

do_pca = False

pca_components = 30

do_stds = True
# Split data according to agents



df_raw = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv', sep=',')



agent_dfs = []

for i in range(0,7):

    agent_dfs.append(df_raw[df_raw['a'+str(i)]==1])

    





print("complete")
# Finding positive correlation of features with label



full_training_dfs = []

training_dfs = []

label_dfs = []

cols = []



if correlation_reduction:

    for df in agent_dfs:

        arr_train_cor = df.corr()['label']

        idx_train_cor_gt0 = arr_train_cor[abs(arr_train_cor) > 0.5].sort_values(ascending=False).index.tolist()

        print("How many feature candidates have positive correlation with SalePrice(including itself)? %d" % len(idx_train_cor_gt0))

#         print(arr_train_cor[idx_train_cor_gt0])

        full_training_dfs.append(df[idx_train_cor_gt0].copy())

        idx_train_cor_gt0.remove('label')

        train_df = df[idx_train_cor_gt0].copy()

        training_dfs.append(train_df)

        label_df = df['label'].copy()

        label_dfs.append(label_df)

    #     print(label_df.head)

else:

    for df in agent_dfs:

        label_df = df['label'].copy()

        label_dfs.append(label_df)

#         print(label_dfs[-1].values.shape)

#         print(label_dfs[-1].head(n=5))

        cols = list(df.columns)

        for i in range(0,7):

            cols.remove('a'+str(i))

        cols.remove('time')

        full_training_dfs.append(df[cols].copy())

        cols.remove('label')

        train_df = df[cols].copy()

#         print(train_df.head(n=5))

        training_dfs.append(train_df)

        

print(len(label_dfs))

    

print("complete")
# PCA



training_data_params_final = []

training_data_params_pca = []

pcas = []

training_dfs_pca = []



for df in training_dfs:

    if do_pca:

        pca = PCA(n_components=pca_components)

#         print(df.head(n=5))

        to_fit = df.values[0:20930,1:]

        to_validate = df.values[20930:,1:]

        print(to_fit[0,:])

        training_op = pca.fit_transform(to_fit)

        validation_op = pca.transform(to_validate)

        print(training_op.shape, validation_op.shape)

        combined_pca = np.concatenate((training_op, validation_op), axis=0)

        training_data_params_pca.append(combined_pca)

        pcas.append(pca)

    else:

        training_data_params_pca.append(df.values[:,1:])

    



print("complete")
# Do standerdisation



stdzers = []



if do_stds:

    for d in training_data_params_pca:

#         stdzed_data,stdzer = standerdize_data(d, None)

        stdzer = preprocessing.MinMaxScaler()

        print(d.shape)

        stdzed_data = stdzer.fit_transform(d)

        training_data_params_final.append(stdzed_data)

        stdzers.append(stdzer)

#     print(training_data_params_final)

else:

    training_data_params_final = training_data_params_pca
# Separate the data into training and validation



training_data_params = []

training_data_times = []

training_data_labels = []

validation_data_params = []

validation_data_times = []

validation_data_labels = []

validation_data_params_2 = []

validation_data_times_2 = []

validation_data_labels_2 = []



log_training_data_params = []

log_training_data_labels = []

log_validation_data_params = []

log_validation_data_labels = []





for i in range(0,7):

    training_data_params.append(training_data_params_final[i][:,:])

    validation_data_params.append(training_data_params_final[i][20930:,:])

    training_data_labels.append(label_dfs[i].values[:])

    validation_data_labels.append(label_dfs[i].values[20930:])

    log_training_data_params.append(np.log1p(training_data_params_final[i][0:20930,:]))

    log_validation_data_params.append(np.log1p(training_data_params_final[i][20930:,:]))

    log_training_data_labels.append(np.log1p(label_dfs[i].values[0:20930]))

    log_validation_data_labels.append(np.log1p(label_dfs[i].values[20930:]))



    

# fig, ax = plt.subplots( nrows=1, ncols=2)  # create figure & 1 axis

# ax[0].plot(training_data_labels[0])

# ax[1].plot(agent_dfs[0]['b3'].values)

# # plt.plot(training_data_labels[0])

# # plt.plot(agent_dfs[0]['b3'].values)

# plt.figure()

# plt.plot(training_data_labels[1])

# plt.figure()

# plt.plot(training_data_labels[2])

# plt.figure()

# plt.plot(training_data_labels[3])

# plt.figure()

# plt.plot(training_data_labels[4])

# plt.figure()

# plt.plot(training_data_labels[5])

# plt.figure()

# plt.plot(training_data_labels[6])



# for i in range(0,7):

# #     plt.figure()

# #     sns.distplot(training_data_labels[i])

# #     plt.figure()

# #     sns.distplot(np.log1p(training_data_labels[i]))

#     fig, ax = plt.subplots( nrows=1, ncols=2)

# #     ax[0].plot(training_data_labels[i])

# #     ax[1].plot(np.log1p(training_data_labels[i]))

#     ax[0].plot(training_data_params[i][:,0])

#     ax[1].plot(np.log1p(training_data_params[i][:,0]))

#     print(training_data_params[i][:,0])





print(len(training_data_params))

print("complete")
# Fit the model



regressors = []

regressors_pca = []



for X_train, y_train in zip(training_data_params, training_data_labels):

#     regressor = Ridge(alpha=50.0, random_state=0, tol=1e-6)  

    regressor = RandomForestRegressor(n_estimators=200, min_samples_split=4, max_depth=100)

#     params = {

#         'max_depth': [20,None],

#         'min_samples_leaf': [2],

#         'min_samples_split': [4],

#         'n_estimators': [100,150],

#         }



#     rf_temp = RandomForestRegressor()

#     rf_temp_tuned = GridSearchCV(rf_temp, params, n_jobs=-1)

#     rf_temp_tuned.fit(X_train,y_train)

#     regressor = RandomForestRegressor(**rf_temp_tuned.best_params_)

    regressor.fit(X_train, y_train) #training the algorithm

    regressors.append(regressor)

    

# for X_train, y_train in zip(training_data_params_pca, training_data_labels):

#     regressor_pca = RandomForestRegressor()  

#     regressor_pca.fit(X_train, y_train) #training the algorithm

#     regressors_pca.append(regressor_pca)

    

    

print("complete")
# Validation



pred_labels = []

pred_labels_2 = []



validation_data_params_pca = []

validation_data_params_final = []



for X_test, regressor in zip(validation_data_params,regressors):

#     print(X_test.shape)

    y_pred = regressor.predict(X_test)

    pred_labels.append(y_pred)

    

for X_test, regressor in zip(training_data_params,regressors):

#     print(X_test.shape)

    y_pred = regressor.predict(X_test)

    pred_labels_2.append(y_pred)



rmse_total = 0.0



for lp, la in zip(pred_labels, validation_data_labels):

    rmse = metrics.mean_squared_error(la, lp)

    rmse_total += rmse**2

    print('Mean Squared Error:', rmse)

#     for p,a in zip(lp,la):

#         print(p, a)

print("Total RMSE:", math.sqrt(rmse_total/7))





rmse_total = 0.0



print(pred_labels_2[0].shape, training_data_labels[0].shape)

for lp, la in zip(pred_labels_2, training_data_labels):

    rmse = metrics.mean_squared_error(la, lp)

    rmse_total += rmse**2

    print('Mean Squared Error:', rmse)

#     for p,a in zip(lp,la):

#         print(p, a)

print("Total RMSE:", math.sqrt(rmse_total/7))
# Test dataset



df_test_raw = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv', sep=',')



agent_test_dfs = []

for i in range(0,7):

    agent_test_dfs.append(df_test_raw[df_test_raw['a'+str(i)]==1])

    



test_output_dfs = []

test_data = []

for df in agent_test_dfs:

    cols = list(df.columns)

    for i in range(0,7):

        cols.remove('a'+str(i))

    cols.remove('time')

    print(cols)

    test_data.append(df[cols].copy().values[:,1:])

    test_output_dfs.append(pd.DataFrame({'id':df['id'].values}))

    

test_pca = []

test_final = []



if do_pca:

    for X,pca in zip(test_data, pcas):

    #     pca = PCA(n_components=pca_components)

        test_pca.append(pca.transform(X))

    #     print(training_data_params_pca[-1].shape)

else:

    test_pca = test_data



if do_stds:

    for X,stdzer in zip(test_pca, stdzers):

#         test_final.append(standerdize_data(X, stdzer))

        test_final.append(stdzer.transform(X))

else:

    test_final = test_pca



# for i in range(0,7):

#     test_final[i] = np.log1p(test_final[i])



pred_output_labels = []

for X_test, regressor in zip(test_final,regressors):

#     print(X_test.shape)

    y_pred = regressor.predict(X_test)

    pred_output_labels.append(y_pred)



for l,df in zip(pred_output_labels, test_output_dfs):

    df['label'] = l

    

final_df = pd.concat(test_output_dfs)

final_df.sort_values(by=['id'], inplace=True)

print(final_df.head(10))



final_df.to_csv("/kaggle/working/output.csv", header=True, index=False)





    




