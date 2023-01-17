# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualisation

import matplotlib.pyplot as plt # graph plotting

from sklearn.model_selection import train_test_split # splitting dataset in train and val

from sklearn.preprocessing import StandardScaler # scaling the input

from sklearn.decomposition import PCA # apply PCA

from scipy.stats.stats import pearsonr # for correlation

from sklearn.utils.testing import ignore_warnings

from sklearn.exceptions import ConvergenceWarning





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def fullprint(df):

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        print(df)
def check_input(df, initial):

#     print("dtypes:")

#     print(df.dtypes)

    

#     df_check = pd.concat([df.dtypes, df.nunique(), df.isnull().sum(), df.mean(axis=0), df.min(axis=0), df.max(axis=0), df.std(axis=0)],axis=1)

#     df_check.columns = ['Type','Unique','NaN', 'Mean', 'Min', 'Max', 'Std. Dev.']

    

    df_check = pd.concat([df.nunique(), df.mean(axis=0), df.min(axis=0), df.max(axis=0), df.std(axis=0)],axis=1)

    df_check.columns = ['Unique','Mean', 'Min', 'Max', 'Std. Dev.']

    

    featureNaN = [i for i in df.columns if df[i].isnull().sum() > 0]

    for i in featureNaN:

        df[i].fillna(df[i].mean(), inplace = True)

    

    if initial == True:

        fullprint(df_check)

        

        agent_freq = [df["a"+str(i)].sum() for i in range(7)]

        print("Agent frequencies: ", agent_freq)

        print("Total: ", sum(agent_freq), "/", len(df))

    

    return df 
def distribute_df(df):

    

    agents = ["a"+str(i) for i in range(7)]

    df_agents = [df.loc[df[i] == 1] for i in agents]

    for i in df_agents:

        i.drop(agents, axis=1, inplace = True)

    return df_agents
train_input = pd.read_csv("/kaggle/input/bits-f464-l1/train.csv")

train_check = check_input(train_input, True)   

train_agents = distribute_df(train_check)
def explore(df):

        

    corr = df.corr()

    corr_label = abs(corr["label"]).to_frame()

#     print(corr_label)

    

    relevant_features = corr_label.loc[(corr_label["label"]>0.6) & (corr_label["label"]!=1)].index.to_list()

    print(relevant_features)

    



# for i in range(7):    

#     explore(train[i])
def add_features(df, corr_threshold, powers):

    

    df = df.drop(['id'],  axis = 1)

    columns = list(df.columns)

    columns.remove('label')

    

    labels = df['label'].to_list()

    

    old_features = np.array([df[i].values for i in columns])

    

    df = df.drop(columns, axis=1)



    for i,a in enumerate(old_features):

        for j in range(1,powers+1):

            c = a**j

            if((pearsonr(c, labels)[0] > corr_threshold) or (columns[i] == 'time')):

                df.insert(0, columns[i]+"^"+str(j), c)

    

#     for i,a in enumerate(old_features):

#         for j,b in enumerate(old_features):

#             if(i>j):

#                 continue

#             c = a*b

#             if(pearsonr(c, labels)[0] > corr_threshold):

#                 df.insert(0, columns[i]+"*"+columns[j], c)

    print(len(df.columns))

                        

    return df

def agent_features(df, corr_threshold):

    

    corr = df.corr()

    corr_label = abs(corr["label"]).to_frame()



    relevant_features = corr_label.loc[(corr_label["label"]>corr_threshold) & (corr_label["label"]!=1)].index.to_list()

#     print(relevant_features)



    return relevant_features
def agent_split(df, features, splitsize):

    X = df.loc[:, features].values

    Y = df.loc[:, ['label']].values

    

    trainX, valX, trainY, valY =  train_test_split(X, Y, test_size=splitsize, shuffle=False, random_state=69)

    

    d = {'trainX': trainX, 'trainY' : trainY, 'valX' : valX, 'valY' : valY}

    return d
def agent_scalers(trainX):

    scaler = StandardScaler()

    scaler.fit(trainX)

    return scaler
def agent_pcas(trainX, scaler, ncomp):

    from sklearn.decomposition import PCA

    pca = PCA(n_components = min(ncomp,trainX.shape[1]))

#     scaler.transform(trainX)

#     pca.fit(trainX)

#     print(pca.explained_variance_ratio_)

    return pca
def predict(reg, valX, scaler, pca):

    

    valX = scaler.transform(valX)

#     valX = pca.transform(valX)



    y = reg.predict(valX)

    return y
def rmse_score(y, valY):

    # RMSE

    from sklearn.metrics import mean_squared_error

    from math import sqrt

    

    rmse = sqrt(mean_squared_error(y, valY))



    return rmse
def r2_score(reg, valX, valY, scaler, pca):

    # R2

    valX = scaler.transform(valX)

#     valX = pca.transform(valX)

    r2 = reg.score(valX, valY)



    return r2
@ignore_warnings(category=ConvergenceWarning)

def agent0_train(trainX, trainY, scaler, pca):

    trainX = scaler.transform(trainX)

#     trainX = pca.transform(trainX)

    trainY = np.ravel(trainY)

    

    #LinearRegression

#     from sklearn.linear_model import LinearRegression

#     reg = LinearRegression().fit(trainX, trainY) 

    

    #RidgeRegression

#     from sklearn.linear_model import Ridge

#     reg = Ridge(alpha=0.5).fit(trainX, trainY)



    #BayesianRidgeRegression

#     from sklearn.linear_model import BayesianRidge

#     reg = BayesianRidge().fit(trainX, trainY)

    

    #KernelRidgeRegression

#     from sklearn.kernel_ridge import KernelRidge

#     reg = KernelRidge(alpha=1).fit(trainX, trainY)

    

    #SupportVectorRegression

#     from sklearn.svm import SVR

#     from sklearn.model_selection import GridSearchCV

#     param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}

#     reg = GridSearchCV(SVR(kernel = 'rbf', max_iter=10000, cache_size = 2048), cv = 5, param_grid= param_grid, refit=True)

#     reg.fit(trainX, trainY)

#     print(reg.best_params_)

#     reg = reg.best_estimator_

    

    #DecisionTreeRegression

#     from sklearn.tree import DecisionTreeRegressor

#     reg = DecisionTreeRegressor(max_depth = 10).fit(trainX, trainY)

    

    #RandomForestRegression

#     from sklearn.ensemble import RandomForestRegressor

#     reg = RandomForestRegressor(n_estimators = 100, max_depth = 10).fit(trainX, trainY)

    

    #ExtraTreesRegression [BANNED -> just see the results]

#     from sklearn.ensemble import ExtraTreesRegressor

#     reg = ExtraTreesRegressor(n_estimators = 10, max_depth = 10).fit(trainX, trainY)



    #AdaBoostRegression

    from sklearn.ensemble import AdaBoostRegressor

#     from sklearn.tree import DecisionTreeRegressor

    from sklearn.ensemble import RandomForestRegressor

#     reg = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    reg = AdaBoostRegressor(base_estimator = RandomForestRegressor(n_estimators = 25, max_depth = 10), n_estimators = 25).fit(trainX, trainY)



    #BaggingRegression

#     from sklearn.ensemble import BaggingRegressor

#     from sklearn.tree import DecisionTreeRegressor

#     from sklearn.ensemble import RandomForestRegressor

#     reg = BaggingRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

#     reg = BaggingRegressor(base_estimator = RandomForestRegressor(n_estimators = 10, max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    print("Training #0 Complete!")

    return reg
@ignore_warnings(category=ConvergenceWarning)

def agent1_train(trainX, trainY, scaler, pca):

    trainX = scaler.transform(trainX)

#     trainX = pca.transform(trainX)

    trainY = np.ravel(trainY)

    

    #LinearRegression

#     from sklearn.linear_model import LinearRegression

#     reg = LinearRegression().fit(trainX, trainY) 

    

    #RidgeRegression

#     from sklearn.linear_model import Ridge

#     reg = Ridge(alpha=0.5).fit(trainX, trainY)



    #BayesianRidgeRegression

#     from sklearn.linear_model import BayesianRidge

#     reg = BayesianRidge().fit(trainX, trainY)

    

    #KernelRidgeRegression

#     from sklearn.kernel_ridge import KernelRidge

#     reg = KernelRidge(alpha=1).fit(trainX, trainY)

    

    #SupportVectorRegression

#     from sklearn.svm import SVR

#     from sklearn.model_selection import GridSearchCV

#     param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}

#     reg = GridSearchCV(SVR(kernel = 'rbf', max_iter=10000, cache_size = 2048), cv = 5, param_grid= param_grid, refit=True)

#     reg.fit(trainX, trainY)

#     print(reg.best_params_)

#     reg = reg.best_estimator_

    

    #DecisionTreeRegression

#     from sklearn.tree import DecisionTreeRegressor

#     reg = DecisionTreeRegressor(max_depth = 10).fit(trainX, trainY)

    

    #RandomForestRegression

#     from sklearn.ensemble import RandomForestRegressor

#     reg = RandomForestRegressor(n_estimators = 100, max_depth = 10).fit(trainX, trainY)

    

    #ExtraTreesRegression [BANNED -> just see the results]

#     from sklearn.ensemble import ExtraTreesRegressor

#     reg = ExtraTreesRegressor(n_estimators = 10, max_depth = 10).fit(trainX, trainY)



    #AdaBoostRegression

#     from sklearn.ensemble import AdaBoostRegressor

#     from sklearn.tree import DecisionTreeRegressor

#     from sklearn.ensemble import RandomForestRegressor

#     reg = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

#     reg = AdaBoostRegressor(base_estimator = RandomForestRegressor(n_estimators = 10, max_depth = 10), n_estimators = 10).fit(trainX, trainY)



    #BaggingRegression

    from sklearn.ensemble import BaggingRegressor

#     from sklearn.tree import DecisionTreeRegressor

    from sklearn.ensemble import RandomForestRegressor

#     reg = BaggingRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    reg = BaggingRegressor(base_estimator = RandomForestRegressor(n_estimators = 25, max_depth = 10), n_estimators = 25).fit(trainX, trainY)

    print("Training #1 Complete!")

    return reg
@ignore_warnings(category=ConvergenceWarning)

def agent2_train(trainX, trainY, scaler, pca):

    trainX = scaler.transform(trainX)

#     trainX = pca.transform(trainX)

    trainY = np.ravel(trainY)

    

    #LinearRegression

#     from sklearn.linear_model import LinearRegression

#     reg = LinearRegression().fit(trainX, trainY) 

    

    #RidgeRegression

#     from sklearn.linear_model import Ridge

#     reg = Ridge(alpha=0.5).fit(trainX, trainY)



    #BayesianRidgeRegression

#     from sklearn.linear_model import BayesianRidge

#     reg = BayesianRidge().fit(trainX, trainY)

    

    #KernelRidgeRegression

#     from sklearn.kernel_ridge import KernelRidge

#     reg = KernelRidge(alpha=1).fit(trainX, trainY)

    

    #SupportVectorRegression

#     from sklearn.svm import SVR

#     from sklearn.model_selection import GridSearchCV

#     param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}

#     reg = GridSearchCV(SVR(kernel = 'rbf', max_iter=10000, cache_size = 2048), cv = 5, param_grid= param_grid, refit=True)

#     reg.fit(trainX, trainY)

#     print(reg.best_params_)

#     reg = reg.best_estimator_

    

    #DecisionTreeRegression

#     from sklearn.tree import DecisionTreeRegressor

#     reg = DecisionTreeRegressor(max_depth = 10).fit(trainX, trainY)

    

    #RandomForestRegression

#     from sklearn.ensemble import RandomForestRegressor

#     reg = RandomForestRegressor(n_estimators = 100, max_depth = 10).fit(trainX, trainY)

    

    #ExtraTreesRegression [BANNED -> just see the results]

#     from sklearn.ensemble import ExtraTreesRegressor

#     reg = ExtraTreesRegressor(n_estimators = 10, max_depth = 10).fit(trainX, trainY)



    #AdaBoostRegression

    from sklearn.ensemble import AdaBoostRegressor

#     from sklearn.tree import DecisionTreeRegressor

    from sklearn.ensemble import RandomForestRegressor

#     reg = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    reg = AdaBoostRegressor(base_estimator = RandomForestRegressor(n_estimators = 25, max_depth = 10), n_estimators = 25).fit(trainX, trainY)



    #BaggingRegression

#     from sklearn.ensemble import BaggingRegressor

#     from sklearn.tree import DecisionTreeRegressor

#     from sklearn.ensemble import RandomForestRegressor

#     reg = BaggingRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

#     reg = BaggingRegressor(base_estimator = RandomForestRegressor(n_estimators = 10, max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    print("Training #2 Complete!")

    return reg
@ignore_warnings(category=ConvergenceWarning)

def agent3_train(trainX, trainY, scaler, pca):

    trainX = scaler.transform(trainX)

#     trainX = pca.transform(trainX)

    trainY = np.ravel(trainY)

    

    #LinearRegression

#     from sklearn.linear_model import LinearRegression

#     reg = LinearRegression().fit(trainX, trainY) 

    

    #RidgeRegression

#     from sklearn.linear_model import Ridge

#     reg = Ridge(alpha=0.5).fit(trainX, trainY)



    #BayesianRidgeRegression

#     from sklearn.linear_model import BayesianRidge

#     reg = BayesianRidge().fit(trainX, trainY)

    

    #KernelRidgeRegression

#     from sklearn.kernel_ridge import KernelRidge

#     reg = KernelRidge(alpha=1).fit(trainX, trainY)

    

    #SupportVectorRegression

#     from sklearn.svm import SVR

#     from sklearn.model_selection import GridSearchCV

#     param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}

#     reg = GridSearchCV(SVR(kernel = 'rbf', max_iter=10000, cache_size = 2048), cv = 5, param_grid= param_grid, refit=True)

#     reg.fit(trainX, trainY)

#     print(reg.best_params_)

#     reg = reg.best_estimator_

    

    #DecisionTreeRegression

#     from sklearn.tree import DecisionTreeRegressor

#     reg = DecisionTreeRegressor(max_depth = 10).fit(trainX, trainY)

    

    #RandomForestRegression

#     from sklearn.ensemble import RandomForestRegressor

#     reg = RandomForestRegressor(n_estimators = 100, max_depth = 10).fit(trainX, trainY)

    

    #ExtraTreesRegression [BANNED -> just see the results]

#     from sklearn.ensemble import ExtraTreesRegressor

#     reg = ExtraTreesRegressor(n_estimators = 10, max_depth = 10).fit(trainX, trainY)



    #AdaBoostRegression

    from sklearn.ensemble import AdaBoostRegressor

#     from sklearn.tree import DecisionTreeRegressor

    from sklearn.ensemble import RandomForestRegressor

#     reg = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    reg = AdaBoostRegressor(base_estimator = RandomForestRegressor(n_estimators = 25, max_depth = 10), n_estimators = 25).fit(trainX, trainY)



    #BaggingRegression

#     from sklearn.ensemble import BaggingRegressor

#     from sklearn.tree import DecisionTreeRegressor

#     from sklearn.ensemble import RandomForestRegressor

#     reg = BaggingRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

#     reg = BaggingRegressor(base_estimator = RandomForestRegressor(n_estimators = 10, max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    print("Training #3 Complete!")

    return reg
@ignore_warnings(category=ConvergenceWarning)

def agent4_train(trainX, trainY, scaler, pca):

    trainX = scaler.transform(trainX)

#     trainX = pca.transform(trainX)

    trainY = np.ravel(trainY)

    

    #LinearRegression

#     from sklearn.linear_model import LinearRegression

#     reg = LinearRegression().fit(trainX, trainY) 

    

    #RidgeRegression

#     from sklearn.linear_model import Ridge

#     reg = Ridge(alpha=0.5).fit(trainX, trainY)



    #BayesianRidgeRegression

#     from sklearn.linear_model import BayesianRidge

#     reg = BayesianRidge().fit(trainX, trainY)

    

    #KernelRidgeRegression

#     from sklearn.kernel_ridge import KernelRidge

#     reg = KernelRidge(alpha=1).fit(trainX, trainY)

    

    #SupportVectorRegression

#     from sklearn.svm import SVR

#     from sklearn.model_selection import GridSearchCV

#     param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}

#     reg = GridSearchCV(SVR(kernel = 'rbf', max_iter=10000, cache_size = 2048), cv = 5, param_grid= param_grid, refit=True)

#     reg.fit(trainX, trainY)

#     print(reg.best_params_)

#     reg = reg.best_estimator_

    

    #DecisionTreeRegression

#     from sklearn.tree import DecisionTreeRegressor

#     reg = DecisionTreeRegressor(max_depth = 10).fit(trainX, trainY)

    

    #RandomForestRegression

#     from sklearn.ensemble import RandomForestRegressor

#     reg = RandomForestRegressor(n_estimators = 100, max_depth = 10).fit(trainX, trainY)

    

    #ExtraTreesRegression [BANNED -> just see the results]

#     from sklearn.ensemble import ExtraTreesRegressor

#     reg = ExtraTreesRegressor(n_estimators = 10, max_depth = 10).fit(trainX, trainY)



    #AdaBoostRegression

    from sklearn.ensemble import AdaBoostRegressor

#     from sklearn.tree import DecisionTreeRegressor

    from sklearn.ensemble import RandomForestRegressor

#     reg = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    reg = AdaBoostRegressor(base_estimator = RandomForestRegressor(n_estimators = 25, max_depth = 10), n_estimators = 25).fit(trainX, trainY)



    #BaggingRegression

#     from sklearn.ensemble import BaggingRegressor

#     from sklearn.tree import DecisionTreeRegressor

#     from sklearn.ensemble import RandomForestRegressor

#     reg = BaggingRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

#     reg = BaggingRegressor(base_estimator = RandomForestRegressor(n_estimators = 10, max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    print("Training #4 Complete!")

    return reg
@ignore_warnings(category=ConvergenceWarning)

def agent5_train(trainX, trainY, scaler, pca):

    trainX = scaler.transform(trainX)

#     trainX = pca.transform(trainX)

    trainY = np.ravel(trainY)

    

    #LinearRegression

#     from sklearn.linear_model import LinearRegression

#     reg = LinearRegression().fit(trainX, trainY) 

    

    #RidgeRegression

#     from sklearn.linear_model import Ridge

#     reg = Ridge(alpha=0.5).fit(trainX, trainY)



    #BayesianRidgeRegression

#     from sklearn.linear_model import BayesianRidge

#     reg = BayesianRidge().fit(trainX, trainY)

    

    #KernelRidgeRegression

#     from sklearn.kernel_ridge import KernelRidge

#     reg = KernelRidge(alpha=1).fit(trainX, trainY)

    

    #SupportVectorRegression

#     from sklearn.svm import SVR

#     from sklearn.model_selection import GridSearchCV

#     param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}

#     reg = GridSearchCV(SVR(kernel = 'rbf', max_iter=10000, cache_size = 2048), cv = 5, param_grid= param_grid, refit=True)

#     reg.fit(trainX, trainY)

#     print(reg.best_params_)

#     reg = reg.best_estimator_

    

    #DecisionTreeRegression

#     from sklearn.tree import DecisionTreeRegressor

#     reg = DecisionTreeRegressor(max_depth = 10).fit(trainX, trainY)

    

    #RandomForestRegression

#     from sklearn.ensemble import RandomForestRegressor

#     reg = RandomForestRegressor(n_estimators = 100, max_depth = 10).fit(trainX, trainY)

    

    #ExtraTreesRegression [BANNED -> just see the results]

#     from sklearn.ensemble import ExtraTreesRegressor

#     reg = ExtraTreesRegressor(n_estimators = 10, max_depth = 10).fit(trainX, trainY)



    #AdaBoostRegression

    from sklearn.ensemble import AdaBoostRegressor

#     from sklearn.tree import DecisionTreeRegressor

    from sklearn.ensemble import RandomForestRegressor

#     reg = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    reg = AdaBoostRegressor(base_estimator = RandomForestRegressor(n_estimators = 25, max_depth = 10), n_estimators = 25).fit(trainX, trainY)



    #BaggingRegression

#     from sklearn.ensemble import BaggingRegressor

#     from sklearn.tree import DecisionTreeRegressor

#     from sklearn.ensemble import RandomForestRegressor

#     reg = BaggingRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

#     reg = BaggingRegressor(base_estimator = RandomForestRegressor(n_estimators = 10, max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    print("Training #5 Complete!")

    return reg
@ignore_warnings(category=ConvergenceWarning)

def agent6_train(trainX, trainY, scaler, pca):

    trainX = scaler.transform(trainX)

#     trainX = pca.transform(trainX)

    trainY = np.ravel(trainY)

    

    #LinearRegression

#     from sklearn.linear_model import LinearRegression

#     reg = LinearRegression().fit(trainX, trainY) 

    

    #RidgeRegression

#     from sklearn.linear_model import Ridge

#     reg = Ridge(alpha=0.5).fit(trainX, trainY)



    #BayesianRidgeRegression

#     from sklearn.linear_model import BayesianRidge

#     reg = BayesianRidge().fit(trainX, trainY)

    

    #KernelRidgeRegression

#     from sklearn.kernel_ridge import KernelRidge

#     reg = KernelRidge(alpha=1).fit(trainX, trainY)

    

    #SupportVectorRegression

#     from sklearn.svm import SVR

#     from sklearn.model_selection import GridSearchCV

#     param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}

#     reg = GridSearchCV(SVR(kernel = 'rbf', max_iter=10000, cache_size = 2048), cv = 5, param_grid= param_grid, refit=True)

#     reg.fit(trainX, trainY)

#     print(reg.best_params_)

#     reg = reg.best_estimator_

    

    #DecisionTreeRegression

#     from sklearn.tree import DecisionTreeRegressor

#     reg = DecisionTreeRegressor(max_depth = 10).fit(trainX, trainY)

    

    #RandomForestRegression

#     from sklearn.ensemble import RandomForestRegressor

#     reg = RandomForestRegressor(n_estimators = 100, max_depth = 10).fit(trainX, trainY)

    

    #ExtraTreesRegression [BANNED -> just see the results]

#     from sklearn.ensemble import ExtraTreesRegressor

#     reg = ExtraTreesRegressor(n_estimators = 10, max_depth = 10).fit(trainX, trainY)



    #AdaBoostRegression

    from sklearn.ensemble import AdaBoostRegressor

#     from sklearn.tree import DecisionTreeRegressor

    from sklearn.ensemble import RandomForestRegressor

#     reg = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    reg = AdaBoostRegressor(base_estimator = RandomForestRegressor(n_estimators = 25, max_depth = 10), n_estimators = 25).fit(trainX, trainY)



    #BaggingRegression

#     from sklearn.ensemble import BaggingRegressor

#     from sklearn.tree import DecisionTreeRegressor

#     from sklearn.ensemble import RandomForestRegressor

#     reg = BaggingRegressor(base_estimator = DecisionTreeRegressor(max_depth = 10), n_estimators = 10).fit(trainX, trainY)

#     reg = BaggingRegressor(base_estimator = RandomForestRegressor(n_estimators = 10, max_depth = 10), n_estimators = 10).fit(trainX, trainY)

    print("Training #6 Complete!")

    return reg
# train_input = pd.read_csv("/kaggle/input/bits-f464-l1/train.csv")



# train_agents = distribute_df(train_check)



# train = [check_input(train_agents[i], False) for i in range(7)]



# for i in range(5,6):

#     sns.scatterplot(x=train[i][:2000]['time'], y=train[i][:2000]['label'])
train_input = pd.read_csv("/kaggle/input/bits-f464-l1/train.csv")



train_agents = distribute_df(train_check)



train = [check_input(train_agents[i], False) for i in range(7)]



f = [agent0_train, agent1_train, agent2_train, agent3_train, agent4_train, agent5_train, agent6_train]



splitsize = 0.01

correlation_thresholds = [0.2,0.25,0.2,0.35,0.2,0.25,0.25]

powers = [15,15,15,15,15,15,15]

pca_components = [5, 5, 5, 5, 5, 5, 5]



train = [add_features(train[i], correlation_thresholds[i], powers[i]) for i in range(7)]



features = [agent_features(train[i], correlation_thresholds[i]) for i in range(7)]



train_val_split = [agent_split(train[i], features[i], splitsize) for i in range(7)]



scalers = [agent_scalers(train_val_split[i]['trainX']) for i in range(7)]



pca = [agent_pcas(train_val_split[i]['trainX'], scalers[i], pca_components[i]) for i in range(7)]



reg = [f[i](train_val_split[i]['trainX'], train_val_split[i]['trainY'], scalers[i], pca[i]) for i in range(7)]



predictions = [predict(reg[i], train_val_split[i]['valX'], scalers[i], pca[i]) for i in range(7)]



rmse_scores = [rmse_score(predictions[i], train_val_split[i]['valY']) for i in range(7)]



r2_scores = [r2_score(reg[i], train_val_split[i]['valX'],  train_val_split[i]['valY'], scalers[i], pca[i]) for i in range(7)]



rmse = 0

r2 = 0



for i in range(7):

    print("Agent#", str(i))

    print("RMSE: ", rmse_scores[i])

    print("R2: ", r2_scores[i])

    rmse += rmse_scores[i]

    r2 += r2_scores[i]

    

print("RMSE: ", rmse/7, "R2: ", r2/7)
def add_features_test(columns, x, features, powers):

        

    l = np.empty(len(features))

    old_features = np.array([x[i] for i in range(len(columns))])

    

    for i,s in enumerate(features):

        f, p = s.split('^')

        f = f[1:]

        l[i] = old_features[int(f)]**int(p)

        

    return l
test_input = pd.read_csv("/kaggle/input/bits-f464-l1/test.csv")



columns = list(test_input.columns)

columns.remove('id')

for i in range(7):

    columns.remove('a'+str(i))

if 'label' in  columns:

    columns.remove('label')





ids = np.empty(len(test_input))

labels = np.empty(len(test_input))



for i in range(len(test_input)):

    for j in range(7):

        if(test_input.iloc[i]['a'+str(j)] == 1):

            z = np.array(test_input.iloc[i][columns])

            x = add_features_test(columns, z, features[j], powers[j])

            y = predict(reg[j], np.array(x.reshape(1,-1)), scalers[j], pca[j])

            ids[i]= test_input.iloc[i]['id']

            labels[i] = y[0]

            break



final = pd.DataFrame(list(zip(ids, labels)), columns =['id', 'label'])   

final['id'] = final['id'].apply(lambda x: int(x))
final.to_csv('submission.csv',index=False)

from IPython.display import FileLink

FileLink('submission.csv')