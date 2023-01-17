#House Pricing Kaggle Cometition
import numpy as np

import pandas as pd

import seaborn as sns



import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import VotingRegressor

from sklearn.model_selection import learning_curve
#Load in data

test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



train_len = len(train_df)

Resulting_Id = test_df['Id']
train_df.head()
train_df['SalePrice'].isnull().sum()
#Create a function to remove outliers from dataset using IQR

def get_outliers(data, n, features):

    outliers = []

    

    for feat in features:

        Q1 = np.percentile(data[feat], 25)

        

        Q3 = np.percentile(data[feat], 75)

        

        IQR = Q3 - Q1

        

        outlier = IQR * 1.5

        outlier_to_lst = data[(data[feat] < Q1 - outlier) | (data[feat] > Q3 + outlier)].index

        

        outliers.extend(outlier_to_lst)

        

    outlier_cnt = Counter(outlier_to_lst)

    outliers = list(j for j, v in outlier_cnt.items() if v > n)

        

    return outliers
outliers = get_outliers(train_df, 2, ['SalePrice', 'LotFrontage', 'LotArea'])

outliers
train_df.shape
test_df.shape
data = pd.concat(objs = [train_df, test_df], sort = False).reset_index(drop=True)

data.shape
#Since we have so many columnns lets see which are categorical and which are numerical



#Numerical Features

num_feats = data.dtypes[data.dtypes != 'object'].index



#Categorical Features 

cat_feats = data.dtypes[data.dtypes == 'object'].index



print('Numerical Features: ', num_feats)

print('Categorical Features: ', cat_feats)
data.head()
print(data['Street'].value_counts())

print(data['Alley'].value_counts())
data = pd.get_dummies(data, columns = ['Street'], prefix = 'St_')
data.head()
data['LotShape'].value_counts()
data = pd.get_dummies(data, columns = ['LotShape'], prefix = 'LS_')
data.head()
data['1stFlrSF'].head()
data['2ndFlrSF'].head()
data['LotArea'].head()
data['HouseSF'] = data['1stFlrSF'] + data['2ndFlrSF']

data.head()
data['MSZoning'].value_counts()
#Replace Null with the most common zoning

data['MSZoning'] = data['MSZoning'].fillna('RL')

data['MSZoning'].value_counts()
#Now make dummie variables

data = pd.get_dummies(data, columns = ['MSZoning'], prefix = 'Zone_')

data.head()
data['Alley'].value_counts()
data['Alley'].isnull().sum()
data.head()
data['SalePrice']
data = data.interpolate(method = 'linear')
data['SalePrice']
data.head()

data['Alley'] = data['Alley'].fillna('Grvl')

data['Alley'].value_counts()
#data = pd.get_dummies(data, columns = ['LotConfig'], prefix = 'Lot_')



data.head()
data = pd.get_dummies(data)
data.head()
data.isnull().sum()
train = data[:train_len]

test = data[train_len:]



#Drop SalePrice from features

train_X = train.drop(columns = 'SalePrice', axis = 1)

test = test.drop(columns = 'SalePrice', axis = 1)



#Get our target

target = train['SalePrice']
train
train_X.shape
target.shape
target
gbr = GradientBoostingRegressor()



gbr_params = {'loss': ['ls'],

             'n_estimators': [100, 200, 300],

             'learning_rate': [0.1, 0.05, 0.01],

             'max_depth': [3, 8]}



best_gbr = GridSearchCV(gbr, param_grid = gbr_params, cv = 5, n_jobs = -1)

best_gbr.fit(train_X, target)



print(best_gbr.best_score_)



best_gbr_model = best_gbr.best_estimator_
rf = RandomForestRegressor(random_state = 42)



rf_grid_params = {'n_estimators': [10, 50, 100, 200, 300],

                 'max_depth': [3, 4, 6, 8],

                 'criterion': ['mse']}



best_rf = GridSearchCV(rf, param_grid = rf_grid_params, cv = 5, n_jobs = -1)

best_rf.fit(train_X, target)



print(best_rf.best_score_)



#Get best estimator from the grid search results

best_rf_model = best_rf.best_estimator_
def plot_loss_curves(model, title, X, y, ylim = None, train_sizes = np.linspace(0.1, 1.0, 2)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

        

    #Axes titles

    plt.xlabel('Training Examples')

    plt.ylabel('Loss')

    

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes = train_sizes)

    

    #Get training scores mean and std

    train_scores_mean = np.mean(train_scores, axis = 1)

    train_scores_std = np.std(train_scores, axis = 1)

    

    #Get test scores mean and std

    test_scores_mean = np.mean(test_scores, axis = 1)

    test_scores_std = np.std(test_scores, axis = 1)

    

    plt.grid()

    

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color = 'b')

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.1, color = 'r')

    

    #Now plot the training and cross val loss curves

    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'b', label = 'Training_Loss')

    plt.plot(train_sizes, test_scores_mean, 'o-', color = 'r', label = 'Test_Loss')

    

    return plt

pl = plot_loss_curves(best_gbr_model, 'Gradient Boosting Regressor', train_X, target)

pl = plot_loss_curves(best_rf_model, 'RF Regressor', train_X, target)
#Voting Regressor

vr = VotingRegressor([('gbr', best_gbr_model), ('rf', best_rf_model)]).fit(train_X, target)

#Make sales price predictions

sales_price = vr.predict(test)



#Create results csv file

results = pd.DataFrame(Resulting_Id)

results['SalePrice'] = sales_price



results.columns = ['Id', 'SalePrice']



results.to_csv('Sales_Price_Pred.csv', index = False)
results