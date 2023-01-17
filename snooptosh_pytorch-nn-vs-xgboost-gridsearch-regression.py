import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

from scipy.stats import shapiro

from sklearn.feature_selection import RFE

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.dummy import DummyRegressor

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV

from sklearn.metrics import explained_variance_score, r2_score, mean_absolute_error, mean_squared_error

import xgboost

from xgboost import XGBRegressor, plot_importance

from collections import Counter

import torch

from torch import nn

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



import os

for dirname, _, filenames in os.walk('/kaggle/input/autompg-dataset'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/autompg-dataset/auto-mpg.csv')

data 

# number of data points - 398
data.columns=['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']



data.replace({'?':np.nan},inplace=True)



data['horsepower'] = pd.to_numeric(data['horsepower'])

data.dtypes ## horsepower is object, change to numeric
data.describe()

plt.figure(figsize=(15,10))

sns.heatmap(data.corr(),annot=True,annot_kws={"size":12})
plt.hist(data['model year'],histtype='bar')
plt.hist(data['origin'],histtype='bar')
plt.hist(data['mpg'],histtype='bar')
data.boxplot(column=['horsepower'])

data.boxplot(column=['acceleration'])

# Outlier detection - courtesy of Yassine's https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling



def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 2 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers
drop = detect_outliers(data,0,['mpg','cylinders','displacement','horsepower','weight','acceleration'])

data = data.drop(drop, axis = 0).reset_index(drop=True)
data['car make'] = data['car name']

data['car make'] = data['car name'].apply(lambda x: x.split()[0]) 

data.drop(columns=['car name'],inplace=True)

data = pd.get_dummies(data,columns=['car make'])

data['mpg'] = np.log(1 + 100*data['mpg'])



X = data.drop(columns=['mpg'])

y = data['mpg']



imp = SimpleImputer(missing_values=np.nan,strategy='median')

X['horsepower'] = imp.fit(X['horsepower'].values.reshape(-1, 1)).transform(X['horsepower'].values.reshape(-1, 1))

print(X.shape)

print(y.shape)

X.head()
xtrain,xtest,ytrain,ytest = train_test_split(X, y, test_size=0.3, random_state=42)

xgbr = XGBRegressor()



xgb_params = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'learning_rate': [.03, 0.05, .07], 

              'max_depth': [5, 6],

              'min_child_weight': [4],

              'subsample': [0.7],

              'colsample_bytree': [0.7]

              }



gsXGB = GridSearchCV(xgbr, xgb_params, cv = 5, scoring='neg_mean_squared_error', 

                     refit=True, n_jobs = 5, verbose=True)

gsXGB.fit(xtrain,ytrain)



XGB_best = gsXGB.best_estimator_



gsXGB.best_score_
ypred = XGB_best.predict(xtest)

explained_variance_score(ytest,ypred)

mean_absolute_error(ytest,ypred)

print(f"Mean Squared using XGBoost model -> {mean_squared_error(ytest,ypred,squared=True)}")
plot_importance(XGB_best)
# Define network dimensions

n_input_dim = xtrain.shape[1]

# Layer size

n_hidden = 4 # Number of hidden nodes

n_output = 1 # Number of output nodes for predicted mpg



# Build mdel

torch_model = torch.nn.Sequential(

    torch.nn.Linear(n_input_dim, n_hidden),

    torch.nn.ELU(),

    torch.nn.Linear(n_hidden, n_output)

)

    

print(torch_model)

loss_func = torch.nn.MSELoss() # Mean Squared Error as Loss metric

learning_rate = 0.02 # play with learning rate

optimizer = torch.optim.Adam(torch_model.parameters(), lr=learning_rate)
train_error = []

iters = 1000



Y_train_t = torch.FloatTensor(ytrain.values).reshape(-1,1) #Converting numpy array to torch tensor



for i in range(iters):

    X_train_t = torch.FloatTensor(xtrain.values)  #Converting numpy array to torch tensor

    y_hat = torch_model(X_train_t)

    loss = loss_func(y_hat, Y_train_t)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()



    train_error.append(loss.item())

    

fig, ax = plt.subplots(2, 1, figsize=(12,8))

ax[0].plot(train_error)

ax[0].set_ylabel('Loss')

ax[0].set_title('Training Loss')
X_test_t = torch.FloatTensor(xtest.values)

ypredict = torch_model(X_test_t)

print(f"Mean Squared Error using PyTorch Basic NN model -> {mean_squared_error(ytest,ypredict.detach().numpy(),squared=True)}")
