# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, KFold, StratifiedKFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# get the data
datadir = "../input"
filename = 'Admission_Predict.csv'
filePath = datadir + '/' + filename
df = pd.read_csv(filePath)
# just take a glance at the data
df.head()
# basic info, specially to check if there are missing values
df.info()
# drop the Serial No. column, it has no relation to the prediction
df = df.drop(columns=['Serial No.'])
# quickly get distributions of each column (feature)
df.hist(bins=50, figsize=(20,15))

# now let pandas describe the data a bit
df.describe()
# get some correlation
df.corr()
sns.heatmap(df.corr(), annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
# plot the scatter matrix for the promising features
attributes = ["Chance of Admit ", "CGPA", "GRE Score",
              "TOEFL Score"]
scatter_matrix(df[attributes], figsize=(12, 8))
# Creating a new field for categorization of the labels
target_label = "Chance of Admit "
target_label_mult10 = "Chance_cat" # new field, which is a function of the original target
df[target_label_mult10] = np.ceil(df[target_label]*10)
df[target_label_mult10].value_counts()
# use this new field for categorization
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df[target_label_mult10]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
# visualize the category distribution in training set
strat_train_set[target_label_mult10].value_counts() / len(strat_train_set)

# visualize the category distribution in test set
strat_test_set[target_label_mult10].value_counts() / len(strat_test_set)
# drop the new category since its just used to properly stratify the train and test set
for set_ in (strat_train_set, strat_test_set):
    set_.drop(target_label_mult10, axis=1, inplace=True)
strat_train_set.head()
models = []
models.append(('linear SVR',SVR(kernel = 'linear',C=1)))
models.append(('Random Forest', RandomForestRegressor(n_estimators = 100, random_state = 42)))
models.append(('Linear regression',LinearRegression()))



X_train = strat_train_set.copy()
X_train = strat_train_set.drop(target_label,axis=1) # drop the target labels
X_train = X_train.values # get the numpy array from the pandas dataframe (needed for kfold cv)
Y_train = strat_train_set[target_label].copy() 
Y_train = Y_train.values # get the numpy array from the pandas dataframe (needed for kfold cv)


X_test = strat_test_set.copy()
X_test = strat_test_set.drop(target_label,axis=1) # drop the target labels

Y_test = strat_test_set[target_label].copy() 


kfold = KFold(n_splits=3, shuffle=True)

rmse_trainAll = []
#rmse_testAll = []
names = []
for name, model in models:
    rmse_train = []
    rmse_test = []
    print('Model: ',name)
    names.append(name)
    # k-fold cross validation on the training set
    for train, test in kfold.split(X_train, Y_train):

        model.fit(X_train[train],Y_train[train])
        y_train_pred = model.predict(X_train[test])

        # train set RMSE prediction
        svr_mse = mean_squared_error(Y_train[test],y_train_pred)
        svr_rmse_train = np.sqrt(svr_mse)
        #print("Train set RMSE: ", svr_rmse_train)  
        rmse_train.append(svr_rmse_train)


    rmse_trainAll.append(rmse_train)
    print('Training RMSE mean (standard deviation): {:.3f}({:.3f})'.format(np.mean(rmse_train),np.std(rmse_train)))


    y_test_pred = model.predict(X_test)
    # test set RMSE prediction
    svr_mse = mean_squared_error(Y_test,y_test_pred)
    svr_rmse_test = np.sqrt(svr_mse)
    print("Test set RMSE: {:.3f} ".format(svr_rmse_test))
    
    print('\n')
    

# get the boxplot of the RMSE scores on the training set
plotName = 'RMSE scores of different models on the cross-validation set'
# define the figure size
fig_size = plt.rcParams["figure.figsize"]
#print(fig_size)
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size


plt.boxplot(rmse_trainAll)
plt.title(plotName)
plt.xticks(range(1,len(names)+1), names)
plt.grid()
# get the cropped data
cropped_df_train = strat_train_set[['GRE Score','TOEFL Score','CGPA']]
cropped_df_train.head()

cropped_df_test = strat_test_set[['GRE Score','TOEFL Score','CGPA']]
cropped_df_test.head()
# test including only the important features 
X_train = cropped_df_train.copy()
X_train = X_train.values # get the numpy array from the pandas dataframe (needed for kfold cv)
X_test = cropped_df_test.copy()

Y_train = strat_train_set[target_label].copy() 
Y_train = Y_train.values # get the numpy array from the pandas dataframe (needed for kfold cv)
Y_test = strat_test_set[target_label].copy() 


print('Number of features in training set: ',X_train.shape[1])
print('Number of features in test set: ',X_test.shape[1])


kfold = KFold(n_splits=3, shuffle=True)

rmse_trainAll = []
#rmse_testAll = []
names = []
for name, model in models:
    rmse_train = []
    rmse_test = []
    print('Model: ',name)
    names.append(name)
    # k-fold cross validation on the training set
    for train, test in kfold.split(X_train, Y_train):

        model.fit(X_train[train],Y_train[train])
        y_train_pred = model.predict(X_train[test])

        # train set RMSE prediction
        svr_mse = mean_squared_error(Y_train[test],y_train_pred)
        svr_rmse_train = np.sqrt(svr_mse)
        #print("Train set RMSE: ", svr_rmse_train)  
        rmse_train.append(svr_rmse_train)


    rmse_trainAll.append(rmse_train)
    print('Training RMSE mean (standard deviation): {:.3f}({:.3f})'.format(np.mean(rmse_train),np.std(rmse_train)))


    y_test_pred = model.predict(X_test)
    # test set RMSE prediction
    svr_mse = mean_squared_error(Y_test,y_test_pred)
    svr_rmse_test = np.sqrt(svr_mse)
    print("Test set RMSE: {:.3f} ".format(svr_rmse_test))
    
    print('\n')
    




# get the boxplot of the RMSE scores on the training set
plotName = 'RMSE scores, on cross-validation, important features only'
# define the figure size
fig_size = plt.rcParams["figure.figsize"]
#print(fig_size)
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size


plt.boxplot(rmse_trainAll)
plt.title(plotName)
plt.xticks(range(1,len(names)+1), names)
plt.grid()
# get the whole training set
X_train = strat_train_set.copy()
X_train = strat_train_set.drop(target_label,axis=1) # drop the target labels
Y_train = strat_train_set[target_label].copy() 


model = LinearRegression() # this model seems to be precise and robust
model.fit(X_train,Y_train)

# get the clean test set again
X_test = strat_test_set.copy()
X_test = strat_test_set.drop(target_label,axis=1) # drop the target labels
Y_test = strat_test_set[target_label].copy() 

# make predictions
y_test_pred = model.predict(X_test)
# see the plot
plt.scatter(Y_test,y_test_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Visualizing the model performance')
plt.grid()