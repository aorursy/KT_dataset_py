import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = 10, 7

import seaborn as sns
%matplotlib inline
# Read train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.info()
test_df.info()
train_missing_columns = train_df.isnull().any().sum()
test_missing_columns = test_df.isnull().any().sum()
print("Train : Number of features with missing values {}".format(train_missing_columns))
print("Test : Number of features with missing values {}".format(test_missing_columns))
plt.figure(figsize=(10,7))
sns.distplot(train_df["target"],kde=True)
plt.show()
plt.figure(figsize=(10,7))
sns.distplot(np.log(train_df["target"]),kde=True)
plt.title("Train set")
plt.show()
train_df["target"] = np.log(train_df["target"])
# TRAIN
raw_lines = train_df.shape[0]
print("Before filtering : " + str(raw_lines))
# Compute IQR
Q1 = train_df['target'].quantile(0.25)
Q3 = train_df['target'].quantile(0.75)
IQR = Q3 - Q1

train = train_df[(train_df['target'] < (Q3 + 1.5 * IQR)) & (train_df['target'] > (Q1 - 1.5 * IQR))]
clean_lines = train.shape[0]
print("After filtering : " + str(clean_lines))
print("We lost {} lines with the IQR filter".format(raw_lines - clean_lines))
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=None, svd_solver="full")
# ID column is a string, and remove the target feature
cols_pca = [col for col in train.columns if col not in ['ID', 'target']]
pca.fit(StandardScaler().fit_transform(train.loc[:, cols_pca]))
cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
n_components = 20

plt.figure(figsize=(12, 6))
plt.bar(np.arange(n_components), list(pca.explained_variance_ratio_[:n_components]), align="center",
        color='red', label="Individual explained variance")
plt.step(np.arange(n_components), cum_var_exp[:n_components], where="mid", label="Cumulative explained variance")
plt.xticks(np.arange(n_components))
plt.legend(loc="best")
plt.xlabel("Principal component index", {"fontsize": 14})
plt.ylabel("Explained variance ratio", {"fontsize": 14})
plt.title("PCA on training data", {"fontsize": 16})
for explained_variance in np.arange(0.80, 1.0, 0.05):
    pca = PCA(explained_variance, svd_solver="full")
    # ID column is a string, and remove the target feature
    cols_pca = [col for col in train.columns if col not in ['ID', 'target']]
    pca.fit(StandardScaler().fit_transform(train.loc[:, cols_pca]))
    cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
    print(pca.n_components_)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(0.15, svd_solver="full")
# ID column is a string, and remove the target feature
cols_pca = [col for col in train.columns if col not in ['ID', 'target']]
X_train = pca.fit_transform(StandardScaler().fit_transform(train.loc[:, cols_pca]))
from sklearn.model_selection import train_test_split
y = train["target"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

#Transform the Prediction to the correct form : we reverse Log() with Exp() 
#final_prediction = np.exp(prediction)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Linear Regression Performance on the test set: {}'.format(rmse))
#rmsle = np.sqrt(mean_squared_log_error(y_test, predictions))
#print('Linear Regression Performance on the test set: {}'.format(rmsle))
from sklearn.ensemble import RandomForestRegressor


def rmsle(h, y): 
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y
    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())


def runRF(x_train, y_train,x_test, y_test,test):
    model=RandomForestRegressor(bootstrap=True, max_features=0.75, min_samples_leaf=11, min_samples_split=13, n_estimators=100)
    model.fit(x_train, y_train)
    y_pred_train=model.predict(x_test)
    mse=rmsle(np.exp(y_pred_train)-1,np.exp(y_test)-1)
    y_pred_test=model.predict(test)
    return y_pred_train,mse,y_pred_test

