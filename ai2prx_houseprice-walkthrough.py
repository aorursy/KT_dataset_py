import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combine = pd.concat([train,test], axis=0, ignore_index=True)
combine.head()
corrmat = combine.corr()
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corrmat, vmax=.8, square=True);
col = ['SalePrice', 'YearRemodAdd', 'YearBuilt', 'TotalBsmtSF', 'FullBath',
       'TotRmsAbvGrd', 'OverallQual', 'GrLivArea', 'GarageArea', 'GarageCars', '1stFlrSF']
combine = combine[col]
combine.head()
combine.isnull().sum().sort_values(ascending=False)
drop_index1 = combine[combine['GarageCars'].isnull()].index.tolist()
drop_index2 = combine[combine['TotalBsmtSF'].isnull()].index.tolist()
combine.drop(drop_index1, inplace=True)
combine.drop(drop_index2, inplace=True)
train = combine.loc[:1459, :]
test = combine.loc[1460:, :]
X = train.iloc[:, 1:]
y = train.iloc[:, [0]]
X = pd.get_dummies(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
y_sc = StandardScaler()
X_train = X_sc.fit_transform(X_train)
X_test = X_sc.transform(X_test)
y_train = y_sc.fit_transform(y_train)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

regressors = [
    LinearRegression(),
    SVR(kernel="rbf"),
    DecisionTreeRegressor(random_state=0),
    RandomForestRegressor(n_estimators=10, random_state=0),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]
for reg in regressors:
    reg.fit(X_train, y_train)
    
    
    print('****Results****')
    y_pred = reg.predict(X_test)
    print("Accuracy: {:.4%}")
    
print("="*30)



