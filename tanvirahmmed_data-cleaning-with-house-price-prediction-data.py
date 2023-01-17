import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings("ignore")

%matplotlib inline
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from scipy.stats import skew

from scipy import stats

from scipy.stats import boxcox

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression, f_regression

from sklearn import ensemble

from sklearn.metrics import mean_squared_error

df = pd.read_csv('/kaggle/input/house-prices-data/train.csv')

dt = pd.read_csv('/kaggle/input/house-prices-data/test.csv')
df.shape,dt.shape
(((df.isnull().sum())*100)/len(df)).sort_values(

            ascending = False, kind = 'mergesort').head(15)
(((dt.isnull().sum())*100)/len(dt)).sort_values(

            ascending = False, kind = 'mergesort').head(30)
df.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis=1,inplace=True)

dt.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis=1,inplace=True)
y = df['SalePrice']

df.drop(['SalePrice'], axis = 1, inplace = True)

data = pd.concat([df,dt], axis = 0)

data.shape
data.describe()
year_all = ['YearBuilt', 'YearRemodAdd','YrSold','MoSold','GarageYrBlt']

for i in data:

  if (data[i].dtypes == object and i != 'FireplaceQu') or i in year_all:

    data[i] = data[i].fillna(data[i].mode()[0])
data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(data['BsmtHalfBath'].mode()[0])

data['BsmtFullBath'] = data['BsmtFullBath'].fillna(data['BsmtFullBath'].mode()[0])

data['GarageCars'] = data['GarageCars'].fillna(data['GarageCars'].mode()[0])
for i in data:

  if data[i].dtypes != object:

     data[i] = data[i].fillna(data[i].mean())

     data[i] = data[i].astype('float64')
# Replace Null using Classification



classifiers = [

    LogisticRegression(),

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.01, probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(ccp_alpha=0.0,max_depth=20, n_estimators=4500),

    GaussianNB(),

    GradientBoostingClassifier(learning_rate=0.03),

    AdaBoostClassifier(learning_rate=0.6)]

#Using only numerical features

fireplace_data = data.select_dtypes(exclude=['object'])

fireplace_data['FireplaceQu'] = data['FireplaceQu'].copy()



def classify_missing(all_int_dataTe):

  fire_train = pd.DataFrame()

  fire_test = pd.DataFrame()

  null_row_list = list(all_int_dataTe[all_int_dataTe['FireplaceQu'].isnull()].index.tolist())

  col = list(all_int_dataTe.columns)

  k = -1

  for j in null_row_list:

    try:

      k+=1 

      null_row_value = all_int_dataTe.iloc[j]

      null_row_value = list(null_row_value.values)

      for i in range(len(null_row_value)):

        fire_test.loc[k,col[i]] = null_row_value[i]

    except:

      continue

  fire_train = all_int_dataTe.dropna()

  fire_test = fire_test.drop(['FireplaceQu'],axis = 1)



  f_train = fire_train.drop(['FireplaceQu'],axis = 1)

  y_train = fire_train['FireplaceQu']



  

  le = preprocessing.LabelEncoder()

  le.fit(y_train)

  y_train = le.transform(y_train)



  

  X_train, X_test, y_train, y_test = train_test_split(f_train, y_train, test_size=0.1, random_state=1)

  r = 0

  for clf in classifiers:

    clf.fit(X_train,y_train)

    print(clf.__class__.__name__,' ', round(clf.score(X_test, y_test) * 100, 2))

    if r < clf.score(X_test, y_test):

      r = clf.score(X_test, y_test)

      model = clf

  print(model)

  model.fit(X_train,y_train)

  Y_prediction = model.predict(fire_test)

  res = le.inverse_transform(Y_prediction)



  for i in range(len(null_row_list)):

    try:

      all_int_dataTe.loc[null_row_list[i],'FireplaceQu'] = res[i]

    except:

      print('error')

  return all_int_dataTe['FireplaceQu']

data['FireplaceQu'] = classify_missing(fireplace_data)
(((data.isnull().sum())*100)/len(data)).sort_values(

            ascending = False, kind = 'mergesort').head(15)