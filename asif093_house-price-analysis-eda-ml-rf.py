# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns



from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))



import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
test_data.head()
#shape of Train Data

train_data.shape
test_data.shape
# train dataframe info

train_data.info() 
test_data.info()
#Defining Fuction for dropping columns

def  drop_null (data):

    for column in data:

        if data[column].count() / len(data) <= 0.3:

            data.drop(column, axis=1, inplace=True)

            print('Dropped Column', column)
#Calling function created for column having null values

#for Train dataset

drop_null(train_data)



#dropping id

train_data.drop('Id', axis=1,  inplace=True)

print(train_data.shape)
#Calling function created for column having null values

#for Train dataset

drop_null(test_data)



#dropping id

test_data.drop('Id', axis=1,  inplace=True)

print(test_data.shape)
# Filling "NA" others missing data from train data

missing_data_stats = train_data.isnull().sum()

cols = missing_data_stats[missing_data_stats>0].index.tolist()

cat_cols = train_data.select_dtypes(exclude=['int64', 'float64']).columns





for c in cols:

    if c in cat_cols:

        mode = train_data[c].mode()[0]

        train_data[c] = train_data[c].fillna(mode)

    else:

        median = train_data[c].median()

        train_data[c] = train_data[c].fillna(median)
#Filling "NA" others missing data from test data

missing_data_stats = test_data.isnull().sum()

cols = missing_data_stats[missing_data_stats>0].index.tolist()

cat_cols = test_data.select_dtypes(exclude=['int64', 'float64']).columns





for c in cols:

    if c in cat_cols:

        mode = test_data[c].mode()[0]

        test_data[c] = test_data[c].fillna(mode)

    else:

        median = test_data[c].median()

        test_data[c] = test_data[c].fillna(median)
train_data.isnull().sum()
test_data.isnull().sum()
train_data.describe()
print(train_data['SalePrice'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(train_data['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.6});
plt.figure(figsize=(16,8))

sns.boxplot(y='SalePrice', x='OverallQual', data=train_data)

plt.title('Sales Price comaparison with Overall Qality of house')

plt.show()
plt.figure(figsize=(16,8))

sns.boxplot(y='SalePrice', x='OverallCond', data=train_data)

plt.title('Sales Price comaparison with Overall condition of house')

plt.show()
plt.figure(figsize=(16,8))

sns.scatterplot(y='SalePrice', x='LotArea', data=train_data)

plt.show()
plt.figure(figsize=(16,6))

sns.barplot(y='SalePrice', x='YearBuilt', data=train_data)

plt.xticks(rotation=90)

plt.title('Sales Price distribution against Year Built')

plt.show()
#Numerical data distribution

df_num= train_data.select_dtypes(include = ['float64', 'int64'])



df_num.head()
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); 
df_corr = train_data.corr()
plt.figure(figsize=(16,8))

sns.heatmap(df_corr, cmap='viridis')

plt.show()
df_num_corr = df_num.corr()['SalePrice'][:-1] 

golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)

golden_features_list
import operator



individual_features_df = []

for i in range(0, len(df_num.columns) - 1): 

    temp = df_num[[df_num.columns[i], 'SalePrice']]

    temp = temp[temp[df_num.columns[i]] != 0]

    individual_features_df.append(temp)



all_correlations = {feature.columns[0]: feature.corr()['SalePrice'][0] for feature in individual_features_df}

all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))

for (key, value) in all_correlations:

    print("{:>15}: {:>15}".format(key, value))
corr = df_num.drop('SalePrice', axis=1).corr()

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
#Importing Library for Encoding Data

from sklearn.preprocessing import LabelEncoder



#Defining instance for Label Encoder

encode_data = LabelEncoder()
# Defining Function For encoding All Categorical columns

def CaTorigical_data(data):

    for c in cat_cols:

        data[c] = encode_data.fit_transform(data[c])
#Applying Encoding function to train data

CaTorigical_data(train_data)



#Validating encoding

train_data.head()
#Applying Encoding function to test data

CaTorigical_data(test_data)



#Validating encoding

test_data.head()
# Importing Libraries 



from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.metrics import r2_score
X= train_data.drop('SalePrice', axis=1)

ytrain = train_data['SalePrice']

x = test_data
# Scalling data with standard scaler

sc = StandardScaler()

X_ = sc.fit_transform(X)

x_= sc.fit_transform(x)



X = pd.DataFrame(data=X_, columns = X.columns)

x = pd.DataFrame(data=x_, columns = x.columns)

X.head()
x.head() 
xtrain = X

xtest= x

print(xtrain.shape, ytrain.shape, xtest.shape)
rf_regr = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)
rf_regr.fit(xtrain, ytrain)
rf_pred = rf_regr.predict(xtest)
rf_regr.score (xtrain, ytrain)
features_tuple=list(zip(X.columns,rf_regr.feature_importances_))

feature_imp=pd.DataFrame(features_tuple,columns=["Feature Names","Importance"])

feature_imp=feature_imp.sort_values("Importance",ascending=False)
plt.figure(figsize=(20,4))

sns.barplot(x="Feature Names",y="Importance", data=feature_imp, color='g')

plt.xlabel("House Price Features")

plt.ylabel("Importance")

plt.xticks(rotation=90)

plt.title("Random Forest Regressor - Features Importance")

plt.show()
param_grid1 = {"n_estimators" : [9, 18, 27, 36, 45, 54, 63, 72, 81, 90],

           "max_depth" : [1, 5, 10, 15, 20, 25, 30],

           "min_samples_leaf" : [1, 2, 4, 6, 8, 10]}



RF = RandomForestRegressor(random_state=0)

# Instantiate the GridSearchCV object: logreg_cv

RF_cv1 = GridSearchCV(RF, param_grid1, cv=5,scoring='r2',n_jobs=4)



# Fit it to the data

RF_cv1.fit(xtrain,ytrain)



#RF_cv1.cv_results_, 

RF_cv1.best_params_, RF_cv1.best_score_
param_grid2 = {"n_estimators" : [45,48,51,54,57,60,63],

           "max_depth" : [16,17,18,19,20,21,22,23,24],

           "min_samples_leaf" : [1,2,3,4]} 



RF = RandomForestRegressor(random_state=0)

# Instantiate the GridSearchCV object: logreg_cv

RF_cv2 = GridSearchCV(RF, param_grid2, cv=5,scoring='r2',n_jobs=4)



# Fit it to the data

RF_cv2.fit(xtrain,ytrain)



#RF_cv2.grid_scores_, 

RF_cv2.best_params_, RF_cv2.best_score_
RF_tuned = RF_cv2.best_estimator_
RF_tuned.fit(xtrain, ytrain)

RF_tpred = RF_tuned.predict(xtest)
RF_tuned.score(xtrain,ytrain)
pred = RF_tuned.predict(xtrain)