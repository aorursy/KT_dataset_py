# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix as conMatrix
from sklearn.metrics import classification_report as ClassR
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from uszipcode import Zipcode, SearchEngine
search = SearchEngine(simple_zipcode=True)

%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/airbnb-listings-in-major-us-cities-deloitte-ml/train.csv")
test_df = pd.read_csv("/kaggle/input/airbnb-listings-in-major-us-cities-deloitte-ml/test.csv")
train_df.head()
test_df.head()
missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df
# 'bathrooms'
train_df['bathrooms'] = train_df['bathrooms'].fillna(0)
test_df['bathrooms'] = test_df['bathrooms'].fillna(0)

# 'bedrooms'
train_df['bedrooms'] = train_df['bedrooms'].fillna(0)
test_df['bedrooms'] = test_df['bedrooms'].fillna(0)

# 'beds' 
train_df['beds'] = train_df['beds'].fillna(0)
test_df['beds'] = test_df['beds'].fillna(0)

# 'host_has_profile_pic'
train_df.loc[train_df.host_has_profile_pic == 't', 'host_has_profile_pic'] = 1
train_df.loc[train_df.host_has_profile_pic == 'f', 'host_has_profile_pic'] = 0
train_df['host_has_profile_pic'] = train_df['host_has_profile_pic'].fillna(0)
test_df.loc[test_df.host_has_profile_pic == 't', 'host_has_profile_pic'] = 1
test_df.loc[test_df.host_has_profile_pic == 'f', 'host_has_profile_pic'] = 0
test_df['host_has_profile_pic'] = test_df['host_has_profile_pic'].fillna(0)

# 'neighbourhood'
train_df['neighbourhood'] = train_df['neighbourhood'].fillna('Unknown')
test_df['neighbourhood'] = test_df['neighbourhood'].fillna('Unknown')

# 'review_scores_rating'
train_df['review_scores_rating'] = train_df['review_scores_rating'].fillna(0)
test_df['review_scores_rating'] = test_df['review_scores_rating'].fillna(0)

# 'host_response_rate'
train_df['host_response_rate'] = train_df['host_response_rate'].str.replace('%','')
train_df['host_response_rate'] = train_df['host_response_rate'].fillna('0')
train_df['host_response_rate'] = pd.to_numeric(train_df['host_response_rate'])
test_df['host_response_rate'] = test_df['host_response_rate'].str.replace('%','')
test_df['host_response_rate'] = test_df['host_response_rate'].fillna('0')
test_df['host_response_rate'] = pd.to_numeric(test_df['host_response_rate'])

# 'thumbnai_url'
train_df['thumbnail_url'] = train_df['thumbnail_url'].fillna('Unknown')
test_df['thumbnail_url'] = test_df['thumbnail_url'].fillna('Unknown')

# 'last_review'
train_df['last_review'] = train_df['last_review'].fillna('00-00-00')
test_df['last_review'] = test_df['last_review'].fillna('00-00-00')

# 'first_review'
train_df['first_review'] = train_df['first_review'].fillna('00-00-00')
test_df['first_review'] = test_df['first_review'].fillna('00-00-00')

# 'host_since'
train_df['host_since'] = train_df['host_since'].fillna('00-00-00')
test_df['host_since'] = test_df['host_since'].fillna('00-00-00')

# 'host_identity_verified'
train_df.loc[train_df.host_identity_verified == 't', 'host_identity_verified'] = 1
train_df.loc[train_df.host_identity_verified == 'f', 'host_identity_verified'] = 0
train_df['host_identity_verified'] = train_df['host_identity_verified'].fillna(0)
test_df.loc[test_df.host_identity_verified == 't', 'host_identity_verified'] = 1
test_df.loc[test_df.host_identity_verified == 'f', 'host_identity_verified'] = 0
test_df['host_identity_verified'] = test_df['host_identity_verified'].fillna(0)

# 'zipcode', ***this might take some time, but works perfect***
train_df['zipcode'] = train_df['zipcode'].fillna(0)
train_df.loc[train_df.zipcode == ' ', 'zipcode'] = 0
idx = train_df.index[train_df['zipcode']==0].tolist()
for i in idx:
    lat = train_df['latitude'][i]
    lon = train_df['longitude'][i]
    result = np.max(search.by_coordinates(lat, lon, radius=30, returns=5))
    train_df['zipcode'][i]=result.values()[0]    
test_df['zipcode'] = test_df['zipcode'].fillna(0)
test_df.loc[test_df.zipcode == ' ', 'zipcode'] = 0
idx = test_df.index[test_df['zipcode']==0].tolist()
for i in idx:
    lat = test_df['latitude'][i]
    lon = test_df['longitude'][i]
    result = np.max(search.by_coordinates(lat, lon, radius=30, returns=5))
    test_df['zipcode'][i]=result.values()[0]

missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df
train_df['int_price'] = np.exp(train_df['log_price'])
train_df.pivot_table(values='int_price',index='property_type',columns='city')
train_df.to_csv('new_train.csv', encoding='utf-8', index=False)
test_df.to_csv('new_test.csv', encoding='utf-8', index=False)
Rtrain_df = pd.read_csv('new_train.csv')
Rtest_df = pd.read_csv('new_test.csv')
Rtrain_df.drop(['int_price'], axis =1, inplace=True)

Rtrain_df.drop(['id'], axis =1, inplace=True)
Rtest_df.drop(['id'], axis =1, inplace=True)

Rtrain_df.drop(['log_price'], axis =1, inplace=True)

Rtrain_df.drop(['neighbourhood'], axis =1, inplace=True)
Rtest_df.drop(['neighbourhood'], axis =1, inplace=True)

Rtrain_df.drop(['description'], axis =1, inplace=True)
Rtest_df.drop(['description'], axis =1, inplace=True)

Rtrain_df.drop(['first_review'], axis =1, inplace=True)
Rtest_df.drop(['first_review'], axis =1, inplace=True)

Rtrain_df.drop(['last_review'], axis =1, inplace=True)
Rtest_df.drop(['last_review'], axis =1, inplace=True)

Rtrain_df.drop(['host_since'], axis =1, inplace=True)
Rtest_df.drop(['host_since'], axis =1, inplace=True)

Rtrain_df.drop(['thumbnail_url'], axis =1, inplace=True)
Rtest_df.drop(['thumbnail_url'], axis =1, inplace=True)

Rtrain_df.drop(['zipcode'], axis =1, inplace=True)
Rtest_df.drop(['zipcode'], axis =1, inplace=True)

Rtrain_df.drop(['amenities'], axis =1, inplace=True)
Rtest_df.drop(['amenities'], axis =1, inplace=True)

Rtrain_df.drop(['name'], axis =1, inplace=True)
Rtest_df.drop(['name'], axis =1, inplace=True)

Rtrain_df.drop(['latitude'], axis =1, inplace=True)
Rtest_df.drop(['latitude'], axis =1, inplace=True)

Rtrain_df.drop(['longitude'], axis =1, inplace=True)
Rtest_df.drop(['longitude'], axis =1, inplace=True)

Rtrain_df.drop(['instant_bookable'], axis =1, inplace=True)
Rtest_df.drop(['instant_bookable'], axis =1, inplace=True)
Rtrain_df.dtypes
rtrain_df = Rtrain_df.copy()
rtest_df = Rtest_df.copy()
print(rtrain_df.shape)
print(rtest_df.shape)
def one_hot(train_df,test_df,columns):
    
    for i,column in enumerate(columns):
        Xtrain = train_df[str(column)].T
        Xtest = test_df[str(column)].T
        
        # train_df
        lb=LabelBinarizer()
        lb.fit(Xtrain)
        X_classes = len(lb.classes_)
        Xenc = lb.transform(Xtrain)
        Xtrain_enc = pd.DataFrame(data = Xenc, columns = lb.classes_)
        train_df.drop([str(column)], axis =1, inplace=True)
        
        # test_df
        Xenc = lb.transform(Xtest)
        Xtest_enc = pd.DataFrame(data = Xenc, columns = lb.classes_)
        test_df.drop([str(column)], axis =1, inplace=True)
        
        print('Number of classes in '+str(column)+ ' = '+ str(X_classes))
        train_df = pd.concat((train_df,Xtrain_enc),axis=1)
        test_df = pd.concat((test_df,Xtest_enc),axis=1) 
    return train_df,test_df
r_train_df , r_test_df = one_hot(rtrain_df,rtest_df,['city','property_type', 'room_type', 'bed_type', 'cancellation_policy', 'host_response_rate'])
print(r_test_df.shape)
print(r_train_df.shape)
pca = PCA()
pca_fit = pca.fit_transform(r_train_df)
pca_fit.shape
X = pca_fit
y = train_df['log_price']
kfold = KFold(n_splits=10,random_state=56,shuffle=True)
average = 0
average1 = 0
for train_idx, test_idx in kfold.split(X,y):    
    X_train, X_CV = X[train_idx], X[test_idx]
    y_train, y_CV = y[train_idx], y[test_idx]
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
        
    pred_CV = lr.predict(X_CV)
    MSE = mse(y_CV, pred_CV)
    average = average + MSE

    score = lr.score(X_CV, y_CV)
    average1 = average1 + score
    
    print('R square score = ',score)
    print('MSE = ',MSE)

MSE_AVG = average/10
Rscore_AVG = average1/10
print('*---------------------------*')
print('Average Rscore = ', Rscore_AVG)
print('Average MSE = ',MSE_AVG)
kfold = KFold(n_splits=10,random_state=56,shuffle=True)
average = 0
average1 = 0

for train_idx, test_idx in kfold.split(X,y):    
    
    X_train, X_CV = X[train_idx], X[test_idx]
    y_train, y_CV = y[train_idx], y[test_idx]
    
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    
    pred_CV = ridge.predict(X_CV)
    MSE = mse(y_CV, pred_CV)
    average = average + MSE
    
    score = ridge.score(X_CV, y_CV)
    average1 = average1 + score
    
    print('R square score = ',score)
    print('MSE = ',MSE)

MSE_AVG = average/10
Rscore_AVG = average1/10
print('*---------------------------*')
print('Average Rscore = ', Rscore_AVG)
print('Average MSE = ', MSE_AVG)

kfold = KFold(n_splits=10,random_state=56,shuffle=True)
average = 0
average1 = 0

for train_idx, test_idx in kfold.split(X,y):    
    
    X_train, X_CV = X[train_idx], X[test_idx]
    y_train, y_CV = y[train_idx], y[test_idx]
    
    lasso = Lasso(0.0001)
    lasso.fit(X_train, y_train)
        
    pred_CV = lasso.predict(X_CV)
    MSE = mse(y_CV, pred_CV)
    average = average + MSE
    
    score = lasso.score(X_CV, y_CV)
    average1 = average1 + score
    
    print('R square score = ',score)
    print('MSE = ',MSE)

MSE_AVG = average/10
Rscore_AVG = average1/10
print('*---------------------------*')
print('Average Rscore = ', Rscore_AVG)
print('Average MSE = ', MSE_AVG)
parameters = {"alpha":np.logspace(-2,2,50)}
lasso_grid = GridSearchCV(lasso, parameters, cv=10) 
lasso_grid.fit(X,y)

print('Hyper Parameters for Lasso:\n',lasso_grid.best_params_)
print('Score for Hyper Parameters from Grid Search:',lasso_grid.best_score_)
lasso_grid.cv_results_
pca0 = PCA()
pca0.fit(r_train_df)
X_train = pca0.transform(r_train_df)
y_train = train_df['log_price']

X_test = pca0.transform(r_test_df)
lasso = Lasso(0.0001)
lasso.fit(X_train, y_train)

# Prices predicted for test dataset is given by 'price_predicted'
price_predicted = lasso.predict(X_test)
print(price_predicted)
preds = pd.DataFrame(price_predicted, columns=['log_price'])
preds
preds.to_csv('submission.csv',index_label='id')