# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

from sklearn import preprocessing as proc

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from collections import Counter

from imblearn.under_sampling import RandomUnderSampler

from imblearn.under_sampling import RandomUnderSampler as rs

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve

import scikitplot as skplt

import matplotlib.pyplot as plt



import xgboost as xgb



print(os.listdir('/kaggle/input'))



# Any results you write to the current directory are saved as output.
PATH = '/kaggle/input/'
df_train_tran = pd.read_csv(PATH + 'train_transaction.csv')

df_train_tran.head(5)
print(df_train_tran.shape)
gc.collect()
cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

n_cat = len(cat_cols)

print(n_cat)
print(type(df_train_tran))
# Imputing and Encoding Categorical Columns

for i in range(0,n_cat):

    print(str(i))

    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    imp.fit(df_train_tran[[cat_cols[i]]])

    df_train_tran[[cat_cols[i]]]= imp.transform(df_train_tran[[cat_cols[i]]])

    df_train_tran[[cat_cols[i]]] = LabelEncoder().fit_transform(df_train_tran[[cat_cols[i]]])
# Imputing and Encoding Non-Categorical Columns

#print(np.array(df_train_tran.columns))

y_col = ['isFraud']

all_cols = set(np.array(df_train_tran.columns)).difference(set(y_col))

non_cat_cols = list(set(all_cols).difference(set(cat_cols)))

non_cat_cols

print(len(non_cat_cols))

for col in non_cat_cols:

    #print(col)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    imp.fit(df_train_tran[[col]])

    df_train_tran[[col]]= imp.transform(df_train_tran[[col]])

    df_train_tran[[col]] = LabelEncoder().fit_transform(df_train_tran[[col]])
df_train_tran.head(5)
#scaling all cols in train_tran dataframe

df_train_tran = np.array(df_train_tran)

train_tran_x = df_train_tran[:,2:]

train_tran_y = df_train_tran[:,1]

del df_train_tran

gc.collect()

#df_train_tran[all_cols] = df_train_tran[all_cols].apply(lambda x: MinMaxScaler().fit_transform(x))
np.shape(train_tran_x)
np.shape(train_tran_y)

m_tran = np.size(train_tran_y)

n_tran = np.shape(train_tran_x)[1]

print(m_tran)

print(n_tran)
def find_FDR(x,y,i):

    idx_0 = (train_tran_y==0)

    idx_1 = (train_tran_y==1)

    x_class_0 = x[idx_0[:,0],i]

    x_class_1 = x[idx_1[:,0],i]

    mu_0 = np.mean(x_class_0)

    sd_0 = np.std(x_class_0)

    mu_1 = np.mean(x_class_1)

    sd_1 = np.std(x_class_1)

    fdr = (mu_1 - mu_0)**2/(sd_0**2 + sd_1**2)

    return fdr
gc.collect()

fdr = []



# Calculate FDR for each column

train_tran_y = train_tran_y.reshape(m_tran, 1)

for i in range(0,n_tran):

    f = find_FDR(train_tran_x, train_tran_y, i)

    fdr.append(f)

print(np.shape(fdr))



# Choose threshold for FDR

tran_tresh = 0.1

idx = [i for i in range(0,n_tran) if fdr[i]>tran_tresh]

tran_x_selected = train_tran_x[:,idx]

len(tran_x_selected)

print(np.shape(tran_x_selected))
# Do PCA on the chosen columns after finding FDR





from numpy import array

from numpy import mean

from numpy import cov

from numpy.linalg import eig



# calculate the mean of each column

M = mean(tran_x_selected.T, axis=1) 

print(M)

# center columns by subtracting column means

C = tran_x_selected - M

# calculate covariance matrix of centered matrix

V = cov(C.T)

print(V)

# eigendecomposition of covariance matrix

values, vectors = eig(V)

#print(vectors)

print(values)

# project data

P = vectors.T.dot(C.T)

P = P.T

print(np.shape(P))



# take top 40 

tran_x_selected = P[:,:40]

print(np.shape(tran_x_selected))
# train test split undersampling

rs = RandomUnderSampler(random_state=42)

X, y = rs.fit_resample(tran_x_selected, train_tran_y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

print(np.shape(X_train))

print(np.shape(y_train))

print(np.shape(X_test))

print(np.shape(y_test))
# random forest classifier

'''regressor = RandomForestRegressor(n_estimators=50, random_state=0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(y_pred)

y_pred = [1 if y_pred[i]>=0.5 else 0 for i in range(0,np.shape(y_pred)[0])]

print(y_pred)'''


gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)

y_pred = gbm.predict(X_test)

y_pred_proba = gbm.predict_proba(X_test)
# metrics

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

print(accuracy_score(y_test, y_pred))

skplt.metrics.plot_roc_curve(y_test, y_pred_proba)

plt.show()
#del X

#del y

del X_train

del X_test

del y_train

del y_test

del tran_x_selected

del train_tran_x

del train_tran_y

gc.collect()
# Loading the test data

df_test_tran = pd.read_csv(PATH + 'test_transaction.csv')

df_test_tran.head(5)
# Test transaction data Imputing and Encoding Categorical Columns

for i in range(0,n_cat):

    #print(str(i))

    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    imp.fit(df_test_tran[[cat_cols[i]]])

    df_test_tran[[cat_cols[i]]]= imp.transform(df_test_tran[[cat_cols[i]]])

    df_test_tran[[cat_cols[i]]] = LabelEncoder().fit_transform(df_test_tran[[cat_cols[i]]])
# Test Imputing and Encoding Non-Categorical Columns

#print(np.array(df_test_tran.columns))

y_col = ['isFraud']

all_cols = set(np.array(df_test_tran.columns)).difference(set(y_col))

non_cat_cols = list(set(all_cols).difference(set(cat_cols)))

non_cat_cols

print(len(non_cat_cols))

for col in non_cat_cols:

    #print(col)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    imp.fit(df_test_tran[[col]])

    df_test_tran[[col]]= imp.transform(df_test_tran[[col]])

#scling all cols in train_tran dataframe

df_test_tran = np.array(df_test_tran)

test_tran_x = df_test_tran[:,2:]

test_tran_y = df_test_tran[:,1]

del df_test_tran

gc.collect()

#df_train_tran[all_cols] = df_train_tran[all_cols].apply(lambda x: MinMaxScaler().fit_transform(x))    df_test_tran[[col]] = LabelEncoder().fit_transform(df_test_tran[[col]])
np.shape(test_tran_y)

m_tran = np.size(test_tran_y)

n_tran = np.shape(test_tran_x)[1]

print(m_tran)

print(n_tran)
tran_x_selected = test_tran_x[:,idx]

len(tran_x_selected)

print(np.shape(tran_x_selected))
tran_x_selected
# Test data: Do PCA on the chosen columns after finding FDR





from numpy import array

from numpy import mean

from numpy import cov

from numpy.linalg import eig



# calculate the mean of each column

M = mean(tran_x_selected.T, axis=1) 

print(M)

# center columns by subtracting column means

C = tran_x_selected - M

# calculate covariance matrix of centered matrix

V = cov(C.T)

print(V)

# eigendecomposition of covariance matrix

values, vectors = eig(V)

#print(vectors)

print(values)

# project data

P = vectors.T.dot(C.T)

P = P.T

print(np.shape(P))



# take top 40 

tran_x_selected = P[:,:40]

print(np.shape(tran_x_selected))
'''y_pred = regressor.predict(tran_x_selected)

print(y_pred)

y_pred = [1 if y_pred[i]>=0.5 else 0 for i in range(0,np.shape(y_pred)[0])]

print(np.shape(y_pred))'''
'''y_pred = np.array(y_pred)

print(y_pred)'''
# test xgboost model

y_pred = gbm.predict(tran_x_selected)

print(y_pred)

y_pred = [1 if y_pred[i]>=0.5 else 0 for i in range(0,np.shape(y_pred)[0])]

print(np.shape(y_pred))
submission = pd.read_csv(PATH + 'sample_submission.csv')

submission['isFraud'] = y_pred#.astype('int64')

submission.to_csv('submission.csv', index=False)
submission