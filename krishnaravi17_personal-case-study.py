

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import xgboost as xgb

color = sns.color_palette()

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.metrics import r2_score

%matplotlib inline

from sklearn.svm import SVR, LinearSVC

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

import scipy.stats as stats

from sklearn.externals import joblib



# Load the Drive helper and mount

#from google.colab import drive

# This will prompt for authorization.

#drive.mount('/content/drive')
#loading data from google drive

#train_df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Personal Case STudy/train.csv')

#print("Train shape : ", train_df.shape)

#test_df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Personal Case STudy/test.csv')

#print("Train shape : ", test_df.shape)

#loading data from HDD

train_df = pd.read_csv('../input/train.csv')

print("Train shape : ", train_df.shape)

test_df = pd.read_csv('../input/test.csv')

print("Train shape : ", test_df.shape)



train_df.info()
train_df.head()
def check_missing_values(df):

    

    if df.isnull().any().any():

        print("There are missing values in the data")  

    else: 

        print("There are no missing values in the data")

#calling functions to check missing values on training and test datasets

check_missing_values(train_df)

check_missing_values(test_df)
#we are checking 'y' column

plt.figure(figsize=(8,6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('y', fontsize=12)

plt.show()



"""here we have observed 1 outlier at apporx 260"""
# we again check by visualising in BoxPlot



plt.figure(figsize=(15,5))

sns.boxplot(train_df.loc[:,'y'])

plt.show()
# we need to remove that outlier 

# i have used zscore method and set threshold 10 acc to our data

# https://www.geeksforgeeks.org/scipy-stats-zscore-function-python/



train_df['x'] = np.abs(stats.zscore(train_df.loc[:,'y']))



outlier_ids = train_df[train_df['x']>10].ID



train_df_final = train_df[~train_df['ID'].isin(list(outlier_ids))]
#now plotting again without outlier

#we are checking 'y' column

plt.figure(figsize=(8,6))

plt.scatter(range(train_df_final.shape[0]), np.sort(train_df_final.y.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('y', fontsize=12)

plt.show()
# we again check by visualising in BoxPlot



plt.figure(figsize=(15,5))

sns.boxplot(train_df_final.loc[:,'y'])

plt.show()
ulimit = 180# we have taken 180 data points

train_df_final['y'].ix[train_df_final['y']>ulimit] = ulimit



plt.figure(figsize=(12,8))#plot size

sns.distplot(train_df_final.y.values, bins=50, kde=True)

plt.xlabel('y value', fontsize=12)

plt.show()
#removing that x helper row for outlier from main row



train_df_final = train_df_final.drop(["x"], axis=1)
dtype_df = train_df_final.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
#here we can see their types

dtype_df.ix[:15,:]

#https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-mercedes

var_name = ['X0','X1','X2','X3','X4','X5','X6','X8']

for val in var_name:

    col_order = np.sort(train_df_final[val].unique()).tolist()

    plt.figure(figsize=(12,6))

    sns.stripplot(x=val, y='y', data=train_df_final, order=col_order)

    plt.xlabel(val, fontsize=12)

    plt.ylabel('y', fontsize=12)

    plt.title("Distribution of y variable with "+val, fontsize=15)

    plt.show()

train_df = pd.read_csv('../input/train.csv')

print("Train shape : ", train_df.shape)

test_df = pd.read_csv('../input/test.csv')

print("Train shape : ", test_df.shape)



y_train = train_df['y'].values

id_test = test_df['ID'].values



usable_columns = list(set(train_df.columns) - set(['ID', 'y']))#taking only important coloumns

print(len(usable_columns))



x_train_final = train_df[usable_columns]

x_test_final = test_df[usable_columns]
# Converting training dataset object categorical values to numerical categorical types

#taken help from link: https://www.kaggle.com/anokas/mercedes-eda-xgboost-starter-0-55



for column in usable_columns:

    cardinality = len(np.unique(x_train_final[column]))

    

    if cardinality == 1:

        x_train_final.drop(column, axis=1) # Column with only one value is useless so we drop it.

        x_test_final.drop(column, axis=1)

        

    if cardinality > 2: # Column is categorical.

        mapper = lambda x: sum([ord(digit) for digit in x])

        x_train_final[column] = x_train_final[column].apply(mapper)

        x_test_final[column] = x_test_final[column].apply(mapper)
# spiltting it into 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(x_train_final, y_train, test_size=0.3, random_state=42)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#taken help from kaggle discussion and kernels for xgboost

#setting up xtrain and xtrain



y_mean = y_train.mean()



d_train = xgb.DMatrix(X_train, label=y_train)

d_cvalid  = xgb.DMatrix(X_test, label=y_test)

d_test = xgb.DMatrix(x_test_final)

# evaluation r2_score metric

def r2_score_metric(y_pred, y):

    y_true = y.get_label()

    return 'r2', r2_score(y_true, y_pred)

  
%%time

#xgb parameters

#just cross validation our model 



params = {

    

   'n_trees': 500, 

    'eta': 0.005,

    'max_depth': 4,

    'subsample': 0.95,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean, # base prediction = mean(target)

    'silent': 1

}



num_boost_round=2000



#Cross Validation of XGBoost 

cv_result = xgb.cv(params, 

                   d_train, 

                   num_boost_round, 

                   nfold = 3,

                   early_stopping_rounds=50,

                   feval=r2_score_metric,#here we have used our metric method

                   verbose_eval=100, 

                   show_stdv=False

                  )

#Training the model

#taken help from link: https://www.kaggle.com/anokas/mercedes-eda-xgboost-starter-0-55



#model = joblib.load('model_xgb.pkl')#from load



watchlist = [(d_train, 'train'), (d_cvalid, 'valid')]



model = xgb.train(params, d_train, num_boost_round, watchlist, early_stopping_rounds=50,

                  feval=r2_score_metric, maximize=True, verbose_eval=10)



#joblib.dump(model, 'model_xgb.pkl')#to load
# Predict on test



y_pred = model.predict(d_test)

# Predicting R2SCORE

#from sklearn.metrics import r2_score



#r2_score = r2_score(y_test, y_pred)#taking r2score on traing data



#print('r2_score = ',r2_score)
#exporting final results into csv file



csvfile = pd.DataFrame()

csvfile['ID'] = test_df['ID']

csvfile['y'] = y_pred

csvfile.to_csv('xgb.csv', index=False)
#https://www.kaggle.com/satadru5/mercedes-benz-xgb-modeling-lb-score-0-54472 

fig, ax = plt.subplots(1, 1, figsize=(8, 13))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
# PCA Implementation

pca = PCA(n_components=2)

pca_data = pca.fit_transform(X_train)
cmap = sns.cubehelix_palette(as_cmap=True)

f, ax = plt.subplots(figsize=(10,7))

points = ax.scatter(pca_data[:,0], pca_data[:,1], c=y_train, s=50, cmap=cmap)

f.colorbar(points)

plt.show()
# TSNE Implementation

model = TSNE(n_components=2,random_state=0,perplexity=30)



tsne_data = model.fit_transform(X_train)


cmap = sns.cubehelix_palette(as_cmap=True)

f, ax = plt.subplots(figsize=(10,7))

points = ax.scatter(tsne_data[:,0], tsne_data[:,1], c=y_train, s=50, cmap=cmap)

f.colorbar(points)

plt.show()
#KNN implementation

#biulding model



knn = KNeighborsRegressor(n_neighbors=5)#k=5 gives best results



knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)



r2_score_knn = round(r2_score(y_test, y_pred),3)#taking r2score

accuracy = round(knn.score(X_train, y_train) *100,2)#taking accuracy



results = {'r2_score':r2_score_knn, 'accuracy':accuracy}

print (results)

#SVR implementation

from sklearn.metrics import r2_score

clf = SVR()



clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



r2_score = round(r2_score(y_test, y_pred),3)#taking r2score

accuracy = round(clf.score(X_train, y_train) * 100, 2)



results = {'r2_score':r2_score, 'accuracy':accuracy}

print(results)
#RFR implementation

from sklearn.metrics import r2_score

clf = RandomForestRegressor(n_estimators = 60 ,max_depth=5,oob_score=True)



clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



r2_score = round(r2_score(y_test, y_pred),3)#taking r2score

accuracy = round(clf.score(X_train, y_train) * 100, 2)



results = {'r2_score':r2_score, 'accuracy':accuracy}

print (results)
#Linear Regression implementation

from sklearn.metrics import r2_score

clf = LinearRegression()

  

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



r2_score = round(r2_score(y_test, y_pred),3)#taking r2score

accuracy = round(clf.score(X_train, y_train) * 100, 2)



results = {'r2_score':r2_score, 'accuracy':accuracy}

print (results)