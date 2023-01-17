# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from time import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import neighbors, datasets, preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, recall_score, r2_score, mean_absolute_error, mean_squared_error, mean_absolute_error
#from sklearn.utils.fixes import loguniform

##Linear Regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDClassifier

## Logistic Regression
from sklearn.linear_model import LogisticRegression

## Decision Tree and random forest packages
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier

## KNN 
from sklearn.neighbors import KNeighborsClassifier

##SVM
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

## Naive Bayesian
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


from sklearn.cluster import KMeans
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv') 
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')   
train.shape
test.shape
train.head()
# select numeric columns
train_numeric = train.select_dtypes(include=[np.number])
train_numeric_cols=train_numeric.columns.values
print("numerical columns: \n")
print(train_numeric_cols)
# select non numeric columns
train_non_numeric = train.select_dtypes(exclude=[np.number])
train_non_numeric_cols = train_non_numeric.columns.values
print("categorical columns: \n")
print(train_non_numeric_cols)

def corr_map(df):
    #cmap=cmap=sns.diverging_palette(20, 220, n=200)
    cmap="RdYlBu"
    plt.figure(figsize=(20,20))
    sns.heatmap(df.corr().round(2), annot=True,vmin=-1, vmax=1, center=0 ,fmt='g', cmap=cmap, annot_kws={"size": 8})
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
corr_map(train)
corr = train.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])
def top_corr_map(df):
    #cmap=cmap=sns.diverging_palette(20, 220, n=200)
    cmap="RdYlBu"
    plt.figure(figsize=(20,20))
    sns.heatmap(df.corr().round(2), annot=True, cmap=cmap, annot_kws={"size": 8})
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    
top_corr_features = train.corr().index[abs(train.corr()["SalePrice"])>0.5]
top_corr_map(train[top_corr_features])
train[train_numeric_cols].hist(figsize=(20,20), xrot=-45);
for col in train_non_numeric_cols:
    train[col].value_counts().plot(kind='bar',figsize=(4,4),title=col) 
    plt.show()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, pct_missing*100))
train_non_numeric_cols
def NaN_to_missing(df):

    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.values
    for col in non_numeric_cols:
        missing = df[col].isnull()
        num_missing = np.sum(missing)
    
        if num_missing > 0:  # only do the imputation for the columns that have missing values.
            df[col] = df[col].fillna('None')
    return df
        
train=NaN_to_missing(train)   
test=NaN_to_missing(test)   
for col in train_non_numeric_cols:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, pct_missing*100))
for col in train_non_numeric_cols:
    train[col].value_counts().plot(kind='bar',figsize=(4,4),title=col) 
    plt.show()
for col in train_numeric_cols:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {:.2f}%'.format(col, pct_missing*100))
train[train['LotFrontage'].isnull()].head()
train['GarageYrBlt']
train[train['MasVnrArea'].isnull()].head()
def num_missing(df):
    df['LotFrontage'] = df['LotFrontage'].fillna(0)
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    return df

num_missing(train)
num_missing(test)
for col in train_numeric_cols:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {:.2f}%'.format(col, pct_missing*100))
train.shape
train[train_non_numeric_cols].shape
y_train=train['SalePrice']
y_train.shape
train.head()
train.shape
X_train=train.drop('SalePrice', axis=1)
X_train.shape

X_train.shape
test.shape
train_numeric_cols[:-1]
# scaling data :
scaler = StandardScaler()
X_train_sc=X_train
X_train_sc[train_numeric_cols[:-1]]= scaler.fit_transform(X_train[train_numeric_cols[:-1]])
X_test_sc=test
X_test_sc[train_numeric_cols[:-1]]=scaler.transform(test[train_numeric_cols[:-1]])
X_train_sc.shape
X_test_sc.shape
X_train_sc.columns
X_test_sc.columns
train_non_numeric_cols
X_train_sc_dum= pd.get_dummies(X_train_sc[train_non_numeric_cols], drop_first=True)

X_test_sc_dum= pd.get_dummies(X_test_sc[train_non_numeric_cols], drop_first=True)
X_train_sc_dum.shape
X_test_sc_dum.shape
list(set(X_train_sc_dum.columns) - set(X_test_sc_dum.columns))
X_train_sc['is_train_set'] = 1
X_test_sc['is_train_set'] = 0
X_train_sc.head()
X_train_sc.shape ; X_test_sc.shape
train_test_sc_concat = pd.concat([X_train_sc, X_test_sc], ignore_index=True)
train_test_sc_concat.shape
train_test_sc_concat.head()
train_non_numeric_cols
train_test_sc_concat_dummy = pd.get_dummies(train_test_sc_concat, columns=train_non_numeric_cols,drop_first=True)
train_test_sc_concat_dummy.shape
X_train_final=train_test_sc_concat_dummy[train_test_sc_concat_dummy['is_train_set'] == 1]
X_test_final=train_test_sc_concat_dummy[train_test_sc_concat_dummy['is_train_set'] == 0]
X_train_final.shape ; y_train.shape ; X_test_final.shape 
####Models#### 

models = {'LR': LinearRegression(),
          'Lasso': Lasso(),
          'Ridge': Ridge(),
          'Elastic_net' :  ElasticNet()
         }


################ hyperParameter  ##################

params= {
    'LR': {'fit_intercept': [True, False]},
    'Lasso' : {'alpha': [0.001,0.01,0.02,0.025,0.05,0.25,0.5,1,5,10]},
    'Ridge' : {'alpha': [100,50,25,10,4,2,1.0,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01,0.001]},
    'Elastic_net': {'alpha' : [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                     'l1_ratio' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] }     
}


# param_grid= [
#     {'clf__C': np.logspace(-2,5,12), 'clf__gamma': np.logspace(-3,1,10), 'clf__kernel': ['rbf']}, 
#     {'clf__C': np.logspace(-2,5,12), 'clf__degree': list(range(1,8)) , 'clf__kernel': ['poly']}
# ]

def gridCV_fit(X_train,y_train):
    Evaluation_CV_result={}
    
    for model_name in models.keys():
        print(model_name)
        grid=GridSearchCV(estimator=models[model_name], param_grid=params[model_name], cv=5, n_jobs=4,verbose=10,return_train_score=True)
        grid.fit(X_train,y_train)  
        print("best estimator are: {}".format(grid.best_params_))
        print("best parameters are: {}".format(grid.best_estimator_))
        print("best scores are: {}".format(grid.best_score_))
        Evaluation_CV_result[model_name]=[grid.best_score_]
    df_CV_result=pd.DataFrame(Evaluation_CV_result, index =['CV_score']) 
    return  df_CV_result
    
# def Evalution_Test(X_train,y_train, X_test, y_test)
#       ## Predictions ##
#         y_pred=grid.predict(X_test)
#         MAE=metrics.mean_absolute_error(y_test, y_pred)
#         MSE=metrics.mean_squared_error(y_test, y_pred)
#         RMSE=np.sqrt(MSE)
#         R2=r2_score(y_test,y_pred)
#         print('MAE: %0.3f' % (MAE))
#         print('MSE: %0.3f' % (MSE))
#         print('RMSE: %0.3f' % (RMSE))
#         print('R2: %0.3f' % (R2))
#         Evaluation_result[model_name]=[grid.best_score_, MAE, MSE, RMSE, R2 ]
#     pd.DataFrame(Evaluation_result, index =['Mean cross-validated', 'MAE', 'MSE', 'RMSE', 'R2'])     


gridCV_fit(X_train_final,y_train)
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X_test_final_imuputed = my_imputer.fit_transform(X_test_final)
X_test_final_imuputed =pd.DataFrame(data=X_test_final_imuputed[:,:], columns=X_test_final.columns)
X_test_final_imuputed
for col in X_test_final_imuputed.columns :
    pct_missing = np.mean(X_test_final_imuputed[col].isnull())
    print('{} - {:.2f}%'.format(col, pct_missing*100))
my_best_model= ElasticNet(alpha=100, copy_X=True, fit_intercept=True, l1_ratio=1,
                       max_iter=1000, normalize=False, positive=False, precompute=False,
                       random_state=0, selection='cyclic', tol=0.0001, warm_start=False).fit(X_train_final,y_train)
# Use the model to make predictions
predicted_prices = my_best_model.predict(X_test_final_imuputed)
predicted_prices
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)