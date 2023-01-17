import numpy as np
import pandas as pd
import math
from pandas import DataFrame
import xgboost
import math
from pandas import DataFrame
import pickle
from scipy import sparse
import pyodbc
import seaborn as sb
%matplotlib inline

from sklearn import preprocessing
from multiprocessing import Pool
from timeit import default_timer as timer
from math import sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelBinarizer
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV



train_data = pd.read_table('../input/jerboris/train.tsv')
test_data = pd.read_table('../input/jerboris/test.tsv')
train_data.head()

test_data.head(1)
# Check Outliers
print (train_data['price'].describe())
sb.distplot(train_data['price'])

### Check the NaN values
def check_nan(data):
    data_check = data.replace('NaN',np.nan)
    data_check = data.replace('No description yet',np.nan)
    data_nan = data_check.isnull().sum()
    data_nan.sort_values(ascending=False, inplace=True)
    print ('\nTotal number of NaN values in the dataset:',data_nan.sum())
    print ('\nTop three features with most NaN values in the dataset:')
    print (data_nan[:5])
    
check_nan(train_data)
check_nan(test_data)
def get_feature_label(data):
    # remove outliers
    data_after = data[(data['price']<400) & (data['price']>1)]
    #data_after = data[data['price']>1]
    # split features and labels
    train_features = data_after.drop(['price'],axis=1)
    train_labels = data_after.price
    return train_features,train_labels
# change this variable to get raw/sample data
train_features,train_labels=get_feature_label(train_data)
train_features=train_features
train_labels=train_labels
test_features=test_data
#b = train_labels[(train_labels < 1.0)]
#train_labels = train_labels.drop(train_labels[(train_labels < 1.0)].index)
train_labels = np.log(train_labels)
train_labels
train_features.head(1)
test_features.head(1)
def category(data):
    cat = data.category_name.str.split('/', expand = True)
    data["main_cat"] = cat[0]
    data["subcat1"] = cat[1]
    data["subcat2"] = cat[2]
    try:
        data["subcat3"] = cat[3]
    except:
        data["subcat3"] = np.nan  
    try:
        data["subcat4"] = cat[4]
    except:
        data["subcat4"] = np.nan  
        
def missing_data(data, _value = 'None'):
    # Handle missing data
    for col in data.columns:
        data[col].fillna(_value,inplace=True)
category(train_features)
category(test_features)

missing_data(train_features)
missing_data(test_features)
## convert categorical var to numeric 

le = preprocessing.LabelEncoder()
def cat_to_num(train,test):
    suf="_le"
    for col in ['brand_name','main_cat','subcat1','subcat2','subcat3','subcat4']:
        train[col+suf] = le.fit_transform(train[col])
        dic = dict(zip(le.classes_, le.transform(le.classes_)))
        test[col+suf] = test[col].map(dic).fillna(0).astype(int) 
        
        print("{} is transformed to {}".format(col,col+suf))
      
## convert categorical var to numeric 
'''le =  LabelBinarizer(sparse_output=True)
def cat_to_num(train,test):
    suf="_le"
    for col in ['brand_name','main_cat','subcat1','subcat2','subcat3','subcat4']:
        train[col+suf] = le.fit_transform(train[col])
        #dic = dict(zip(le.classes_, le.transform(le.classes_)))
        test[col+suf] = le.transform(test[col]) 
        
        print("{} is transformed to {}".format(col,col+suf))'''
cat_to_num(train_features,test_features)
## Length of item discription
train_features['Length_of_item_description']=train_features['item_description'].apply(len)
test_features['Length_of_item_description']=test_features['item_description'].apply(len)
## Combine numeric features
def numeric_to_features(data):
    numeric_features = list(data.apply(lambda x:(x['shipping'],x['item_condition_id'],x['main_cat_le'],\
                                                 x['subcat1_le'],x['subcat2_le'],x['subcat3_le'],\
                                                 x['subcat4_le'],x['Length_of_item_description'],\
                                                 x['brand_name_le']), axis=1))
    return numeric_features


train_numeric_features = numeric_to_features(train_features)
test_numeric_features = numeric_to_features(test_features)

#train_text =text_process(train_features)
#test_text =text_process(test_features)

# Tfidf
    # save the vectorize
    # pickle.dump(tfidf,open('vectorizer.pkl', "bw",-1))
    # tfidf=pickle.load(open('vectorizer.pkl','br'))

tfidf_d = TfidfVectorizer(sublinear_tf=True,ngram_range=(1,3),min_df=0, stop_words = 'english',max_features = 500)
Description_matrix_train=  tfidf_d.fit_transform(train_features['item_description'])
Description_matrix_test=  tfidf_d.transform(test_features['item_description'])
tfidf_n = CountVectorizer(min_df = 10)
name_features_train = tfidf_n.fit_transform(train_features['name'])
name_features_test = tfidf_n.transform(test_features['name'])
train_text_features = sparse.hstack([ name_features_train, Description_matrix_train], format='csr')
#mask = np.array(np.clip(train_text_features.getnnz(axis=0) - 100, 0, 1), dtype=bool)
#train_text_features= train_text_features[:, mask]
test_text_features = sparse.hstack([ name_features_test, Description_matrix_test], format='csr')
#mask = np.array(np.clip(test_text_features.getnnz(axis=0) - 100, 0, 1), dtype=bool)
#test_text_features= test_text_features[:, mask]'''

#  Stacker for sparse data

train_final_features = sparse.hstack((train_numeric_features,train_text_features)).tocsr()

test_final_features = sparse.hstack((test_numeric_features,test_text_features)).tocsr()


# Check
print (train_final_features.shape)
print (train_labels.shape)

### XGBRegressor Model
    
xgb = xgboost.XGBRegressor(n_estimators=800, learning_rate=0.25, gamma=0,booster='gbtree',n_jobs=4,subsample=1,colsample_bytree=1,min_child_weight=1, max_depth=15,seed=1505)



xgb.fit(train_final_features,train_labels)
predictions = xgb.predict(test_final_features)
#results = (xgb.predict(test_final_features))
results = np.exp(predictions)
results[results<0]=0
outfile_name = 'submit_xgboost_regression.csv'
prediction = pd.DataFrame(np.array(results), columns = ['price'])
prediction.index.name = 'test_id'
prediction.to_csv(outfile_name, encoding='utf-8')
