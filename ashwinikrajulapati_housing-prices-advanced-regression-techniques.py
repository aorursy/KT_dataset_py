# This Python 3 environment comes with many helpful analytics libraries installed



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import lightgbm as lgb



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



#  For text feature extraction

#     Ref: https://stackoverflow.com/questions/22489264/is-a-countvectorizer-the-same-as-tfidfvectorizer-with-use-idf-false

from sklearn.feature_extraction.text import TfidfVectorizer



# 2.2 For preprocessing--conversion of categorical to dummy

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

from scipy import sparse

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA as sklearnPCA

# 2.3 For splitting arrays into train/test

from sklearn.model_selection import train_test_split,GridSearchCV

#from sklearn.decomposition import TruncatedSVD



# 2.4 Manipulate sparse matricies

from scipy.sparse import  hstack



# 3. For modeling

from sklearn.ensemble import RandomForestRegressor



# 3.2 ridge regression 

from sklearn.linear_model import Ridge



# 4. Miscellenous libraries

# 4.1 Timing processes

import time

# 4.2 OS related and releasing memory

import os, gc

# System related



import sys
train_dt= pd.read_csv("../input/train.csv")  

test= pd.read_csv("../input/test.csv")



# 6.1 List columns of the two datasets

test.columns.values

train_dt.columns.values
#Head of train dataset

train_dt.head()
# Structure of train dataset

train_dt.shape
train_dt.SalePrice.describe()
train_dt.BsmtFinSF1.describe()
#Check skewness of Sales price

target = np.log(train_dt.SalePrice)

plt.hist(target, color='blue')

plt.show()


# Get a list of columns having NAs

def ColWithNAs(x):            

    z =x.isnull()

    df = np.sum(z, axis = 0)       # Sum vertically, across rows

    col = df[df > 0].index.values 

    return (col)

    



## Fill in missing values

def MissingValues(t , filler = "other"):

    return(t.fillna(value = filler))



def MissingValuesMean(t):

    return(t.fillna(t.mean(),inplace=True))







#  Convert a pandas series of text values into tfidf ndarray

def Tf_Idf(x):

    """x is pandas 'series' having text entries

    """

    # Convert a collection of text documents to a matrix of token counts

    #  Instantiate CountVectorizer object

    countvect = TfidfVectorizer(max_features = MAX_FEATURES,  # build vocabulary of top max_features ordered by tf

                              ngram_range = (1,NGRAMS),

                              stop_words = "english",

                              lowercase = True,

                              norm= "l2"          # Norm used to normalize term vectors.

                              )

    tfidf = countvect.fit_transform(x)

    # Return tfidf sparse matrix

    return (tfidf)



    

# Convert categorical features to dummy

def DoDummy(x):

   #data_dummy =pd.get_dummies(x)

   # Try: le.fit_transform(list('abc'))

    le = LabelEncoder()

    # Apply across all columns of x

    y = x.apply(le.fit_transform)

    # Try:  enc.fit_transform([[1,2],[2,1]]).toarray()

    enc = OneHotEncoder(categorical_features = "all")  # ‘all’: All features are treated as categorical.

    enc.fit(y)

    trans = enc.transform(y)

    return(trans)
#Numeric columns

numeric_train = train_dt.select_dtypes(include=[np.number])

numeric_train.dtypes
#Check the co-relation 

corr_tr = numeric_train.corr()

#first five features are the most positively correlated with SalePrice

corr_tr['SalePrice'].sort_values(ascending=False)[:5]


#  Assign 'price' column to a variable and drop it from tr

#     Also drop train_id/test_id columns

y = train_dt['SalePrice']              # This is also the target variable



train_dt.drop( ['SalePrice', 'Id'], inplace = True, axis = 'columns')

test.drop( ['Id'], inplace = True, axis = 'columns')  



# Stack both train and test one upon another

df=[train_dt,test]

Data_df=pd.concat(df, axis = 'index') 



train_dt.isnull().sum().sort_values()



test.isnull().sum().sort_values()



Data_df
#delete the columns Alley/PoolQC/MiscFeature in train dataset , has all NA values

drop_column = ['MiscFeature','PoolQC', 'Alley','Fence','FireplaceQu']

Data_df.drop( ['MiscFeature','PoolQC', 'Alley','Fence','FireplaceQu'], inplace = True, axis = 'columns')

    

#Numeric columns

numeric_df = Data_df.select_dtypes(include=[np.number])

numeric_col=Data_df.select_dtypes(include=[np.number]).columns.values

numeric_df.dtypes

# Get columns with missing values

Data_df[numeric_col] = MissingValuesMean(Data_df[numeric_col])

 #Data_df[numeric_col] = MissingValuesAsMean(Data_df[numeric_df])



col = ColWithNAs(Data_df)

col                         # category_name, brand_name, item_description

print("Columns with NA :",col,"\n")

Data_df[col] = MissingValues(Data_df[col])



Data_df.isnull().sum().sort_values()

Data_df.shape
categorical_columns_name = Data_df.columns.difference(numeric_col)

categorical_columns_name
#unique caolumns

unique_columns_name = Data_df[categorical_columns_name].nunique()

unique_columns_name
X_std = StandardScaler().fit_transform(Data_df[numeric_col])

sklearn_pca = sklearnPCA(n_components=4)

num_pca = sklearn_pca.fit_transform(X_std)



num_pca.shape

type(num_pca)

sparse_pca=sparse.csr_matrix(num_pca)
#  Convert categorical to dummy

start = time.time()

df_dummy = DoDummy(Data_df[categorical_columns_name])

end = time.time()   # 3 minutes

dummy_time = (end-start)/60.0

print("Time taken to convert categorical to dummy features: ", dummy_time , " minutes" )

df_dummy.shape
# Concatenate Categorical + Numerical Data

df_sp = sparse.hstack([df_dummy,sparse_pca], format = 'csr')



df_sp.shape

type(df_sp)
##  Unstack tr and test, sparse matrices

df_train = df_sp[ : train_dt.shape[0] , : ]

df_test = df_sp[train_dt.shape[0] :, : ]

df_train.shape

df_test.shape


#  PArtition datasets into train + validation

y_train = np.log1p(y)    # Criterion is rmsle

X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(

                                     df_train, y_train,

                                     test_size=0.40,

                                     random_state=42

                                     )

type(X_train_sparse)


## AA. Ensemble based prediction

MAXDEPTH = 10

#  Instantiate a RandomRegressor object

regr = RandomForestRegressor(n_estimators=300,       # No of trees in forest

                             criterion = "mse",       # Can also be mae

                             max_features = "sqrt",  # no of features to consider for the best split

                             max_depth= MAXDEPTH,    #  maximum depth of the tree

                             min_samples_split= 2,   # minimum number of samples required to split an internal node

                             min_impurity_decrease=0, # Split node if impurity decreases greater than this value.

                             oob_score = True,       # whether to use out-of-bag samples to estimate error on unseen data.

                             n_jobs = -1,            #  No of jobs to run in parallel

                             random_state=0,

                             verbose = 10            # Controls verbosity of process

                             )





#  Do regression

start = time.time()

regr.fit(X_train_sparse,y_train_sparse)

end = time.time()

rf_model_time=(end-start)/60.0

print("Time taken to model: ", rf_model_time , " minutes" ) # 6 minutes



# Sales price

regr.predict(df_test)

#Prediction and performance

rf_sparse=regr.predict(X_test_sparse)

squared = np.square(rf_sparse - y_test_sparse)

rf_error = np.sqrt(np.sum(squared)/len(y_test_sparse))

rf_error


## B. Linear Models

# Ridge Regression

model = Ridge(alpha = 1.0,            # Regularization strength. Try 0.0 and 40

              solver = "lsqr",        # auto,svd,cholesky,lsqr,sparse_cg,sag,saga

              fit_intercept=False     # Data is already normalized and centered

              )



model.fit(X_train_sparse, y_train_sparse)

ridge_pre = model.predict(X_test_sparse)

squared = np.square(ridge_pre-y_test_sparse)

ridge_error = np.sqrt(np.sum(squared)/len(y_test_sparse))

ridge_error


## CC. Gradient Boosting Model



# Lightgbm model



params = {

    'learning_rate': 0.25,

    'application': 'regression',

    'is_enable_sparse' : 'true',

    'max_depth': 3,

    'num_leaves': 60,

    'verbosity': -1,

    'bagging_fraction': 0.5,

    'nthread': 4,

    'metric': 'RMSE'

}



d_train = lgb.Dataset(X_train_sparse, label=y_train_sparse)

d_test = lgb.Dataset(X_test_sparse, label = y_test_sparse)

watchlist = [d_train, d_test]





start = time.time()

model = lgb.train(params,

                  train_set=d_train,

                  num_boost_round=240,

                  valid_sets=watchlist,

                  early_stopping_rounds=20,

                  verbose_eval=10)

end = time.time()

end - start



                  

lgb_pred = model.predict(X_test_sparse)

squared = np.square(lgb_pred - y_test_sparse)

lgb_error = np.sqrt(np.sum(squared)/len(y_test_sparse))

lgb_error
