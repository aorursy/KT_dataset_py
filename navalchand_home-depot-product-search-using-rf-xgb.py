# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

from nltk.stem.snowball import SnowballStemmer



stemmer = SnowballStemmer('english')



df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")

df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")

df_attr = pd.read_csv('../input/attributes.csv')

df_pro_desc = pd.read_csv('../input/product_descriptions.csv')
df_attr.head()
## get shape of actual train dataframe

num_train = df_train.shape[0]

df_train.head()


def str_stemmer(s):

    ''' To stem and lamatize the sentences so that we can avoid the difference between computing , computed , computs'''

    return " ".join([stemmer.stem(word) for word in s.lower().split()])



def str_common_word(str1, str2):

    '''Get count of words common in two input strings. Basic word matching'''

    return sum(int(str2.find(word)>=0) for word in str1.split())
### concatenate both train and test data set.

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)



### add all product info to the above dataframe

df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')



### applying str_stemmer to stem and lamitize the values

df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))

df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))

df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))

### calculating the length of search term

df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

### combine search_term , product_title and product_description

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

### get common words in search_term and product_title

df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))

### get count of common words in search_term and product_description

df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

### display first rows in dataframe

df_all.head()
### taking a sub-set of dataframe to train from above df_all

df_train = df_all.iloc[:num_train]

### seperate test data from df_all

df_test = df_all.iloc[num_train:]

id_test = df_test['id']



### test and train data

y_train = df_train['relevance'].values

X_train = df_train[[w for w in list(df_train.columns) if w not in ['search_term','product_title','product_description','product_info' , 'id','relevance']]].values

X_test = df_test[[w for w in list(df_test.columns) if w not in ['search_term','product_title','product_description','product_info' ,'id','relevance']]].values



### training a random forest regressor

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)

clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

### to test model performance 

y_true_pred = clf.predict(X_train)



print ('RMSE using Random Forest Regressor consedering basic features : ' , np.sqrt(((y_train - y_true_pred) ** 2).mean()))

#pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)
from fuzzywuzzy import fuzz

def fuzzy_partial_ratio(string_1 , string_2):

    return fuzz.partial_ratio(string_1, string_2)



fuzzy_partial_ratio('delta vero 1-handl shower onli faucet trim kit...' , 'shower onli faucet')
def fuzzy_token_sort_ratio(string_1,string_2):

    return fuzz.token_sort_ratio(string_1,string_2)

fuzzy_token_sort_ratio('delta vero 1-handl shower onli faucet trim kit...' , 'shower onli faucet')
### adding new features

### 1. Fuzzy partial ratio on 'search_term' and 'product_title' 

### 2. Fuzzy partial ratio on  'search_term' and 'product_description'

df_all['fuzzy_ratio_in_title'] = df_all['product_info'].map(lambda x:fuzzy_partial_ratio(x.split('\t')[0],x.split('\t')[1]))

df_all['fuzzy_ratio_in_description'] = df_all['product_info'].map(lambda x:fuzzy_partial_ratio(x.split('\t')[0],x.split('\t')[2]))



df_all.head()
### adding new features

### 1. Fuzzy token_sort_ratio on 'search_term' and 'product_title' 

### 2. Fuzzy token_sort_ratio on  'search_term' and 'product_description'



df_all['fuzzy_token_sort_ratio_in_title'] = df_all['product_info'].map(lambda x:fuzzy_token_sort_ratio(x.split('\t')[0],x.split('\t')[1]))

df_all['fuzzy_token_sort_ratio_in_description'] = df_all['product_info'].map(lambda x:fuzzy_token_sort_ratio(x.split('\t')[0],x.split('\t')[2]))



df_all.head()
columns_to_train = ['len_of_query' , 'word_in_title' , 'word_in_description' , 'fuzzy_ratio_in_title' , 'fuzzy_ratio_in_description' , 'fuzzy_token_sort_ratio_in_title' , 'fuzzy_token_sort_ratio_in_description']

### taking a sub-set of dataframe to train from above df_all

df_train = df_all.iloc[:num_train]

### seperate test data from df_all

df_test = df_all.iloc[num_train:]

id_test = df_test['id']



### test and train data

### training using all previous and fuzzy features

y_train = df_train['relevance'].values

X_train = df_train[columns_to_train].values

X_test = df_test[columns_to_train].values



### training a random forest regressor

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)

clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

### to test model performance 

y_true_pred = clf.predict(X_train)



print ('RMSE using Random Forest Regressor consedering basic and fuzzy features : ' , np.sqrt(((y_train - y_true_pred) ** 2).mean()))
columns_to_train = ['len_of_query'  , 'fuzzy_ratio_in_title' , 'fuzzy_ratio_in_description', 'fuzzy_token_sort_ratio_in_title' , 'fuzzy_token_sort_ratio_in_description']

### taking a sub-set of dataframe to train from above df_all

df_train = df_all.iloc[:num_train]

### seperate test data from df_all

df_test = df_all.iloc[num_train:]

id_test = df_test['id']



### test and train data

### training using fuzzy features

y_train = df_train['relevance'].values

X_train = df_train[columns_to_train].values

X_test = df_test[columns_to_train].values



### training a random forest regressor

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)

clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

### to test model performance 

y_true_pred = clf.predict(X_train)



print ('RMSE using Random Forest Regressor consedering only fuzzy features : ' , np.sqrt(((y_train - y_true_pred) ** 2).mean()))
import xgboost as xgb

from sklearn.datasets import dump_svmlight_file

from xgboost import plot_importance



columns_to_train = ['len_of_query' , 'word_in_title' , 'word_in_description' , 'fuzzy_ratio_in_title' , 'fuzzy_ratio_in_description' , 'fuzzy_token_sort_ratio_in_title' , 'fuzzy_token_sort_ratio_in_description']

### taking a sub-set of dataframe to train from above df_all

df_train = df_all.iloc[:num_train]

### seperate test data from df_all

df_test = df_all.iloc[num_train:]

id_test = df_test['id']



### test and train data

### training using all previous and fuzzy features

y_train = df_train['relevance'].values

X_train = df_train[columns_to_train]

X_test = df_test[columns_to_train]
bst = xgb.XGBRegressor(max_depth = 6,

                    n_estimators = 100).fit(X_train , y_train)
### to test model performance 

y_true_pred = bst.predict(X_train)

print ('RMSE using XGBoost Regressor consedering basic and fuzzy features : ' , np.sqrt(((y_train - y_true_pred) ** 2).mean()))

## plotting the feature importance

plot_importance(bst)



### adding new features

### 1. Fuzzy token_sort_ratio on 'search_term' and 'product_title' 

### 2. Fuzzy token_sort_ratio on  'search_term' and 'product_description'



df_all['fuzzy_ratio_in_title_description'] = df_all['product_info'].map(lambda x:fuzzy_partial_ratio(x.split('\t')[0]," ".join(x.split('\t')[1:])))

df_all['fuzzy_token_sort_ratio_in_title_description'] = df_all['product_info'].map(lambda x:fuzzy_token_sort_ratio(x.split('\t')[0]," ".join(x.split('\t')[1:])))



df_all.head()
import xgboost as xgb

from sklearn.datasets import dump_svmlight_file

from xgboost import plot_importance



columns_to_train = ['len_of_query' , 'word_in_title' , 'word_in_description' , 'fuzzy_ratio_in_title' , 'fuzzy_ratio_in_description' , 'fuzzy_token_sort_ratio_in_title' , 'fuzzy_token_sort_ratio_in_description' , 'fuzzy_ratio_in_title_description' , 'fuzzy_token_sort_ratio_in_title_description']

### taking a sub-set of dataframe to train from above df_all

df_train = df_all.iloc[:num_train]

### seperate test data from df_all

df_test = df_all.iloc[num_train:]

id_test = df_test['id']



### test and train data

### training using all previous and fuzzy features

y_train = df_train['relevance'].values

X_train = df_train[columns_to_train]

X_test = df_test[columns_to_train]



bst = xgb.XGBRegressor(max_depth = 6,

                    n_estimators = 50).fit(X_train , y_train)
### to test model performance 

y_true_pred = bst.predict(X_train)

print ('RMSE using XGBoost Regressor consedering basic and fuzzy features : ' , np.sqrt(((y_train - y_true_pred) ** 2).mean()))
import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.cross_validation import train_test_split





svr = SVR(kernel='linear')

lm = LinearRegression()

#svr.fit(X_train,y_train)

lm.fit(X_train, y_train)



### to test model performance 

y_true_pred = lm.predict(X_train)

print ('RMSE using linear Regressor consedering basic and fuzzy features : ' , np.sqrt(((y_train - y_true_pred) ** 2).mean()))

### to test model performance 

y_true_pred = svr.predict(X_train)

print ('RMSE using SVM Regressor consedering basic and fuzzy features : ' , np.sqrt(((y_train - y_true_pred) ** 2).mean()))



y_pred = bst.predict(X_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)
### searching for best parameters



from sklearn.model_selection import KFold, train_test_split, GridSearchCV

xgb_model = xgb.XGBRegressor()

clf = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6,8],

                    'n_estimators': [20,50,100,200]}, verbose=1)

clf.fit(X_train , y_train)

print(clf.best_score_)

print(clf.best_params_)
### to test model performance 

y_true_pred = clf.predict(X_train)

print ('RMSE for all features using XGB : ' , np.sqrt(((y_train - y_true_pred) ** 2).mean()))
#Choose all predictors except target & IDcols

predictors = ['len_of_query' , 'word_in_title' , 'word_in_description' , 'fuzzy_ratio_in_title' , 'fuzzy_ratio_in_description' , 'fuzzy_token_sort_ratio_in_title' , 'fuzzy_token_sort_ratio_in_description']

xgb1 = xgb.XGBRegressor(

 learning_rate =0.01,

 n_estimators=100,

 max_depth=4,

 min_child_weight=2,

 gamma=0.3,

 subsample=0.8,

 colsample_bytree=0.8,

 nthread=4,

 scale_pos_weight=1,

 seed=27)



xgb1.fit(X_train , y_train)