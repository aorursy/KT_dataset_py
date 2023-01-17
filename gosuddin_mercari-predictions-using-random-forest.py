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
%pylab inline
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#read the data
df_train = pd.read_csv('../input/train.tsv',sep="\t")
df_test = pd.read_csv('../input/test.tsv',sep="\t")
def compute_rmsle(y_true,y_predict):
    """
    description:
    The function returns the Root Mean Squared Logarthmic Error (RMSLE)
    
    input:
    y_true    = np-array of true values of dimension (n,)
    y_predict = np-array of true values of dimension (n,)
    
    output:
    rmsle     = root mean squared logarthmic error
    """
    n = len(y_true)
    rmsle = np.sqrt(np.sum(log(y_predict+1) + log(y_true +1 )) / n )
    return rmsle
def transform_brandname(x):
    """
    description:
    The function returns a cleaner version of the brand name
    
    input:
    x = uncleaned version of brand name
    
    output:
    x = cleaned version of brand name
    """
    if x is NaN:
        return 'no_brand'
    elif x is nan:
        return 'no_brand'
    else:
        return x

def transform_catlevel(x):
    """
    description:
    The function returns a cleaner version of the category level
    
    input:
    x = uncleaned version of category level
    
    output:
    x = cleaned version of category level
    """
    if x is None:
        return 'no_level'
    else:
        return x
def cleandata(df):
    """
    description:
    The function cleans the dataset
    Transformations done:
        1. replacing all null descriptions with "No description yet"
        2. split categories and impute missing values
        3. replacing missing brands with no_brands
        4. replacing all categories with no_level
        5. remove the temporary columns
    input:
    df  = dataframe of the data
    
    output:
    df = cleaned dataframe
    """
    
    #1. replacing all null descriptions with "No description yet"
    df.item_description = df.item_description.replace(np.NaN,"No description yet")
    
    
    #2. split categories and impute missing values
    df['c_c'] = list(map(lambda x:str(x).split('/'),df.category_name))
    df[['l1','l2','l3','l4','l5']] = pd.DataFrame(df.c_c.values.tolist(),index=df.index)
    df.drop('c_c',axis = 1,inplace=True)
    
    #3. replacing missing brands with no_brands
    df['pro_bn']  =list(map(lambda x: transform_brandname(x), df.brand_name))
    df.drop('brand_name',axis = 1,inplace=True)
    
    #4. replacing all categories with no_level
    df['pro_l1'] = list(map(lambda x: transform_catlevel(x), df.l1) )
    df['pro_l2'] = list(map(lambda x: transform_catlevel(x), df.l2) )
    df['pro_l3'] = list(map(lambda x: transform_catlevel(x), df.l3) )
    df['pro_l4'] = list(map(lambda x: transform_catlevel(x), df.l4) )
    df['pro_l5'] = list(map(lambda x: transform_catlevel(x), df.l5) )
    
    #5. remove the temporary columns
    df.drop('l1',axis = 1,inplace=True)
    df.drop('l2',axis = 1,inplace=True)
    df.drop('l3',axis = 1,inplace=True)
    df.drop('l4',axis = 1,inplace=True)
    df.drop('l5',axis = 1,inplace=True)
    
    return df
#call clean data
df_train = cleandata(df_train)
df_test = cleandata(df_test)
#extract all the labels and transform it with a label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
all_names = df_train.name.tolist() +  df_test.name.tolist()
all_names = list(set(all_names))
le.fit(all_names)
df_train['name'] = le.fit_transform(df_train.name)
df_test['name'] = le.fit_transform(df_test.name)

del le
#similarly perform label encoding operation for the category levels
c_l = ['pro_l1','pro_l2','pro_l3','pro_l4','pro_l5']

for l in c_l:
    print('processing %s' % l)
    le = LabelEncoder()
    all_l = df_train[l].tolist() +  df_test[l].tolist()
    all_l = list(set(all_names))
    le.fit(all_l)
    df_train[l] = le.fit_transform(df_train[l])
    df_test[l] = le.fit_transform(df_test[l])

    del le
#create a tf-idf feature vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1),max_features= 100000, min_df = 0.15, stop_words = 'english')
tfidf_matrix =  tf.fit_transform(df_train.item_description)
feature_names = tf.get_feature_names() 
#create a list of this
feature_names
txt_features = feature_names
#apply the transform on the train and test
for txt_f in txt_features:
    df_train[txt_f] = list(map(lambda x: 1 if x.find(txt_f) != -1 else 0,df_train.item_description))
    df_test[txt_f] = list(map(lambda x: 1 if x.find(txt_f) != -1 else 0,df_test.item_description))
#create final dataset for cleaning and building a predictive model
X = pd.get_dummies(df_train.shipping)
#X = np.column_stack([X,pd.get_dummies(df_train.shipping)])
X = np.column_stack([X,df_train['name']])
X = np.column_stack([X,df_train[c_l]])
X = np.column_stack([X,df_train[txt_features]])
y = log(df_train.price+1)
#create a train-test split on the train dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#produce a simple random forest regressor
from sklearn.ensemble import RandomForestRegressor
RF_regr = RandomForestRegressor(n_estimators= 300, max_features= 'sqrt', n_jobs= -1, max_depth=16, min_samples_split=5, min_samples_leaf=5)
RF_regr.fit(X_train, y_train)
#generate predictions
y_test_pred = RF_regr.predict(X_test)
#y_test_pred = exp(y_test_pred) - 1 
#calculate the mean squared error
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_test_pred)
#create test for generating predictions
X = pd.get_dummies(df_test.shipping)
#X = np.column_stack([X,pd.get_dummies(df_test.shipping)])
X = np.column_stack([X,df_test['name']])
X = np.column_stack([X,df_test[c_l]])
X = np.column_stack([X,df_test[txt_features]])
#perform inverse transformations
log_test_prices = RF_regr.predict(X)
test_prices = exp(log_test_prices) - 1
#create submission csv
df_test['price'] = list(map(lambda x: x-1 , test_prices))
df_test.to_csv('mer_all_labels.csv',columns=['test_id','price'],index=False)
