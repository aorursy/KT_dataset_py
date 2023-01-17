import numpy as np  

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/used-electronics-data/Train.csv')

test = pd.read_csv('/kaggle/input/used-electronics-data/Test.csv')

sub = pd.read_excel('/kaggle/input/used-electronics-data/Sample_Submission.xlsx')
train.shape, test.shape, sub.shape
train.head(5)
test.head(3)
train.isnull().sum()
train.nunique()
#train = train[(train['Price'] > 2500) & (train['Price'] < 100000)]

#train = train[train['Price'] > 399]
df = train.append(test,ignore_index=True)

df.shape
#import re



#def remove_non_ascii(text):

#    return re.sub(r'[^\x00-\x7F]+',' ', text)

#df['Model_Info'] = df['Model_Info'].apply(remove_non_ascii)

#df['Additional_Description'] = df['Additional_Description'].apply(remove_non_ascii)



from sklearn.feature_extraction.text import TfidfVectorizer

tf1 = TfidfVectorizer(min_df=1, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', 

                     ngram_range=(1,1), use_idf=1, smooth_idf=1, sublinear_tf=1)

df_text1 = tf1.fit_transform(df['Model_Info'])

df_text1 = pd.DataFrame(data=df_text1.toarray(), columns=tf1.get_feature_names())



#tf2 = TfidfVectorizer()

#df_text2 = tf2.fit_transform(df['Additional_Description'])

#df_text2 = pd.DataFrame(data=df_text2.toarray(), columns=tf2.get_feature_names())
df = pd.concat([df, df_text1], axis=1) 

df.shape
df['Brand_Count'] = df['Brand'].map(df['Brand'].value_counts())

df['Locality_Count'] = df['Locality'].map(df['Locality'].value_counts())

df['CityCount'] = df['City'].map(df['City'].value_counts())

df['State_Count'] = df['State'].map(df['State'].value_counts())
#for c in ['Brand', 'Locality', 'City', 'State']:

#    df[c] = df[c].astype('category')
df.drop(['Model_Info','Additional_Description'], axis=1, inplace=True)
#df = pd.get_dummies(df, columns=['Brand', 'Locality', 'City', 'State'], drop_first=True)
train_df = df[df['Price'].isnull()!=True]

test_df = df[df['Price'].isnull()==True]

test_df.drop(['Price'], axis=1, inplace=True)
train_df['Price'] = np.log1p(train_df['Price'])
X = train_df.drop(labels=['Price'], axis=1)

y = train_df['Price'].values



X.shape, y.shape
from math import sqrt 

import lightgbm as lgb

from sklearn.metrics import mean_squared_error, mean_squared_log_error
Xtest = test_df
from xgboost import XGBRegressor

from sklearn.model_selection import KFold



errxgb = []

y_pred_totxgb = []



fold = KFold(n_splits=15, shuffle=True, random_state=42)



for train_index, test_index in fold.split(X):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    xgb = XGBRegressor(random_state=42)

    xgb.fit(X_train, y_train)



    y_pred_xgb = xgb.predict(X_test)

    print("RMSLE: ", sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_xgb))))



    errxgb.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_xgb))))

    p = xgb.predict(Xtest)

    y_pred_totxgb.append(p)
np.mean(errxgb,0)
final = np.exp(np.mean(y_pred_totxgb,0))
sub['Price'] = final
sub.head()
sub.to_excel('Output.xlsx', index=False)