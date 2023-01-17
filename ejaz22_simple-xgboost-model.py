# imports

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

import pandas as pd

import numpy as np 

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# read files

submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv",index_col='id')

df_train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv",index_col='id')

df_test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv",index_col='id')

df = pd.concat([df_train, df_test],axis=0,ignore_index=True) # combine training and testing data
# check submission file first, which gives an idea as what to do, we need to predict probability

submission.head()
# function to describe variables

def desc(df):

    summ = pd.DataFrame(df.dtypes,columns=['Data_Types'])

    summ = summ.reset_index()

    summ['Columns'] = summ['index']

    summ = summ[['Columns','Data_Types']]

    summ['Missing'] = df.isnull().sum().values    

    summ['Uniques'] = df.nunique().values

    return summ



# function to analyse missing values

def nulls_report(df):

    nulls = df.isnull().sum()

    nulls = nulls[df.isnull().sum()>0].sort_values(ascending=False)

    nulls_report = pd.concat([nulls, nulls / df.shape[0]], axis=1, keys=['Missing_Values','Missing_Ratio'])

    return nulls_report
# test data, there are no missing values

desc(df_test)
# distribution of target variable

df_train['target'].value_counts(normalize = True).plot(kind='barh',title='Distribution of Target Variable')
# Binary Encdoing

# bin_o and bin_1 need not be converted as these are already converted

# bin_3and bin_4 are binary variables representing T/F and Y/N. We can convert them to 0 or 1.

df['bin_3'] = df['bin_3'].map({'T':1,'F':0})

df['bin_4'] = df['bin_4'].map({'Y':1,'N':0})
# Ordinal Encoding

# ord_0 need not to be converted

# ord_1 and ord_2 has ordinal data. We can manually encode these variables.

# ( ord_3,ord_4,ord_5 are of hight cardinality)



# ord_1 and ord_2

d1 = {'Grandmaster': 5, 'Expert': 4 , 'Novice':1 , 'Contributor':2 , 'Master': 3}

d2 = {'Cold': 2, 'Hot':4, 'Lava Hot': 6, 'Boiling Hot': 5, 'Freezing': 1, 'Warm': 3}

df['ord_1'] = df['ord_1'].map(d1)

df['ord_2'] = df['ord_2'].map(d2)



# ord_3 and ord_4

df['ord_3'] = df['ord_3'].astype('category')

df['ord_4'] = df['ord_4'].astype('category')

d3 = dict(zip(df['ord_3'],df['ord_3'].cat.codes))

d4 = dict(zip(df['ord_4'],df['ord_4'].cat.codes))

df['ord_3'] = df['ord_3'].map(d3)

df['ord_4'] = df['ord_4'].map(d4)



df['ord_3'] = df['ord_3'].astype(int)

df['ord_4'] = df['ord_4'].astype(int)



#  ord_5

li = sorted(list(set(df['ord_5'].values)))

d5 = dict(zip(li, range(len(li))))  # mapping dict for ord_5

df['ord_5'] = df['ord_5'].map(d5)
# one hot encoding for column : nom_0 to nom_4

df = pd.get_dummies(df, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],

                        prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], 

                        drop_first=True)
# encoding hex feature

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

features_hex = ['nom_5','nom_6','nom_7','nom_8','nom_9']



for col in features_hex:

    le.fit(df[col])

    df[col] = le.transform(df[col])
# convert cyclical features such as day and month into 2d sin-cos features

df['day_sin'] = np.sin(2*np.pi * df['day']/7)

df['day_cos'] = np.cos(2*np.pi * df['day']/7)

df['month_sin'] = np.sin(2*np.pi * df['month']/12)

df['month_cos'] = np.cos(2*np.pi * df['month']/12)

df.drop(columns=['day','month'],inplace=True)



# plot features in 2d

df.sample(1000).plot.scatter('month_sin','month_cos').set_aspect('equal')
# get training ,testing, validation and target

from sklearn.model_selection import train_test_split

y_train = df_train['target']

X_train = df[:len(df_train)].drop(['target'],axis=1)

X_test = df[len(df_train):].drop(['target'],axis=1)



# split train and validation data

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.001, random_state=0)
# xgboost

from xgboost import XGBClassifier

xgb = XGBClassifier(objective= 'binary:logistic',

                    learning_rate=0.1,

                    max_depth=3,

                    n_estimators=200,

                    scale_pos_weight=2,

                    random_state=1,

                    colsample_bytree=0.5)



xgb.fit(X_train,y_train)
#from sklearn.linear_model import LogisticRegression

#lr=LogisticRegression(C=0.125, solver="lbfgs", max_iter=500) 

#lr.fit(X_train, y_train)
#submission

y_pred = xgb.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({'id':df_test.index,'target':y_pred})

submission.to_csv('submission.csv', index=False)