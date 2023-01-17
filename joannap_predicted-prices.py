# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
f = open('../input/train.csv')

train = pd.read_csv(f,parse_dates=['timestamp'])
#show type of data in dataframe train

dtype_df = train.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
train=train.sort_values(by='timestamp') #sort values by date
train['price_sqm']=train.price_doc/train.full_sq #create new column with price of square metre
#create new column of date as a one number e.g 2011-08-20 as 20110820

train["yearmonthday"] = train["timestamp"].dt.year*1000 + train["timestamp"].dt.month*100+train['timestamp'].dt.day
#find columns with non numeric values

a=[item for item in train.columns if item not in (train._get_numeric_data()).columns]



a=a[1:]

for item in a:

    train.drop(a,axis=1)#remove column of non numeric values

    train1=train.join(pd.get_dummies(train[a]))#replace this column by series of values created by get_dummies function

#train1.corr()
#plot missing values from train data

missing = train1.isnull().sum(axis=0).reset_index()

missing.columns = ['column_name', 'missing_count']

missing = missing.ix[missing['missing_count']>0]

ind = np.arange(missing.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing.missing_count.values, color='y')

ax.set_yticks(ind)

ax.set_yticklabels(missing.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
train1=train1.fillna(train1.mean()) #fill dataframe where is 'Nan' value
#measure correlation between columns of train data

corr=train1.corr()

correlation_pd=pd.Series(corr.price_doc) #list of correlation for price_doc column

correlation_psqm=pd.Series(corr.price_sqm) #list of correlation for price_sqm column
correlation_pd.nlargest(20) #show 20 the most correlated values with column price_doc
correlation_psqm.nlargest(20) #show 20 the most correlated values with column price_sqm
e = open('../input/macro.csv')

macro = pd.read_csv(e,parse_dates=['timestamp'])

#create new column of date as a one number e.g 2011-08-20 as 20110820

macro["yearmonthday"] = macro["timestamp"].dt.year*1000 + macro["timestamp"].dt.month*100+macro['timestamp'].dt.day
#find missing values

missing_df = macro.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count']>0]

ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_df.missing_count.values, color='y')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
macro=macro.fillna(macro.mean()) #fill dataframe where is 'Nan' value
# choose from macro data only timestamp the same as in train data

date1=macro.timestamp[macro.timestamp=='2011-08-20'].index

date2=macro.timestamp[macro.timestamp=='2015-06-30'].index

macro1=macro.iloc[date1[0]:date2[0],]
#find columns with non numeric values

b=[item for item in macro1.columns if item not in (macro1._get_numeric_data()).columns]



b=b[1:] #first one is timestamp and it can't be changed

for item in b:

    macro1.drop(b,axis=1)#remove column of non numeric values

    macro2=train.join(pd.get_dummies(macro1[b]))#replace this column by series of values created by get_dummies function

#macro2.corr()
#join train and macro data

train2=train1.rename(index=str, columns={"timestamp": "times"})

train_macro=macro2.merge(train2,how='left', left_on='timestamp', right_on='times')
#chcecking types in dataframe

dtype_df = train_macro.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
from sklearn.preprocessing import LabelEncoder
#remove object type, which disturbs to create model

for f in train_macro.columns:

    if train_macro[f].dtype=='object':

        lbl = LabelEncoder()

        lbl.fit(list(train_macro[f].values)) 

        train_macro[f] = lbl.transform(list(train_macro[f].values))

 
#chcecking again types in dataframe

dtype_df = train_macro.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
d = open('../input/test.csv')

test = pd.read_csv(d,parse_dates=['timestamp'])
test=test.fillna(test.mean()) #fill dataframe where is 'Nan' value
test['price_doc'] = np.nan
from xgboost import XGBClassifier

#from sklearn.model_selection import train_test_split

features = ['num_room_x','full_sq_x','sport_count_5000_x','trc_count_5000_x','office_sqm_5000_x']

x_train = train_macro[features]

y_train=train_macro['price_doc']

x_test=test[features]

# split data into train and test sets

#seed = 7

#test_size = 0.33

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data

model = XGBClassifier()

model.fit(train_macro, test['price_doc'])

# make predictions for test data

y_pred = model.predict(test['price_doc'])

from sklearn.tree import DecisionTreeClassifier

features1 = ['num_room_x','full_sq_x','sport_count_5000','trc_count_5000','office_sqm_5000','sport_obejct_ratio']

featurevals = train_macro[features1]

labels = train_macro['price_doc']

dt = DecisionTreeClassifier(min_samples_split=19) # parameter is optional

dt.fit(featurevals,labels)

predictions = dt.predict(test['price_doc'])