# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

color = sns.color_palette()

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data=pd.read_csv('../input/train.csv',parse_dates=['timestamp'])

test_data=pd.read_csv('../input/test.csv',parse_dates=['timestamp'])

macro_data=pd.read_csv('../input/macro.csv',parse_dates=['timestamp'])

print (train_data.shape)

print (test_data.shape)

print (macro_data.shape)
##We will merge the test,train data with macro data

train_data=pd.merge(train_data,macro_data,how='left',on='timestamp')

test_data=pd.merge(test_data,macro_data,how='left',on='timestamp')



##Let us visyalize the target variables using a boxplot to see the outliers

plt.figure(figsize=(10,8))

sns.boxplot(train_data['price_doc'],orient='v')

plt.show()
##As we can see there are a lot of values which qualify as outliers.We will remove the variables which are more than

##99 percentile of the data



ulimit=np.percentile(train_data['price_doc'].values,99)

llimit=np.percentile(train_data['price_doc'].values,1)

train_data=train_data[(train_data['price_doc']<ulimit) & (train_data['price_doc']>llimit)]
##Visualizing the target data

plt.figure(figsize=(10,6))

sns.distplot(train_data['price_doc'],kde=False,bins=50)

plt.xlabel('price')

plt.show()
#We can see the data is positively skewed and the range in large.We can also use the logarithmic plot to visualize the data better.

##Lets plot log of target variable

plt.figure(figsize=(10,6))

sns.distplot(np.log(train_data['price_doc']),kde=False,bins=50)

plt.xlabel('price')

plt.show()
train_data['year_str']=train_data['timestamp'].apply(lambda x :  x.strftime('%Y-%m-%d'))

#test_data['year_str']=test_data['timestamp'].apply(lambda x :  x.strftime('%Y-%m-%d'))

train_data['year_str'].head()
##We can see the data better with the logarithmic plot.



##Lets see the increase of price over time





train_data['yearmonth']=train_data['year_str'].apply(lambda x:x[:4]+x[5:7])



grouped_data_yearmonth=train_data.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()

grouped_data_yearmonth.columns=['yearmonth','median price']

grouped_data_yearmonth.head()

#test_data['yearmonth']=test_data['year_str'].apply(lambda x:x[:4]+x[5:7])

###Barplot for price increase

plt.figure(figsize=(10,8))

sns.barplot(x=grouped_data_yearmonth['yearmonth'],y=grouped_data_yearmonth['median price'],color='b')

plt.xticks(rotation='vertical')

plt.show()
##We will now visualize the number of houses built each year

grouped_data_count=train_data.groupby('build_year')['id'].aggregate('count').reset_index()

grouped_data_count.columns=['build_year','count']
##Lets check the minimum and maximum build year dates

print (grouped_data_count.iloc[grouped_data_count['build_year'].idxmax()])

print (grouped_data_count.iloc[grouped_data_count['build_year'].idxmin()])



train_data=train_data[(train_data['build_year']<2019)&(train_data['build_year']>1690)]

#test_data=test_data[(test_data['build_year']<2019)&(test_data['build_year']>1690)]

train_data.head()
train_data.shape


##These values clearly suggests that this is not

#correct and needs to be rectified during our data cleaning process



#Lets visualize this data

grouped_data_count=grouped_data_count[(grouped_data_count['build_year']>1950) & (grouped_data_count['build_year']<2018) ]

plt.figure(figsize=(10,8))

sns.barplot(grouped_data_count['build_year'],grouped_data_count['count'],color='g')

plt.xticks(rotation='vertical')

plt.show()
##Lets visualize the internal characteristics of the house and its relation with price

plt.figure(figsize=(10,8))

internal_characteristics=['full_sq', 'life_sq', 'floor', 'max_floor', 'material',

                          'num_room', 'kitch_sq','price_doc']

heatmap_data=train_data[internal_characteristics].corr()

sns.heatmap(heatmap_data,annot=True)

plt.show()
##We can see a high co-relation between the full_sq and the  num of rooms
##Lets start working on data cleaning and removing bad data from dataset

##We will start by visualizing the missing data in all the columns



train_missing=train_data.isnull().sum()/len(train_data)

train_missing=train_missing.drop(train_missing[train_missing==0].index).sort_values(ascending=False).reset_index()

train_missing.columns=['column name','missing percentage']

plt.figure(figsize=(12,8))

sns.barplot(train_missing['column name'],train_missing['missing percentage'],palette='inferno')

plt.xticks(rotation='vertical')

plt.show()
##Count of different datatypes

plt.figure(figsize=(10,8))

sns.countplot(train_data.dtypes,)

plt.show()
test_data.head().T
from sklearn.preprocessing import LabelEncoder



def encodeData(data):

    for c in data.columns:

        if data[c].dtype=='object':

            lbl=LabelEncoder()

            lbl.fit(list(data[c].values))

            data[c]=lbl.transform(list(data[c].values))

    return data 

 

train_df=encodeData(train_data)

#test_df=encodeData(test_data)





y_train=train_df['price_doc']

X_train=train_df.drop(['id','timestamp','price_doc'],axis=1)



#train_y=train_data['price_doc']

#train_X=train_data.drop(['id','timestamp','price_doc'],axis=1)
import xgboost as xgb

xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}

dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #

# plot the important features #

fig, ax = plt.subplots(figsize=(12,25))

xgb.plot_importance(model,max_num_features=10,  height=0.8, ax=ax)

plt.show()
##Lets visualize the missing values of these data

imp_variables=['full_sq','life_sq','floor','build_year','max_floor','kitch_sq','state',

               'kindergarten_km','railroad_km','micex']

##We need to add yearmonth to test_data





train_df1=train_df[imp_variables]

test_df1=test_data[imp_variables]
test_df1=encodeData(test_df1)
X_train=train_df1

y_train=train_df['price_doc']

x_test=test_df1
xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(X_train, y_train)

dtest = xgb.DMatrix(x_test)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)



num_boost_rounds = len(cv_output)

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round= num_boost_rounds)
y_predict = model.predict(dtest)
id_test = test_data.id
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})



output.to_csv('xgbSub_2.csv', index=False)