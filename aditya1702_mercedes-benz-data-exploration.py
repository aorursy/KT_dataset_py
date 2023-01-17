%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt



import numpy as np

import pandas as pd

import math



from sklearn.preprocessing import LabelEncoder

import xgboost as xgb



#Disable pandas warnings

pd.options.mode.chained_assignment = None



import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head()
sns.countplot(train.dtypes)
train.describe()
sum(train.isnull().sum()), sum(test.isnull().sum())
train.shape, test.shape
for col in train.select_dtypes(['object']).columns:

    lb=LabelEncoder()

    lb.fit(train[col])

    train[col]=lb.transform(train[col])

    

x_train=train.drop(['y','ID'],1)

y_train=train['y']



xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}

dtrain = xgb.DMatrix(x_train, y_train, feature_names=x_train.columns.values)

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=100)
fig, ax = plt.subplots(figsize=(12,15))

xgb.plot_importance(model, height=0.8, ax=ax, max_num_features=30)

plt.show()
plt.figure(figsize=(10,10))

top_features=['X8','X5','X0','X6','X1','X3','X2','y']

train_top_features_df=train[top_features]

test_top_features_df=test[top_features[0:len(top_features)-1]]

sns.heatmap(train_top_features_df.corr(), annot=True)
train_top_features_df.X5.describe()
train_top_features_df.X5.value_counts()


x5_df=train_top_features_df.groupby(by='X5')['y'].aggregate(np.mean).reset_index()

x5_df=x5_df.sort_values(by='X5')

plt.figure(1,figsize=(10,5))

ind=np.arange(x5_df.shape[0])

sns.barplot(x5_df.X5.values, x5_df.y.values)

plt.xlabel("X5")

plt.ylabel("Mean Time")

plt.title("Mean Time vs X5")







x6_df=train_top_features_df.groupby(by='X6')['y'].aggregate(np.mean).reset_index()

x6_df=x6_df.sort_values(by='X6')

plt.figure(2,figsize=(10,5))

ind=np.arange(x6_df.shape[0])

sns.barplot(x6_df.X6.values, x6_df.y.values)

plt.xlabel("X6")

plt.ylabel("Mean Time")

plt.title("Mean Time vs X6")







x1_df=train_top_features_df.groupby(by='X1')['y'].aggregate(np.mean).reset_index()

x1_df=x1_df.sort_values(by='X1')

plt.figure(3,figsize=(10,5))

ind=np.arange(x1_df.shape[0])

sns.barplot(x1_df.X1.values, x1_df.y.values)

plt.xlabel("X1")

plt.ylabel("Mean Time")

plt.title("Mean Time vs X1")





x2_df=train_top_features_df.groupby(by='X2')['y'].aggregate(np.mean).reset_index()

x2_df=x2_df.sort_values(by='X2')

plt.figure(4,figsize=(10,5))

ind=np.arange(x2_df.shape[0])

sns.barplot(x2_df.X2.values, x2_df.y.values)

plt.xlabel("X2")

plt.ylabel("Mean Time")

plt.title("Mean Time vs X2")





x5_df=train_top_features_df.groupby(by='X5')['y'].aggregate(np.mean).reset_index()

x5_df=x5_df.sort_values(by='X5')

plt.figure(5,figsize=(10,5))

ind=np.arange(x5_df.shape[0])

sns.barplot(x5_df.X5.values, x5_df.y.values)

plt.xlabel("X5")

plt.ylabel("Mean Time")

plt.title("Mean Time vs X5")







x3_df=train_top_features_df.groupby(by='X3')['y'].aggregate(np.mean).reset_index()

x3_df=x3_df.sort_values(by='X3')

plt.figure(6,figsize=(10,5))

ind=np.arange(x3_df.shape[0])

sns.barplot(x3_df.X3.values, x3_df.y.values)

plt.xlabel("X3")

plt.ylabel("Mean Time")

plt.title("Mean Time vs X3")







x8_df=train_top_features_df.groupby(by='X8')['y'].aggregate(np.mean).reset_index()

x8_df=x8_df.sort_values(by='X8')

plt.figure(7,figsize=(10,5))

ind=np.arange(x8_df.shape[0])

sns.barplot(x8_df.X8.values, x8_df.y.values)

plt.xlabel("X8")

plt.ylabel("Mean Time")

plt.title("Mean Time vs X8")

plt.show()
plt.figure(1,figsize=(10,5))

ind=np.arange(x1_df.shape[0])

plt.scatter(train_top_features_df.X1.values, train_top_features_df.y.values)

plt.xlabel("X1")

plt.ylabel("Time")

plt.title("Mean Time vs X1")



plt.figure(2,figsize=(10,5))

ind=np.arange(x1_df.shape[0])

plt.scatter(train_top_features_df.X2.values, train_top_features_df.y.values)

plt.xlabel("X2")

plt.ylabel("Time")

plt.title("Mean Time vs X2")



plt.figure(3,figsize=(10,5))

ind=np.arange(x1_df.shape[0])

plt.scatter(train_top_features_df.X3.values, train_top_features_df.y.values)

plt.xlabel("X3")

plt.ylabel("Time")

plt.title("Mean Time vs X3")





plt.figure(4,figsize=(10,5))

ind=np.arange(x1_df.shape[0])

plt.scatter(train_top_features_df.X5.values, train_top_features_df.y.values)

plt.xlabel("X5")

plt.ylabel("Time")

plt.title("Mean Time vs X5")





plt.figure(5,figsize=(10,5))

ind=np.arange(x1_df.shape[0])

plt.scatter(train_top_features_df.X6.values, train_top_features_df.y.values)

plt.xlabel("X6")

plt.ylabel("Time")

plt.title("Mean Time vs X6")

plt.show()
binary_df=pd.DataFrame()

zeros={}

ones={}



for col in train.select_dtypes(['int64']).columns:

    num_ones=len(train[col][train[col]==1])

    num_zeros=len(train[col][train[col]==0])

    

    if num_zeros+num_ones == len(train[col]):

        zeros[col]=num_zeros

        ones[col]=num_ones        
binary_df['columns']=zeros.keys()

binary_df['ones']=ones.values()

binary_df['zeros']=zeros.values()

binary_df=binary_df.sort_values(by='zeros')







ind=np.arange(binary_df.shape[0])



width = 0.35

plt.figure(figsize=(6,100))

p1 = plt.barh(ind, binary_df.ones.values, width, color='yellow')

p2 = plt.barh(ind, binary_df.zeros.values, width, left=binary_df.ones.values, color="green")

plt.yticks(ind, binary_df['columns'])

plt.legend((p1[0], p2[0]), ('Zero count', 'One Count'))

plt.show()
plt.figure(figsize=(15,10))

train['y'].hist(bins=100)

plt.xlabel('Y')

plt.title('Distribution of Y variable')

plt.show()
plt.figure(figsize=(15,10))

np.log1p(train['y']).hist(bins=100)

plt.xlabel('Log of Y')

plt.title('Distribution of Log of Y variable')

plt.show()
var_name = "ID"

plt.figure(figsize=(10,10))

sns.regplot(x=var_name, y='y', data=train, scatter_kws={'alpha':0.5, 's':30})

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
num_ones_row={}

num_zeros_row={}

y={}

for index in train.index:

    num_ones_row[index]=len(train.ix[index][train.ix[index]==1])

    num_zeros_row[index]=len(train.ix[index][train.ix[index]==0])

    y[index]=train.ix[index, 'y']
binary_row_df=pd.DataFrame()

binary_row_df['Indexes']=num_ones_row.keys()

binary_row_df['ones']=num_ones_row.values()

binary_row_df['zeros']=num_zeros_row.values()

binary_row_df['y']=y.values()

binary_row_df=binary_row_df.sort_values(by='ones')

binary_row_df.head(10)
width = 0.35

plt.figure(figsize=(10,5))

plt.scatter(binary_row_df.ones, binary_row_df.y)

plt.xlabel('Number of ones')



plt.figure(figsize=(10,5))

plt.scatter(binary_row_df.zeros, binary_row_df.y)

plt.xlabel('Number of zeros')

plt.show()

plt.figure(figsize=(15,5))

sns.pointplot(binary_row_df.ones, binary_row_df.y)

plt.xlabel('Number of ones')



plt.figure(figsize=(20,5))

sns.pointplot(binary_row_df.zeros, binary_row_df.y)

plt.xlabel('Number of zeros')



plt.show()