import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# Because my Jupyter was showing only last line output, so need to add
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# read train and testing data

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
testID = test['Id']
train.head()
test.head()
train.columns
train.shape
test.shape
train.dtypes
# replace NaN value! If striing column, then using mode otherwise median
for col in train:
    if train[col].dtype == 'object':
        train[col] = train[col].fillna(train[col].mode())
    else:
        train[col] = train[col].fillna(train[col].median())
from sklearn.preprocessing import LabelEncoder


def label_encoding(df_train,df_test):
    le_count=0;
    for col in df_train:
        if df_train[col].dtype == 'object':
            if len(list(df_train[col].unique())) <= 2:
                le = LabelEncoder()
                le.fit(list(df_train[col].unique())+list(df_test[col].unique()))

                df_train[col] = le.transform(df_train[col].astype(str))
                df_test[col] = le.transform(df_test[col].astype(str))
                le_count +=1;
               
    
    print("Total label encoded columns : %d " %le_count)
label_encoding(train,test)
train.shape
test.shape
import copy

train_objs_num = len(train)
dataset = pd.concat(objs=[train, test], axis=0)
dataset = pd.get_dummies(dataset)
train = copy.copy(dataset[:train_objs_num])
test = copy.copy(dataset[train_objs_num:])
test = test.drop(['SalePrice'],axis=1)
train.shape
test.shape
# display the distribution of salePrice
from scipy.stats import norm
sns.distplot(train['SalePrice'],fit=norm)
corr = train.corr()
corr = corr.sort_values('SalePrice')
cols = corr['SalePrice'][corr['SalePrice'].values > 0.2].index.values
cols
heatMapCols=np.append(cols[-10:], np.array(['SalePrice']))
cm = np.corrcoef(train[heatMapCols[::-1]].T)
plt.figure(figsize=(16,16))
sns.set(font_scale=1)
with sns.axes_style("white"):
    sns.heatmap(cm,yticklabels=heatMapCols[::-1],xticklabels=heatMapCols[::-1],fmt='.2f',annot_kws={'size':10},annot=True,square=True,cmap=None)
train_label = train['SalePrice']

cols = np.delete(cols,len(cols)-1)

train_sample = train[cols]

test_sample = test[cols]

test_sample.head()
train_sample.head()
from sklearn.preprocessing import  Imputer
imputer = Imputer(strategy = 'median')

imputer.fit(train_sample)

train_sample = imputer.transform(train_sample)
test_sample = imputer.transform(test_sample)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(train_sample)

train_sample = scaler.transform(train_sample)
test_sample = scaler.transform(test_sample)
from sklearn.cross_validation import train_test_split

X_train, X_test , y_train, y_test = train_test_split(train_sample,train_label,train_size = 0.8)

X_train.shape
X_test.shape
test_sample.shape
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(train_sample,train_label)
y_preds = model.predict(X_test)
from sklearn import metrics

print("Root Mean square error: " , np.sqrt(metrics.mean_squared_error(y_test,y_preds)))
test_pred = model.predict(test_sample)
submit = pd.DataFrame()
submit['ID'] = testID
submit['SalePrice'] = test_pred
submit.head()
submit.to_csv('attemp1.csv', index = False)
cols = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF']

test_cols = test[cols];

imputer = Imputer(strategy = 'median')

imputer.fit(test_cols)

test_cols = imputer.transform(test_cols)


count=0

f, axes = plt.subplots(2, 3,figsize=(15,12))

for i in range(2):
    for j in range(3):   
        sns.kdeplot(test_cols[:,count],test_pred,ax=axes[i][j])
        axes[i][j].set_xlabel("%s" %cols[count])
        axes[i][j].set_ylabel("Sale Price")
        count+=1
        
    
f.tight_layout()