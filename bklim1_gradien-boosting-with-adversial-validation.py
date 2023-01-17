%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from xgboost import XGBRegressor

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import  mean_squared_error, roc_auc_score

from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier



color = sns.color_palette()
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv('../input/test.csv')

id_test = test_df.id



print('Train df shape:',train_df.shape)

print('Test df shape:',test_df.shape)
dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
train_df.describe().round(1)
test_df.describe().round(1)
print(train_df.loc[train_df['build_year'] == 20052009].id)

print(train_df.loc[train_df['state'] == 33].id)
train_df.loc[train_df['id'] == 10092, 'build_year'] = 2007

train_df.loc[train_df['id'] == 10092, 'state'] = 3

train_df.loc[train_df['id'] == 10093, 'build_year'] = 2009
sns.distplot(train_df.price_doc.values, kde=None)

plt.xlabel('price')
ulimit = np.percentile(train_df.price_doc.values, 99)

llimit = np.percentile(train_df.price_doc.values, 1)

train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit

train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit
sns.distplot(np.log(train_df.price_doc.values), kde=None)

plt.xlabel('price')



train_df['price_doc_log'] = np.log1p(train_df['price_doc'])
train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4]+x[5:7])

grouped_df = train_df.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()



sns.barplot(grouped_df.yearmonth.values, grouped_df.price_doc.values,color=color[0])

plt.ylabel('Median Price')

plt.xlabel('Year Month')

plt.xticks(rotation='vertical')
corrmat = train_df.corr()

sns.heatmap(corrmat, vmax=.8, square=True,xticklabels=False,yticklabels=False,cbar=False,annot=False);
train_na = (train_df.isnull().sum() / len(train_df)) * 100

train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)

sns.barplot(y=train_na.index, x=train_na,color=color[0])

plt.xlabel('% missing')
kitch_ratio = train_df['full_sq']/train_df['kitch_sq']

train_df['kitch_sq']=train_df['kitch_sq'].fillna(train_df['full_sq'] /kitch_ratio.median())



lifesq_ratio = train_df['full_sq']/train_df['life_sq']

train_df['life_sq']=train_df['life_sq'].fillna(train_df['full_sq'] /lifesq_ratio.median())



train_df["extra_sq"] = train_df["full_sq"] - train_df["life_sq"]

test_df["extra_sq"] = test_df["full_sq"] - test_df["life_sq"]



train_df=train_df.fillna(train_df.median())

test_df=test_df.fillna(test_df.median())
for f in train_df.columns:

    if train_df[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df[f].values)) 

        train_df[f] = lbl.transform(list(train_df[f].values))

        

for c in test_df.columns:

    if test_df[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(test_df[c].values)) 

        test_df[c] = lbl.transform(list(test_df[c].values))

        #x_test.drop(c,axis=1,inplace=True)
train_dfadv = train_df.drop(["timestamp","price_doc","price_doc_log","yearmonth"],axis=1)

test_dfadv = test_df

train_dfadv['istrain'] = 1

test_dfadv['istrain'] = 0

whole_df = pd.concat([train_dfadv, test_dfadv], axis = 0)

whole_df = whole_df.fillna(whole_df.median())

valY = whole_df['istrain'].values

valX = whole_df.drop(['istrain',"id", "timestamp"],axis=1).values



X_vtrain, X_vtest, y_vtrain, y_vtest = train_test_split(valX, valY, test_size=0.20)



GBclf= GradientBoostingClassifier()
GBclf.fit(X_vtrain,y_vtrain)
vpred_y = GBclf.predict(X_vtest)

roc_auc_score(vpred_y,y_vtest)
X=train_df.drop(["id", "timestamp", "price_doc","price_doc_log","yearmonth"], axis=1)

y=train_df.price_doc_log.values
val_prob = GBclf.predict_proba(X)

adversarial_set = train_df

adversarial_set['prob'] = val_prob.T[1]



adversarial_set=adversarial_set.drop(["id", "timestamp", "price_doc","yearmonth"], axis=1)



adversarial_set_length =int(adversarial_set.shape[0]*0.20)

adversarial_set = adversarial_set.sort_values(by='prob')

validation_set = adversarial_set[:adversarial_set_length]   #odwrócona walidacja !!!!

train_set = adversarial_set[adversarial_set_length:]



trainY  =train_set['price_doc_log'].values

trainX = train_set.drop(['price_doc_log','prob'],axis=1).values



validationY  =validation_set['price_doc_log'].values

validationX = validation_set.drop(['price_doc_log','prob'],axis=1).values
GBmodel = GradientBoostingRegressor().fit(trainX,trainY)

print(mean_squared_error(GBmodel.predict(validationX),validationY))
importances = GBmodel.feature_importances_

importances_by_trees=[tree[0].feature_importances_ for tree in GBmodel.estimators_]

std = np.std(importances_by_trees,axis=0)

indices = np.argsort(importances)[::-1]





sns.barplot(importances[indices][:50],X.columns[indices[:50]].values)

plt.title("Feature importances")
test_X = test_df.drop(["id", "timestamp",'istrain'],axis=1).values

y_predict = GBmodel.predict(test_X)

output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})