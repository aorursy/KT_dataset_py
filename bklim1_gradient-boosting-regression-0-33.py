%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV

from sklearn.metrics import  precision_score,recall_score,average_precision_score,roc_auc_score

from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier

from pylab import rcParams



rcParams['figure.figsize'] = 10, 10

color = sns.color_palette()
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



id_test = test_df.id



print('train_df shape:',train_df.shape)

print('test_df shape:',test_df.shape)
dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

print('Variables data type:')

dtype_df.groupby("Column Type").aggregate('count').reset_index()
train_df.describe().round(1)
print(train_df.loc[train_df['build_year'] == 20052009].id)

print(train_df.loc[train_df['state'] == 33].id)

print('build_year:',train_df.ix[10090].build_year)

print('state:',train_df.ix[10090].state)



train_df.loc[train_df['id'] == 10092, 'build_year'] = 2007

train_df.loc[train_df['id'] == 10092, 'state'] = 3

train_df.loc[train_df['id'] == 10093, 'build_year'] = 2009
train_df.describe().round(1)
train_na = (train_df.isnull().sum() / len(train_df)) * 100

train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)

sns.barplot(y=train_na.index, x=train_na,color=color[0])

plt.xlabel('% missing')
for f in train_df.columns:

    if train_df[f].dtype=='object':

        lbl = LabelEncoder()

        lbl.fit(list(train_df[f].values)) 

        train_df[f] = lbl.transform(list(train_df[f].values))

        

for c in test_df.columns:

    if test_df[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(test_df[c].values)) 

        test_df[c] = lbl.transform(list(test_df[c].values))
kitch_ratio = train_df['full_sq']/train_df['kitch_sq']

train_df['kitch_sq']=train_df['kitch_sq'].fillna(train_df['full_sq'] /kitch_ratio.median())

test_df['kitch_sq']=test_df['kitch_sq'].fillna(test_df['full_sq'] /kitch_ratio.median())



lifesq_ratio = train_df['full_sq']/train_df['life_sq']

train_df['life_sq']=train_df['life_sq'].fillna(train_df['full_sq'] /lifesq_ratio.median())

test_df['life_sq']=test_df['life_sq'].fillna(test_df['full_sq'] /lifesq_ratio.median())



train_df=train_df.fillna(train_df.median(),inplace=True)

test_df=test_df.fillna(test_df.median(),inplace=True)
sns.distplot(train_df.price_doc.values, kde=None)

plt.xlabel('price')
ulimit = np.percentile(train_df.price_doc.values, 99)

llimit = np.percentile(train_df.price_doc.values, 1)

train_df.loc[train_df['price_doc'] >ulimit, 'price_doc'] = ulimit

train_df.loc[train_df['price_doc'] <llimit, 'price_doc'] = llimit



sns.distplot(np.log(train_df.price_doc.values),  bins=50,kde=None)

plt.xlabel('price')



train_df['price_doc_log'] = np.log1p(train_df['price_doc'])
print(train_df['price_doc'].value_counts().head(10))



train_df['label_value'] = 0

train_df.loc[train_df['price_doc'] == 1000000, 'label_value'] = 1

train_df.loc[train_df['price_doc'] == 2000000, 'label_value'] = 2
data_X = train_df.drop(["id","timestamp","price_doc","price_doc_log",'label_value'],axis=1)

data_y = train_df['price_doc_log']
# GBmodel = GradientBoostingRegressor()

# param_dist = {"learning_rate": np.linspace(0.05, 0.15,5),

#               "max_depth": range(3, 5),

#               "min_samples_leaf": range(3, 5)}



# rand = RandomizedSearchCV(GBmodel, param_dist, cv=7,n_iter=10, random_state=5)

# rand.fit(data_X,data_y)

# rand.grid_scores_



# print(rand.best_score_)

# print(rand.best_params_)
GBmodel = GradientBoostingRegressor(min_samples_leaf= 4, learning_rate= 0.1, max_depth= 4)

GBmodel.fit(data_X,data_y)
sns.distplot(GBmodel.predict(data_X),kde=None)
clfdata_X = train_df.drop(['id','timestamp','label_value','price_doc_log','price_doc'],axis=1)

clfdata_y = train_df['label_value']



clfX_train, clfX_test, clfY_train, clfY_test = train_test_split(clfdata_X, clfdata_y, test_size=0.30,random_state=31)



GBclf= GradientBoostingClassifier()
GBclf.fit(clfX_train,clfY_train)

GBclf.score(clfX_test,clfY_test)
print(precision_score(GBclf.predict(clfX_test),clfY_test.values,average='macro'))

print(recall_score(GBclf.predict(clfX_test),clfY_test.values,average='macro'))



print(precision_score(GBclf.predict(clfX_test),clfY_test.values,average='micro'))

print(recall_score(GBclf.predict(clfX_test),clfY_test.values,average='micro'))
pred = GBmodel.predict(data_X)

lab = GBclf.predict(clfdata_X)

pred_Y = pd.DataFrame({'pred': np.expm1(pred), 'label':lab})







pred_Y.loc[pred_Y['label'] == 1, 'pred'] = 1000000

pred_Y.loc[pred_Y['label'] == 2, 'pred'] = 2000000

sns.distplot(np.log(pred_Y.pred),kde=None)

sns.distplot(train_df.price_doc_log.values,kde=None)
importances = GBmodel.feature_importances_

importances_by_trees=[tree[0].feature_importances_ for tree in GBmodel.estimators_]

std = np.std(importances_by_trees,axis=0)

indices = np.argsort(importances)[::-1]





sns.barplot(importances[indices][:20],data_X.columns[indices[:20]].values)

plt.title("Feature importances - regression")
clf_importances = GBclf.feature_importances_

clf_importances_by_trees=[tree[0].feature_importances_ for tree in GBclf.estimators_]

clf_std = np.std(clf_importances_by_trees,axis=0)

clf_indices = np.argsort(clf_importances)[::-1]





sns.barplot(clf_importances[clf_indices][:20],clfdata_X.columns[clf_indices[:20]].values)

plt.title("Feature importances - classification")
predict = GBmodel.predict(test_df.drop(["id", "timestamp"],axis=1))

label = GBclf.predict(test_df.drop(['id','timestamp'],axis=1))

output = pd.DataFrame({'id': id_test, 'price_doc': np.expm1(predict), 'label':label})







output.loc[output['label'] == 1, 'price_doc'] = 1000000

output.loc[output['label'] == 2, 'price_doc'] = 2000000

output = output.drop(['label'],axis=1)

output.to_csv('output.csv', index=False)

train_dfadv = train_df.drop(["timestamp","price_doc","price_doc_log"],axis=1)

test_dfadv = test_df

train_dfadv['istrain'] = 1

test_dfadv['istrain'] = 0

whole_df = pd.concat([train_dfadv, test_dfadv], axis = 0)

whole_df = whole_df.fillna(whole_df.median())

valY = whole_df['istrain']

valX = whole_df.drop(['istrain',"id", "timestamp"],axis=1)



X_vtrain, X_vtest, y_vtrain, y_vtest = train_test_split(valX.values, valY.values, test_size=0.20)
GBclf= GradientBoostingClassifier()

GBclf.fit(X_vtrain,y_vtrain)

vpred_y = GBclf.predict(X_vtest)

roc_auc_score(vpred_y,y_vtest)
importances = GBclf.feature_importances_

importances_by_trees=[tree[0].feature_importances_ for tree in GBclf.estimators_]

std = np.std(importances_by_trees,axis=0)

indices = np.argsort(importances)[::-1]





sns.barplot(importances[indices][:20],valX.columns[indices[:20]].values)

plt.title("Feature importances")
X=train_df.drop(["id", "timestamp", "price_doc","price_doc_log"], axis=1)

y=train_df.price_doc_log.values
val_prob = GBclf.predict_proba(X)

adversarial_set = train_df

adversarial_set['prob'] = val_prob.T[1]



adversarial_set=adversarial_set.drop(["id", "timestamp", "price_doc"], axis=1)



adversarial_set_length =int(adversarial_set.shape[0]*0.20)

adversarial_set = adversarial_set.sort_values(by='prob')

validation_set = adversarial_set[:adversarial_set_length] 

train_set = adversarial_set[adversarial_set_length:]



trainY  =train_set['price_doc_log'].values

trainX = train_set.drop(['price_doc_log','prob'],axis=1).values



validationY  =validation_set['price_doc_log'].values

validationX = validation_set.drop(['price_doc_log','prob'],axis=1).values