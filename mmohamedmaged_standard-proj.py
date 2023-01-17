import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

import os
print(os.listdir("../input/santander-customer-transaction-prediction"))
warnings.filterwarnings('ignore')

train_data = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test_data =  pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
a=[train_data,test_data]
for i in a :
    print (i.shape)
    
 
train_data.head(10)
train_data.describe()
test_data.describe()
sns.countplot=train_data["target"].value_counts().plot.bar()
def mis_val (data):
    a=[]

    for f in data.columns:
        a.append(train_data[f].isnull().sum())


    print('check for missing data: ',a)


    
mis_val(train_data)
mis_val(test_data)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(train_data.iloc[:,2:],train_data.target)

plt.figure(figsize=(17,14))

plt.plot(rf.feature_importances_)
plt.xticks(np.arange(train_data.iloc[:,2:].shape[1]), train_data.iloc[:,2:].columns.tolist(), rotation=90)
feature_importances=list(rf.feature_importances_)
h=pd.DataFrame(feature_importances,columns=['x'])
h.sort_values(by=['x'],ascending=False).head(10)

h.sort_values(by=['x'],ascending=True).head(10)
features =['var_81','var_12','var_139','var_53','var_110','var_26','var_174','var_22'
          ,'var_164','var_109']


print('mean',train_data.var_81.mean())
print('std',train_data.var_81.std())
for feature in features:
    print (train_data[feature].nunique())

train_data.var_12.value_counts().head(40)
var_12_unique = train_data.var_12.unique()
var_12_unique_sorted = np.sort(var_12_unique)
                           
a=np.diff(var_12_unique_sorted)
pd.DataFrame(a).value_counts()
c=np.diff(a/0.0001)
pd.DataFrame(c).max()
t0 = train_data.loc[train_data['target'] == 0]
t1 = train_data.loc[train_data['target'] == 1]

i=0
plt.subplots(2,5,figsize=(12,10))
for feature in features:
    i+=1
    plt.subplot(2,5,i)
    sns.distplot(a=(t0[feature]),label=0, kde=False)
    sns.distplot(a=(t1[feature]),label=1, kde=False)
    plt.legend()



def plot_feature_scatter(df1, df2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,5,figsize=(14,14))

    for feature in features:
        i += 1
        plt.subplot(2,5,i)
        plt.scatter(df1[feature], df2[feature], marker='+')
        plt.xlabel(feature, fontsize=9)
    plt.show();

plot_feature_scatter(train_data,test_data,features)
def coo_between_target_feat(df1, features,feats):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(5,5,figsize=(14,14))

    for feature in features:
        for x in feats:
            i += 1
            plt.subplot(5,5,i)
            plt.scatter(df1[feature], df1[x], marker='o')
            plt.xlabel(feature, fontsize=8)
            plt.ylabel(x,fontsize=8)
    plt.show();

feat=['var_105','var_17','var_10','var_158','var_7']
feats=['var_31','var_42','var_140','var_74','var_30']
coo_between_target_feat(train_data,feat,feats)
train_enc =  pd.DataFrame(index = train_data.index)
from tqdm import tqdm_notebook
dup_cols = {}


for i, c1 in enumerate(tqdm_notebook(train_enc.columns)):
    for c2 in train_enc.columns[i + 1:]:
        if c2 not in dup_cols and np.all(train_enc[c1] == train_enc[c2]):
            dup_cols[c2] = c1
dup_cols
feats_counts = train_data.nunique(dropna = False)
feats_counts.sort_values()
X, y = train_data.iloc[:,2:],train_data.target
X_1=train_data.loc[train_data.target==1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123,shuffle=True)


xg_cls = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.01,
                max_depth = 3, alpha = 10, n_estimators = 10,gamma=0,reg_lambda= 1, scale_pos_weight= 5)

xg_cls.fit(X_train,y_train)
preds = xg_cls.predict(X_test)


CM = confusion_matrix(y_test, preds)
print('Confusion Matrix is : \n', CM)



predss=xg_reg.predict(test_data.iloc[:,1:])

GBRModel = GradientBoostingRegressor(n_estimators=100,max_depth=2,learning_rate = 1.5 ,random_state=33)
GBRModel.fit(X_train, y_train)

#Calculating Details
print('GBRModel Train Score is : ' , GBRModel.score(X_train, y_train))
print('GBRModel Test Score is : ' , GBRModel.score(X_test, y_test))

y_pred_prob = GBRModel.predict(test_data.iloc[:,1:])
sub_df = pd.DataFrame({"ID_code":test_data["ID_code"].values})
sub_df["target"] = y_pred_prob
sub_df.to_csv("asa.csv", index=False)

