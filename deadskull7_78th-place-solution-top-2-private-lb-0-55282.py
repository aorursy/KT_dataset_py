import matplotlib.pyplot as plt
import cv2
from pylab import rcParams

rcParams['figure.figsize'] = 50,20
img=cv2.imread("../input/private-score/score.JPG")
plt.imshow(img)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input/"))
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/mercedes-benz-greener-manufacturing/train.csv')
test = pd.read_csv('../input/mercedes-benz-greener-manufacturing/test.csv')
df = train
print(train.shape)
train.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)       #  numeric dataframe
objects = ['O']
df_cat = df.select_dtypes(include=objects)
print(df_num.shape,df_cat.shape)
print(df_cat.columns,'\n','--------------------------------------------------------------------------------','\n',df_num.columns)
for i in df_cat.columns:
    print('The unique values in '+i+' are: ',df[i].nunique(),'\n',df_cat[i].unique(),'\n',"--------------------------------------------------------------------------------")
print(df.isnull().sum().sum(axis=0))
temp=df.y.values
df_cat['y']=temp
print(df_cat.head())
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(14,9)})
plt.subplot(221)
plt.title("Outlier Detection in target column via Boxplot")
plt.ylabel("Values of y")
plt.grid(True)
sns.boxplot(y=df["y"],color='gold')
plt.subplot(222)
plt.title("Outlier Detection in target column via Histogram")
plt.grid(True)
ax = sns.distplot(df.y,color='green',bins=22)
plt.show()
sns.set(rc={'figure.figsize':(20,7)})
plt.title("y Analysis")
plt.ylabel("Values of y")
plt.scatter(range(df.shape[0]),np.sort(df.y.values),color='orange')
print((df.loc[df.y>150,'y'].values))
df=df[df.y<150]
print("Removing outliers based on above information and setting 150 as a threshold value . . . . . . . . . . . . . . . . . . . . ")
print(df.shape)
df_cat=df_cat[df_cat.y<150]
df_num=df_num[df_num.y<150]
sns.set(rc={'figure.figsize':(20,7)})
sns.regplot(x='ID', y='y', data=df,color='maroon')
from scipy import stats
rcParams['figure.figsize'] = 15, 7
res = stats.probplot(df['y'], plot=plt)
res = stats.probplot(np.log1p(train["y"]), plot=plt)
rcParams['figure.figsize'] = 22, 8
for i in df_cat.columns:
    if i not in 'y':
        plt.figure()
        plt.xlabel=i
        sns.stripplot(x=i, y="y", data=df,jitter=True, linewidth=1,order=np.sort(df[i].unique()))
        sns.boxplot(x=i, y="y", data=df, order=np.sort(df[i].unique()))
        plt.show()
pd.crosstab([df_cat.X2], [df_cat.X0], margins=True).style.background_gradient(cmap='autumn_r')
temp = []
for i in df_num.columns:
    if df[i].var()==0:
        temp.append(i)
print(len(temp))
print(temp)
count=0
low_var_col=[]
for i in test.columns:
    if test[i].dtype == 'int64':
        if test[i].var()<0.01:
            low_var_col.append(i)
            count+=1
print(count)

df.drop(low_var_col,axis=1,inplace=True)
df_num.drop(low_var_col,axis=1,inplace=True)
test.drop(low_var_col,axis=1,inplace=True)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)
dic={}
for i in df_num.columns:
    if i!='y':
        if df[i].corr(df.y)>0.25 or df[i].corr(df.y)<-0.25:
            dic[i]=df[i].corr(df.y)
print("Important Features with there respective correlations are ",'\n','---------------------------------------------------------','\n',dic)
print(df.X119.corr(df.X118),'\n', df.X29.corr(df.X54) ,'\n', df.X54.corr(df.X76) ,'\n', df.X263.corr(df.X279))
# Dublicate features
d = {}; done = []
cols = df.columns.values
for c in cols: d[c]=[]
for i in range(len(cols)):
    if i not in done:
        for j in range(i+1, len(cols)):
            if all(df[cols[i]] == df[cols[j]]):
                done.append(j)
                d[cols[i]].append(cols[j])
dub_cols = []
for k in d.keys():
    if len(d[k]) > 0: 
        dub_cols += d[k]        
print('Dublicates:','\n', dub_cols)
corrs=[]
high_corr=[]
for i in range(0,len(dub_cols)):
    for j in range(i+1,len(dub_cols)):
        if df[dub_cols[i]].corr(df[dub_cols[j]]) >=0.90:
            corrs.append(df[dub_cols[i]].corr(df[dub_cols[j]]))
            high_corr.append((dub_cols[i],dub_cols[j]))
print(corrs)
print("\n")
print(high_corr)
df.drop(['X279','X76','X37','X134','X147','X222','X244','X326'] , axis=1 , inplace=True)
test.drop(['X279','X76','X37','X134','X147','X222','X244','X326'] , axis=1 , inplace=True)
df_num.drop(['X279','X76','X37','X134','X147','X222','X244','X326'] , axis=1 , inplace=True)
from sklearn import preprocessing
categorical=[]
for i in df.columns:
    if df[i].dtype=='object':
        le = preprocessing.LabelEncoder()
        le.fit(list(df[i].values) + list(test[i].values))
        print("Categories in the encoded order from 1 to the size of "+i+" are : ")
        print(le.classes_)
        print("--------------------------------------------------------------------------")
        df[i] = le.transform(list(df[i].values))
        test[i] = le.transform(list(test[i].values))
        categorical.append(i)
correlation_map = df[df.columns[1:10]].corr()
obj = np.array(correlation_map)
obj[np.tril_indices_from(obj)] = False
fig,ax= plt.subplots()
fig.set_size_inches(9,10)
sns.heatmap(correlation_map, mask=obj,vmax=.7, square=True,annot=True)
import xgboost as xgb
train_y = df["y"].values
train_X = df.drop(['y'], axis=1)

def xgb_r2_score(preds, final):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

xgb_params = {
    'n_trees': 520, 
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.98,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': np.mean(train_y), # base prediction = mean(target)
    'silent': 1
}

final = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params), final, num_boost_round=200, feval=xgb_r2_score, maximize=True)

fig, ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(model, max_num_features=40, height=0.8, ax=ax, color = 'coral')
print("Feature Importance by XGBoost")
plt.show()

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
model.fit(train_X, train_y)
feat_names = train_X.columns.values

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:40]

plt.subplots(figsize=(10,10))
plt.title("Feature importances by RandomForestRegressor")
plt.ylabel("Features")
plt.barh(range(len(indices)), importances[indices], color="green", align="center")
plt.yticks(range(len(indices)), feat_names[indices], rotation='horizontal')
plt.ylim([-1, len(indices)])
plt.show()
df['X314_plus_X315'] = df.apply(lambda row: row.X314 + row.X315, axis=1)
test['X314_plus_X315'] = test.apply(lambda row: row.X314 + row.X315, axis=1)
print("Correalation between X314_plus_X315 and y is :  ",df.y.corr(df['X314_plus_X315']))
print("Which makes it pretty much high !! Awesome !!")
#df['X122_plus_X128'] = df.apply(lambda row: row.X122 + row.X128, axis=1)
#test['X122_plus_X128'] = test.apply(lambda row: row.X122 + row.X128, axis=1)
#print("Correlation between X122_plus_X128 and y is :  ",df.y.corr(df['X122_plus_X128']))
df['X118_plus_X314_plus_X315'] = df.apply(lambda row: row.X118 + row.X314 + row.X315, axis=1)
test['X118_plus_X314_plus_X315'] = test.apply(lambda row: row.X118 + row.X314 + row.X315, axis=1)
print("Correalation between X118_plus_X314_plus_X315 and y is :  ",df.y.corr(df['X118_plus_X314_plus_X315']))
print("Which makes it pretty much high !! Awesome !!")
df["X10_plus_X54"] = df.apply(lambda row: row.X10 + row.X54, axis=1)
test["X10_plus_X54"] = test.apply(lambda row: row.X10 + row.X54, axis=1)
print("Correalation between X10_plus_X54 and y is :  ",df.y.corr(df['X10_plus_X54']))
df["X10_plus_X29"] = df.apply(lambda row: row.X10 + row.X29, axis=1)
test["X10_plus_X29"] = test.apply(lambda row: row.X10 + row.X29, axis=1)
print("Correalation between X10_plus_X29 and y is :  ",df.y.corr(df['X10_plus_X29']))
train_X['X314_plus_X315']=df['X314_plus_X315']
#train_X['X122_plus_X128']=df['X122_plus_X128']
train_X['X118_plus_X314_plus_X315']=df['X118_plus_X314_plus_X315']
train_X["X10_plus_X54"] = df["X10_plus_X54"]
train_X["X10_plus_X29"] = df["X10_plus_X29"]
corr_val=[]
same_features=[]
for i in range(0,len(df_num.columns)-1):
    for j in range(i+1,len(df_num.columns)):
        temp_corr=df[df_num.columns[i]].corr(df[df_num.columns[j]])
        if temp_corr>=0.95 or temp_corr<=-0.95: 
            same_features.append((df_num.columns[i],df_num.columns[j]))
            corr_val.append(temp_corr)
print(len(corr_val))
print(same_features)
booler = np.ones(400)
for i in same_features:
    if booler[int(i[1][1:])]==1:
        booler[int(i[1][1:])]=0
        df_num.drop(i[1],axis=1,inplace=True)
        df.drop(i[1],axis=1,inplace=True)
        test.drop(i[1],axis=1,inplace=True)
        train_X.drop(i[1],axis=1,inplace=True)
    elif booler[int(i[0][1:])]==1:
        booler[int(i[0][1:])]=0
        df_num.drop(i[0],axis=1,inplace=True)
        df.drop(i[0],axis=1,inplace=True)
        test.drop(i[0],axis=1,inplace=True)
        train_X.drop(i[0],axis=1,inplace=True)
model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
model.fit(train_X, train_y)
feature_names = train_X.columns.values

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:40]

plt.subplots(figsize=(10,10))
plt.title("Feature importances by RandomForestRegressor")
plt.ylabel("Features")
plt.barh(range(len(indices)), importances[indices], color="green", align="center")
plt.yticks(range(len(indices)), feature_names[indices], rotation='horizontal')
plt.ylim([-1, len(indices)])
plt.show()

final = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params), final, num_boost_round=1350, feval=xgb_r2_score, maximize=True)

fig, ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(model, max_num_features=40, height=0.8, ax=ax,color = 'coral')
print("Feature Importance by XGBoost")
plt.show()
print(train_X.shape , test.shape)
list(set(train_X.columns)-set(test.columns))
'''from sklearn.preprocessing import OneHotEncoder
total_hot=np.concatenate( (train_X.values[:,1:9], test.values[:,1:9]), axis=0)
enc = OneHotEncoder()
enc.fit(total_hot)
total_hot=enc.transform(total_hot)'''
'''total_hot.todense().shape'''
'''train_hot=total_hot.todense()[:4194,:]
test_hot=total_hot.todense()[4194:8404,:]
print(train_hot.shape)
train_X_hot=np.concatenate( (train_X.values[:,0].reshape(4194,1),train_hot) , axis=1)
test_hot=np.concatenate( (test.values[:,0].reshape(4209,1),test_hot) , axis=1)
train_X_hot=np.concatenate( (train_X_hot,train_X.values[:,9:]) , axis=1)
test_hot=np.concatenate( (test_hot,test.values[:,9:]) , axis=1)'''
'''print(train_X_hot.shape, test_hot.shape)'''
'''from sklearn.decomposition import PCA
pca=PCA(n_components=6 , random_state=7)
pca.fit(train_X_hot)
pca_train_X = pca.transform(train_X_hot)
pca_test = pca.transform(test_hot)

print(pca.explained_variance_ratio_.sum())
print("--------------------------------------------------------------")
print(pca.components_)
print("--------------------------------------------------------------")
print(pca.components_.shape)
print("--------------------------------------------------------------")
print(pca_train_X.shape , pca_test.shape)
'''
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.2, random_state=420)

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test = xgb.DMatrix(test)

xgb_params = {
    'n_trees': 500, 
    'eta': 0.0050,
    'max_depth': 3,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': np.mean(train_y), # base prediction = mean(target)
    'silent': 1
}

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(xgb_params, d_train, 1050 , watchlist, early_stopping_rounds=70, feval=xgb_r2_score, maximize=True, verbose_eval=10)
d_train = xgb.DMatrix(train_X, label=train_y)
#d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test = xgb.DMatrix(test)

xgb_params = {
    'n_trees': 500, 
    'eta': 0.0050,
    'max_depth': 3,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': np.mean(train_y), 
    'silent': 1
}

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

watchlist = [(d_train, 'train')]

clf = xgb.train(xgb_params, d_train, 1050 , watchlist, early_stopping_rounds=70, feval=xgb_r2_score, maximize=True, verbose_eval=10)
Answer = clf.predict(d_test)

sub = pd.DataFrame()
sub['ID'] = test.ID
sub['y'] = Answer
sub.to_csv('mercedes_benz_The_best_or_Nothing.csv', index=False)
sub.head()
