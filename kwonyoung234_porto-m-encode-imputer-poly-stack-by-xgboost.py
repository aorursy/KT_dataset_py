import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

# from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectFromModel



from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score



# from lightgbm import LGBMClassifier

# from xgboost import XGBClassifier

# from sklearn.linear_model import LogisticRegression



pd.set_option('display.max_columns', 100)
import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
data = []

for feature in train.columns:

    if feature == 'id':

        use = 'id'

    elif feature == 'target':

        use = 'target'

    else:

        use = 'input'

    

    if 'bin' in feature or feature == 'target':

        type = 'binary'

    elif 'cat' in feature or feature == 'id':

        type = 'categorical'

    elif train[feature].dtype == float or isinstance(train[feature].dtype,float):

        type = 'real'

    elif train[feature].dtype == int:

        type = 'integer'

        

    preserve = True

    if feature =='id':

        preserve = False

    

    dtype = train[feature].dtype

    

    category = 'none'

    if 'ind' in feature:

        category = 'individual'

    elif 'reg' in feature:

        category = 'registration'

    elif 'car' in feature:

        category = 'car'

    elif 'calc' in feature:

        category = 'calculated'

    

    feature_dict = {

        'var_name':feature,

        'use':use,

        'type':type,

        'preserve':preserve,

        'dtype':dtype,

        'category':category

    }

    

    data.append(feature_dict)



metadata = pd.DataFrame(data,columns=['var_name', 'use', 'type', 'preserve', 'dtype', 'category'])

metadata.set_index('var_name',inplace=True)
metadata
metadata[(metadata['type']=='categorical') & (metadata.preserve)]
pd.DataFrame({'count':metadata.groupby('category')['category'].size()}).reset_index()
pd.DataFrame({'count':metadata.groupby(['use','type'])['use'].size()}).reset_index()
plt.figure()

fig,ax = plt.subplots(figsize=[6,6])

sns.countplot('target',data=train,ax=ax)

plt.ylabel('Number of Values',fontsize=12)

plt.xlabel('Target Value',fontsize=12)

plt.tick_params(axis='both', which='major', labelsize=12)

height = [p.get_height() for p in ax.patches]
print('Percentage of Target 0 of the total {}'.format(height[0]/sum(height)*100))

print('Percentage of Target 1 of the total {}'.format(height[1]/sum(height)*100))
train[metadata[(metadata.type=='real')&(metadata.preserve)].index].describe()
real_corr = train[metadata[(metadata.type=='real')&(metadata.preserve)].index].corr()
fig = plt.figure()

fig,ax = plt.subplots(figsize=[15,15])

sns.heatmap(real_corr,annot=True,square=True,center=0,cmap=plt.cm.summer,ax=ax)
"""

ps_car_12 and ps_car_13 (0.67) 

ps_reg_01 and ps_reg_03 (0.64)

ps_car_13 and ps_car_15 (0.53)

ps_reg_02 and ps_reg_03 (0.52)

ps_reg_01 and ps_reg_02 (0.47)

"""

s = train.sample(frac=0.1)
sns.lmplot(x='ps_car_12',y='ps_car_13',hue='target',data=s,palette='Set1',scatter_kws={'alpha':0.3})
sns.lmplot(x='ps_car_12',y='ps_car_13',hue='target',data=s,palette='Set1',scatter_kws={'alpha':0.3})

plt.suptitle('plot for ps_car_12 and ps_car_13')
sns.lmplot(x='ps_reg_01',y='ps_reg_03',hue='target',data=s,palette='Set1',scatter_kws={'alpha':0.3})

plt.suptitle('plot for ps_reg_01 and ps_reg_03')
sns.lmplot(x='ps_car_13',y='ps_car_15',hue='target',data=s,palette='Set1',scatter_kws={'alpha':0.3})

plt.suptitle('plot for ps_car_13 and ps_car_15')
sns.lmplot(x='ps_reg_02',y='ps_reg_03',hue='target',data=s,palette='Set1',scatter_kws={'alpha':0.3})

plt.suptitle('plot for ps_reg_02 and ps_reg_03')
sns.lmplot(x='ps_reg_01',y='ps_reg_02',hue='target',data=s,palette='Set1',scatter_kws={'alpha':0.3})

plt.suptitle('plot for ps_reg_01 and ps_reg_02')
train[metadata[(metadata['type'] == 'integer') & (metadata.preserve)].index].describe()
int_corr = train[metadata[(metadata['type'] == 'integer') & (metadata.preserve)].index].corr()

fig = plt.figure()

fig,ax = plt.subplots(figsize=[20,20])

ax = sns.heatmap(int_corr,cmap=plt.cm.summer,center=0,square=True,annot=True)
train[metadata[(metadata['type'] == 'binary') & (metadata.preserve)].index].describe()
bin_col = [col for col in train.columns if '_bin' in col]

zero_list = []

one_list = []

for col in bin_col:

    zero_list.append((train[col]==0).sum()/train.shape[0]*100)

    one_list.append((train[col]==1).sum()/train.shape[0]*100)

fig = plt.figure()

fig, ax = plt.subplots(figsize=[6,6])

p1 = sns.barplot(x=bin_col,y=zero_list,ax=ax,color='blue')

p1.set_xticklabels(p1.get_xticklabels(),rotation=90)

p2 = sns.barplot(x=bin_col,y=one_list,bottom=zero_list,ax=ax,color='red')

plt.ylabel('Percent of one/zero[%]')

plt.xlabel('Binary Features')
var = [col for col in train.columns if '_bin' in col]

i = 0

t1 = train.loc[train.target==1]

t0 = train.loc[train.target==0]



fig = plt.figure()

fig,ax = plt.subplots(6,3,figsize=[16,24])



for feature in var:

    i+= 1

    plt.subplot(6,3,i)

    sns.kdeplot(t1[feature],bw=0.5,label='target=1')

    sns.kdeplot(t0[feature],bw=0.5,label='target=0')

    plt.ylabel('Density plot',fontsize=12)

    plt.xlabel(feature,fontsize=12)
vars = metadata[(metadata['type'] == 'categorical') & (metadata.preserve)].index

for col in vars:

    fig,ax = plt.subplots(figsize=(6,6))

    cat_perc = train[[col,'target']].groupby(col,as_index=False).mean()

    cat_perc.sort_values(by='target',ascending=False,inplace=True)

    sns.barplot(x=col,y='target',data=cat_perc,ax=ax,order=cat_perc[col])
var = metadata[(metadata.type=='categorical') & (metadata.preserve)].index

i = 0

t1 = train.loc[train['target']==1] 

t0 = train.loc[train['target']==0]

fig = plt.figure()

fig,ax = plt.subplots(5,3,figsize=[16,20])



for col in var:

    i+=1

    plt.subplot(5,3,i)

    sns.kdeplot(t1[col],bw=0.5,label='target = 1')

    sns.kdeplot(t0[col],bw=0.5,label='target = 0')

    plt.ylabel('Density plot', fontsize=12)

    plt.xlabel(col, fontsize=12)
var = metadata[(metadata.category == 'registration')&(metadata.preserve)].index

fig = plt.figure()

fig,ax = plt.subplots(1,3,figsize=[12,4])

i = 0



for col in var:

    i+=1

    plt.subplot(1,3,i)

    sns.kdeplot(train[col],bw=0.5,label='train')

    sns.kdeplot(test[col],bw=0.5,label='test')

    plt.ylabel('Distribution')

    plt.xlabel(col)
var = metadata[(metadata.category == 'individual')&(metadata.preserve)].index

fig = plt.figure()

fig,ax = plt.subplots(5,4,figsize=[18,20])

i = 0



for col in var:

    i+=1

    plt.subplot(5,4,i)

    sns.kdeplot(train[col],bw=0.5,label='train')

    sns.kdeplot(test[col],bw=0.5,label='test')

    plt.ylabel('Distribution')

    plt.xlabel(col)
var = metadata[(metadata.category == 'car')&(metadata.preserve)].index

fig = plt.figure()

fig,ax = plt.subplots(4,4,figsize=[18,18])

i = 0



for col in var:

    i+=1

    plt.subplot(4,4,i)

    sns.kdeplot(train[col],bw=0.5,label='train')

    sns.kdeplot(test[col],bw=0.5,label='test')

    plt.ylabel('Distribution')

    plt.xlabel(col)
var = metadata[(metadata.category == 'calculated')&(metadata.preserve)].index

fig = plt.figure()

fig,ax = plt.subplots(5,4,figsize=[18,20])

i = 0



for col in var:

    i+=1

    plt.subplot(5,4,i)

    sns.kdeplot(train[col],bw=0.5,label='train')

    sns.kdeplot(test[col],bw=0.5,label='test')

    plt.ylabel('Distribution')

    plt.xlabel(col)
desired_apriori=0.10



idx_0 = train[train.target == 0].index

idx_1 = train[train.target == 1].index



no_0 = len(train.loc[idx_0])

no_1 = len(train.loc[idx_1])



undersampling_rate = ((1-desired_apriori)*no_1)/(desired_apriori*no_0)

undersampled_no_0 = int(undersampling_rate*no_0)

print('Rate to undersample records with target=0: {}'.format(undersampling_rate))

print('Number of records with target=0 after undersampling: {}'.format(undersampled_no_0))



undersampled_idx = shuffle(idx_0,random_state=242,n_samples=undersampled_no_0)



idx_list = list(undersampled_idx)+list(idx_1)



train = train.loc[idx_list].reset_index(drop=True)
df_null = (train == -1).sum().sort_values(ascending=False).reset_index()

df_null.columns = ['features','count']

df_null = df_null[df_null['count'] > 0]

df_null
for col in df_null.features:

    print('Variable {} has {} records among {} ({:.2%}) with missing values'.format(col,(train[col]==-1).sum(),len(train[col]),((train[col]==-1).sum()/len(train[col]))))
metadata.loc[col_to_drop,'preserve']
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]

metadata.loc[col_to_drop,'preserve'] = False

train = train.drop(col_to_drop,axis=1)

test = test.drop(col_to_drop,axis=1)
vars_to_drop = ['ps_car_03_cat','ps_car_05_cat']

train = train.drop(vars_to_drop,axis=1)

test = test.drop(vars_to_drop,axis=1)

metadata.loc[(vars_to_drop),'preserve'] = False
print('Train dataset(row,col):',train.shape,"\nTest dataset(row,col):",test.shape)
dummy = metadata.loc[df_null.features,:]

dummy[(dummy.type=='real')|(dummy.type=='integer')]
mean_imputer = Imputer(missing_values=-1,strategy='mean',axis=0)



dummy = metadata.loc[df_null.features,:]



real_int_cols = dummy[(dummy.type=='real')|(dummy.type=='integer')].index

for col in real_int_cols:

    if metadata.loc[col].type =='real':

        train[col] = mean_imputer.fit_transform(pd.DataFrame(train[col])).ravel()

        train[col] = train[col].astype(float)

        test[col] = mean_imputer.transform(pd.DataFrame(test[col])).ravel()

        test[col] = test[col].astype(float)

        

    if metadata.loc[col].type == 'integer':

        train[col] = mean_imputer.fit_transform(pd.DataFrame(train[col])).ravel()

        train[col] = train[col].astype(float)

        test[col] = mean_imputer.transform(pd.DataFrame(test[col])).ravel()

        test[col] = test[col].astype(float)
frequent_imputer = Imputer(missing_values=-1,strategy='most_frequent',axis=0)



droped_cats = dummy[dummy.type=='categorical'].index[~dummy[dummy.type=='categorical'].index.isin(['ps_car_03_cat','ps_car_05_cat'])]



for col in droped_cats:

    train[col] = frequent_imputer.fit_transform(pd.DataFrame(train[col])).ravel()

    train[col] = train[col].astype(int)



    test[col] = frequent_imputer.fit_transform(pd.DataFrame(test[col])).ravel()

    test[col] = test[col].astype(int)
for col in droped_cats:

    print('Variable {} has {} records among {} ({:.2%}) with missing values'.format(col,(train[col]==-1).sum(),len(train[col]),((train[col]==-1).sum()/len(train[col]))))

for col in real_int_cols:

    print('Variable {} has {} records among {} ({:.2%}) with missing values'.format(col,(train[col]==-1).sum(),len(train[col]),((train[col]==-1).sum()/len(train[col]))))
# Script by https://www.kaggle.com/ogrellier

# Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

def add_noise(series, noise_level):

    return series * (1 + noise_level * np.random.randn(len(series)))



def target_encode(trn_series=None, 

                  tst_series=None, 

                  target=None, 

                  min_samples_leaf=1, 

                  smoothing=1,

                  noise_level=0):

    """

    Smoothing is computed like in the following paper by Daniele Micci-Barreca

    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf

    trn_series : training categorical feature as a pd.Series

    tst_series : test categorical feature as a pd.Series

    target : target data as a pd.Series

    min_samples_leaf (int) : minimum samples to take category average into account

    smoothing (int) : smoothing effect to balance categorical average vs prior  

    """ 

    assert len(trn_series) == len(target)

    assert trn_series.name == tst_series.name

    temp = pd.concat([trn_series, target], axis=1)

    # Compute target mean 

    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    # Compute smoothing

    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data

    prior = target.mean()

    # The bigger the count the less full_avg is taken into account

    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing

    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply averages to trn and tst series

    ft_trn_series = pd.merge(

        trn_series.to_frame(trn_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=trn_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_trn_series.index = trn_series.index 

    ft_tst_series = pd.merge(

        tst_series.to_frame(tst_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=tst_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
train_encoded, test_encoded = target_encode(train['ps_car_11_cat'],

                                           test['ps_car_11_cat'],

                                           target=train.target,

                                           min_samples_leaf=100,

                                           smoothing=10,

                                           noise_level=0.01)

train['ps_car_11_cat_te'] = train_encoded

train.drop('ps_car_11_cat',axis=1,inplace=True)

metadata.loc['ps_car_11_cat', 'preserve'] =False



temp_dict = metadata.loc['ps_car_11_cat',:].to_dict()

temp_dict['var_name'] = 'ps_car_11_cat_te'

temp_dict['type'] ='real'

temp_dict['preserve'] = True

temp_dict['dtype'] = train['ps_car_11_cat_te'].dtype

test['ps_car_11_cat_te'] = test_encoded

test.drop('ps_car_11_cat',axis=1,inplace=True)



metadata = pd.concat([metadata,pd.DataFrame(temp_dict,index=range(1)).set_index('var_name')])
print('Train dataset(row,col):',train.shape,"\nTest dataset(row,col):",test.shape)
cat_features = metadata[(metadata.type=='categorical')&(metadata.preserve)].index

for col in cat_features:

    temp = pd.get_dummies(pd.Series(train[col]),prefix=col)

    train = pd.concat([train,temp],axis=1)

    train.drop(col,inplace=True,axis=1)

    

for col in cat_features:

    temp = pd.get_dummies(pd.Series(test[col]),prefix=col)

    test = pd.concat([test,temp],axis=1)

    test.drop(col,inplace=True,axis=1)
print('Train dataset(row,col):',train.shape,"\nTest dataset(row,col):",test.shape)
real_cols = metadata[(metadata.type=='real')&(metadata.preserve)].index
poly = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)



new_poly_df_train = pd.DataFrame(poly.fit_transform(train[real_cols]),columns=poly.get_feature_names(real_cols))

new_poly_df_test = pd.DataFrame(poly.transform(train[real_cols]),columns=poly.get_feature_names(real_cols))



new_poly_df_train = new_poly_df_train.drop(real_cols,axis=1)

new_poly_df_test =new_poly_df_test.drop(real_cols,axis=1)



train = pd.concat([train,new_poly_df_train],axis=1)

test = pd.concat([test,new_poly_df_train],axis=1)
print('Train dataset(row,col):',train.shape,"\nTest dataset(row,col):",test.shape)
# id_test = test['id'].values

# target_train = train['target'].values



# train.drop(['id','target'],axis=1,inplace=True)

# test.drop(['id'],axis=1,inplace=True)
X = train.drop(['id','target'],axis=1)

train_labels = X.columns

y = train['target']



X_test = test.drop('id',axis=1)

y_test = np.zeros(X_test.shape[0])



sub = test['id'].to_frame()

sub['target'] = 0
print('Train dataset(row,col):',X.shape,"\nTest dataset(row,col):",X_test.shape)
def gini(y, pred):

    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)

    g = g[np.lexsort((g[:,2], -1*g[:,1]))]

    gs = g[:,0].cumsum().sum() / g[:,0].sum()

    gs -= (len(y) + 1) / 2.

    return gs / len(y)



def gini_xgb(pred, y):

    y = y.get_label()

    return 'gini', gini(y, pred) / gini(y, y)
import xgboost

from xgboost import XGBClassifier



XgbC = XGBClassifier( 

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

xgb_params = XgbC.get_xgb_params()

xgtrain = xgboost.DMatrix(X.values,label=y.values)

cvresult = xgboost.cv(xgb_params,xgtrain,num_boost_round=XgbC.get_params()['n_estimators'],nfold=5,metrics='auc',early_stopping_rounds=50)



print(cvresult.shape)



from sklearn import metrics



XgbC.set_params(n_estimators=cvresult.shape[0])



#Fit the algorithm on the data

XgbC.fit(X, y,eval_metric='auc')

        

#Predict training set:

dtrain_predictions = XgbC.predict(X)

dtrain_predprob = XgbC.predict_proba(X)[:,1]

        

#Print model report:

print ("\nModel Report")

print ("Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions))

print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))
indices = np.argsort(XgbC.feature_importances_)[::-1]



new_list = []

for f in range(X.shape[1]):

    print(f+1,train_labels[indices[f]],XgbC.feature_importances_[indices[f]])

    new_list.append(train_labels[indices[f]])
feat_imp = pd.Series(XgbC.feature_importances_).sort_values(ascending=False)



plt.figure(figsize=[12,8])

feat_imp.plot(kind='bar', title='Feature Importances')

plt.ylabel('Feature Importance Score')

plt.xticks(range(0,91),new_list,rotation=60)
XgbC = XGBClassifier( 

 learning_rate =0.1,

 n_estimators=cvresult.shape[0],

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)



xgb_params = XgbC.get_xgb_params()



from sklearn.model_selection import StratifiedKFold



KFOLD = StratifiedKFold(n_splits=5,random_state=12)



X = X.values



for i,(train_index,valid_index) in enumerate(KFOLD.split(X,y)):

    

    X_train, X_valid = X[train_index],X[valid_index]

    y_train, y_valid = y[train_index], y[valid_index]

    

    d_train = xgboost.DMatrix(X_train,label=y_train)

    d_valid = xgboost.DMatrix(X_valid,label=y_valid)

    

    watchlist = [(d_train,'train'),(d_valid,'valid')]

    

    xgboost_model = xgboost.train(xgb_params,d_train,cvresult.shape[0],watchlist,early_stopping_rounds=50,feval=gini_xgb,maximize=True,verbose_eval=100)

    sub['target'] +=xgboost_model.predict(xgboost.DMatrix(X_test.values),ntree_limit=xgboost_model.best_ntree_limit+50) / 5
submission = pd.DataFrame()

submission['id'] = sub['id']

submission['target'] = sub['target']

submission.to_csv('stacked.csv', index=False)