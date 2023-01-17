!wget https://he-s3.s3.amazonaws.com/media/hackathon/hackerearth-machine-learning-challenge-pet-adoption/pet-adoption-9-5838c75b/a01c26dcd27711ea.zip
!unzip a01c26dcd27711ea.zip
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

!pip install rfpimp

!pip install catboost

from sklearn.metrics import mean_absolute_error,accuracy_score

import lightgbm as lgb

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,GroupKFold,train_test_split,StratifiedShuffleSplit

from rfpimp import *

from tqdm import tqdm

from catboost import *

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('Dataset/train.csv')

test = pd.read_csv('Dataset/test.csv')
train.head(5)
test.head(5)
train.isnull().sum(),test.isnull().sum(),train.shape,test.shape,train.dtypes
df=pd.concat([train,test])

#df=train.append(test,ignore_index=True)
#df['pet_id']=df.pet_id.str.extract('(\d+)').astype(int)
k=['issue_date','listing_date']

for i in k:

  df[i] = pd.to_datetime(df[i])

df['diff']=df['listing_date']-df['issue_date']

df['diff']=abs(df['diff'].astype(int))/1000000000000
k=['issue_date','listing_date']

for i in k:

  df[i+'_'+'year'] = df[i].dt.year

  df[i+'_'+'day'] = df[i].dt.day

  df[i+'_'+'weekofyear'] = df[i].dt.weekofyear

  df[i+'_'+'month'] = df[i].dt.month

  df[i+'_'+'dayofweek'] =df[i].dt.dayofweek

  df[i+'_'+'weekend'] = (df[i].dt.weekday >=5).astype(int)

  df[i+'_'+'hour'] = df[i].dt.hour

  df[i+'_'+'minute'] = df[i].dt.minute

for i in k:

  del df[i]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["col_cond"] = df["condition"].fillna('-9999').astype(str)+"_"+df["color_type"]

df["col_cond"] = le.fit_transform(df["col_cond"])
df['condition']=df['condition'].fillna(3.0)

df['condition']=df['condition']+1

#df['condition']=df['condition'].fillna(method='bfill')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['color_type'] = le.fit_transform(df['color_type'])

le.classes_
#extraa

import math

df['length(m)']=df['length(m)']*100

df['area']=(df['height(cm)']*df['length(m)'])+(df['X1']+df['X2'])

#df['val']=df['height(cm)']*df['length(m)']*2*math.pi

df['Ag_x']=df['color_type']*df['condition']

df['Ag_y']=df['condition']*df['X1']

df['Ag_z']=df['condition']*df['X2']

m=(df['issue_date_year'].min())-1

df['Ag_a']=(df['issue_date_year']-m)*df['color_type']

df['Ag_b']=(df['issue_date_year']-m)*df['condition']

df["X12col"] = df["X1"]+df["X2"] + df["col_cond"]

#df['power']=df['diff']/(df['Ag_a']*(df['color_type']+df['condition']))#new

#df['xor']=df['diff']/(df['X1']*df['condition'])#new
temp = df.groupby(['color_type']).agg({'X1':['count','mean','sum','median'], #median

                                            'X2':['count','mean','sum'],#median

                                       #'X12col':['count','mean','sum','median'],#new

                                       #'diff':['count','mean','sum','median'], #new

                                   #'length(m)':['count','sum','min','max','mean'],

                                   #'height(cm)':['count','sum','min','max','mean'],

                                   #'issue_date_weekofyear':['min','max','count'],

                                   #'issue_date_day':['min','max','count'],

                                   #'listing_date_weekofyear':['min','max','count'],

                                   'condition':['count','mean',],#median

                                       'color_type':['count','mean','sum']})

temp.columns = ['_'.join(x) for x in temp.columns]

df = pd.merge(df,temp,on=['color_type'],how='left')
temp = df.groupby(['condition']).agg({

                                       'color_type':['count','sum','mean','max'],

                                      #'X1':['count','mean','sum','median'],

                                      #'X2':['count','mean','sum','median']

                                      })

temp.columns = ['_condd_'.join(x) for x in temp.columns]

df = pd.merge(df,temp,on=['condition'],how='left')
df = pd.get_dummies(df, columns=['condition','color_type','X1','X2'])

del df['issue_date_hour']

del df['issue_date_minute']
train = df[df['breed_category'].isnull()==False]

test = df[df['breed_category'].isnull()==True]

del test['breed_category']

del test['pet_category']

train_df=train.copy()

test_df=test.copy()
from math import sqrt 

from sklearn.metrics import f1_score
train_df=train.copy()

test_df=test.copy()
x=train_df['pet_category']

del train_df['pet_category']

index=test_df['pet_id']

del train_df['pet_id']

del test_df['pet_id']
X = train_df.drop(labels=['breed_category'], axis=1)

y = train_df['breed_category'].values



from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.10, random_state=101)
X_train.shape, y_train.shape, X_cv.shape, y_cv.shape
categorical_features_indices = np.where(X_train.dtypes == 'category')[0]

categorical_features_indices
from catboost import CatBoostClassifier

cat = CatBoostClassifier(loss_function='MultiClass', 

                         eval_metric='TotalF1', 

                         classes_count=3,

                         depth=10,

                         random_seed=121, 

                         iterations=3500, 

                         learning_rate=0.1,

                         leaf_estimation_iterations=1,

                         l2_leaf_reg=1,

                         bootstrap_type='Bayesian', 

                         bagging_temperature=1, 

                         random_strength=1,

                         od_type='Iter', 

                         border_count=100,

                         od_wait=500)

cat.fit(X_train, y_train, verbose=100,

        use_best_model=True,

        cat_features=categorical_features_indices,

        eval_set=[(X_train, y_train),(X_cv, y_cv)],

        plot=False)

predictions = cat.predict(X_cv)

print('accuracy:', f1_score(y_cv, predictions, average='weighted'))
print('accuracy:', f1_score(y_cv, predictions, average='weighted'))
import seaborn as sns

feature_imp = pd.DataFrame(sorted(zip(cat.feature_importances_, X.columns), reverse=True)[:50], 

                           columns=['Value','Feature'])

plt.figure(figsize=(15,15))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('Catboost Features')

plt.tight_layout()

plt.show()
Xtest = test_df
from sklearn.model_selection import KFold



errcat = []

y_pred_totcat = []



fold = KFold(n_splits=10, shuffle=True, random_state=101)



for train_index, test_index in fold.split(X):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    cat = CatBoostClassifier(loss_function='MultiClass', 

                         eval_metric='TotalF1', 

                         classes_count=3,

                         depth=6,

                         random_seed=121, 

                         iterations=3500, 

                         learning_rate=0.1,

                         leaf_estimation_iterations=1,

                         l2_leaf_reg=1,

                         bootstrap_type='Bayesian', 

                         bagging_temperature=0.8, 

                         random_strength=1,

                         od_type='Iter', 

                         border_count=100,

                         od_wait=500)

    cat.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=200, cat_features=categorical_features_indices)



    y_pred_cat = cat.predict(X_test)

    print("Accuracy: ", f1_score(y_test,y_pred_cat, average='weighted'))



    errcat.append(f1_score(y_test,y_pred_cat, average='weighted'))

    p = cat.predict(Xtest)

    y_pred_totcat.append(p)
np.mean(errcat,0)
cat_final = np.mean(y_pred_totcat,0).round().astype(int)

cat_final
xxx = pd.DataFrame(data=cat_final, columns=['breed_category'])
train_df=train.copy()

test_df=test.copy()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_df['pet_category'] = le.fit_transform(train_df['pet_category'])

le.classes_
x=train_df['breed_category']

del train_df['breed_category']

index=test_df['pet_id']

del train_df['pet_id']

del test_df['pet_id']
X = train_df.drop(labels=['pet_category'], axis=1)

y = train_df['pet_category'].values



from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.10, random_state=101)
X_train.shape, y_train.shape, X_cv.shape, y_cv.shape
categorical_features_indices = np.where(X_train.dtypes == 'category')[0]

categorical_features_indices
from catboost import CatBoostClassifier

cat = CatBoostClassifier(loss_function='MultiClass', 

                         eval_metric='TotalF1', 

                         classes_count=4,

                         depth=6,

                         random_seed=42, 

                         iterations=3500, 

                         learning_rate=0.1,

                         leaf_estimation_iterations=1,

                         l2_leaf_reg=1,

                         bootstrap_type='Bayesian', 

                         bagging_temperature=0.8, 

                         random_strength=1,

                         #od_pval=0.00001,

                         od_type='Iter', 

                         border_count=100,

                         od_wait=500)

cat.fit(X_train, y_train, verbose=100,

        use_best_model=True,

        cat_features=categorical_features_indices,

        eval_set=[(X_train, y_train),(X_cv, y_cv)],

        plot=False)
predictions = cat.predict(X_cv)

print('accuracy:', f1_score(y_cv, predictions, average='weighted'))
import seaborn as sns

feature_imp = pd.DataFrame(sorted(zip(cat.feature_importances_, X.columns), reverse=True)[:50], 

                           columns=['Value','Feature'])

plt.figure(figsize=(15,15))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('Catboost Features')

plt.tight_layout()

plt.show()
Xtest = test_df
from sklearn.model_selection import KFold



errcat = []

y_pred_totcat = []



fold = KFold(n_splits=10, shuffle=True, random_state=42)



for train_index, test_index in fold.split(X):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    cat = CatBoostClassifier(loss_function='MultiClass', 

                         eval_metric='TotalF1', 

                         classes_count=4,

                         depth=6,

                         random_seed=42, 

                         iterations=3500, 

                         learning_rate=0.07,

                         leaf_estimation_iterations=1,

                         l2_leaf_reg=1,

                         bootstrap_type='Bayesian', 

                         bagging_temperature=0.8, 

                         random_strength=1,

                         #od_pval=0.1,

                         od_type='Iter', 

                         border_count=150,

                         od_wait=100)

    cat.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=100, cat_features=categorical_features_indices)



    y_pred_cat = cat.predict(X_test)

    print("Accuracy: ", f1_score(y_test,y_pred_cat, average='weighted'))



    errcat.append(f1_score(y_test,y_pred_cat, average='weighted'))

    p = cat.predict(Xtest)

    y_pred_totcat.append(p)
np.mean(errcat,0)
cat_final_x = np.mean(y_pred_totcat,0).round().astype(int)

cat_final_x
yyy = pd.DataFrame(data=cat_final_x, columns=['pet_category'])
id=test['pet_id']
submission = pd.DataFrame({

        "pet_id":id,

        "breed_category": xxx['breed_category'],

        "pet_category": yyy['pet_category']

    })

submission.to_csv('./submission.csv', index=False)

print(submission)
train_df=train.copy()

test_df=test.copy()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_df['pet_category'] = le.fit_transform(train_df['pet_category'])

le.classes_
x=train_df['breed_category']

del train_df['breed_category']

index=test_df['pet_id']

del train_df['pet_id']

del test_df['pet_id']
df_train = train_df

df_test = test_df
X_train = train_df.drop(['pet_category'],axis=1)

y_train = train_df['pet_category']
x=[]

for i in test_df.columns:

  x.append(i)

x=np.array(x)

feats=x
splits = 15

folds =StratifiedKFold(n_splits=splits, random_state=42,shuffle=True)

oof_preds = np.zeros((len(df_test), 4))

feature_importance_df = pd.DataFrame()

feature_importance_df['Feature'] = X_train.columns

final_preds = []

random_state = [22,44,66,77,88,99,101]

counter = 0



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values,y_train)):

        print("iter_ {}".format(fold_))

        X_trn,y_trn = X_train[feats].iloc[trn_idx],y_train.iloc[trn_idx]

        X_val,y_val = X_train[feats].iloc[val_idx],y_train.iloc[val_idx]

        clf = lgb.LGBMClassifier(boosting='gbdt',learning_rate=0.1, n_estimators=1000, random_state=101, subsample=0.9,max_depth=-1,num_leaves=31)#

         #,

         #min_data_in_leaf=11,

         #bagging_fraction=0.90,

         #bagging_freq=2,

         #bagging_seed=3,

         #feature_fraction=0.90,

         #feature_fraction_seed=2,

         #early_stopping_round=200,

         #max_bin=1000)#(n_estimators=1000,max_depth=4,random_state=42)#dart

        clf.fit(X_trn, y_trn,eval_set=[(X_trn, y_trn), (X_val, y_val)],verbose=0,

                eval_metric='multi_error',early_stopping_rounds=100)

        

        imp = importances(clf,X_val,y_val)

        imp.rename(columns={'Importance':f'Importance_{fold_}'},inplace=True)

        feature_importance_df = pd.merge(feature_importance_df,imp,on='Feature')

        final_preds.append(accuracy_score(y_pred=clf.predict(X_val),y_true=y_val))

        

        oof_preds += clf.predict_proba(df_test[feats])

oof_preds = oof_preds/splits

print(sum(final_preds)/splits)
preds_x = [np.argmax(x) for x in oof_preds]
train_df=train.copy()

test_df=test.copy()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_df['breed_category'] = le.fit_transform(train_df['breed_category'])

le.classes_
x=train_df['pet_category']

del train_df['pet_category']

index=test_df['pet_id']

del train_df['pet_id']

del test_df['pet_id']
df_train = train

df_test = test
X_train = train.drop(['breed_category'],axis=1)

y_train = train['breed_category']
splits = 20

folds =StratifiedKFold(n_splits=splits, random_state=42,shuffle=True)

oof_preds = np.zeros((len(df_test), 3))

feature_importance_df = pd.DataFrame()

feature_importance_df['Feature'] = X_train.columns

final_preds = []

random_state = [22,44,66,77,88,99,101,201]

counter = 0



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values,y_train)):

        print("iter_ {}".format(fold_))

        X_trn,y_trn = X_train[feats].iloc[trn_idx],y_train.iloc[trn_idx]

        X_val,y_val = X_train[feats].iloc[val_idx],y_train.iloc[val_idx]

        clf = lgb.LGBMClassifier(boosting='gbdt',learning_rate=0.1, n_estimators=1000, random_state=101, subsample=0.9)

        clf.fit(X_trn, y_trn,eval_set=[(X_trn, y_trn), (X_val, y_val)],verbose=0,

                eval_metric='multi_error',early_stopping_rounds=100)

        

        imp = importances(clf,X_val,y_val)

        imp.rename(columns={'Importance':f'Importance_{fold_}'},inplace=True)

        feature_importance_df = pd.merge(feature_importance_df,imp,on='Feature')

        final_preds.append(accuracy_score(y_pred=clf.predict(X_val),y_true=y_val))



        oof_preds += clf.predict_proba(df_test[feats])

oof_preds = oof_preds/splits

print(sum(final_preds)/splits)
preds_y = [np.argmax(x) for x in oof_preds]
id=test['pet_id']
submission = pd.DataFrame({

        "pet_id":id,

        "breed_category": preds_y,

        "pet_category":preds_x

    })

submission.to_csv('./submission.csv', index=False)

print(submission)