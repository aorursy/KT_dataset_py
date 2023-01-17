# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier;
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold;

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_train=pd.read_csv("../input/train.csv")
df_test=pd.read_csv("../input/test.csv")
#df_hist=pd.read_csv("../input/historical_user_logs.csv")
# df_hist_agg=df_hist.groupby(['user_id','product','action']).size().reset_index(name='count')

df_train.isnull().sum()
#df_hist_agg.head()
#dummy=pd.get_dummies(df_hist_agg['action'])
#df_hist_agg_t=pd.concat([df_hist_agg,dummy],axis=1)
df_hist_train=df_train
df_hist_test=df_test
#df_hist_train=pd.merge(df_hist_agg,df_train,on='user_id')
#df_hist_test=pd.merge(df_hist_agg,df_test,on='user_id')
target_new=df_hist_train['is_click']


print(pd.DataFrame(df_hist_train.corr()["is_click"].sort_values(ascending=False)));

#imputing missing values and feature engineering 
#df_hist_train['webpage_id'].value_counts()
most_view_webpage=['13787','60305']
usual_view_webpage=['28529','6970','53587','1734','45962']
less_view_weppage=['51181','11085']
df_hist_train['most_view_webpage']=df_hist_train['webpage_id'].map(lambda x : 1 if x in most_view_webpage else 0 )
df_hist_train['usual_view_webpage']=df_hist_train['webpage_id'].map(lambda x : 1 if x in usual_view_webpage else 0 )
df_hist_train['less_view_weppage']=df_hist_train['webpage_id'].map(lambda x : 1 if x in less_view_weppage else 0 )
df_hist_train.drop('webpage_id',axis=1,inplace=True)

most_view_webpage=['13787','60305']
usual_view_webpage=['28529','6970','53587','1734','45962']
less_view_weppage=['51181','11085']
df_hist_test['most_view_webpage']=df_hist_test['webpage_id'].map(lambda x : 1 if x in most_view_webpage else 0 )
df_hist_test['usual_view_webpage']=df_hist_test['webpage_id'].map(lambda x : 1 if x in usual_view_webpage else 0 )
df_hist_test['less_view_weppage']=df_hist_test['webpage_id'].map(lambda x : 1 if x in less_view_weppage else 0 )
df_hist_test.drop('webpage_id',axis=1,inplace=True)
df_hist_train['product_category_2'].fillna(4,inplace=True)
df_hist_train['product_category_2'].value_counts()
df_hist_train['Top_most_category']=df_hist_train['product_category_2'].map(lambda x : 1 if x==82527.0 else 0 )
df_hist_train['Second_most_category']=df_hist_train['product_category_2'].map(lambda x : 1 if x==146115.0 else 0 )
df_hist_train['Third_class_category']=df_hist_train['product_category_2'].map(lambda x : 1 if x==270915.0 else 0 )
df_hist_train['unkown_category']=df_hist_train['product_category_2'].map(lambda x : 1 if x==4 else 0 )
df_hist_train.drop('product_category_2',axis=1,inplace=True)

df_hist_test['product_category_2'].fillna(4,inplace=True)
df_hist_test['product_category_2'].value_counts()
df_hist_test['Top_most_category']=df_hist_test['product_category_2'].map(lambda x : 1 if x==82527.0 else 0 )
df_hist_test['Second_most_category']=df_hist_test['product_category_2'].map(lambda x : 1 if x==146115.0 else 0 )
df_hist_test['Third_class_category']=df_hist_test['product_category_2'].map(lambda x : 1 if x==270915.0 else 0 )
df_hist_test['unkown_category']=df_hist_test['product_category_2'].map(lambda x : 1 if x==4 else 0 )
df_hist_test.drop('product_category_2',axis=1,inplace=True)
df_hist_train['campaign_id'].value_counts()  
large_campaign_id=[359520,405490]
medium_campaign_id=[360936,118601,98970,404347,82320]
small_campaign_id=[396664,105960]
df_hist_train['large_campaign_id']=df_hist_train['campaign_id'].map(lambda x : 1 if x in large_campaign_id  else 0 )
df_hist_train['medium_campaign_id']=df_hist_train['campaign_id'].map(lambda x : 1 if x in medium_campaign_id  else 0 )
df_hist_train['small_campaign_id']=df_hist_train['campaign_id'].map(lambda x : 1 if x in small_campaign_id  else 0 )
df_hist_train.drop('campaign_id',axis=1,inplace=True)


df_hist_test['campaign_id'].value_counts()  
large_campaign_id=[359520,405490]
medium_campaign_id=[360936,118601,98970,404347,82320]
small_campaign_id=[396664,105960,414159]
df_hist_test['large_campaign_id']=df_hist_test['campaign_id'].map(lambda x : 1 if x in large_campaign_id  else 0 )
df_hist_test['medium_campaign_id']=df_hist_test['campaign_id'].map(lambda x : 1 if x in medium_campaign_id  else 0 )
df_hist_test['small_campaign_id']=df_hist_test['campaign_id'].map(lambda x : 1 if x in small_campaign_id  else 0 )
df_hist_test.drop('campaign_id',axis=1,inplace=True)
df_hist_train['gender'].head(10)
df_hist_train['gender'].fillna('Male',inplace=True)
df_hist_train['gender']=df_hist_train['gender'].map({'Male':1,'Female':0})

df_hist_test['gender'].head(10)
df_hist_test['gender'].fillna('Male',inplace=True)
df_hist_test['gender']=df_hist_test['gender'].map({'Male':1,'Female':0})

#ct=pd.crosstab(df_hist_train['count'],df_hist_train["is_click"],margins=True)
#df_hist_train[df_hist_train['count']==12452]['user_id']
df_hist_train['age_level'].fillna(2,inplace=True)
df_hist_train['age_level'].value_counts()
low_age_group=[0]
middle_age_group=[1,2,3,4]
high_age_group=[5,6]
df_hist_train['low_age_group']=df_hist_train['age_level'].map(lambda x : 1 if x in low_age_group  else 0 )
df_hist_train['middle_age_group']=df_hist_train['age_level'].map(lambda x : 1 if x in middle_age_group  else 0 )
df_hist_train['high_age_group']=df_hist_train['age_level'].map(lambda x : 1 if x in high_age_group  else 0 )
df_hist_train.drop('age_level', axis=1,inplace=True)

df_hist_test['age_level'].fillna(2,inplace=True)
df_hist_test['age_level'].value_counts()
low_age_group=[0]
middle_age_group=[1,2,3,4]
high_age_group=[5,6]
df_hist_test['low_age_group']=df_hist_test['age_level'].map(lambda x : 1 if x in low_age_group  else 0 )
df_hist_test['middle_age_group']=df_hist_test['age_level'].map(lambda x : 1 if x in middle_age_group  else 0 )
df_hist_test['high_age_group']=df_hist_test['age_level'].map(lambda x : 1 if x in high_age_group  else 0 )
df_hist_test.drop('age_level', axis=1,inplace=True)

df_hist_train['user_depth'].value_counts()
df_hist_train['user_depth'].fillna(3,inplace=True)
df_hist_train['user_depth_1']=df_hist_train['user_depth'].map(lambda x : 1 if x==1.0  else 0 )
df_hist_train['user_depth_2']=df_hist_train['user_depth'].map(lambda x : 1 if x==2.0  else 0 )
df_hist_train['user_depth_3']=df_hist_train['user_depth'].map(lambda x : 1 if x==3.0 else 0 )
df_hist_train.drop('user_depth',axis=1,inplace=True)


df_hist_test['user_depth'].value_counts()
df_hist_test['user_depth'].fillna(3,inplace=True)
df_hist_test['user_depth_1']=df_hist_test['user_depth'].map(lambda x : 1 if x==1.0  else 0 )
df_hist_test['user_depth_2']=df_hist_test['user_depth'].map(lambda x : 1 if x==2.0  else 0 )
df_hist_test['user_depth_3']=df_hist_test['user_depth'].map(lambda x : 1 if x==3.0 else 0 )
df_hist_test.drop('user_depth',axis=1,inplace=True)
columns=df_hist_train.columns
print(columns)
columns=df_hist_test.columns
print(columns)
#df_hist_train['action'].value_counts()
#df_hist_train['view']=df_hist_train['action'].map(lambda x : 1 if x=='view'  else 0 )
#df_hist_train['interest']=df_hist_train['action'].map(lambda x : 1 if x=='interest'  else 0 )
#df_hist_train.drop('action', axis=1,inplace=True)

#df_hist_test['action'].value_counts()
#df_hist_test['view']=df_hist_test['action'].map(lambda x : 1 if x=='view'  else 0 )
#df_hist_test['interest']=df_hist_test['action'].map(lambda x : 1 if x=='interest'  else 0 )
#df_hist_test.drop('action', axis=1,inplace=True)
df_hist_orig=df_hist_train
drop_column=['user_id','DateTime','is_click']
df_hist_train.drop(drop_column,axis=1,inplace=True)



df_hist_orig_test=df_hist_test
drop_column=['user_id','DateTime']
df_hist_test.drop(drop_column,axis=1,inplace=True)
df_hist_train.drop('city_development_index',axis=1,inplace=True)
df_hist_test.drop('city_development_index',axis=1,inplace=True)
df_hist_train.drop('product',axis=1,inplace=True)
df_hist_train.drop('product_category_1',axis=1,inplace=True)
df_hist_train.drop('user_group_id',axis=1,inplace=True)

df_hist_test.drop('product',axis=1,inplace=True)
df_hist_test.drop('product_category_1',axis=1,inplace=True)
df_hist_test.drop('user_group_id',axis=1,inplace=True)
print(df_hist_train.head());
print(df_hist_test.head());

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel




def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

def recover_train_test_target():
    train=df_hist_train
    test=df_hist_test
    print(train.shape)
    print(test.shape)
    print(test['session_id'].head(20))
    targets =target_new
    print(targets.shape)  
    return train, test, targets
train, test, targets = recover_train_test_target()

print(train.info())
print(test.info())

#clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
#clf = clf.fit(train, targets)
dt = DecisionTreeClassifier(random_state=17, class_weight='balanced');
max_depth_values = [5, 6, 7, 8, 9];
max_features_values = [4, 5, 6, 7];
tree_params = {'max_depth': max_depth_values,
               'max_features': max_features_values};
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17);
dt_grid_search = GridSearchCV(dt, tree_params, n_jobs=-1, scoring ='roc_auc', cv=skf);
dt_grid_search.fit(train, targets);
print(test.info());
print(train.info());
y_pred=dt_grid_search.predict(test);
my_submission=pd.DataFrame({"session_id":test["session_id"],"is_click":y_pred});
my_submission.to_csv('submission_dt.csv',index=False);
