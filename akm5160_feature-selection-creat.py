import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

data1=pd.read_csv('../input/train_LZdllcl.csv')

data3=pd.read_csv('../input/train_LZdllcl.csv')

data1.head()
data1[['department', 'region', 'education', 'gender',

       'recruitment_channel']].head()

cat_col=['department','region','education','gender','recruiment_channel']

cols=['department', 'region', 'education', 'gender',

       'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating',

       'length_of_service', 'KPIs_met >80%', 'awards_won?',

       'avg_training_score']
dict_department=dict((v,k) for k,v in (dict(enumerate(list(data3.department.unique())))).items())

dict_region=dict((v,k) for k,v in (dict(enumerate(list(data3.region.unique())))).items())

dict_recr=dict((v,k) for k,v in (dict(enumerate(list(data3.recruitment_channel.unique())))).items())

dict_edu=dict((v,k) for k,v in (dict(enumerate(list(data3.education.unique())))).items())

dict_gen=dict((v,k) for k,v in (dict(enumerate(list(data3.gender.unique())))).items())

            
data3['department']=data3['department'].map(dict_department)

data3['region']=data3['region'].map(dict_region)

data3['recruitment_channel']=data3['recruitment_channel'].map(dict_recr)

data3['education']=data3['education'].map(dict_edu)

data3['gender']=data3['gender'].map(dict_gen)
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,matthews_corrcoef

x_train,x_test,y_train,y_test=train_test_split(data3[cols], data3['is_promoted'], test_size=0.20, random_state=123)

x_train.head()
from xgboost import XGBClassifier

clf=XGBClassifier().fit(x_train[cols],y_train)

print (classification_report(y_test,clf.predict(x_test[cols])))

print(matthews_corrcoef(y_test,clf.predict(x_test[cols])))
data1.dtypes

#There are continuous variable and categorical variable. 

#We will first start with numerical variable.
print (len(data1.department.unique()))

dum_department=pd.get_dummies(data1['department'], prefix='department', drop_first=False)

dum_department.head()
data1[['department','is_promoted']].head()

dpt=pd.DataFrame(data1.department.value_counts())

dpt=dpt.reset_index()

dpt.columns=['department','count']

target_label=data1[['department','is_promoted']].groupby(['department']).sum()

target_label=target_label.reset_index()

final_encoded=pd.merge(dpt,target_label,on='department',how='left')

final_encoded['mean_encoded']=final_encoded['is_promoted']/final_encoded['count']

final_encoded=final_encoded[['department','mean_encoded']]

final_encoded
data1=pd.merge(data1,final_encoded,on='department',how='left')

data1[['department','mean_encoded']].head()
sns.distplot(data1[data1['is_promoted']==1]['avg_training_score'], color='r')

sns.distplot(data1[data1['is_promoted']==0]['avg_training_score'], color='g')

#Observation- Idea behind doing dist plot, plotting separately for categories to identify regions where there are no overlaps. If we can find pockets of

#non-overlap, then the variable can clearly differenitate or classify the target and would add value to the model.
# from sklearn.preprocessing import CategoricalEncoder

data1=pd.merge(data1,data1[['region','department','avg_training_score']].groupby(['region','department']).mean(),how='left',on=['region','department'])

data1=data1.rename(columns={'avg_training_score_x':'avg_training_score','avg_training_score_y':'mean_reg_dpt'})

data1['new_avg_trng_score']=data1['avg_training_score']/data1['mean_reg_dpt']

data1.head()
plt.subplots(figsize=(16,7))

sns.distplot(data1[data1.is_promoted==0]['new_avg_trng_score'],color='g',label='Not Promoted')

sns.distplot(data1[data1.is_promoted==1]['new_avg_trng_score'],color='r',label='Promoted')
print (data1.age.describe())
sns.distplot(data1[data1['is_promoted']==1]['age'], color='r')

sns.distplot(data1[data1['is_promoted']==0]['age'], color='g')
data1['age_bin']=pd.qcut(data1['age'], q=[0,.10,.20,.30,.40,.50,.60,.70,.80,.90,1], labels=False)
sns.distplot(data1[data1['is_promoted']==1]['age_bin'], color='r')

sns.distplot(data1[data1['is_promoted']==0]['age_bin'], color='g')

sns.distplot(data1[data1['is_promoted']==1]['length_of_service'], color='r')

sns.distplot(data1[data1['is_promoted']==0]['length_of_service'], color='g')
data1.columns

cols=['gender','no_of_trainings', 'age',

       'length_of_service', 'KPIs_met >80%', 'awards_won?',

       'avg_training_score']

cols_a=['gender','no_of_trainings', 'age',

       'length_of_service', 'KPIs_met >80%', 'awards_won?', 'mean_reg_dpt','new_avg_trng_score']

category_cols = ['gender','recruitment_channel', 'region', 'department']
data1.gender.unique()
data2=data1

data2['gender']=data2['gender'].map({'m':1,'f':0})

#data2['recruitment_channel']=data2['recruitment_channel'].map({'sourcing':1,'other':0,'referred':2})

dum_recr=pd.get_dummies(data2['recruitment_channel'], prefix_sep='recr', drop_first=True)

#data2['department']=data2['department'].map({'Sales & Marketing':0, 'Operations':1, 'Technology':2, 'Analytics':3,

#       'R&D':4, 'Procurement':5, 'Finance':6, 'HR':7, 'Legal':8})

dum_dpt=pd.get_dummies(data2['department'], prefix_sep='dpt', drop_first=True)

data2=pd.concat([dum_dpt, dum_recr,data2], axis=1)

cols=cols+list(dum_recr.columns)+list(dum_dpt.columns)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data2[cols], data2['is_promoted'], test_size=0.20, random_state=123)

x_train.head()
#automated feature selection using Standard Scikit Package. One of the most popular such algorithm is Random Forest

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

x_train,x_test,y_train,y_test=train_test_split(data1[cols_a], data1['is_promoted'], test_size=0.20, random_state=123)

clf=XGBClassifier().fit(x_train[cols_a],y_train)

print (classification_report(y_test,clf.predict(x_test[cols_a])))

print(matthews_corrcoef(y_test,clf.predict(x_test[cols_a])))
category_cols = ['gender']
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier().fit(x_train[cols_a],y_train)

print (classification_report(y_test,clf.predict(x_test[cols_a])))

print(matthews_corrcoef(y_test,clf.predict(x_test[cols_a])))

clf.feature_importances_

sns.barplot(y=cols_a , x=clf.feature_importances_)

# Gender, awards_won, recruitment channel and no_of_traings recieved are few features marked as least important. 

#Let's analyse them, before we cross them off from our list

#gender- it points to fact that the dataset we have, belongs to a region where getting promoted is gender insensitive. More of just work culture.