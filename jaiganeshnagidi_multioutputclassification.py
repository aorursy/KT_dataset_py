

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv("/kaggle/input/hackerearth-ml-challenge-pet-adoption/train.csv")
train.head()
train.info()
train.shape
for col in train.columns:
    print(col,':',len(train[col].unique()))
#removing unwanted text in pet_id
train['pet_id']=train['pet_id'].str.replace('[^0-9]',"")

#converting into int data type
train['pet_id'] = train.pet_id.astype(int)
                                      
#converting both the columns into datetime format
train['issue_date']=pd.to_datetime(train['issue_date'])
train['listing_date']=pd.to_datetime(train['listing_date'])


#taking duration
train['duration']=train['listing_date']-train['issue_date']
train['duration']
#considering only no of days ---duration of days
train['duration'] = train['duration'].dt.days
train['duration']
#drop issue_date and listing date columns

train.drop(['issue_date','listing_date'],axis=1,inplace=True)
#Checking for missing values
train.isna().sum()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sns.countplot(x=train['condition'],data=train)
plt.title("Condition values composition")
plt.show()
train=train.fillna(2.0)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sns.countplot(x=train['condition'],data=train)
plt.title("Condition values composition")
plt.show()
train.isna().sum()
#frequency Encoding
feq_encode = train.groupby('color_type').size()/len(train)
print(feq_encode)

train.loc[:,'color_type'] = train['color_type'].map(feq_encode)
plt.figure(figsize=(10,8))
sns.distplot(train['length(m)'])
plt.title("Length data Distribution")
plt.show()

plt.figure(figsize=(10,8))
sns.distplot(train['height(cm)'])
plt.title("Height data Distribution")
plt.show()
sns.set_style('darkgrid')
plt.figure(figsize=(10,8))
sns.countplot("condition",hue="pet_category",data=train)
plt.show()
plt.figure(figsize=(10,8))
sns.countplot("condition",hue="breed_category",data=train)
plt.show()
plt.figure(figsize=(18,10))
sns.heatmap(train.corr(),annot=True)
plt.figure(figsize=(10,8))
sns.regplot(x="X1",y="X2",data=train)
plt.title("realtion between X1 and X2")
plt.show()
import xgboost as xgb
x=train.drop(['pet_category','breed_category'],axis=1)
y=train.breed_category #target label
x.head(5)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
## Hyper Parameters

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.50 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
from sklearn.model_selection import RandomizedSearchCV
xgb_model = xgb.XGBClassifier()

random_search=RandomizedSearchCV(xgb_model,param_distributions=params,n_iter=5,n_jobs=-1,cv=5,verbose=3)
random_search.fit(x_train,y_train)
random_search.best_params_ #printing best parameters
random_search.best_estimator_  #best estimator 
first_model=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.15, max_delta_step=0, max_depth=10,
              min_child_weight=1,  monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
first_model.fit(x_train,y_train)
y_pred=first_model.predict(x_test)
print("Accuracy score:",accuracy_score(y_test,y_pred))
cm=confusion_matrix(y_pred,y_test)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True)
plt.show()
y=train.pet_category #target label
sns.countplot('pet_category',data=train)
x.head(5)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
xgb_model = xgb.XGBClassifier()

random_search=RandomizedSearchCV(xgb_model,param_distributions=params,n_iter=5,n_jobs=-1,cv=5,verbose=3)
random_search.fit(x_train,y_train)
random_search.best_estimator_
random_search.best_params_
second_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.2, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=10,
              min_child_weight=3,monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
second_model.fit(x_train,y_train)
ypred=second_model.predict(x_test)
print("Accuracy score:",accuracy_score(y_test,y_pred))