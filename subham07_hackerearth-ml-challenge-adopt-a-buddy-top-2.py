import pandas as pd

df_train=pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/train.csv')

df_test=pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/test.csv')
print(df_train.head())

print(df_train.tail())
print(df_train.columns)
print(df_train['pet_category'].unique())
y_train=df_train['pet_category'].values

print(y_train)
print(df_train['condition'].unique())

print(df_train['color_type'].unique())

print(df_train['breed_category'].unique())
print(df_train['length(m)'].unique())

print(df_train['height(cm)'].unique())

print(df_train['X1'].unique())

print(df_train['X2'].unique())
print(df_train['length(m)'].isna().sum())

print(df_train['height(cm)'].isna().sum())

print(df_train['X1'].isna().sum())

print(df_train['X2'].isna().sum())



print(df_train['condition'].isna().sum())

print(df_train['color_type'].isna().sum())

print(df_train['breed_category'].isna().sum())
print(df_test['length(m)'].isna().sum())

print(df_test['height(cm)'].isna().sum())

print(df_test['X1'].isna().sum())

print(df_test['X2'].isna().sum())



print(df_test['condition'].isna().sum())

print(df_test['color_type'].isna().sum())

#print(df_test['breed_category'].isna().sum())
print(df_train.groupby(['condition']).size())



print(df_train[df_train['condition'].isnull()])
print(df_train[df_train['condition'].isnull()]['breed_category'].unique())
df_train[df_train['breed_category']==2].count()
import numpy as np

df_train['condition']=df_train['condition'].replace(np.nan,3)
df_test['condition']=df_test['condition'].replace(np.nan,3)
print(df_train.groupby(['condition']).size())



print(df_train[df_train['condition'].isnull()])
df_train['diff_days']=np.abs((pd.to_datetime(df_train['listing_date'].values)-pd.to_datetime(df_train['issue_date'].values)).days)



print(df_train['diff_days'].values)
df_test['diff_days']=np.abs((pd.to_datetime(df_test['listing_date'].values)-pd.to_datetime(df_test['issue_date'].values)).days)



print(df_test['diff_days'].values)
print(df_train['issue_date'][5], " ", df_train['listing_date'][5], " ",df_train['diff_days'][5])
df_train_new=df_train.drop(columns=['issue_date','listing_date'])



print(df_train_new.head())

print(df_train_new.columns)
df_test_new=df_test.drop(columns=['issue_date','listing_date'])



print(df_test_new.head())

print(df_test_new.columns)
from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

df_train_new["color_type_code"] = lb_make.fit_transform(df_train_new["color_type"])

df_train_new[["color_type", "color_type_code"]].head(11)
df_test_new["color_type_code"] = lb_make.transform(df_test_new["color_type"])

df_test_new[["color_type", "color_type_code"]].head(11)
df_train_new['color_type_code'].unique()



df_train_new=df_train_new.drop(columns=['color_type'])



print(df_train_new.head(25))
df_test_new['color_type_code'].unique()



df_test_new=df_test_new.drop(columns=['color_type'])



print(df_test_new.head(25))
print(df_train_new.columns)
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import skew 

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

#Check the distribution 

sns.distplot(df_train_new['length(m)'], color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="Length")

ax.set(title="Length distribution")

sns.despine(trim=True, left=True)

plt.show()



print("skew value: ", skew(df_train_new['length(m)']))
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import skew 

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

#Check the distribution 

sns.distplot(df_train_new['height(cm)'], color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="Height")

ax.set(title="Height distribution")

sns.despine(trim=True, left=True)

plt.show()



print("skew value: ", skew(df_train_new['height(cm)']))
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import skew 

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

#Check the distribution 

sns.distplot(df_train_new['X1'], color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="X1")

ax.set(title="X1 distribution")

sns.despine(trim=True, left=True)

plt.show()



print("skew value: ", skew(df_train_new['X1']))
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import skew 

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

#Check the distribution 

sns.distplot(df_train_new['X2'], color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="X2")

ax.set(title="X2 distribution")

sns.despine(trim=True, left=True)

plt.show()



print("skew value: ", skew(df_train_new['X2']))
# to check skewness of X1 Score

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import skew 

X1_trans=np.log(1+df_train_new['X1'].values)

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

#Check the distribution 

sns.distplot(X1_trans, color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="X1")

ax.set(title="X1 distribution")

sns.despine(trim=True, left=True)

plt.show()



print("skew value: ", skew(X1_trans))
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import skew 

X1_trans_test=np.log(1+df_test_new['X1'].values)

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

#Check the distribution 

sns.distplot(X1_trans, color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="X1")

ax.set(title="X1 distribution")

sns.despine(trim=True, left=True)

plt.show()



print("skew value: ", skew(X1_trans_test))
df_train_norm=df_train_new

df_train_norm['X1']=X1_trans



df_test_norm=df_test_new

df_test_norm['X1']=X1_trans_test
df_train_norm['Low_Height']=np.where(df_train_norm['height(cm)']<=15,1,0)

df_train_norm['Medium_Height']=np.where(((df_train_norm['height(cm)']>15) & (df_train_norm['height(cm)']<=30)),1,0)

df_train_norm['High_Height']=np.where(df_train_norm['height(cm)']>30,1,0)



df_test_norm['Low_Height']=np.where(df_test_norm['height(cm)']<=15,1,0)

df_test_norm['Medium_Height']=np.where(((df_test_norm['height(cm)']>15) & (df_test_norm['height(cm)']<=30)),1,0)

df_test_norm['High_Height']=np.where(df_test_norm['height(cm)']>30,1,0)
df_train_norm['Low_Length']=np.where(df_train_norm['length(m)']<=0.3,1,0)

df_train_norm['Medium_Length']=np.where((df_train_norm['length(m)']>0.3) & (df_train_norm['length(m)']<=0.6),1,0)

df_train_norm['High_Length']=np.where(df_train_norm['length(m)']>0.6,1,0)



df_test_norm['Low_Length']=np.where(df_test_norm['length(m)']<=0.3,1,0)

df_test_norm['Medium_Length']=np.where((df_test_norm['length(m)']>0.3) & (df_test_norm['length(m)']<=0.6),1,0)

df_test_norm['High_Length']=np.where(df_test_norm['length(m)']>0.6,1,0)
print(df_train_norm.head(20))
from sklearn.model_selection import train_test_split



Y=df_train_norm['pet_category'].values

X=df_train_norm.drop(columns=['pet_category','pet_id','breed_category'])



X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=0)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)



import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



model=XGBClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print(f1_score(y_pred,y_test,average='weighted'))

print(accuracy_score(y_pred,y_test))







from sklearn.model_selection import train_test_split



Y=df_train_norm['breed_category'].values

X=df_train_norm.drop(columns=['pet_category','pet_id','breed_category'])



X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=0)





print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



model2=XGBClassifier()

model2.fit(X_train,y_train)

y_pred=model2.predict(X_test)

print(f1_score(y_pred,y_test,average='weighted'))

print(accuracy_score(y_pred,y_test))
from sklearn.model_selection import train_test_split



Y=df_train_norm['pet_category'].values

X=df_train_norm.drop(columns=['pet_category','pet_id','breed_category'])



X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=0)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)



import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



model=XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.4,  

                      colsample_bytree = 0.4,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=1000, 

                      reg_alpha = 0.3,

                      max_depth=4, 

                      gamma=4)



model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print(f1_score(y_pred,y_test,average='weighted'))

print(accuracy_score(y_pred,y_test))







from sklearn.model_selection import train_test_split



Y=df_train_norm['breed_category'].values

X=df_train_norm.drop(columns=['pet_category','pet_id','breed_category'])



X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=0)





print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



model2=XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.4,  

                      colsample_bytree = 0.4,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=1000, 

                      reg_alpha = 0.3,

                      max_depth=4, 

                      gamma=4)

model2.fit(X_train,y_train)

y_pred=model2.predict(X_test)

print(f1_score(y_pred,y_test,average='weighted'))

print(accuracy_score(y_pred,y_test))
from sklearn.model_selection import train_test_split



Y=df_train_norm['pet_category'].values

X=df_train_norm.drop(columns=['pet_category','pet_id','breed_category'])





import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



modelx=XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.4,  

                      colsample_bytree = 0.4,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=1000, 

                      reg_alpha = 0.3,

                      max_depth=4, 

                      gamma=4)



modelx.fit(X,Y)







from sklearn.model_selection import train_test_split



Y=df_train_norm['breed_category'].values

X=df_train_norm.drop(columns=['pet_category','pet_id','breed_category'])





from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



modelx2=XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.4,  

                      colsample_bytree = 0.4,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=1000, 

                      reg_alpha = 0.3,

                      max_depth=4, 

                      gamma=4)

modelx2.fit(X,Y)

df_test_norm['length(cm)']=df_test_norm['length(m)']*100

df_train_norm['length(cm)']=df_train_norm['length(m)']*100





df_test_today=df_test_norm.drop(columns=['length(m)'])

df_train_today=df_train_norm.drop(columns=['length(m)'])
df_train_today['ratio']=df_train_today['X2']/(1+df_train_today['X1'])

df_test_today['ratio']=df_test_today['X2']/(1+df_test_today['X1'])
df_train_today['lhratio']=df_train_today['height(cm)']/(1+df_train_today['length(cm)'])

df_test_today['lhratio']=df_test_today['height(cm)']/(1+df_test_today['length(cm)'])
from sklearn.model_selection import train_test_split



Y=df_train_today['pet_category'].values

X=df_train_today.drop(columns=['pet_category','pet_id','breed_category'])





import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



modelx=XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.47,  

                      colsample_bytree = 0.4,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=1500, 

                      reg_alpha = 0.3,

                      max_depth=7, 

                      gamma=4,

                     random_state=42)



modelx.fit(X,Y)





from sklearn.model_selection import train_test_split



Y=df_train_today['breed_category'].values

X=df_train_today.drop(columns=['pet_category','pet_id','breed_category'])





from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



modelx2=XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.4,  

                      colsample_bytree = 0.4,

                      subsample = 0.8,

                      objective='binary:logistic', 

                      n_estimators=1000, 

                      reg_alpha = 0.3,

                      max_depth=4, 

                      gamma=4,

                      random_state=42)

modelx2.fit(X,Y)

from sklearn.model_selection import train_test_split



#Y_test_fin=df_test_new['pet_category'].values

idx=df_test_today['pet_id'].values

X_test_fin=df_test_today.drop(columns=['pet_id'])





y_pred_fin=modelx.predict(X_test_fin)





from sklearn.model_selection import train_test_split



#Y_test_fin=df_test_new['pet_category'].values

idx=df_test_today['pet_id'].values

X_test_fin=df_test_today.drop(columns=['pet_id'])





y_pred_fin2=modelx2.predict(X_test_fin)





df_sub = pd.DataFrame({'pet_id': idx,

                   'breed_category': y_pred_fin2,

                   'pet_category': y_pred_fin})

df_sub.to_csv('submit.csv',index=False)