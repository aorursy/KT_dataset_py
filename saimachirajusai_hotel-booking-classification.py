import pandas as pd
import numpy as np
df=pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
print(df.shape)
df.head()
df.isnull().sum().sort_values(ascending=False).head()
df["agent"].fillna(df["agent"].median(),inplace=True)
df["country"].fillna(method="ffill",inplace=True)
df["children"].fillna(df["children"].median(),inplace=True)
df.isnull().sum().sort_values(ascending=False).head()
df["Family_Size"]=df["adults"]+df["children"]+df["babies"]
columns=["arrival_date_year","arrival_date_month","arrival_date_week_number","arrival_date_day_of_month","reservation_status_date","adults","children","babies","company"]
df.drop(columns,axis=1,inplace=True)
df.shape
y=df["is_canceled"]
df.drop(["is_canceled"],axis=1,inplace=True)
i=df.select_dtypes(int)
f=df.select_dtypes(float)
i=pd.concat([i,f],axis=1)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
i=sc.fit_transform(i)
i=pd.DataFrame(i)
i.head()
X=df.select_dtypes(object)
X=pd.get_dummies(dummy)
X.head()
X=pd.concat([X,i],axis=1)
X.head()
from sklearn.model_selection import KFold,cross_val_score
kfold=KFold(n_splits=10, shuffle=True, random_state=0)
from xgboost import XGBClassifier
clf= XGBClassifier()
score=cross_val_score(clf,X,y,cv=kfold)
print(score)
print(round(np.mean(score)*100,2))
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
clf=GradientBoostingClassifier()
score=cross_val_score(clf,X,y,cv=kfold,n_jobs=-1)
print(score)
print(round(np.mean(score)*100,2))
