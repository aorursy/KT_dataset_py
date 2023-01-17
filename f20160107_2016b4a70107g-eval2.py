import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/eval-lab-2-f464/train.csv')

df.head()
df.info()
missing_count = df.isnull().sum()

missing_count[missing_count > 0]

missing_count

df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
df.fillna(value=df.mean(),inplace=True)

df.head()
df.isnull().any().any()
df.columns
corr=df.corr()

plt.figure(figsize=(18,18))

mask = np.zeros_like(corr)

cmap=sns.diverging_palette(220,10,as_cmap=True)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

  ax = sns.heatmap(corr,cmap=cmap,mask=mask, vmax=1,vmin=-1, square=True)
num_features = ['chem_0', 'chem_1','chem_2','chem_3', 'chem_4', 

    'chem_6','chem_7','attribute']

X= df[num_features]

y = df["class"]
X.head()
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MinMaxScaler

scaler = RobustScaler()

scaler1 = MinMaxScaler()

from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(X,y,test_size = 0.2,random_state=42)



x_train_scaled = x_train

x_val_scaled = x_val



x_train_scaled[num_features] = scaler.fit_transform(x_train[num_features])

x_val_scaled[num_features] = scaler.transform(x_val[num_features])



x_scaled=scaler.fit_transform(X[num_features])



from sklearn.ensemble import ExtraTreesClassifier

clf7= ExtraTreesClassifier(n_estimators=10, bootstrap=True, oob_score=True,  class_weight='balanced',random_state=42).fit(x_train,y_train)

y_pred_7= clf7.predict(x_val)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_val, y_pred_7))
df1=pd.read_csv('../input/eval-lab-2-f464/test.csv')

df1.fillna(value=df.mean(),inplace=True)
X_test = df1[num_features]

x_scaled=scaler.fit_transform(X[num_features])

x_test_scaled=scaler.transform(df1[num_features])
from sklearn.ensemble import RandomForestClassifier

clf=ExtraTreesClassifier(n_estimators=1000, bootstrap=True, oob_score=True, class_weight='balanced').fit(X,y)

pred1=clf.predict(X_test)

pred1
df1['class']=np.array(pred1)

df1.head()
out=df1[['id','class']]

out=out.round({'class': 0})

out.head()
out.to_csv('final_sub1.csv',index=False)
from xgboost import XGBClassifier

import xgboost as xgb

clf2 = XGBClassifier(silent=True, scale_pos_weight=1,learning_rate=0.03,colsample_bytree = 1,subsample = 0.8,objective='multi:softprob',  n_estimators=10000,reg_alpha = 0.3, max_depth=3,gamma=1).fit(X[num_features],y)  

pred2 = clf2.predict(X_test[num_features])

pred2
df1['class']=np.array(pred2)

df1.head()
out=df1[['id','class']]

out=out.round({'class': 0})

out.head()
out.to_csv('final_sub2.csv',index=False)