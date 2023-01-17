import pandas as pd
import numpy as np
df= pd.read_csv('../input/crv.csv')
df.head()
df_train = df.loc[df['ORIGIN'] == 'train']
df_test = df.loc[df['ORIGIN'] == 'test']
df_train.shape
df_test.shape
target_train = df['CARAVAN'].loc[df['ORIGIN'] == 'train']
target_train.shape
target_test = df['CARAVAN'].loc[df['ORIGIN'] == 'test']
target_test.shape
df.isnull().sum()
#since it is showing more columns we will check which column's isnull().sum() value is greater than zero
df_nulls = df.isnull().sum().to_frame('nulls')
df_nulls[df_nulls.nulls > 0]
X_train = df_train.drop(['ORIGIN'], axis=1)
X_test = df_test.drop(['ORIGIN'], axis=1)
len(X_train)
len(X_test)
y_train = target_train
y_test = target_test
X_train.nunique() 

coltypes = (X_train.nunique() < 5)  
coltypes    # All True are cat and all False are num


cat_cols = coltypes[coltypes==True].index.tolist()
num_cols = coltypes[coltypes==False].index.tolist()
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier as rf
ohe = onehot(sparse = False)
ohe.fit_transform(X_train[cat_cols])
SS= ss()
SS.fit_transform(X_train[num_cols])
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt.predict(X_test)
score = dt.score(X_test, y_test)
print(score)
df_train = df.loc[df['ORIGIN'] == 'train']
df_test = df.loc[df['ORIGIN'] == 'test']
target_train = df['CARAVAN'].loc[df['ORIGIN'] == 'train']
target_test = df['CARAVAN'].loc[df['ORIGIN'] == 'test']
X_train = df_train.drop(['ORIGIN'], axis=1)
X_test = df_test.drop(['ORIGIN'], axis=1)
y_train = target_train
y_test = target_test
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
yy=rf.predict(X_test)
score = rf.score(X_test, y_test)
score
df_train = df.loc[df['ORIGIN'] == 'train']
df_test = df.loc[df['ORIGIN'] == 'test']
target_train = df['CARAVAN'].loc[df['ORIGIN'] == 'train']
target_test = df['CARAVAN'].loc[df['ORIGIN'] == 'test']
X_train = df_train.drop(['ORIGIN'], axis=1)
X_test = df_test.drop(['ORIGIN'], axis=1)
y_train = target_train
y_test = target_test
from sklearn.preprocessing import StandardScaler
ct= ColumnTransformer([('abc', StandardScaler(),num_cols) ], remainder = 'passthrough')
from sklearn.pipeline import make_pipeline
pipe = make_pipeline( ct, rf)
ct.fit(X_train,y_train)
pipe.fit(X_train,y_train)
yy = pipe.predict(X_test)
np.sum(yy == y_test)/len(y_test)
df_train = df.loc[df['ORIGIN'] == 'train']
df_test = df.loc[df['ORIGIN'] == 'test']
target_train = df['CARAVAN'].loc[df['ORIGIN'] == 'train']
target_test = df['CARAVAN'].loc[df['ORIGIN'] == 'test']
X_train = df_train.drop(['ORIGIN'], axis=1)
X_test = df_test.drop(['ORIGIN'], axis=1)
y_train = target_train
y_test = target_test
X_train.nunique() 

coltypes = (X_train.nunique() < 5)  
coltypes   


cat_cols = coltypes[coltypes==True].index.tolist()
num_cols = coltypes[coltypes==False].index.tolist()
from sklearn.preprocessing import OneHotEncoder
ct= ColumnTransformer([('abc', StandardScaler(),num_cols),('cde', OneHotEncoder(handle_unknown='ignore'),cat_cols) ], remainder = 'passthrough')
ct.fit(X_train,y_train)
from sklearn.ensemble import RandomForestClassifier as rf
pipe = Pipeline([ ('ct',ct), ('rf', rf() )])
y_train.shape
pipe.fit(X_train,y_train)
yy=pipe.predict(X_test)
np.sum(yy == y_test)/len(y_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,yy)
cm
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
from sklearn.model_selection import train_test_split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(df.drop(['ORIGIN','CARAVAN'],axis='columns'),df.CARAVAN, test_size=0.4)
X_train_c.nunique() 

coltypes_c = (X_train_c.nunique() < 5)  
coltypes_c   
cat_cols_c = coltypes_c[coltypes_c==True].index.tolist()
num_cols_c = coltypes_c[coltypes_c==False].index.tolist()
ct_c= ColumnTransformer([('abc', StandardScaler(),num_cols_c),('cde', OneHotEncoder(handle_unknown='ignore'),cat_cols_c) ], remainder = 'passthrough')
pipe_c = Pipeline([ ('ct',ct_c), ('rf', rf() )])
ct_c.fit(X_train_c,y_train_c)
pipe_c.fit(X_train_c,y_train_c)
from sklearn.ensemble import RandomForestClassifier as rf
X_test_c.isnull().sum()
yy_c=pipe_c.predict(X_test_c)
np.sum(yy_c == y_test_c)/len(y_test_c)
from sklearn.metrics import confusion_matrix
cm_c=confusion_matrix(y_test_c,yy_c)
cm_c
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm_c,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test_c, yy_c)