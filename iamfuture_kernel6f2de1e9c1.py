# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
import re

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

#import turicreate as tc



df = pd.read_csv("../input/Train_psolI3n.csv")

df1 = pd.DataFrame(df)

print(df1.isnull().sum())

df1 = df1.dropna()

print(df1.head())

df3 = pd.read_csv("../input/Test_09JmpYa.csv")

df2 = pd.DataFrame(df3)

#print(df1)
df1 = pd.DataFrame(df)

print(df1.axes)

df1 =df1.dropna()

print(df1.head())
import numpy as np



ignored = len(df1[df1['Email_Status']==0])

read  = len(df1[df1['Email_Status']==1])

acknowledge = len(df1[df1['Email_Status']==2])



print("IGNORED:",ignored)

print("READ:",read)

print("Acknowledge:",acknowledge)



labels = ['IGNORED','READ','ACKNOWLEDGE']

li = [ignored,read,acknowledge]



index = np.arange(len(labels))



plt.bar(index,li)

plt.xlabel('Mail-Tracking',fontsize =12)

plt.ylabel('Count',fontsize =12)

plt.xticks(index,labels,fontsize=12,rotation=40)

plt.show()



print(df1.head())

import seaborn as sc



X2 = df1.iloc[:,1:7]

Y2 = df1.iloc[:,-1]



corrmat  = df1.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(6,6))



g= sc.heatmap(df1[top_corr_features].corr(),annot = True,cmap ="RdYlGn")



X_val = df1.iloc[:,1:7]

Y_val = df1.iloc[:,-1]

Y_val = Y_val.astype('int')



#X_train,X_test,Y_train,Y_test = train_test_split(X_val,Y_val,test_size=0.35,random_state = 32)



y_col = df1.Email_Status

print(y_col.shape)

df1.corr()

df1.nunique()



df1.describe()

df1.columns
df2 = pd.DataFrame(df3)

print(df2.index)

#to_drop = df1()

df2.drop(['Customer_Location','Email_Campaign_Type','Time_Email_sent_Category'],1,inplace=True)

df2.head()

df2.shape

df1.Email_Type.value_counts()
df1.groupby('Email_Type').agg(['nunique'])
df1.tail()
df1.groupby('Email_Type')['Email_Source_Type'].agg(['size','count','mean'])
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split







numeric_features = ['Subject_Hotness_Score','Total_Past_Communications','Word_Count','Total_Links','Total_Images']

category_features = ['Email_Type','Email_Source_Type']



numeric_transformer = Pipeline(steps=[('imputer',SimpleImputer(fill_value='N/A')),('scaler',StandardScaler())])

category_transformer = Pipeline(steps=[('imputer',SimpleImputer(fill_value='missing')),('onehot',OneHotEncoder(handle_unknown='ignore'))])





preprocess = ColumnTransformer(transformers=[('num',numeric_transformer,numeric_features),('cat',category_transformer,category_features)])



clf = Pipeline(steps=[('preprocessor',preprocess),('classifier',LogisticRegression(solver='lbfgs'))])



X_train,X_test,Y_train,Y_test = train_test_split(df1,y_col,test_size=0.3,random_state=75)



clf.fit(X_train,Y_train)

print("Score:",clf.score(X_test,Y_test))

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split







numeric_features = ['Subject_Hotness_Score','Total_Past_Communications','Word_Count','Total_Links','Total_Images']

category_features = ['Email_Type','Email_Source_Type']



numeric_transformer = Pipeline(steps=[('imputer',SimpleImputer(fill_value='N/A')),('scaler',StandardScaler())])

category_transformer = Pipeline(steps=[('imputer',SimpleImputer(fill_value='missing')),('onehot',OneHotEncoder(handle_unknown='ignore'))])





preprocess = ColumnTransformer(transformers=[('num',numeric_transformer,numeric_features),('cat',category_transformer,category_features)])



clf = Pipeline(steps=[('preprocessor',preprocess),('classifier',LogisticRegression(solver='lbfgs'))])



from xgboost import XGBClassifier



gbm = XGBClassifier(max_depth=3,n_estimator=300,learning_rate=0.05).fit(preprocess.fit_transform(df1),y_col)



import eli5



eli5.show_weights(gbm,top=7)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split







numeric_features = ['Subject_Hotness_Score','Total_Past_Communications','Word_Count','Total_Links','Total_Images']

category_features = ['Email_Type','Email_Source_Type']



numeric_transformer = Pipeline(steps=[('imputer',SimpleImputer(fill_value='N/A')),('scaler',StandardScaler())])

category_transformer = Pipeline(steps=[('imputer',SimpleImputer(fill_value='missing')),('onehot',OneHotEncoder(handle_unknown='ignore'))])





preprocess = ColumnTransformer(transformers=[('num',numeric_transformer,numeric_features),('cat',category_transformer,category_features)])



clf = Pipeline(steps=[('preprocessor',preprocess),('classifier',LogisticRegression(solver='lbfgs'))])



from xgboost import XGBClassifier



gbm = XGBClassifier(max_depth=3,n_estimator=300,learning_rate=0.1).fit(preprocess.fit_transform(df1),y_col)

clf.fit(X_train,Y_train)

print(clf.score(X_test,Y_test))

import eli5



eli5.show_weights(gbm,top=7)
from sklearn.model_selection import cross_val_score





print(cross_val_score(gbm,preprocess.fit_transform(df1),y_col,cv=3))

df1.columns

features = ['Email_Type','Subject_Hotness_Score','Email_Source_Type','Total_Past_Communications','Word_Count','Total_Links','Total_Images']

numeric_features = ['Subject_Hotness_Score','Total_Past_Communications','Word_Count','Total_Links','Total_Images']

category_features = ['Email_Type','Email_Source_Type']



train_test_concat = pd.concat([df1[features],df2[features]])

train_test_concat.info()

train_test_concat.shape

train_test_concat.head()
import psutil

import os





from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(train_test_concat[numeric_features])

print(u'memory：{}gb'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)) 

trained_scaled = df1[['Subject_Hotness_Score','Total_Past_Communications','Word_Count','Total_Links','Total_Images']].copy()

trained_scaled = scaler.transform(trained_scaled)

trained_scaled = pd.DataFrame(trained_scaled,columns = ['Subject_Hotness_Score','Total_Past_Communications','Word_Count','Total_Links','Total_Images'])

trained_scaled = pd.concat([trained_scaled,df1[['Email_Type','Email_Source_Type']]],axis=1)

print(trained_scaled.info())

print(trained_scaled.shape)

trained_scaled.shape
test_scaled = df2[['Subject_Hotness_Score','Total_Past_Communications','Word_Count','Total_Links','Total_Images']].copy()

test_scaled = scaler.transform(test_scaled)

test_scaled = pd.DataFrame(test_scaled,columns =['Subject_Hotness_Score','Total_Past_Communications','Word_Count','Total_Links','Total_Images'])

test_scaled = pd.concat([test_scaled,df2[['Email_Type','Email_Source_Type']]],axis=1)

print(test_scaled.info())

print(test_scaled.shape)

test_scaled.describe()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



X_val = df.iloc[:,1:7]

X_train2,Y_train2,X_test2,Y_test2 = train_test_split(X_val,y_col,test_size=0.3,random_state=33)

scaler = MinMaxScaler()

X_scale = scaler.fit_transform(df1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV





param_grid = {

    'n_estimators':[900,1500],

    'max_depth':range(1,5,2),

    'max_features': ('log2','sqrt'),

    'class_weight':[{1:w} for w in [1,1.5]]

}



Gridr= GridSearchCV(RandomForestClassifier(random_state=96),param_grid)

Gridr.fit(data_with_impute,y_col)



print("Best Parameter:",str(Gridr.best_params_))



rfo = RandomForestClassifier(random_state=96,**Gridr.best_params_)

rfo.fit(data_with_impute,y_col)



rfcl_fea = pd.DataFrame(rfo.feature_importance_)

print(rfcl_fea)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



X_train2,Y_train2,X_test2,Y_test2 = train_test_split(data_with_impute,y_col,test_size=0.3,random_state=33)

scaler = MinMaxScaler()

X_scale = scaler.fit_transform(data_with_impute)