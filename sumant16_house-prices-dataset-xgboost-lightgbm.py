import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from pandas_profiling import ProfileReport

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-dataset/sample_submission.csv")

test = pd.read_csv("../input/house-prices-dataset/test.csv")

train = pd.read_csv("../input/house-prices-dataset/train.csv")
train.head(5)
test.head(5)
target=pd.DataFrame(train['SalePrice'])
train_df=train.drop(['SalePrice'],axis=1)
train.info()
comb_df=train_df.append(test)
comb_df.shape
num_features=[]
cat_features=[]
for col in comb_df.columns:

    if(comb_df[col].dtypes!='object'):

        num_features.append(col)

    else:

        cat_features.append(col)
print("Total numerical features",len(num_features))
print(num_features)
print("Total number of ctegorical features",len(cat_features))
print(cat_features)
ProfileReport(comb_df)
comb_df[num_features].head()
for col in ['MSSubClass','OverallQual','OverallCond']:

    comb_df[col]=comb_df[col].astype('object')
num_features=[]
cat_features=[]
for col in comb_df.columns:

    if(comb_df[col].dtypes!='object'):

        num_features.append(col)

    else:

        cat_features.append(col)
print(len(num_features))
print(len(cat_features))
col_drop=['Id','Alley','Fence','MiscFeature','PoolQC']
comb_df=comb_df.drop(col_drop,axis=1)
comb_df['Age']=comb_df['YrSold']-comb_df['YearBuilt']
comb_df=comb_df.drop(['3SsnPorch','Condition2','LowQualFinSF','MiscVal','PoolArea','Utilities','YearBuilt','YrSold','YearRemodAdd'],axis=1)
comb_df.shape
num_features=[]
cat_features=[]
for col in comb_df.columns:

    if(comb_df[col].dtypes!='object'):

        num_features.append(col)

    else:

        cat_features.append(col)
print(len(num_features))
print(len(cat_features))
comb_df_num=comb_df[num_features]
imputer=IterativeImputer()
comb_df_num_imp=pd.DataFrame(imputer.fit_transform(comb_df_num))
comb_df_num_imp.columns=comb_df_num.columns
comb_df_num_imp.index=comb_df_num.index
comb_df_cat=comb_df[cat_features]
comb_df_cat=comb_df_cat.fillna('Unknown')

le=LabelEncoder()
for col in comb_df_cat.columns:

    comb_df_cat[col]=le.fit_transform(comb_df_cat[col])

    
comb_new=pd.DataFrame()
comb_new=pd.concat([comb_df_cat,comb_df_num_imp],axis=1)
comb_new.head(2)
scaler=StandardScaler()
comb_new_scaled=pd.DataFrame(scaler.fit_transform(comb_new))
comb_new_scaled_train=comb_new_scaled.iloc[:1460,:]
X=comb_new_scaled_train
y=target
comb_new_scaled_test=comb_new_scaled.iloc[1460:,:]
test_data=comb_new_scaled_test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
XGB = XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=1000,reg_alpha=0.001,reg_lambda=0.000001,n_jobs=-1,min_child_weight=3)
XGB.fit(X_train,y_train)
print(XGB.score(X_train,y_train))
print(XGB.score(X_test,y_test))
y_pred = pd.DataFrame( XGB.predict(test_data))
y_pred