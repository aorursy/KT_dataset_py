# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def create_folds(data):
    data['kfold']=-1
    data=data.sample(frac=1).reset_index(drop=True)
    
    # Sturge's rule to calculate the approximate number of bins
    num_bins=np.floor(1+np.log2(len(data)))
    
    #create bin targets with pd.cut
    data.loc[:,'bins']=pd.cut(data['TARGET(PRICE_IN_LACS)'],bins=num_bins,labels=False)
    
    kf=model_selection.StratifiedKFold(n_splits=5)
    for f, (t_,v_) in enumerate(kf.split(X=data,y=data.bins.values)):
        data.loc[v_,'kfold']=f
    data=data.drop('bins',axis=1)
    return data
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/kaggle/input/house-price-prediction-challenge/train.csv")
df.head()
test=pd.read_csv("/kaggle/input/house-price-prediction-challenge/test.csv")
test.head()
df["ADDRESS"].value_counts()
col=["POSTED_BY","UNDER_CONSTRUCTION","RERA","BHK_OR_RK","BHK_NO.","READY_TO_MOVE","RESALE"]
DEPENDENT_VARIABLE = 'TARGET(PRICE_IN_LACS)'
CATEGORICAL_INDEPENDENT_VARIABLES = ['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'BHK_OR_RK', 'READY_TO_MOVE', 'RESALE']
CONTINUOUS_INDEPENDENT_VARIABLES = ['SQUARE_FT', 'LONGITUDE', 'LATITUDE']
from sklearn import preprocessing
df=df.drop("ADDRESS",axis=1)
test=test.drop("ADDRESS",axis=1)
    
    
    
    
df.head()
test.head()
print(df["POSTED_BY"].value_counts())
print(test["POSTED_BY"].value_counts())
print(df["BHK_OR_RK"].value_counts())
print(test["BHK_OR_RK"].value_counts())
map1={"Dealer":0,"Owner":1,"Builder":2}
map2={"BHK":0,"RK":1}
df.loc[:,"POSTED_BY"]=df["POSTED_BY"].map(map1)
test.loc[:,"POSTED_BY"]=test["POSTED_BY"].map(map1)
df.loc[:,"BHK_OR_RK"]=df["BHK_OR_RK"].map(map2)
test.loc[:,"BHK_OR_RK"]=test["BHK_OR_RK"].map(map2)
df.head()
test.head()
df['price'] = df['TARGET(PRICE_IN_LACS)']
df.drop('TARGET(PRICE_IN_LACS)',axis=1,inplace=True)
df.head()
corr = df.corr()
corr['price'].sort_values(ascending=False)
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scale = StandardScaler()


X = df.drop('price',axis=1)
y = df['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
X_train.shape
X_test.shape




x_train = scale.fit_transform(X_train)
x_test = scale.fit_transform(X_test)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_log_error
lr = LinearRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)
pred.shape
y_test.shape
print("The R2score By Linear Regression is",r2_score(y_test,pred))
rf = RandomForestRegressor(max_depth=11,random_state=0)
rf.fit(X_train,y_train)

rf_pred = rf.predict(X_test)
print("The R2score By Random Forest Regressor is " ,r2_score(y_test,rf_pred))
import xgboost as xgb
model=xgb.XGBRegressor(max_depth=9,num_parallel_tree=3)
model.fit(X_train,y_train)
pred=model.predict(X_test)
print(f"Accuracy: {r2_score(pred,y_test)}")
model = ExtraTreesRegressor(max_depth=25, min_samples_split=5, random_state=19).fit(X_train,y_train)
pred=model.predict(X_test)
print(f"Accuracy: {r2_score(pred,y_test)}")



model = XGBRFRegressor(max_depth=20).fit(X_train,y_train)
pred=model.predict(X_test)
print(f"Accuracy: {r2_score(pred,y_test)}")
model =ExtraTreesRegressor(max_depth=25).fit(X_train,y_train)
pred=model.predict(X_test)
print(f"Accuracy: {r2_score(pred,y_test)}")


test = pd.read_csv("/kaggle/input/house-price-prediction-challenge/test.csv")
test.head()
postedby_dummy = pd.get_dummies(test['POSTED_BY'],drop_first=True,prefix='postedby')
types = pd.get_dummies(test['BHK_OR_RK'],drop_first=True,prefix='type')
test= pd.concat([test,postedby_dummy],axis=1)

test = pd.concat([test,types],axis=1)

test.drop("POSTED_BY",axis=1,inplace=True)

test.drop('BHK_OR_RK',axis=1,inplace=True)
test.drop("ADDRESS",axis=1,inplace=True)


test.head()
test = scale.fit_transform(test)
Preds = rf.predict(test)
submission = pd.DataFrame()
submission["TARGET(PRICE_IN_LACS)"] = Preds
submission.to_csv('house_price.csv', index = False)
submission