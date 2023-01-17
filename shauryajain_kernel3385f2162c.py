import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from time import time as time
BDS = pd.read_csv('../input/big-mart-sales-prediction/Train.csv')
BDS_test = pd.read_csv('../input/big-mart-sales-prediction/Test.csv')
BDS.head()
BDS_test.head()
BDS.shape
BDS_test.shape
BDS.info()
BDS_test.info()
BDS.isnull().sum()
BDS_test.isnull().sum()
BDS_Final = pd.concat([BDS,BDS_test],ignore_index=True)
BDS_Final.isnull().sum()
sns.distplot(BDS['Item_Outlet_Sales'])
print('Skewness',BDS['Item_Outlet_Sales'].skew())
BDS['Item_Outlet_Sales']=np.sqrt(BDS['Item_Outlet_Sales'])
sns.distplot(BDS['Item_Outlet_Sales'])
print('Skewness',BDS['Item_Outlet_Sales'].skew())
BDS_Final['Item_Outlet_Sales'] = BDS_Final['Item_Outlet_Sales'].fillna(BDS_Final['Item_Outlet_Sales'].median())
BDS_Final['Item_Outlet_Sales'] = np.sqrt(BDS_Final['Item_Outlet_Sales'])
BDS_Final.head()
BDS_Final['Item_Weight']= BDS_Final['Item_Weight'].fillna(BDS_Final['Item_Weight'].median())
sns.distplot(BDS_Final['Item_Weight'])
BDS_Final['Outlet_Size'].value_counts()
BDS_Final['Outlet_Size'].fillna(BDS_Final['Outlet_Size'].mode()[0],inplace=True)
sns.countplot(BDS_Final['Outlet_Size'])
BDS_Final = BDS_Final.drop(columns=['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'])
BDS_Final.info()
plt.figure(figsize=(7,7))
sns.countplot(BDS_Final['Item_Fat_Content'])
BDS_Final['Item_Fat_Content'] = BDS_Final['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF':'Low Fat','reg':'Regular'})
plt.figure(figsize=(7,7))
sns.countplot(BDS_Final['Item_Fat_Content'])
BDS_Final.head(10)
obj=[]
num = []
for col in BDS_Final.columns:
    if BDS_Final[col].dtypes == 'O':
        obj.append(col)
    else:
        num.append(col)
        
BDS_num = BDS_Final[num]
BDS_obj = BDS_Final[obj]
plt.figure(figsize=(10,10))
sns.heatmap(BDS_num.corr(),annot=True)
BDS_Final = pd.get_dummies(BDS_Final,columns=obj)
BDS_Final.head()
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.preprocessing import PolynomialFeatures
#ploy = PolynomialFeatures(4)
X = BDS_Final.drop('Item_Outlet_Sales',axis=1)
y = BDS_Final['Item_Outlet_Sales']
#ploy.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y)
scaler = StandardScaler()
lr = LinearRegression(normalize=True)
svr = SVR()
#knr = KNeighborsRegressor()
dt = DecisionTreeRegressor(criterion='mse',max_depth=3)
rf = RandomForestRegressor(n_estimators=10,max_depth=5)
gbr = GradientBoostingRegressor()
#pipeline_lr = make_pipeline(scaler,lr)
#pipeline_svr = make_pipeline(scaler,svr)
#pipeline_knr = make_pipeline(scaler,knr)
#pipeline_dt = make_pipeline(scaler,dt)
#pipeline_rf = make_pipeline(scaler,rf)
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error
from sklearn import metrics
def score_reg(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Root Mean Squared Log Error',np.sqrt(mean_squared_log_error( y_test, y_pred )))
lr.fit(X_train,y_train)
score_reg(lr, X_test, y_test)
#pipeline_lr.fit(X_train,y_train)
#pred = pipeline_lr.predict(X_test)
#print(pipeline_lr.score(X_train,y_train))
#print(pipeline_lr.score(X_test,y_test))
svr.fit(X_train,y_train)
score_reg(svr, X_test, y_test)
#pipeline_svr.fit(X_train,y_train)
#pred = pipeline_svr.predict(X_test)
#print(pipeline_svr.score(X_train,y_train))
#print(pipeline_svr.score(X_test,y_test))
dt.fit(X_train,y_train)
score_reg(dt, X_test, y_test)
#pipeline_dt.fit(X_train,y_train)
#pred = pipeline_dt.predict(X_test)
#print(pipeline_dt.score(X_train,y_train))
#print(pipeline_dt.score(X_test,y_test))
rf.fit(X_train,y_train)
score_reg(rf, X_test, y_test)
#pipeline_rf.fit(X_train,y_train)
#pred = pipeline_rf.predict(X_test)
#print(pipeline_rf.score(X_train,y_train))
#print(pipeline_rf.score(X_test,y_test))
gbr.fit(X_train,y_train)
score_reg(gbr,X_test, y_test)
from sklearn.model_selection import GridSearchCV
params = ({'n_estimators':[20,30,40],'criterion':['mse','mae'],'max_depth':[2,3,4,5]})
grid_search = GridSearchCV(estimator=rf,param_grid=params,n_jobs=-1)

grid_search.fit(X_train,y_train)
grid_search.predict(X_test, y_test)

print("Best parameter of the model :",grid_search.best_params_)
test = BDS_test
test.head()
test.isnull().sum()
test['Item_Weight']= test['Item_Weight'].fillna(test['Item_Weight'].median())
test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0],inplace=True)
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF':'Low Fat','reg':'Regular'})
test = test.drop(columns=['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'])
test.head()
test = pd.get_dummies(test,columns=obj)
test_pred = rf.predict(test)
pred = pd.DataFrame(test_pred)

sub = pd.read_csv('../input/big-mart-sales-prediction/Submission.csv')

sub['Item_Outlet_Sales'] = pred
sub.to_csv('big-mart-Submission_pred.csv', index=False)
