import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O and data manipulation 

#visulaizations 
import matplotlib.pyplot as plt   
import seaborn as sns
%matplotlib inline

#
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


import warnings
warnings.filterwarnings('ignore')
# let's define some helping functions 
def cor_map(df):
    ## objective : drawing a heating map of correlation between the numerical features and each other
    ## input : data 
    
    cor=df.corr()
    _,ax=plt.subplots(figsize=(12,10))
    cmap=sns.diverging_palette(192,6,as_cmap=True)
    _=sns.heatmap(cor,cmap=cmap,ax=ax,square=True,annot=True)
    
def print_coef(model,x):
    ## objective : printing the coefficients of each feature to determine it's importance 
    ## input: ML model , X(all data except the target variable) 
    coeff_df = pd.DataFrame(x.columns)
    coeff_df.columns = ['Variable']
    coeff_df["Coeff"] = pd.Series(model.coef_)
    coeff_df.sort_values(by='Coeff', ascending=True)
    print(coeff_df)
    
def metrics(y_true,y_pred):
    ## objective: printing the R^2, mean squared error and mean absolute error 
    ## input: the actual target variable Series and the predicted target variable Series 
    print("R2 score:",r2_score(y_true,y_pred))
    print('mean squared error',mean_squared_error(y_true,y_pred))
    print("mean absolute error",mean_absolute_error(y_true,y_pred))
# loading the data 
data=pd.read_csv('../input/diamonds.csv')
data.head()
# removing the unwanted column 
data.drop(['Unnamed: 0'],axis=1,inplace=True)
data.head()

# show the classes of every categorical feature 
for i in data.select_dtypes(include=['O']).columns:
    print(i,data[i].unique())
#check data types and  if there's no missing data
data.info()
x=data.drop('price',axis=1)
y=data['price']
data.describe()
data.describe(include=['O'])
cor_map(data)
print(data[['cut','price']].groupby('cut',as_index=False).mean().sort_values(by='price',ascending=False))
print(data[['color','price']].groupby('color',as_index=False).mean().sort_values(by='price',ascending=False))
print(data[['clarity','price']].groupby('clarity',as_index=False).mean().sort_values(by='price',ascending=False))
#how many exactly missing data we have
data.loc[(data['x']==0)|(data['y']==0)|(data['z']==0)].shape[0]
# we will exclude them from the dataset since 20 aren't important
data=data.loc[(data['x']!=0)&(data['y']!=0)&(data['z']!=0.0)]
data['p/ct']=data['price']/data['carat']
print(data[['cut','p/ct']].groupby('cut',as_index=False).mean().sort_values(by='p/ct',ascending=False))
print(data[['color','p/ct']].groupby('color',as_index=False).mean().sort_values(by='p/ct',ascending=False))
print(data[['clarity','p/ct']].groupby('clarity',as_index=False).mean().sort_values(by='p/ct',ascending=False))
data['cut']=data['cut'].map({'Ideal':1,'Good':2,'Very Good':3,'Fair':4,'Premium':5})
data['color']=data['color'].map({'E':1,'D':2,'F':3,'G':4,'H':5,'I':6,'J':7})
data['clarity']=data['clarity'].map({'VVS1':1,'IF':2,'VVS2':3,'VS1':4,'I1':5,'VS2':6,'SI1':7,'SI2':8})
data.head()
#also we can merge the thre dimensions into volume 
data['volume']=data['x']*data['y']*data['z']
data['table*y']=data['table']*data['y']
data['depth*y']=data['depth']*data['y']
data['cut/wt']=data['cut']/data['carat']
data['color/wt']=data['color']/data['carat']
data['clarity/wt']=data['clarity']/data['carat']
data.drop(['carat','cut','color','clarity','depth','table','x','y','z','p/ct'],axis=1,inplace=True)
cor_map(data)
X_train,X_test,y_train,y_test=train_test_split(data.drop(['price'],axis=1),data['price'],test_size=0.25,random_state=1)
scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)
print("The shape of the train set",X_train.shape)
print("The shape of the test set",X_test.shape)
reg_all=LinearRegression()
reg_all.fit(X_train_scaled,y_train) #fitting the model for the x and y train

pred=reg_all.predict(X_test_scaled) #predicting y(the target variable), on x test

# Rsquare=reg_all.score(X_test,y_test)
R2=r2_score(y_test,pred)
# print("Rsquare: %f" %(Rsquare))
# print("R2:",R2)
print ('=========\nTest results')
metrics(y_test,pred)
print ('=========\nTrain results')
metrics(y_train,reg_all.predict(X_train_scaled))
print_coef(reg_all,X_train)
kn_model=KNeighborsRegressor(n_neighbors=3)
kn_model.fit(X_train_scaled,y_train)
pred=kn_model.predict(X_test_scaled)
print ('=========\nTest results')
metrics(y_test,pred)
print ('=========\nTrain results')
metrics(y_train,kn_model.predict(X_train_scaled))
gbr = GradientBoostingRegressor(random_state=0)
gbr.fit(X_train_scaled,y_train)
pred=gbr.predict(X_test_scaled)
print ('=========\nTest results')
metrics(y_test,pred)
print ('=========\nTrain results')
metrics(y_train,gbr.predict(X_train_scaled))
xgb = XGBRegressor(random_state=0,n_jobs=-1)
xgb.fit(X_train_scaled,y_train)
pred=xgb.predict(X_test_scaled)
print ('=========\nTest results')
metrics(y_test,pred)
print ('=========\nTrain results')
metrics(y_train,xgb.predict(X_train_scaled))


clf = XGBRegressor(random_state=0,n_jobs=-1)
cv_sets = ShuffleSplit(X_train.shape[0], n_iter =10,test_size = 0.20, random_state = 7)
parameters = {'n_estimators':list(range(100,1000,100)),
#              'max_depth':np.linspace(1,32,32,endpoint=True,dtype=np.int),
             'learning_rate':[0.05,0.1,0.25,0.5,0.75],}
#              'reg_lambda':[1,10,15,20,25]}
scorer=make_scorer(r2_score)
grid_obj=GridSearchCV(clf, parameters, scoring=scorer,verbose=1,cv=cv_sets)
grid_obj= grid_obj.fit(X_train_scaled,y_train)
clf_best = grid_obj.best_estimator_
print(clf_best)
clf_best.fit(X_train_scaled,y_train)
print(clf_best)
print ('=========\nTrain results')
metrics(y_train,clf_best.predict(X_train_scaled))
print ('=========\nTest results')
metrics(y_test,pred)


