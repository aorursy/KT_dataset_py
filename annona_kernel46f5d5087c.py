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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('/kaggle/input/diamonds/diamonds.csv')
df.head()
df.columns
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.shape
df.describe()
df.info()
#the cut,color,clarity are the categorical features and rest are the numerical features and the price is the target variable
df.isna().sum()#there are no null values.
import missingno as msno
msno.matrix(df)
df.describe()
df.loc[(df['x']==0)|(df['y']==0)|(df['z']==0)]
len(df[(df['x']==0)|(df['y']==0)|(df['z']==0)])
#there are 20 rows where the values are zero for features x,y,z.
#dropping these values from the dataset.
df=df[(df[['x','y','z']]!=0).all(axis=1)]
df.shape
sns.factorplot(data=df,kind='box',size=7,aspect=2.5)
corr=df.corr()
sns.heatmap(data=corr,square=True,annot=True,cbar=True)
#the price is highly correlated to carat and the dimensions x,y,z and among it carat has the most significance.
#price is inversly realated to depth.
sns.kdeplot(df['carat'])
sns.kdeplot(df['carat'])
sns.jointplot(x='carat',y='price',data=df)
sns.factorplot(x='cut',data=df,kind='count')
sns.factorplot(x='cut',y='price',data=df,kind='box')
#premium cut diamonds are the most expensive.
sns.factorplot(x='color',data=df,kind='count')
sns.factorplot(x='color',y='price',data=df,kind='box')
sns.factorplot(x='clarity',y='price',data=df,kind='box')
#vs1 and vs2 affect the diamonds price equally and has the highest effect on price.
sns.distplot(df['depth'],kde=False)
sns.regplot(x='depth',y='price',data=df)
sns.jointplot(x='depth',y='price',data=df,kind='regplot')
#here we see that the price vary heavily with the same depth.
sns.kdeplot(df['table'],shade=True,color='red')
sns.kdeplot(df['x'],shade=True,color='r')
sns.kdeplot(df['y'],shade=True,color='g')
sns.kdeplot(df['z'],shade=True,color='b')
plt.xlim(2,11)
#we are gonna create a new feature from the 3 features x,y,z as they are highly correlated.
df['volume']=df['x']*df['y']*df['z']
df.head()
df.drop(['x','y','z'],axis=1,inplace=True)
df.head()
plt.figure(figsize=(5,5))
plt.hist(x=df['volume'],color='g',bins=30)
sns.jointplot(x='volume',y='price',data=df)
#here we can see that the price increase with the increase in the volume of the diamond.
from sklearn.preprocessing import LabelEncoder
label_cut=LabelEncoder()
label_color=LabelEncoder()
label_clarity=LabelEncoder()
df['cut']=label_cut.fit_transform(df['cut'])
df['color']=label_color.fit_transform(df['color'])
df['clarity']=label_clarity.fit_transform(df['clarity'])
df.head()
#split the data set into train and test:
x=df.drop(['price'],axis=1)
y=df['price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
#model_building:
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score

r2_scores=[]
#linear regression:
lr=LinearRegression()
lr.fit(x_train,y_train)
accuracies=cross_val_score(estimator=lr,X=x_train,y=y_train,cv=5,verbose=1)
y_pred=lr.predict(x_test)
print('linear regression')
print('Score:',lr.score(x_test,y_test))
print(accuracies)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)**0.5
r2=r2_score(y_test,y_pred)
print('mse: ',mse)
print('mae: ',mae)
print('rmse: ',rmse)
print('r2: ', r2)

r2_scores.append(r2)

#lasso regression:
lar=Lasso(normalize=True)
lar.fit(x_train,y_train)
accuracies=cross_val_score(estimator=lar,X=x_train,y=y_train,cv=5,verbose=1)
y_pred=lar.predict(x_test)
print('lasso regression')
print('Score:',lar.score(x_test,y_test))
print(accuracies)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)**0.5
r2=r2_score(y_test,y_pred)
print('mse: ',mse)
print('mae: ',mae)
print('rmse: ',rmse)
print('r2: ', r2)
r2_scores.append(r2)
#gradientboost regression:
adr=AdaBoostRegressor(n_estimators=1000)
adr.fit(x_train,y_train)
accuracies=cross_val_score(estimator=adr,X=x_train,y=y_train,cv=5,verbose=1)
y_pred=adr.predict(x_test)
print('gradientboost regression')
print('Score:',adr.score(x_test,y_test))
print(accuracies)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)**0.5
r2=r2_score(y_test,y_pred)
print('mse: ',mse)
print('mae: ',mae)
print('rmse: ',rmse)
print('r2: ', r2)
r2_scores.append(r2)
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls',verbose = 1)
gbr.fit(x_train , y_train)
accuracies=cross_val_score(estimator=gbr,X=x_train,y=y_train,cv=5,verbose=1)
y_pred=gbr.predict(x_test)
print('gradientboost regression')
print('Score:',gbr.score(x_test,y_test))
print(accuracies)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)**0.5
r2=r2_score(y_test,y_pred)
print('mse: ',mse)
print('mae: ',mae)
print('rmse: ',rmse)
print('r2: ', r2)
r2_scores=[]
r2_scores.append(r2)
#randomforest regression:
rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)
accuracies=cross_val_score(estimator=rfr,X=x_train,y=y_train,cv=5,verbose=1)
y_pred=rfr.predict(x_test)
print('randomforest regression')
print('Score:',rfr.score(x_test,y_test))
print(accuracies)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)**0.5
r2=r2_score(y_test,y_pred)
print('mse: ',mse)
print('mae: ',mae)
print('rmse: ',rmse)
print('r2: ', r2)
r2_scores.append(r2)
no_of_test=[100]
params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}
rfr=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='r2')
rfr.fit(x_train,y_train)
print('Score :' , clf_rf.score(x_test, y_test))
pred=clf_rf.predict(x_test)
r2 = r2_score(y_test, pred)
print('R2 :' ,r2)
r2_scores.append(r2)

#kneighboursregressor.
knr=KNeighborsRegressor()
knr.fit(x_train , y_train)
accuracies = cross_val_score(estimator = knr, X = x_train, y = y_train, cv = 5,verbose = 1)
y_pred = knr.predict(x_test)
print('kneighbours regression')
print('Score:',rfr.score(x_test,y_test))
print(accuracies)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)**0.5
r2=r2_score(y_test,y_pred)
print('mse: ',mse)
print('mae: ',mae)
print('rmse: ',rmse)
print('r2: ', r2)
r2_scores.append(r2)

n_neighbors=[]
for i in range (0,50,5):
    if(i!=0):
        n_neighbors.append(i)
params_dict={'n_neighbors':n_neighbors,'n_jobs':[-1]}
knr=GridSearchCV(estimator=KNeighborsRegressor(),param_grid=params_dict,scoring='r2')
knr.fit(x_train,y_train)
print('Score :' , knr.score(x_test, y_test))
pred=knr.predict(x_test)
r2 = r2_score(y_test, pred)
print('R2 :' ,r2)
r2_scores.append(r2)

r2_scores
models = ['Linear Regression' , 'Lasso Regression' , 'AdaBoost Regression' , 'Ridge Regression' , 'GradientBoosting Regression',
          'RandomForest Regression' ,
         'KNeighbours Regression']
algos=pd.DataFrame({'Algorithms':models,"R2-scores":r2_scores})
algos.sort_values(by='R2-scores',ascending=False)
sns.barplot(x='R2-scores',y="Algorithms",data=algos)
sns.factorplot(x='Algorithms',y='R2-scores',data=algos,size=6,aspect=4)
#ridge regressor gave the highest score.