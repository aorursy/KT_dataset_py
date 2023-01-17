import warnings

warnings.filterwarnings("ignore")

import shutil

import os

import pandas as pd

import matplotlib

matplotlib.use(u'nbAgg')

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pickle

from sklearn.manifold import TSNE

from sklearn import preprocessing

import pandas as pd

from multiprocessing import Process# this is used for multithreading

import multiprocessing

import codecs# this is used for file operations 

import random as r

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from collections import Counter



from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

%matplotlib inline

from sklearn.metrics import r2_score

from sklearn import metrics
data = pd.read_csv("/kaggle/input/insurance/insurance.csv")

data.head()
data.shape
data.info()

data.describe()
data.head()
print("Find most important features relative to target")

corr = data.corr()

corr.sort_values(["charges"], ascending = False, inplace = True)

corr

print(corr['charges'])
sns.distplot(data['charges'])
data['charges'].skew()
sns.countplot(data['sex'])
g=sns.FacetGrid(data,col='sex')

g=g.map(sns.distplot,"charges")

sns.catplot(y='charges',x='sex',data=data,kind="swarm")
sns.catplot(y='charges',x='sex',col='smoker',data=data,kind="swarm")
sns.catplot(y='charges',x='sex',col='smoker',data=data,kind="box")
sns.catplot(y='charges',x='sex',col='smoker',data=data,kind="violin")
data[(data['smoker']=='yes')&(data['sex']=='male')]['charges'].mean()
data[(data['smoker']=='yes')&(data['sex']=='female')]['charges'].mean()
sns.distplot(data['age'])
sns.scatterplot(x='age', y='charges',data=data)
data['age'].describe()
data['age_cat']=np.NAN

lst = [data]



for col in lst:

    col.loc[(col['age'] >= 18) & (col['age'] <= 26), 'age_cat'] = 'Young'

    col.loc[(col['age'] > 26) & (col['age'] <= 40), 'age_cat'] = 'Adult'

    col.loc[(col['age'] > 40) & (col['age'] <= 55), 'age_cat'] = 'Senior Adult'

    col.loc[col['age'] > 55, 'age_cat'] = 'Elder'
data['age_cat'].value_counts()
sns.catplot(x='age_cat', y='charges',data=data,kind='bar')
sns.catplot(x='age_cat', y='charges',data=data,kind='box')
sns.catplot(x='age_cat', y='charges',data=data,kind='violin')
sns.catplot(x='age_cat', y='charges',data=data,kind='swarm')
sns.catplot(x='age_cat', y='charges',data=data,kind='box',col='smoker')
sns.catplot(x='age_cat', y='charges',data=data,kind='box',col='smoker',hue='sex')
sns.catplot(x='age_cat', y='charges',data=data,kind='violin',col='smoker',hue='sex')
data[(data['age_cat']=='Adult')&(data['smoker']=='yes')&(data['sex']=='male')]['charges'].describe()
data[(data['age_cat']=='Adult')&(data['smoker']=='yes')&(data['sex']=='female')]['charges'].describe()
data.head()
sns.scatterplot(x='bmi',y='charges',data=data)
sns.distplot(data['bmi'])
data['bmi'].describe()
data.loc[(data['bmi']>= 15)&(data['bmi']<19), 'bmi_cat'] = 'underweight'

data.loc[(data['bmi']>= 19)&(data['bmi']<25), 'bmi_cat'] = 'healthy'

data.loc[(data['bmi']>= 25)&(data['bmi']<30), 'bmi_cat'] = 'overweight'

data.loc[(data['bmi']>= 30)&(data['bmi']<40), 'bmi_cat'] = 'obese'

data.loc[(data['bmi']>= 40), 'bmi_cat'] = 'ext_obese'
data['bmi_cat'].value_counts()
data['bmi_cat'].unique()
sns.catplot(x='bmi_cat',y='charges',kind='bar',data=data)
sns.catplot(x='bmi_cat',y='charges',kind='box',data=data)
sns.catplot(x='bmi_cat',y='charges',kind='swarm',data=data)
sns.catplot(x='bmi_cat',y='charges',col='smoker',kind='box',data=data)
sns.catplot(x='bmi_cat',y='charges',col='age_cat',hue='smoker',kind='box',data=data)
sns.countplot(data['children'])
sns.stripplot(x="children", y="charges", data=data, size = 5, jitter = True)
data['children'].value_counts()
data['children']=data['children'].map({0:'0',1:'1',2:'2',3:'3+',4:'3+',5:'3+'})
data['children'].value_counts()
sns.catplot(x='children',y='charges',data=data,kind='bar')
sns.countplot(x='children',hue='smoker',data=data)
data['smoker'].value_counts()
data.head()
sns.catplot(x='children',y='bmi',data=data,kind='swarm')
sns.catplot(x='children',y='bmi',hue='smoker',data=data,kind='box')
sns.catplot(x='children',y='charges',data=data,kind='bar')
data['child_cat']=np.NAN

data.loc[(data['children']=='0')|(data['children']=='1'), 'child_cat'] = 'less'

data.loc[(data['children']=='2')|(data['children']=='3+'), 'child_cat'] = 'more'
data.head()
data['region'].value_counts()
sns.countplot(x='region',hue='smoker',data=data)
sns.catplot(x='region',y='charges',hue='smoker',data=data,kind='box')
sns.catplot(x='region',y='charges',hue='smoker',data=data,kind='swarm')
sns.catplot(x='region',y='charges',col='smoker',data=data,kind='bar')
sns.catplot(x='region',y='charges',col='bmi_cat',hue='smoker',data=data,kind='bar')
sns.catplot(x='region',y='charges',data=data,kind='bar')
data1=data.copy()
data1.head()
data1=data1.drop(['age','bmi','children'],axis=1)

data1['age_cat'].unique()
data1['bmi_cat'].unique()
data1.info()
data1['age_cat']=data1['age_cat'].map({'Young':0, 'Adult':1, 'Senior Adult':2, 'Elder':3})

data1['bmi_cat']=data1['bmi_cat'].map({'underweight':0, 'healthy':1, 'overweight':2, 'obese':3,'ext_obese':4})

data1['child_cat']=data1['child_cat'].map({'less':0, 'more':1})

#data_hot=data1[['sex','smoker','region','age_cat','bmi_cat','child_cat']]
#data_hot1 = pd.get_dummies(data_hot)
#data_final=pd.concat([data_hot1,data1['charges']],axis=1)
data1.head()
data1['smoker']=data1['smoker'].map({'no':0, 'yes':1})
data1.head()
data_hot = pd.get_dummies(data1['region'])

data_hot1 = pd.get_dummies(data1['sex'])
data1
data1=data1.drop(['region','sex'],axis=1)
data_final=pd.concat([data1,data_hot,data_hot1],axis=1)
data_final.head()
from scipy import stats

from scipy.stats import norm, skew #for some statistics



sns.distplot(data['charges'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(data_final['charges'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('charge distribution')



fig = plt.figure()

res = stats.probplot(data['charges'], plot=plt)

plt.show()
data_final['charges'] = np.log1p(data_final['charges'] )

y=data_final['charges']



sns.distplot(data_final['charges'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(data_final['charges'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')



fig = plt.figure()

res = stats.probplot(data['charges'], plot=plt)

plt.show()
data_final.head()
data_final.shape
#data_final=data_final.drop(['sex_male','smoker_no','region_southwest','age_cat_Young','bmi_cat_healthy','child_cat_more'],axis=1)
Data_out=data_final['charges']

input_data=data_final.drop(['charges'],axis=1)
from sklearn.preprocessing import PolynomialFeatures

quad = PolynomialFeatures (degree = 2)

x_quad = quad.fit_transform(input_data)

R2_Scores = []

models1 = ['Linear Regression' , 'GradientBoosting Regression' ,'DecisionTreeRegressor','SVR','RandomForestRegressor','KNeighbours Regression']
#data_y = result['Class']

# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]

X_train, X_test, y_train, y_test = train_test_split(input_data,Data_out,test_size=0.10)

# split the train data into train and cross validation by maintaining same distribution of output varaible 'y_train' [stratify=y_train]

#X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train,test_size=0.10)
print('Number of data points in train data:', X_train.shape[0])

print('Number of data points in test data:', X_test.shape[0])

#print('Number of data points in cross validation data:', X_cv.shape[0])
data_final.shape
# X_train.describe()





y_train= y_train.values.reshape(-1,1)

y_test= y_test.values.reshape(-1,1)

#y_cv= y_cv.values.reshape(-1,1)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

y_train = sc_X.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)

#X_cv = sc_X.fit_transform(X_cv)

#y_cv = sc_y.fit_transform(y_cv)
from sklearn import ensemble

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error, r2_score
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)
clf_pred=clf.predict(X_test)

clf_pred= clf_pred.reshape(-1,1)
r2=r2_score(y_test, clf_pred)

print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))

print('MSE:', metrics.mean_squared_error(y_test, clf_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))

print('r2 score:',r2)

R2_Scores.append(r2)
plt.figure(figsize=(15,8))

plt.scatter(y_test,clf_pred, c= 'brown')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
from sklearn.tree import DecisionTreeRegressor

dtreg = DecisionTreeRegressor(random_state = 100)

dtreg.fit(X_train, y_train)
dtr_pred = dtreg.predict(X_test)

dtr_pred= dtr_pred.reshape(-1,1)
r2=r2_score(y_test, dtr_pred)

print('MAE:', metrics.mean_absolute_error(y_test, dtr_pred))

print('MSE:', metrics.mean_squared_error(y_test, dtr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))

print('r2 score:',r2)



R2_Scores.append(r2)
plt.figure(figsize=(15,8))

plt.scatter(y_test,dtr_pred,c='green')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
from sklearn.svm import SVR

svr = SVR(kernel = 'rbf')

svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)

svr_pred= svr_pred.reshape(-1,1)
r2=r2_score(y_test, svr_pred)

print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))

print('MSE:', metrics.mean_squared_error(y_test, svr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))

print('r2 score:',r2)



R2_Scores.append(r2)
plt.figure(figsize=(15,8))

plt.scatter(y_test,svr_pred, c='red')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 1500, random_state = 0)

rfr.fit(X_train, y_train)
rfr_pred= rfr.predict(X_test)

rfr_pred = rfr_pred.reshape(-1,1)
r2=r2_score(y_test, rfr_pred)

print('MAE:', metrics.mean_absolute_error(y_test, rfr_pred))

print('MSE:', metrics.mean_squared_error(y_test, rfr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))

print('r2 score:',r2)



R2_Scores.append(r2)
plt.figure(figsize=(15,8))

plt.scatter(y_test,rfr_pred, c='orange')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
from sklearn.neighbors import KNeighborsRegressor

clf_knn = KNeighborsRegressor()

clf_knn.fit(X_train , y_train)

accuracies = cross_val_score(estimator = clf_knn, X = X_train, y = y_train, cv = 5,verbose = 1)

y_pred = clf_knn.predict(X_test)

print('')

print('###### KNeighbours Regression ######')

print('Score : %.4f' % clf_knn.score(X_test, y_test))

print(accuracies)



mse = metrics.mean_squared_error(y_test, y_pred)

mae = metrics.mean_absolute_error(y_test, y_pred)

rmse = metrics.mean_squared_error(y_test, y_pred)**0.5

r2 = r2_score(y_test, y_pred)



print('')

print('MSE    : %0.2f ' % mse)

print('MAE    : %0.2f ' % mae)

print('RMSE   : %0.2f ' % rmse)

print('R2     : %0.2f ' % r2)
from sklearn.model_selection import GridSearchCV

n_neighbors=[]

for i in range (0,50,5):

    if(i!=0):

        n_neighbors.append(i)

params_dict={'n_neighbors':n_neighbors,'n_jobs':[-1]}

clf_knn=GridSearchCV(estimator=KNeighborsRegressor(),param_grid=params_dict,scoring='r2')

clf_knn.fit(X_train,y_train)

print('Score : %.4f' % clf_knn.score(X_test, y_test))

pred=clf_knn.predict(X_test)

r2 = r2_score(y_test, pred)

print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)
plt.figure(figsize=(15,8))

plt.scatter(y_test,pred, c='orange')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()