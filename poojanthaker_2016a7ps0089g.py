import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model
df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

df.head()

df.isnull().sum()
df.fillna(value=df.mean(),inplace=True)

df.isnull().sum()
# df['type']

temp_code = {'old':0,'new':1}

df['type']=df['type'].map(temp_code)

test['type']=test['type'].map(temp_code)
test.fillna(value=test.mean(),inplace=True)
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split



y = df['rating']

X = df.drop(['rating'],axis=1)

ids = test['id']





 

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing

clf = GradientBoostingClassifier(n_estimators=571,max_depth=9)

dlf = RandomForestRegressor(n_estimators=283,max_depth=6,random_state=0)



mm_scaler = preprocessing.StandardScaler()

X = mm_scaler.fit_transform(X)

test=mm_scaler.transform(test)



dlf.fit(X,y)

u1 = dlf.predict(test)

clf.fit(X,y)

u=clf.predict(test)
vv = ids.values

ans = []

ans2= []

for i in range(len(vv)):

  ans.append([vv[i],round(u[i])])

  ans2.append([vv[i],round(u1[i])])            

finans = pd.DataFrame(data=ans,columns=['id','rating'])

finans2 = pd.DataFrame(data=ans2,columns=['id','rating'])

# print(finans)

finans2.to_csv('submission2.csv',index=False)

finans.to_csv('submission.csv',index=False)


def rms(a,b):

  sum=0;

  for i in range(len(a)):

    sum+=(ff(a[i])-b[i])**2

  return sum/len(a)
import math

def ff(a):

  t = math.floor(a)

  b=a-t

  if(b>0.6):

    return t+1

  else:

    return t

  
# from sklearn import preprocessing

 

# X['feature7']=(X['feature7']-X['feature7'].min())/(X['feature7'].max()-X['feature7'].min())

 

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42) 



# mm_scaler = preprocessing.RobustScaler()

# X_train_minmax = mm_scaler.fit_transform(X_train.values)

# X_test=mm_scaler.transform(X_test)

# # # print(X_train)

# # X_train=mm_scaler.transform(X_train)

# # # # print(X_train)

# # import sklearn

# # from sklearn.ensemble import GradientBoostingClassifier

# # # from sklearn.model_selection import RandomizedSearchCV

# # from sklearn.model_selection import cross_val_score

# # from sklearn import metrics

# # # clf = clf.fit(X_train,y_train.values)

# # # u=clf.predict(X_test)



# # from sklearn.ensemble import GradientBoostingRegressor

# # # clf= RandomForestRegressor(max_depth=100, random_state=0,n_estimators=1000)



# param_grid = {'n_estimators':np.arange(1,1000),'max_depth':np.arange(1,10)}



# clf = RandomForestClassifier()



# clf_cv = GridSearchCV(clf,param_grid,cv=5)



# clf_cv.fit(X,y)



# # scores = cross_val_score(clf,X,y,cv=5,scoring='mean_squared_error')

# # sorted(sklearn.metrics.SCORERS.keys())

# # print(scores.sum())



# print(clf_cv.best_params_)





# clf = clf.fit(X_train,y_train.values)

# u=clf.predict(X_test)



# # reg2 = linear_model.LinearRegression()

# # reg.fit(X_train,y_train.values)



# # u = reg.predict(X_test)

# # print(type(u))

# # print(type/(X_test))

# # print(u)

# # print(y_test.v/alues)

# print(rms(u,y_test.values))

# X.describe()