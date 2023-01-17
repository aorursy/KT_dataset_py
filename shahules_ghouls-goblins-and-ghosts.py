import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore',message='DeprecationWarning')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

train=pd.read_csv('../input/train.csv').copy()
test=pd.read_csv('../input/test.csv').copy()

train.describe()
train.info()
train['type'].unique()
train=train.drop(['id'],axis=1)
test_df=test.drop(['id'],axis=1)
train.head()
np.sort(train['color'].unique())==np.sort(test['color'].unique())
color_le=preprocessing.LabelEncoder()
color_le.fit(train['color'])
train['color_le']=color_le.transform(train['color'])

sns.pairplot(train.drop(['color'],axis=1),palette='muted',diag_kind='kde',hue='type')
train.drop(['color_le'],axis=1,inplace=True)
sns.heatmap(train.corr(),annot=True)

df=pd.get_dummies(train.drop(['type'],axis=1))
X_train,X_test,y_train,y_test=train_test_split(df,train['type'],random_state=0)
tr=DecisionTreeClassifier(random_state=0)
tr.fit(X_train,y_train)
y_pre=tr.predict(X_test)

print("accuracy score is ",metrics.accuracy_score(y_test,y_pre))
print('\n',metrics.classification_report(y_test,y_pre))
sns.barplot(y=X_test.columns,x=tr.feature_importances_)
params={'max_depth':np.linspace(1, 16, 16, endpoint=True),'min_samples_split':np.linspace(.1, 1,10, endpoint=True),"max_features":[1,4,6]}
accuracy=metrics.make_scorer(metrics.accuracy_score)
tr=DecisionTreeClassifier()
clf=GridSearchCV(tr,param_grid=params,scoring=accuracy,cv=5,n_jobs=-1)
clf.fit(X_train,y_train)
print('best score',clf.best_score_)
print('param',clf.best_params_)
rf=RandomForestClassifier(random_state=0)
rf.fit(X_train,y_train)
y_pre=rf.predict(X_test)

print('rf basline score',metrics.accuracy_score(y_test,y_pre))
print('\n',metrics.classification_report(y_test,y_pre))
gb=GradientBoostingClassifier()
gb.fit(X_train,y_train)
y_pre=gb.predict(X_test)

print('score',metrics.accuracy_score(y_test,y_pre))
rf.get_params()
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)
rf1=RandomForestClassifier(random_state=0)
rf1_rand=RandomizedSearchCV(rf1,param_distributions=random_grid,n_iter=100,cv=3,n_jobs=-1)
rf1_rand.fit(X_train,y_train)
print(rf1_rand.best_params_)
best_random=rf1_rand.best_estimator_
best_random.fit(X_train,y_train)
y_pre=best_random.predict(X_test)
print(metrics.accuracy_score(y_test,y_pre))
params={'max_features':['auto',],'bootstrap':[False],'max_depth':[50,60,70,56],'min_samples_leaf':[1,2],'n_estimators':[100,120,130,140],'min_samples_split':[5,10,15,20]}
rf=RandomForestClassifier()
gcv=GridSearchCV(rf,param_grid=params,cv=5,n_jobs=-1,scoring=accuracy)
gcv.fit(X_train,y_train)
print('score',gcv.best_params_)
y_pre=gcv.predict(X_test)
print('score',metrics.accuracy_score(y_test,y_pre))
gcv.param_grid
test_=pd.get_dummies(test_df)
pre=gcv.predict(test_)
df=pd.DataFrame({'id':test['id'],'type':pre},columns=['id','type'])
csv=df[['id','type']].to_csv('submission.csv',index=False)

