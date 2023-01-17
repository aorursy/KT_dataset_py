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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
def name_check(x):
	if x['Name'].lower().find('mr.')	>=0:
		return 'mr'
	elif x['Name'].lower().find('mrs.')>=0:
		return 'mrs'
	elif x['Name'].lower().find('miss')>=0:
		return 'miss'
	elif x['Name'].lower().find('master')>=0:
		return 'master'
	elif (x['Sex'].lower() == 'female'):
		return 'mrs'
	return 'other'
	


train['name_c'] = train.apply(lambda x: name_check(x),axis=1)
test['name_c'] = test.apply(lambda x: name_check(x),axis=1)
train['cabin_c'] = train.apply(lambda x: 0 if pd.isna(x['Cabin']) else 1, axis=1)
test['cabin_c'] = test.apply(lambda x: 0 if pd.isna(x['Cabin']) else 1, axis=1)
def ticket_number(x,i):
	if i == 0 :
		temp = len(train.groupby(['Ticket']).get_group(x['Ticket']))
	else :
		temp = len(test.groupby(['Ticket']).get_group(x['Ticket']))
#	print(temp)
	return temp	
train['ticket_no_c'] = train.apply(lambda x : ticket_number(x,0), axis=1)
test['ticket_no_c'] = test.apply(lambda x : ticket_number(x,1), axis=1)
def fare_range(x) :
	if x['Fare'] <= 30.0 :
		return '0-30'
	elif x['Fare']>30.0 and x['Fare'] <= 50.0 :
		return '30-50'
	return '>50'

train['fare_c'] = train.apply(lambda x : fare_range(x), axis=1)
test['fare_c'] = test.apply(lambda x : fare_range(x), axis=1)
def count(x,s):
	if x[s] > 2 :
		return '>2'
	elif x[s] == 0 :
		return '0'
	elif x[s] == 1 :
		return '1'
	return '2'
#	return x[s]
	
train['parch_c'] = train.apply(lambda x : count(x,'Parch'), axis = 1)
train['sibsp_c'] = train.apply(lambda x : count(x,'SibSp'), axis = 1)

test['parch_c'] = test.apply(lambda x : count(x,'Parch'), axis = 1)
test['sibsp_c'] = test.apply(lambda x : count(x,'SibSp'), axis = 1)
train.drop(columns = ['Name','Cabin','Ticket','Fare','Parch','SibSp'], inplace = True)
test.drop(columns = ['Name','Cabin','Ticket','Fare','Parch','SibSp'], inplace = True)
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp_mf = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
cols_train = ['PassengerId','Pclass','Sex','Embarked','name_c','cabin_c','ticket_no_c','fare_c','parch_c','sibsp_c','Age','Survived']
train = train[cols_train]
cols_test = ['PassengerId','Pclass','Sex','Embarked','name_c','cabin_c','ticket_no_c','fare_c','parch_c','sibsp_c','Age']
test = test[cols_test]

imp_mf = imp_mf.fit(train.iloc[:,3:4])
train.iloc[:,3:4] = imp_mf.transform(train.iloc[:,3:4])
test.iloc[:,3:4] = imp_mf.transform(test.iloc[:,3:4])

imp_mean = imp_mean.fit(train.iloc[:,10:11])

#enc = enc.fit(train.iloc[:,1:10])

#
#from sklearn.compose import ColumnTransformer
#ct = ColumnTransformer([('one_hot_encoding',enc,slice(0,9)),('impute',imp_mean,[9])],remainder = 'passthrough')

#train.iloc[:,1:11] = ct.fit_transform(train.iloc[:,1:11])
#test = ct.transform(test)

train.iloc[:,10:11] = imp_mean.transform(train.iloc[:,10:11])
test.iloc[:,10:11] = imp_mean.transform(test.iloc[:,10:11])
from sklearn.preprocessing import StandardScaler
sd = StandardScaler()
train.iloc[:,10:11] = sd.fit_transform(train.iloc[:,10:11])
test.iloc[:,10:11] = sd.transform(test.iloc[:,10:11])
def age_transform(x):
	if x['Age'] >= 1.0 :
		return '>=1'
	elif 0 <= x['Age'] < 1.0 :
		return '0-1'
	elif -1.0 <= x['Age'] < 0.0 :
		return '-1-0'
	return "<-1"
	
train['age_c'] = train.apply(lambda x : age_transform(x), axis = 1)
test['age_c'] = test.apply(lambda x : age_transform(x), axis = 1)
train.drop(columns = ['Age'], inplace = True)
test.drop(columns = ['Age'], inplace = True)

cols_train = ['PassengerId','Pclass','Sex','Embarked','name_c','cabin_c','ticket_no_c','fare_c','parch_c','sibsp_c','age_c','Survived']
train = train[cols_train]
cols_test = ['PassengerId','Pclass','Sex','Embarked','name_c','cabin_c','ticket_no_c','fare_c','parch_c','sibsp_c','age_c']
test = test[cols_test]
X = train.iloc[:,1:11]
y = train.iloc[:,11:12]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 0)


#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder(handle_unknown = 'ignore')
#
##enc.fit(X_train)
#X_train = enc.fit_transform(X_train)
#X_test = enc.transform(X_test)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X_train = enc.fit_transform(X_train)
X_test = enc.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42)
X_train,y_train = sm.fit_resample(X_train,y_train)
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
pdtc = [{'criterion' : ['gini','entropy'], 'max_depth' : [3,200],'max_features' : ['auto','sqrt','log2']}]
clf1 = GridSearchCV(DecisionTreeClassifier(random_state = 0),pdtc)


from sklearn.ensemble import RandomForestClassifier
prfc = [{'n_estimators' : [3,300],'criterion' : ['gini','entropy'], 'max_depth' : [3,200],'max_features' : ['auto','sqrt','log2']}]
clf2 = GridSearchCV(RandomForestClassifier(random_state = 0),prfc)

from sklearn.svm import SVC
psvc = [{'C' : [0.1,5.0], 'kernel' : ['linear','poly','rbf','sigmoid']}]
clf3 = GridSearchCV(SVC(random_state = 0),psvc)

from sklearn.ensemble import GradientBoostingClassifier
pgbc = [{'loss' : ['deviance','exponential'], 'learning_rate' : [0.1,1],'n_estimators' : [10,300],'warm_start' : [True, False]}]
#for a in range()
clf4 = GridSearchCV(GradientBoostingClassifier(random_state = 0),pgbc)

#estimator = [('dtc',clf1),('rfc',clf2),('svc',clf3)]
estimator = [('dtc',clf1),('rfc',clf2),('svc',clf3),('gbc',clf4)]

from sklearn.ensemble import StackingClassifier
clf5 = StackingClassifier(estimators = estimator,final_estimator = clf4,passthrough = True)
clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)
clf4.fit(X_train,y_train)
y_pred = ((clf1.predict(X_test)+clf2.predict(X_test)+clf3.predict(X_test)+clf4.predict(X_test))>=2)*1
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
# print(cr)
test_X = enc.transform(test.iloc[:,1:])
y_temp = clf1.predict(test_X)+clf2.predict(test_X)+clf3.predict(test_X)+clf4.predict(test_X)
y_ans = (y_temp>=4)*1
test_y = pd.DataFrame({'PassengerId' : test.iloc[:,0].to_numpy(), 'Survived' : y_ans})
test_y.to_csv('/kaggle/output/v1.csv',index = False)