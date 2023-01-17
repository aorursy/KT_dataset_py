# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/asteroid-dataset/dataset.csv',low_memory=False)
df.head()
df.isnull().sum()
#dropping features with too many missing value and id, spkid and full_name
df.drop(['name','prefix','diameter','albedo','diameter_sigma','id','spkid','full_name'],axis=1,inplace=True)
#basic statistical description of features
df.describe()
#correlations between features
corr=df.corr()
corr.style.background_gradient(cmap='PuBu')
#dropping features that are highly correlated
corr = df.corr().abs()
upper=corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.88)]
df.drop(to_drop, axis=1, inplace=True)
corr=df.corr()
corr.style.background_gradient(cmap='PuBu')
df.info()
df[['pdes','neo','pha','orbit_id','equinox','class']]
#we can drop this column
df['equinox'].value_counts()
df['class'].value_counts()
#highly left skewed 
sns.distplot(df['epoch'],kde=False)
#dropping equinox, epoch, orbit_id and pdes
df.drop(['pdes','orbit_id','equinox','epoch'],axis=1,inplace=True)
df.head()
sns.distplot(df['H'])
df['e']=df['e'].apply(np.sqrt)
sns.distplot(df['e'])
sns.distplot(df['a'])
sns.distplot(df[df['q']<10]['q'])
df.isnull().sum()
#dropping all the null values as we have lots of data
df.dropna(inplace=True)
df.isnull().sum()
#we have too less asteroid that are potential hazard that mean we have a skewed class
df['pha'].value_counts()
df.info()
df['neo'].value_counts()
#dealing with categorical variable 
classes = pd.get_dummies(df['class'], drop_first = True)
classes.head()
df['neo']=df['neo'].apply(lambda x: 1 if x=='Y' else 0)
df['pha']=df['pha'].apply(lambda x: 1 if x=='Y' else 0)
df = pd.concat([df.drop('class', axis = 1), classes], axis = 1)
df.info()
df.isnull().sum()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#feature scaling
scaler.fit(df[['H','e','a','q','i','om','w','ma','n','tp','sigma_e','sigma_a','sigma_q','sigma_i',
                'sigma_om','sigma_w','sigma_n','rms']])
scaled_df = scaler.transform(df[['H','e','a','q','i','om','w','ma','n','tp','sigma_e','sigma_a','sigma_q','sigma_i',
                             'sigma_om','sigma_w','sigma_n','rms']])
scaled_df
new_df = pd.DataFrame(scaled_df, columns = ['H','e','a','q','i','om','w','ma','n','tp','sigma_e','sigma_a','sigma_q','sigma_i',
                                            'sigma_om','sigma_w','sigma_n','rms'])
final_df = pd.concat([new_df , df[['neo' ,'pha', 'APO', 'AST', 'ATE', 'CEN', 'IEO', 'IMB', 'MBA', 'MCA', 'OMB', 'TJN', 'TNO']]], axis = 1)
final_df.info()
final_df.isnull().sum()
final_df.dropna(inplace=True)
final_df['pha'].value_counts()
# dividing data into independent feature and target variable
X=final_df.drop('pha',axis=1)
y=final_df['pha']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_test.value_counts()
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# lg = LogisticRegression()
# grid={'C':10.0**np.arange(-3,3),'penalty':['l1','l2'],'solver':['linlinear','lbfgs']}
# cv=KFold(n_splits=5,shuffle=False)
# clf=GridSearchCV(lg,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
# clf.fit(X_train,y_train)
# y_pred=clf.predict(X_test)
# #Evaluation of Logistic Regression model
# print('Logistic Regression')
# print('\n')
# c=confusion_matrix(y_test, y_pred)
# print(c)
# print('TN:',c[0][0])
# print('TP:',c[1][1])
# print('FN:',c[1][0])
# print('FP:',c[0][1])
# print('\n')
# print(classification_report(y_test,y_pred ))
# print('\n')
# #training and testing both are very low but recall is very low.
# #we desire, of all the asteroid that are actually potential hazard what fraction did we actually detect correctly as potential hazard.
# print('training error:',1-accuracy_score(y_train,clf.predict(X_train)))
# print('testing error:',1-accuracy_score(y_test,y_pred)) 
# #Recall is very low either need to improve model or try different model
# #Not a very good model
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
#Evaluation of Regression model
print('RandomForest Classifier')
print('\n')
c=confusion_matrix(y_test, y_pred)
print(c)
print('TN:',c[0][0])
print('TP:',c[1][1])
print('FN:',c[1][0])
print('FP:',c[0][1])
print('\n')
print(classification_report(y_test,y_pred ))
print('\n')
#training and testing both are very low but recall is very low.
#we desire, of all the asteroid that are actually potential hazard what fraction did we actually detect correctly as potential hazard.
print('training error:',1-accuracy_score(y_train,rfc.predict(X_train)))
print('testing error:',1-accuracy_score(y_test,y_pred)) 
#Recall and Precision has improved as compared to LR model
#still not a very good model
# y_train.value_counts()
class_weight=dict({0:1,1:400})
rfc=RandomForestClassifier(class_weight=class_weight)
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
#Evaluation of Logistic Regression model
print('RandomForest Classifier')
print('\n')
c=confusion_matrix(y_test, y_pred)
print(c)
print('TN:',c[0][0])
print('TP:',c[1][1])
print('FN:',c[1][0])
print('FP:',c[0][1])
print('\n')
print(classification_report(y_test,y_pred ))
print('\n')
#training and testing both are very low but recall is very low.
#we desire, of all the asteroid that are actually potential hazard what fraction did we actually detect correctly as potential hazard.
print('training error:',1-accuracy_score(y_train,rfc.predict(X_train)))
print('testing error:',1-accuracy_score(y_test,y_pred)) 
#Recall is very low either need to improve model or try different model
#Not a very good model
cw=dict({0:1,1:400})
rfc = RandomForestClassifier(class_weight=cw)
grid={'max_features':['auto','sqrt'],'max_depth':[10,20],
      'min_samples_split':[10,20],'min_samples_leaf':[20,50,80]}
cv=KFold(n_splits=5,shuffle=False)
clf=GridSearchCV(rfc,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
clf.fit(X_train,y_train)
clf.best_params_
#Best hyperParameters
# {'max_depth': 20,
#  'max_features': 'sqrt',
#  'min_samples_leaf': 20,
#  'min_samples_split': 10,
#  'class_weight':{0:1,1:400}}
y_pred=clf.predict(X_test)
#Evaluation of Random Forest Classifier model
print('RandomForest Classifier')
print('\n')
c=confusion_matrix(y_test, y_pred)
print(c)
print('TN:',c[0][0])
print('TP:',c[1][1])
print('FN:',c[1][0])
print('FP:',c[0][1])
print('\n')
print(classification_report(y_test,y_pred ))
print('\n')
#training and testing error are low so that's a good thing
#we desire, of all the asteroid that are actually potential hazard what fraction did we actually detect correctly as potential hazard.
print('training error:',1-accuracy_score(y_train,clf.predict(X_train)))
print('testing error:',1-accuracy_score(y_test,y_pred)) 
#This RF classifier has a accuracy of 0.988 and has a high recall that is desirable.
rdfc=RandomForestClassifier(max_depth=20,max_features='sqrt',min_samples_leaf=20,min_samples_split=10,
                            class_weight={0:1,1:400})
rdfc.fit(X_train,y_train)
y_pred=rdfc.predict(X_test)
#Evaluation of Random Forest Classifier model
print('RandomForest Classifier')
print('\n')
c=confusion_matrix(y_test, y_pred)
print(c)
print('TN:',c[0][0])
print('TP:',c[1][1])
print('FN:',c[1][0])
print('FP:',c[0][1])
print('\n')
print(classification_report(y_test,y_pred ))
print('\n')
#training and testing error are low so that's a good thing
#we desire, of all the asteroid that are actually potential hazard what fraction did we actually detect correctly as potential hazard.
print('training error:',1-accuracy_score(y_train,rdfc.predict(X_train)))
print('testing error:',1-accuracy_score(y_test,y_pred)) 
#This RF classifier has a accuracy of 0.988 and has a high recall that is desirable.
