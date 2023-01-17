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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB,MultinomialNB
from time import time
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
from sklearn.tree import plot_tree, export_text
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic.head()
titanic1 = pd.read_csv('/kaggle/input/titanic/test.csv')
titanic1.head()
titanic.shape
titanic1.shape
titanic.columns
titanic1.columns
titanic.describe()
titanic1.describe()
titanic.info()
titanic.isnull().sum()
titanic1.isnull().sum()
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())
titanic1['Age'] = titanic1['Age'].fillna(titanic1['Age'].mean())
titanic.isnull().sum()
titanic1.isnull().sum()
titanic1['Fare'] = titanic1['Fare'].fillna(titanic1['Fare'].median())
titanic1.isnull().sum()
titanic.isnull().sum()
titanic['Cabin'] = titanic['Cabin'].fillna(titanic['Cabin'].mode()[0])
titanic1['Cabin'] = titanic1['Cabin'].fillna(titanic1['Cabin'].mode()[0])
titanic.isnull().sum()
titanic1.isnull().sum()
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])
titanic.isnull().sum()
titanic['Survived'].value_counts()
titanic['Survived'].value_counts(normalize = True)
women = titanic.loc[titanic.Sex == 'female']["Survived"]
survival_rate_women = sum(women)/len(women)
print("% of women who survived:", survival_rate_women)
men = titanic.loc[titanic.Sex == 'male']["Survived"]
survival_rate_men = sum(men)/len(men)
print("% of men who survived:", survival_rate_men)
sns.countplot(titanic['Survived'])
plt.figure(figsize=(15,8))
titanic.boxplot()
q = int(titanic['Age'].quantile(0.25))
q = int(titanic['Age'].quantile(0.9))
titanic[titanic['Age']>q]['Age'].count()
titanic[titanic['Age']<q]['Age'].count()
titanic['Age'] = np.where(titanic['Age']>q,q,titanic['Age'])
titanic['Age'] = np.where(titanic['Age']<q,q,titanic['Age'])
q = int(titanic['SibSp'].quantile(0.25))
q = int(titanic['SibSp'].quantile(0.9))
titanic[titanic['SibSp']>q]['SibSp'].count()
titanic[titanic['SibSp']<q]['SibSp'].count()
titanic['SibSp'] = np.where(titanic['SibSp']>q,q,titanic['SibSp'])
titanic['SibSp'] = np.where(titanic['SibSp']<q,q,titanic['SibSp'])
q = int(titanic['Parch'].quantile(0.25))
q = int(titanic['Parch'].quantile(0.9))
titanic[titanic['Parch']>q]['Parch'].count()
titanic[titanic['Parch']<q]['Parch'].count()
titanic['Parch'] = np.where(titanic['Parch']>q,q,titanic['Parch'])
titanic['Parch'] = np.where(titanic['Parch']<q,q,titanic['Parch'])
q = int(titanic['Fare'].quantile(0.25))
q = int(titanic['Fare'].quantile(0.99))
titanic[titanic['Fare']>q]['Fare'].count()
titanic[titanic['Fare']<q]['Fare'].count()
titanic['Fare'] = np.where(titanic['Fare']>q,q,titanic['Fare'])
titanic['Fare'] = np.where(titanic['Fare']<q,q,titanic['Fare'])
plt.figure(figsize=(15,8))
titanic.boxplot()
## checking how many people survived based on Age
sns.countplot(y=titanic['Age'],hue=titanic['Survived'])
titanic.groupby('Survived')['Age'].value_counts()
## checking how many people survived based on Pclass
sns.countplot(y=titanic['Pclass'],hue=titanic['Survived'])
titanic.groupby('Pclass')['Survived'].value_counts()
## checking how many people survived based on SibSp
sns.countplot(y=titanic['SibSp'],hue=titanic['Survived'])
titanic.groupby('SibSp')['Survived'].value_counts()
## checking how many people survived based on Parch
sns.countplot(y=titanic['Parch'],hue=titanic['Survived'])
titanic.groupby('Parch')['Survived'].value_counts()
## checking how many people survived based on Parch
sns.countplot(y=titanic['Fare'],hue=titanic['Survived'])
titanic.groupby('Fare')['Survived'].value_counts()
sns.countplot(y=titanic['Sex'],hue=titanic['Survived'])
titanic.groupby('Sex')['Survived'].value_counts()

le = LabelEncoder()
for column in titanic.columns:
    if titanic[column].dtype == type(object):
        titanic[column] = le.fit_transform(titanic[column])

titanic.head(10)
X = titanic.drop('Survived',axis=1)
y = titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=0) 
X_train.head()
X_test.head()
y_train.head()
y_test.head()
classifier_log = LogisticRegression()
model = classifier_log.fit(X_train,y_train)
#predicting on the test data
y_pred_log = classifier_log.predict(X_test)
#checking the accuracy of the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred_log,y_test)*100)
probs_log = classifier_log.predict_proba(X_test)
probs_log
classifier_tree = DecisionTreeClassifier()
model = classifier_tree.fit(X_train,y_train)
# predicting the test data
y_pred_tree = classifier_tree.predict(X_test)
print(accuracy_score(y_pred_tree,y_test)*100)
features  = X_test.columns
plt.figure(figsize=(10,10))
plot_tree(classifier_tree, feature_names=features,filled = True)
bnb = BernoulliNB() 
print("Start training...")
tStart = time()
bnb.fit(X_train, y_train)
tEnd = time()
print("Training time: ", round(tEnd-tStart, 3), "s")

# making predictions on the testing set 
y_pred = bnb.predict(X_test) 
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred)*100)

CM=confusion_matrix(y_test,y_pred)
print("Confusion Matrix is:", CM, sep='\n') 
CM=confusion_matrix(y_test,y_pred)
print("Confusion Matrix is:", CM, sep='\n') 

print("----------")
print("   TN FP")
print("   FN TP")



print("----------")

TN = 92
TP = 49
FN = 20
FP = 18

sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)

print("The probability of predicting people surviving correctly is ",specificity)
print("The probability of predicting people not surviving correctly is ",sensitivity)

y_pred_NB = bnb.predict(X_test)
y_pred_NB.shape
clf=BaggingClassifier(oob_score=True,n_jobs=-1,n_estimators=20,random_state=400,
                      base_estimator=DecisionTreeClassifier())
clf.fit(X_train,y_train)
#Average number of correct predictions
clf.score(X_test,y_test)
clf.oob_score_
set(range(100,200,20))
#Parameter tuning
for w in range(100,200,20):
    clf=BaggingClassifier(oob_score=True,n_jobs=-1,n_estimators=w,random_state=400,
                          base_estimator=DecisionTreeClassifier())
    clf.fit(X_train,y_train)
    oob=clf.oob_score_
    print('For n_estimators = '+str(w))
    print('OOB score is '+str(oob))
    print('************************')
#Finalizing on a tree model with 180 trees
clf=BaggingClassifier(oob_score=True,n_jobs=-1,n_estimators=180,random_state=400,
                      base_estimator=DecisionTreeClassifier())
clf.fit(X_train,y_train)
clf_bagging  = clf.predict(X_test)
clf_bagging.shape
#Average Number of correct predictions
clf.score(X_test,y_test) 
# Feature Importance
len(clf.estimators_)
# We can extract feature importance from each tree then take a mean for all trees
import numpy as np
imp=[]
for i in clf.estimators_:
    imp.append(i.feature_importances_) #feature importance at each tree
imp
imp=np.mean(imp,axis=0)
imp
feature_importance=pd.Series(imp,index=X.columns.tolist()).sort_values(ascending=False)
feature_importance
clf=RandomForestClassifier(n_estimators=80,oob_score=True,n_jobs=-1,random_state=400)
clf.fit(X_train,y_train)
clf.oob_score_
for w in range(50,300,10):
    clf=RandomForestClassifier(n_estimators=w,oob_score=True,n_jobs=-1,random_state=400)
    clf.fit(X_train,y_train)
    oob=clf.oob_score_
    print('For n_estimators = '+str(w))
    print('OOB score is '+str(oob))
    print('************************')
#Finalize 80 trees
clf=RandomForestClassifier(n_estimators=80,oob_score=True,n_jobs=-1,random_state=400,max_depth=3,max_features=4)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
#Area Under the curve
from sklearn import metrics
metrics.roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
clf.feature_importances_
clf_rf = clf.predict(X_test)
clf_rf.shape
pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
clf=AdaBoostClassifier(n_estimators=50,random_state=400)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
clf=GradientBoostingClassifier(n_estimators=80,random_state=400)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
from sklearn.model_selection import GridSearchCV
mod=GridSearchCV(clf,param_grid={'n_estimators':[60,80,100,120,140,160]},cv=3) 
mod.fit(X_train,y_train)
mod.best_estimator_
clf=GradientBoostingClassifier(n_estimators=80,random_state=400)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
import xgboost as xg
xgb=xg.XGBClassifier(objective='binary:logistic',reg_lambda=0.2,max_depth=4,random_state=200) #L2 regularization
#Grid Search CV: Parameter tuning
from sklearn import model_selection
xgb=model_selection.GridSearchCV(xgb, param_grid={'max_depth':[2,3,5,9],'n_estimators':[50,100,150,500],'reg_lambda':[0.1,0.2]})
xgb.fit(X_train,y_train)
xgb.best_params_
xgb.score(X_test,y_test)
from sklearn import metrics
metrics.roc_auc_score(y_test,xgb.predict_proba(X_test)[:,1])
from sklearn.ensemble import RandomForestClassifier

y = titanic["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
X = pd.get_dummies(titanic[features])
X_test = pd.get_dummies(titanic[features])

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X = my_imputer.fit_transform(X)
imputed_X_test = my_imputer.transform(X_test)

model = RandomForestClassifier(n_estimators=6, max_depth=20, random_state=1)
model.fit(imputed_X, y)
predictions = model.predict(imputed_X_test)

output = pd.DataFrame({'PassengerId': titanic.PassengerId, 'Survived': predictions})
output.to_csv('submission1.csv', index=False)
print("Your submission was successfully saved!")
