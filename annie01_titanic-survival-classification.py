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
train_data=pd.read_csv('../input/titanic/train.csv')
test_data=pd.read_csv('../input/titanic/test.csv')
train_data['train_test']=1
test_data['train_test']=0
test_data['Survived']=np.NaN
all_data=pd.concat([train_data, test_data])
all_data.columns
train_data.info()
train_data.describe()
#feature engineering on person's title 
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

#impute nulls for continuous data 
all_data.Age = all_data.Age.fillna(train_data.Age.median())
all_data.Fare = all_data.Fare.fillna(train_data.Fare.median())
#Dropping rows with missing values for 'Embarked' and 'Fair'
all_data.dropna(subset=['Embarked'],inplace = True)
all_data.dropna(subset=['Fare'],inplace = True)
#Replacing age with ordinals
all_data.loc[ all_data['Age'] <= 16, 'Age'] = 0
all_data.loc[(all_data['Age'] > 16) & (all_data['Age'] <= 32), 'Age'] = 1
all_data.loc[(all_data['Age'] > 32) & (all_data['Age'] <= 48), 'Age'] = 2
all_data.loc[(all_data['Age'] > 48) & (all_data['Age'] <= 64), 'Age'] = 3
all_data.loc[ all_data['Age'] > 64, 'Age']
all_data.head()
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
#all_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
all_data['IsAlone'] = 0
all_data.loc[all_data['FamilySize'] == 1, 'IsAlone'] = 1

all_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
all_data.loc[ all_data['Fare'] <= 7.91, 'Fare'] = 0
all_data.loc[(all_data['Fare'] > 7.91) & (all_data['Fare'] <= 14.454), 'Fare'] = 1
all_data.loc[(all_data['Fare'] > 14.454) & (all_data['Fare'] <= 31), 'Fare']   = 2
all_data.loc[ all_data['Fare'] > 31, 'Fare'] = 3
all_data['Fare'] = all_data['Fare'].astype(int)
 
all_data.head(10)
all_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Parch', 'SibSp', 'FamilySize'], axis=1, inplace=True)

all_data.head()
all_data.info()
#created dummy variables from categories (also can use OneHotEncoder)
all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','IsAlone','Fare','Embarked','name_title','train_test']])
all_dummies.head()
#Split to train test again
x_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis =1)
x_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)


y_train = all_data[all_data.train_test==1].Survived
y_train.shape
# Scale data 
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age','IsAlone','Fare']]= scale.fit_transform(all_dummies_scaled[['Age','IsAlone','Fare']])


x_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)
x_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#I usually use Naive Bayes as a baseline for my classification tasks
gnb = GaussianNB()
cv = cross_val_score(gnb,x_train,y_train,cv=5)
#print(cv)
#print(cv.mean())
acc_gaussian=round(cv.mean() * 100, 2)
print(acc_gaussian)
lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,x_train,y_train,cv=5)
acc_lr=round(cv.mean() * 100, 2)
print(acc_lr)
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,x_train,y_train,cv=5)
acc_dtree=round(cv.mean() * 100, 2)
print(acc_dtree)
knn = KNeighborsClassifier()
cv = cross_val_score(knn,x_train,y_train,cv=5)
acc_k_neighbor=round(cv.mean() * 100, 2)
print(acc_k_neighbor)
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,x_train,y_train,cv=5)
acc_random_forest=round(cv.mean() * 100, 2)
print(acc_random_forest)
svc = SVC(probability = True)
cv = cross_val_score(svc,x_train_scaled,y_train,cv=5)
acc_svc=round(cv.mean() * 100, 2)
print(acc_svc)
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state =1)
cv = cross_val_score(xgb,x_train,y_train,cv=5)
acc_xgboost=round(cv.mean() * 100, 2)
print(acc_xgboost)
models = pd.DataFrame({   
    'Model': ['Gaussian', 'Logistic Regression', 'Decision Tree', 'KNN','Random Forest','SVC', 'XGBoost'],
    'Score': [acc_gaussian, acc_lr, acc_dtree, acc_k_neighbor, acc_random_forest, acc_svc, acc_xgboost]})
sorted_model=models.sort_values(by='Score', ascending=False)
sorted_model
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
fig = plt.bar(sorted_model['Model'], sorted_model['Score'],color='orange')
plt.grid()
plt.show()
#parameter tuning
from sklearn.model_selection import GridSearchCV  
#simple performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))
svc = SVC(probability = True)
param_grid = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                  'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]
clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_svc = clf_svc.fit(x_train_scaled,y_train)
clf_performance(best_clf_svc,'SVC')

rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [400,450,500,550],
               'criterion':['gini','entropy'],
                                  'bootstrap': [True],
                                  'max_depth': [15, 20, 25],
                                  'max_features': ['auto','sqrt', 10],
                                  'min_samples_leaf': [2,3],
                                  'min_samples_split': [2,3]}
                                  
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(x_train_scaled,y_train)
clf_performance(best_clf_rf,'Random Forest')
xgb = XGBClassifier(random_state = 1)

param_grid = {
    'n_estimators': [450,500,550],
    'colsample_bytree': [0.75,0.8,0.85],
    'max_depth': [None],
    'reg_alpha': [1],
    'reg_lambda': [2, 5, 10],
    'subsample': [0.55, 0.6, .65],
    'learning_rate':[0.5],
    'gamma':[.5,1,2],
    'min_child_weight':[0.01],
    'sampling_method': ['uniform']
}

clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(x_train_scaled,y_train)
clf_performance(best_clf_xgb,'XGB')
Y_pred_xgb = best_clf_xgb.predict(x_test).astype(int)
acc_xgb = round(best_clf_xgb.score(x_train_scaled, y_train) * 100, 2)
acc_xgb
Y_pred_svc = best_clf_svc.predict(x_test).astype(int)
acc_svc = round(best_clf_svc.score(x_train_scaled, y_train) * 100, 2)
acc_svc
Y_pred_rf = best_clf_rf.predict(x_test).astype(int)
acc_rf = round(best_clf_rf.score(x_train_scaled, y_train) * 100, 2)
acc_rf
submission_xgb = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_pred_xgb
    })
submission_svc = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_pred_svc
    })
submission_rf = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_pred_rf
    })
submission_xgb.to_csv('submission_xgb.csv', index=False)
submission_svc.to_csv('submission_svc.csv', index=False)
submission_rf.to_csv('submission_rf.csv', index=False)