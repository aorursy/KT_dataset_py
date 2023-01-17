import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/heart.csv')
data.head()
data.info()
data.describe()
data.isnull().sum()
plt.style.use('fivethirtyeight')

plt.figure(figsize=(10,10))
data['cp'].value_counts().plot(kind='bar', title='Chest Pain Types',align='center');
data['fbs'].value_counts().plot(kind='bar', title='Fast Blood Sugar (fbs)',align='center');
data['exang'].value_counts().plot(kind='bar', title='Exercise induced angina',align='center');
plt.bar(data['age'],data['oldpeak'],color='r')

plt.xlabel('Age')

plt.ylabel('ST Depression Peak')
sns.distplot(data['thalach'],bins=20,hist=True,)
fig,ax = plt.subplots()

ax.scatter(data['age'],data['chol'],c=data.age)

ax.set_xlabel('Age')

ax.set_ylabel('Cholestrol Level')

ax.set_title('Age vs Cholestrol Level')
data['slope'].value_counts().plot(kind='bar', title='Slope of the peak',align='center');
data['target'].value_counts().plot(kind='bar', title='Has the disease or not?',align='center');

plt.xlabel('Yes or No')

plt.ylabel('Count')
sns.pairplot(data)
cp_categories = pd.get_dummies(data['cp'], prefix = "cp")

thal_categroies = pd.get_dummies(data['thal'], prefix = "thal")

Slope_categories = pd.get_dummies(data['slope'], prefix = "slope")

data.drop(['cp','thal','slope'],axis=1,inplace=True)
data = pd.concat([data,cp_categories,thal_categroies,Slope_categories], axis = 1)
data.head()
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import xgboost as xgb

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score
xtreme = xgb.XGBClassifier(learning_rate=0.1,n_estimators=100)

knn = KNeighborsClassifier(n_neighbors=2)

rf = RandomForestClassifier(random_state=1,n_estimators=1000)

lr = LogisticRegression()

dt = DecisionTreeClassifier(criterion='gini',max_depth=8)

svc = SVC(random_state=1)



#ada = AdaBoostClassifier(n_)

#model = VotingClassifier(estimators=[lr,knn,rf])
y = data['target']

X = data.drop(['target'],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
names = ['KNN','LogisticRegression','Decision Tree','Gradient Boost','Random Forest','SVM']

scores = []

def accuracy(model,X_train,y_train,X_test,y_test):

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    scores.append(accuracy_score(y_test,y_pred))
models = [knn, lr, dt, xtreme, rf, svc]

for i in models:

    accuracy(i,X_train, y_train, X_test,y_test)
scores
plt.plot(names,scores)

plt.xticks(rotation=90)
from sklearn.model_selection import GridSearchCV
neighbors = np.arange(1,20)

parameters = {'n_neighbors':neighbors}
knn_grid = GridSearchCV(estimator=knn,param_grid=parameters,cv=5)

#knn_grid.fit(X_train,y_train)
knn_grid.fit(X_train,y_train)
knn_grid.best_params_
y_pred = knn_grid.predict(X_test)
knn_new = accuracy_score(y_test,y_pred)
max_fea = np.random.randint(1,11) #select a random int value

m_split = np.random.randint(2, 11)
params = {"max_depth": [1,8,None], #just gave numbers i felt that would be good. More numbers we give, more time it would take

             "max_features": [max_fea],

              "min_samples_split": [m_split],

              "bootstrap": [True, False],

              "criterion": ["gini", "entropy"]}

#No of iterations: 3 * 1 * 1 * 2 * 2 * no of folds.
rf_grid = GridSearchCV(estimator=rf,param_grid=params,cv=3)
rf_grid.fit(X_train,y_train)
rf_grid.best_params_
y_pred = rf_grid.predict(X_test)

rf_new = accuracy_score(y_test,y_pred)
params = {

        'learning_rate': [0.1,0.05,0.01],

        'min_child_weight': [1, 5, 10],

        'gamma': [1,5],

        'max_depth': [3, 4, 5],

        'n_estimator':[100,1000]

        }
xg_grid = GridSearchCV(estimator=xtreme,param_grid=params,cv=3)

xg_grid.fit(X_train,y_train)
xg_grid.best_params_
y_pred = xg_grid.predict(X_test)

xgb_new = accuracy_score(y_test,y_pred)
penalty = ['l1', 'l2']

C = np.logspace(0, 4, 10)

params = dict(C=C, penalty=penalty)
lr_grid = GridSearchCV(estimator=lr,param_grid=params,cv=3)

lr_grid.fit(X_train,y_train)
lr_grid.best_params_
y_pred = lr_grid.predict(X_test)

lr_new = accuracy_score(y_test,y_pred)
Cs =[0.01]

gammas = [0.001, 0.01]

kernels = ['linear', 'rbf']

params = {'C': Cs, 'gamma' : gammas,'kernel':kernels}
grid = GridSearchCV(estimator=svc,param_grid=params,cv=3)

grid.fit(X_train,y_train)
grid.best_params_
y_pred = grid.predict(X_test)

svm_new = accuracy_score(y_test,y_pred)
max_dep = np.arange(3, 10)

cri = ['gini','entropy']



params= {'max_depth': max_dep,'criterion':cri}
dt_grid = GridSearchCV(estimator=dt,param_grid=params,cv=3)
dt_grid.fit(X_train,y_train)
dt_grid.best_params_
y_pred = dt_grid.predict(X_test)

dt_new = accuracy_score(y_test,y_pred)
names = ['KNN','LogisticRegression','Decision Tree','Gradient Boost','Random Forest','SVM']
new_scores= [knn_new,lr_new,dt_new,xgb_new,rf_new,svm_new]
fig, ax = plt.subplots()

ax.plot(names,scores,marker = 'o',markersize=15)

ax.plot(names,new_scores,marker='o',markersize=15)

ax.legend(['Old Scores','New Scores'])

fig.set_size_inches(18.5, 10.5)