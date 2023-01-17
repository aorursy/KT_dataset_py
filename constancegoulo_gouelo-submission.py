

import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





import seaborn as sns



from sklearn.pipeline import Pipeline





from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split



from sklearn.model_selection import KFold



from sklearn.model_selection import cross_val_score



from sklearn.model_selection import GridSearchCV



from sklearn.metrics import classification_report



from sklearn.metrics import accuracy_score



from sklearn.metrics import confusion_matrix



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier



from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.shape
train.head()
train.tail()
train.describe().transpose()
train.groupby('Survived').size()
train.info()
train.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')
# setting the style of axes(plotting area) as 'whitegrid'.

sns.set_style('whitegrid')



# let's count the #person survived

sns.countplot(x='Survived',data= train, palette='RdBu_r')
# count # survived person catergorised by 'Sex'

sns.countplot(x='Survived', hue='Sex',data= train, palette='RdBu_r')
sns.countplot(x='Survived', hue='Pclass',data= train, palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=False,bins=30,color='darkred')
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=train)
sns.countplot(x='Survived',hue='SibSp',data=train)
train['Fare'].hist(color='green',bins=35)
sns.boxplot(x='Pclass', y='Age',data=train,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)



# we have also to clean the test data

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
train.info()
train.drop('Cabin',axis=1,inplace=True)

# drop Cabin from test data also

test.drop('Cabin',axis=1,inplace=True)
train.head()
train.isnull().sum()
train.dropna(inplace=True)

test.isnull().sum()
test[test['Fare'].isnull()]
test.set_value(152,'Fare',50)
train.isnull().sum()

test.isnull().sum()
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True) 

embark = pd.get_dummies(train['Embarked'],drop_first=True) 
sex_test = pd.get_dummies(test['Sex'],drop_first=True) 

embark_test = pd.get_dummies(test['Embarked'],drop_first=True) 
train.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)



test.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)

train = pd.concat([train,sex,embark],axis=1)

test = pd.concat([test,sex_test,embark_test],axis=1)
train.head()
test.head()
predictors = train.drop(['Survived'],axis=1)
predictors.head() 
target = train['Survived']
target.head()


X = predictors.values

Y = target.values

validation_size = 0.20

seed = 42 

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=validation_size,random_state=seed)
type(X)
models = []



models.append(('LR',LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=10000)))

models.append(('LDA',LinearDiscriminantAnalysis()))

models.append(('KNN',KNeighborsClassifier()))

models.append(('CART',DecisionTreeClassifier()))

models.append(('NB',GaussianNB()))

models.append(('SVM',SVC(gamma='scale')))



results = []

names = []



for name, model in models:

    kfold = KFold(n_splits = 10, random_state=seed)

    

    cv_results = cross_val_score(model,X_train,y_train,cv=kfold, scoring="accuracy")

    

    results.append(cv_results)

    

    names.append(name)

    

    print(name, cv_results.mean()*100.0, "(",cv_results.std()*100.0,")")
figure = plt.figure()

figure.suptitle('Algorithm Comparison')

ax = figure.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)
num_folds=10

seed=42

scoring='accuracy'
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=10000))])))

pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA',LinearDiscriminantAnalysis())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsClassifier())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeClassifier())])))

pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',GaussianNB())])))

pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC(gamma='scale'))])))

results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())

    print(msg)
fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]

kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

param_grid = dict(C=c_values, kernel=kernel_values)

model = SVC()

kfold = KFold(n_splits=num_folds, random_state=seed)

grid = GridSearchCV(estimator= model, param_grid=param_grid, scoring=scoring,cv=kfold)

grid_result = grid.fit(rescaledX,y_train)

print("Best: %f using %s"% (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds= grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']



for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r"%(mean,stdev,param))
ensembles = []

ensembles.append(('AB', AdaBoostClassifier()))

ensembles.append(('GBM', GradientBoostingClassifier()))

ensembles.append(('RF', RandomForestClassifier()))

ensembles.append(('ET', ExtraTreesClassifier()))

results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)

    print(msg)
fig = plt.figure()

fig.suptitle('Ensemble Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
scaler = StandardScaler().fit(X_train)



rescaledX = scaler.transform(X_train)



model = SVC(C=2.0,kernel='rbf')



model.fit(rescaledX,y_train)





rescaledValidationX = scaler.transform(X_test)

predictions = model.predict(rescaledValidationX)

print(accuracy_score(y_test,predictions)*100)
from xgboost import XGBClassifier
new_model = XGBClassifier(n_estimators = 1000, learning_rate=0.05)
new_model.fit(X_train,y_train,

             early_stopping_rounds = 5,

             eval_set = [(X_test,y_test)],

             verbose = False)
xgb_predictions = new_model.predict(X_test)
xgb_predictions
accuracy_score(xgb_predictions,y_test)


test_predictions = new_model.predict(test.values)
test_predictions
xgb_submission= pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
xgb_submission.head(15)
xgb_submission['Survived'] = test_predictions
xgb_submission.head(15)
xgb_submission.to_csv('gender_submission.csv',index=False)