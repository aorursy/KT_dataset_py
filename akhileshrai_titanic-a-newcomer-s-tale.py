# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings #supress warnings

warnings.filterwarnings(action="ignore")



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import pandas as pd

#from math import pi

from collections import Counter

import numpy as np

import seaborn as sns



import warnings

warnings.filterwarnings('ignore') 



df = pd.read_csv('../input/train.csv')#extract values

Test = pd.read_csv('../input/test.csv',index_col = None)#do not drop passenger id

Y = df['Survived'] # dependent variable



cols_with_missing = [col for col in df.columns if df[col].isnull().any()]

print(cols_with_missing)

cols_with_missing2 = [col for col in Test.columns if Test[col].isnull().any()]

print(cols_with_missing2)
for col in cols_with_missing:

    if(df[col].dtype == np.dtype('O')):

         df[col]= df[col].fillna(df[col].value_counts().index[0]) 

   #replace nan with most frequent

    else:

        df[col] = df[col].fillna(0) 

        #replace nan with median

print(df.isnull().any())

print(Test.isnull().any())


for col in cols_with_missing2:

    if(Test[col].dtype == np.dtype('O')):

         Test[col]= Test[col].fillna(Test[col].value_counts().index[0]) #replace nan with most frequent

    else:

        Test[col] = Test[col].fillna(0) 

        #replace nan with median

print(df.isnull().any())

print(Test.isnull().any())



#encoding and categorizng data



df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

Test['Title'] = Test['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title



df['Title'] = df.apply(replace_titles,axis = 1)

Test['Title'] = Test.apply(replace_titles,axis = 1)

features= [ 'Title', 'Pclass','Sex','Fare','Embarked']

X = df[features]
df.groupby(df['Survived']).count()



549/(549+342)
#plot how many survive

names = 'Died', 'Survived'

size = list(dict(Counter(df['Survived'])).values())

fig = plt.figure(figsize=(18,8))

fig.patch.set_facecolor('black')

plt.rcParams['text.color'] = 'white'

my_circle=plt.Circle( (2,2), 0.9, color='black')

plt.pie(size, labels=names)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.legend(loc='lower left')

plt.title("Training Data- 'Died vs Survived' ")

plt.show()
names = 'Mr', 'Mrs', 'Miss', 'Master'

size = list(dict(Counter(df['Title'])).values())

fig = plt.figure(figsize=(18,8))

fig.patch.set_facecolor('black')

plt.rcParams['text.color'] = 'white'

my_circle=plt.Circle( (3,3), 0.9, color='black')

plt.pie(size, labels=names)

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.legend(loc='lower left')

plt.show()

plt.rcParams['figure.figsize'] = (18, 7)

plt.style.use('seaborn-dark-palette')

sns.countplot(df['Title'], palette = 'rainbow')

plt.title('Title Variations', fontsize = 20)
# Title,PClass vs Survived

f,ax=plt.subplots(1,2,figsize=(25,10))

df.groupby(['Title','Survived'])['Title'].count().plot.bar(ax=ax[0])

ax[0].set_title("Title vs Survival/Death")

df.groupby(['Pclass','Survived'])['Pclass'].count().plot.bar(ax=ax[1])

ax[1].set_title('Passenger Class vs Survival')



f1,ax1=plt.subplots(1,2,figsize=(25,10))

df.groupby(['Embarked','Survived'])['Embarked'].count().plot.bar(ax=ax1[0])

ax1[0].set_title("Embarked vs Survival/Death")

df.groupby(['Sex','Survived'])['Sex'].count().plot.bar(ax=ax1[1])

ax1[1].set_title('Sex vs Survival')
sns.pairplot(df, kind="scatter",hue= 'Embarked', markers=["o", "s", "D"],palette="Set2")

plt.show()
sns.relplot(x="PassengerId", y="Age", hue="Title", data=df)
sns.relplot(x="PassengerId", y="Age", hue="Embarked",col="Title",col_wrap = 2, palette="ch:r=-.5,l=.75", data=df)




# Fitting the models

select_model = []



from sklearn.preprocessing import LabelEncoder





LE = LabelEncoder()

X['Embarked'] = LE.fit_transform(X['Embarked'])

X["Sex"] = LE.fit_transform(X["Sex"])

X['Title'] = LE.fit_transform(X['Title'])

Test['Embarked'] = LE.fit_transform(Test['Embarked'])

Test["Sex"] = LE.fit_transform(Test["Sex"])

Test['Title'] = LE.fit_transform(Test['Title'])



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state =0)



X.describe()
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)



sns.heatmap(X.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
pd.crosstab([df.Title,df.Pclass,df.Embarked],

            [df.Sex,df.Survived],margins=True).style.background_gradient(cmap='summer_r')

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix   





LR = LogisticRegression()

# LR.fit(x_train,y_train)

# y_pred = LR.predict(x_test)

# confusion_matrix(y_test,y_pred)

# print(LR.score(x_test,y_test))

LR.fit(X,Y)



#select_model.append(LR.score(x_test,y_test))

from sklearn.naive_bayes import GaussianNB



GBayes_clf = GaussianNB()

#GBayes_clf.fit(x_train, y_train)

GBayes_clf.fit(X, Y)

# print (GBayes_clf.score(x_test,y_test))

# select_model.append(GBayes_clf.score(x_test,y_test))

# confusion_matrix(y_pred,y_test)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score



k_range = range(1,31)

weights_options=['uniform','distance']

param = {'n_neighbors':k_range, 'weights':weights_options}



cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

grid_KNNsearch = GridSearchCV(KNeighborsClassifier(), param, cv=cv) 

# grid_KNNsearch.fit(x_train,y_train)

grid_KNNsearch.fit(X,Y)

grid_KNNsearch = grid_KNNsearch.best_estimator_

# y_pred = grid_KNNsearch.predict(x_test)

# print(grid_KNNsearch.score(x_test,y_test))



# select_model.append(grid_KNNsearch.score(x_test,y_test))



# confusion_matrix(y_pred,y_test)
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

Cs = [0.001, 0.01, 0.1, 1,1.5,2,2.5,3,4,5, 10] 

gammas = [0.0001,0.001, 0.01, 0.1, 1]

param_grid = {'C': Cs, 'gamma' : gammas}

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

grid_search = GridSearchCV(SVC(kernel = 'rbf', probability=True), param_grid, cv=cv) 

# grid_search.fit(x_train,y_train)

grid_search.fit(X,Y)

SVC_grid = grid_search.best_estimator_

cross_val_score(SVC_grid, X, Y, cv=5).mean()

# print(SVC_grid.score(x_test,y_test))

# select_model.append(SVC_grid.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

    

tree_clf = DecisionTreeClassifier(max_depth = 5)

tree_clf.fit(X,Y)

score_tree = cross_val_score(tree_clf, X, Y, cv=5).mean()

print(score_tree)

select_model.append(score_tree)


from sklearn.ensemble import BaggingClassifier

n_estimators = [10,30,50,70,80,150,160, 170,175,180,185];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)



params = {'n_estimators':n_estimators}



bagging_grid = GridSearchCV(BaggingClassifier(base_estimator= None,

                                      bootstrap_features=False),

                                 param_grid=params,

                                 cv=cv,

                                 n_jobs = -1)

bagging_grid.fit(X,Y)

bagging_estimator = bagging_grid.best_estimator_



bagging_estimator_score = cross_val_score(bagging_estimator, X, Y, cv=5).mean()



cross_val_score(bagging_estimator, X, Y, cv=5).mean()
from sklearn.gaussian_process import GaussianProcessClassifier

GaussianProcessClassifier = GaussianProcessClassifier()

GaussianProcessClassifier.fit(X,Y)

# y_pred = GaussianProcessClassifier.predict(x_test)

# print(accuracy_score(y_pred, y_test))

# select_model.append(accuracy_score(y_pred, y_test))



cross_val_score(GaussianProcessClassifier, X, Y, cv=5).mean()


from sklearn.ensemble import ExtraTreesClassifier

ExtraTreesClassifier = ExtraTreesClassifier()

ExtraTreesClassifier.fit(X, Y)

# y_pred = ExtraTreesClassifier.predict(x_test)

# extraTree_accy = round(accuracy_score(y_pred, y_test), 3)

# select_model.append(extraTree_accy)

# print(extraTree_accy)



cross_val_score(ExtraTreesClassifier, X, Y, cv=5).mean()

from xgboost.sklearn import XGBClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV



params={

    'max_depth': [5],

    'subsample': [1.0],

    'colsample_bytree': [0.5],

    'n_estimators': [1000],

    'reg_alpha': [0.01, 0.02, 0.03, 0.04]

}



xgb_clf = XGBClassifier()

grid_search = GridSearchCV(xgb_clf,

                  params,

                  cv=5,

                  n_jobs=1,

                  verbose=2)





grid_search.fit(X,Y)

#print("\nGrid Search Best parameters set :")

#print(grid_search.best_params_)



model = grid_search.best_estimator_



cross_val_score(model, X, Y, cv=5).mean()




from sklearn.ensemble import RandomForestClassifier

n_estimators = [140,145,150,155,160];

max_depth = range(1,10);

criterions = ['gini', 'entropy'];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)





parameters = {'n_estimators':n_estimators,

              'max_depth':max_depth,

              'criterion': criterions

              

        }

grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

grid.fit(X,Y) 

rf_grid = grid.best_estimator_

# print(rf_grid.score(x_test,y_test))

# select_model.append(rf_grid.score(x_test,y_test))

cross_val_score(rf_grid, X, Y, cv=5).mean()


from sklearn.ensemble import AdaBoostClassifier

n_estimators = [100,140,145,150,160, 170,175,180,185];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

learning_r = [0.1,1,0.01,0.5]



parameters = {'n_estimators':n_estimators,

              'learning_rate':learning_r

              

        }

adaBoost_grid = GridSearchCV(AdaBoostClassifier(base_estimator= None, ),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

adaBoost_grid.fit(X,Y) 

AdaBoost_estimator = adaBoost_grid.best_estimator_

# print(AdaBoost_estimator.score(x_test,y_test))

# select_model.append(AdaBoost_estimator.score(x_test,y_test))

cross_val_score(AdaBoost_estimator, X, Y, cv=5).mean()
from sklearn.ensemble import VotingClassifier



voting_classifier = VotingClassifier(estimators=[

    ('lr_grid', LR),

    ('svc', SVC_grid),('AdaBoost', AdaBoost_estimator),

    ('random_forest', rf_grid),

    ('knn_classifier', grid_KNNsearch),

    ('bagging_classifier', bagging_grid),

    ('Decision Tree', tree_clf),('XGBClassifier',model),

    ('ExtraTrees_Classifier', ExtraTreesClassifier),

    ('gaussian_bayes_classifier',GBayes_clf),

    ('gaussian_process_classifier', GaussianProcessClassifier)

],voting='hard')



voting_classifier = voting_classifier.fit(X,Y)   

# y_pred = voting_classifier.predict(x_test)

# voting_accy = accuracy_score(y_pred, y_test)

# print(voting_accy)

# select_model.append(voting_accy)

cross_val_score(voting_classifier, X, Y, cv=5).mean()
test_prediction = voting_classifier.predict(Test[features])
submission = pd.DataFrame({

        "PassengerId": Test['PassengerId'],

        "Survived": test_prediction

    })

print(submission)

submission.to_csv('submission.csv', index=False)