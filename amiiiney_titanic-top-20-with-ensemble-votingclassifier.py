

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

pd.options.mode.chained_assignment = None # Warning for chained copies disabled



from sklearn import preprocessing

import sklearn.model_selection as ms

from sklearn import linear_model

import sklearn.metrics as sklm

from sklearn import feature_selection as fs

from sklearn import metrics

from sklearn.model_selection import cross_validate

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score



import numpy.random as nr

import scipy.stats as ss

import math



import warnings

warnings.simplefilter(action='ignore')



%matplotlib inline
a = pd.read_csv("/kaggle/input/titanic/train.csv")

b = pd.read_csv("/kaggle/input/titanic/test.csv")
na = a.shape[0] #Shape of train, to be used later to split the combined dataset

nb = b.shape[0]

frames= [a,b]

#Combine train and test sets

c1=pd.concat(frames, axis=0, sort=False).reset_index(drop=True)

#Drop the target "Survived" from the combined dataset

target = a[['Survived']]

c1.drop(['Survived'], axis=1, inplace=True)

print("The shape of the training set is", na)

print("Total size is :",c1.shape)
print(c1.isnull().sum())
Survived=a['Survived'].value_counts()

Survived=pd.DataFrame(Survived)

Survived=Survived.reset_index()



pclass= a['Pclass'].value_counts()

pclass= pd.DataFrame(pclass)

pclass= pclass.reset_index()



sex= a['Sex'].value_counts()

sex= pd.DataFrame(sex)

sex= sex.reset_index()



embarked=a['Embarked'].value_counts()

embarked= pd.DataFrame(embarked)

embarked= embarked.reset_index()



sibling=a['SibSp'].value_counts()

sibling= pd.DataFrame(sibling)

sibling= sibling.reset_index()



parch=a['Parch'].value_counts()

parch= pd.DataFrame(parch)

parch= parch.reset_index()

plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,15))

ax1 = plt.subplot2grid((3,2),(0,0))

plt.pie(Survived.Survived,colors=("pink","r"), autopct='%2.1f%%',labels=Survived['index'], shadow=True)

plt.title('Titanic survival rates', fontsize=13, weight='bold' )



ax1 = plt.subplot2grid((3,2),(0,1))

plt.pie(pclass.Pclass,colors=("cyan","lime", 'orange'), autopct='%2.1f%%',labels=pclass['index'], shadow=True)

plt.title('Passengers class 1,2 and 3 distributions', fontsize=13, weight='bold' )



ax1 = plt.subplot2grid((3,2),(1,0))

plt.pie(sex.Sex,colors=("darkcyan","purple"), autopct='%2.1f%%',labels=sex['index'], shadow=True)

plt.title('Passengers sex', fontsize=13, weight='bold' )



ax1 = plt.subplot2grid((3,2),(1,1))

plt.pie(embarked.Embarked,colors=("deepskyblue","lawngreen", 'crimson'), autopct='%2.1f%%',labels=embarked['index'], shadow=True)

plt.title('Passengers embarking port', fontsize=13, weight='bold' )



ax1 = plt.subplot2grid((3,2),(2,0))

plt.pie(sibling.SibSp,colors=("gold","sienna","plum","deepskyblue","lawngreen", 'crimson'), autopct='%2.1f%%',labels=sibling['index'], shadow=True)

plt.title('Passengers siblings number', fontsize=13, weight='bold' )



ax1 = plt.subplot2grid((3,2),(2,1))

plt.pie(parch.Parch,colors=("gold","sienna","plum","deepskyblue","lawngreen", 'crimson', 'red'), autopct='%2.1f%%',labels=parch['index'], shadow=True)

plt.title('Passengers parents/children number', fontsize=13, weight='bold' )





plt.show()
sns.catplot(x='Pclass', kind='count',hue='Survived', data=a, palette='ch:.384')

plt.title('Survival rates per Pclass', fontsize=10, weight='bold' )

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
sns.catplot(x='Sex', kind='count',hue='Survived', data=a, palette='ch:.991')

plt.title('Survival rates per sex', fontsize=10, weight='bold' )

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
sns.set_style('whitegrid')

f,ax=plt.subplots(figsize=(10,5))

c1['Age'].plot(kind='hist', color='darkorchid', alpha=0.7)

plt.title('Distribution of Titanic passengers ages', fontsize=10, weight='bold' )

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
sns.boxplot(y='Age', x='Survived', data=a, palette='ch:.39041')

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()


sns.catplot(x='SibSp', kind='count',hue='Survived', data=a, palette='Paired')

plt.title('Survival rates with respect to number of siblings', fontsize=10, weight='bold' )

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
sns.catplot(x='Parch', kind='count',hue='Survived', data=a, palette='PuRd')

plt.title('Survival rates with respect to the number of parents and children', fontsize=10, weight='bold' )

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
sns.set_style('whitegrid')

fig = plt.figure(figsize=(10,5))

c1['Fare'].plot(kind='hist', color='grey', alpha=0.7)

plt.title('Fare distribution', fontsize=15, weight='bold' )

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()

sns.set_style('whitegrid')

sns.boxplot(x='Pclass', y='Fare',hue='Survived', palette='ch:.838', data=a)

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
sns.catplot(x='Embarked', kind='count',hue='Survived', data=a, palette='Set1')

plt.title('Survival rates with respect to the embarking port', fontsize=10, weight='bold' )

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()

c1['family_size']=c1['SibSp'] + c1['Parch'] + 1
a['family_size']=a['SibSp'] + a['Parch'] + 1

sns.catplot(x='family_size', kind='count',hue='Survived', data=a, palette='Accent')

plt.title('Survival rates per family size', fontsize=15, weight='bold' )

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
c1.isnull().sum()
sns.catplot(y='Cabin', x='Fare',height=4, hue='Survived', col='Pclass', data=a)

plt.show()
c=c1.drop('Cabin', axis=1)
c['Age'].fillna(28, inplace=True)
c['Embarked'].fillna(method='ffill', inplace=True)
c['Fare'].fillna(method='ffill', inplace=True)
#We cleaned our combined dataset from missing values

print(c.isnull().sum())
#First, we keep just the features that will be used in Machine Learning

c2=c[['Pclass','Sex','Age','Embarked','family_size','Parch','SibSp', 'Fare']]



#Change the data type of Pclass to object

c2['Pclass']=c2['Pclass'].astype(object)
c3=pd.get_dummies(c2)

print("the shape of the original dataset",c2.shape)

print("the shape of the encoded dataset",c3.shape)

print("We have ",c3.shape[1]- c2.shape[1], 'new encoded features')
c3.sample(2)
#Splitting the combined dataset to the original train and test datasets

Train = c3[:na]  #na is the number of rows of the original training set

Test = c3[na:] 
sns.boxplot(y='Fare', data=c)

plt.title('Fare boxplot', fontsize=10, weight='bold' )

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
s=a[['Pclass','Fare','Survived']]

s.sort_values(by='Fare', ascending=False).head(10)
Train['Fare'].sort_values(ascending=False).head(20)
Train1=Train[Train['Fare'] < 200]
print('We dropped ',Train.shape[0] - Train1.shape[0],'fare outliers')
sns.boxplot(y='SibSp', data=c2, color='orange')

plt.title('SibSp boxplot', fontsize=10, weight='bold' )

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
Train1['SibSp'].sort_values(ascending=False).head(7)
train=Train1[Train1['SibSp'] <= 5]
print('We dropped ', Train1.shape[0]- train.shape[0], 'sibSp outliers')

print('And in total, we dropped ', Train.shape[0]-train.shape[0], 'outliers')
print(train.shape, target.shape)
#Index positions are taken from the above 2 list of outiers (Fare and SibSp)

pos=[863,846,159,792,180,324,201, 679, 258, 737, 341,438,88,27,311,742,299, 118,

     716,557,380,700,527,377,698,730,779]

target.drop(target.index[pos], inplace=True)



print("Shape of training dataset",train.shape[0])

print("Shape of target", target.shape[0])
f,ax=plt.subplots(figsize=(10,7))

sns.heatmap(train.corr(), annot=True, fmt = ".2f", cmap='viridis')

plt.title('Correlation between features', fontsize=10, weight='bold' )

plt.show()
train.sample(5)
x=train

y=np.array(target)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .33, random_state=0)
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

# transform "x_train"

x_train = scaler.fit_transform(x_train)

# transforming "x_test"

x_test = scaler.transform(x_test)



# transforming "test (dataset)"

test = scaler.transform(Test)
#We will need those libraries 

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import classification_report



from sklearn.linear_model import LogisticRegression



lr_c=LogisticRegression( C=1, class_weight={0:0.62, 1:0.38}, max_iter=5000,

                     penalty='l2',

                   random_state=None, solver='lbfgs', verbose=0,

                   warm_start=True)

lr_c.fit(x_train,y_train.ravel())

lr_pred=lr_c.predict(x_test)

lr_ac=accuracy_score(y_test.ravel(), lr_pred)

print('LogisticRegression_accuracy test:',lr_ac)

print("AUC",roc_auc_score(y_test.ravel(), lr_pred))
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

cv = StratifiedShuffleSplit(n_splits = 15, test_size = .25, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%



x = scaler.fit_transform(x)

accuracies = cross_val_score(LogisticRegression(solver='liblinear',class_weight={0:0.62, 1:0.38}), x,y, cv  = cv)

print ("CV accuracy on 15 chunks: {}".format(accuracies))

print ("Mean CV accuracy: {}".format(round(accuracies.mean(),5)))
from sklearn import svm 

from sklearn.svm import SVC, LinearSVC





svm=SVC(kernel='rbf',C=10,gamma=0.1)

svm.fit(x_train,y_train.ravel())

svm_pred=svm.predict(x_test)

print('Accuracy is ',metrics.accuracy_score(svm_pred,y_test.ravel()))

print('AUC: ',roc_auc_score(y_test.ravel(), svm_pred))
SVMC = SVC(probability=True,class_weight={0:0.62, 1:0.38})

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001,0.008, 0.01,0.02,0.05, 0.1, 1],

                  'C': [1, 10, 12, 14, 20, 25, 30, 40,50,60, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=5, scoring="accuracy", n_jobs=-1)



gsSVMC.fit(x_train,y_train)



SVMC_best = gsSVMC.best_estimator_

SVMC_best.fit(x_train, y_train.ravel())

pred_svm = SVMC_best.predict(x_test)

acc_svm = accuracy_score(y_test.ravel(), pred_svm)

print('Accuracy', acc_svm)

print('AUC: ',roc_auc_score(y_test.ravel(), pred_svm))

print(SVMC_best)
from sklearn.ensemble import RandomForestClassifier



rdf_c=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)

rdf_c.fit(x_train,y_train.ravel())

rdf_pred=rdf_c.predict(x_test)

rdf_ac=accuracy_score(rdf_pred,y_test.ravel())



print('Accuracy of random forrest classifier: ',rdf_ac)

print('AUC: ',roc_auc_score(y_test.ravel(), rdf_pred))
from sklearn.neighbors import KNeighborsClassifier



knn_clf = KNeighborsClassifier()



parameters_knn = {"n_neighbors": [3, 5, 10, 15], "weights": ["uniform", "distance"], "algorithm": ["auto", "ball_tree", "kd_tree"],

                  "leaf_size": [20, 30, 50]}



grid_knn = GridSearchCV(knn_clf, parameters_knn, scoring='accuracy', cv=5, n_jobs=-1)

grid_knn.fit(x_train, y_train)



knn_clf = grid_knn.best_estimator_



knn_clf.fit(x_train, y_train.ravel())

pred_knn = knn_clf.predict(x_test)

acc_knn = accuracy_score(y_test.ravel(), pred_knn)







print("The accuracy of KNeighbors is: " + str(acc_knn))

print('AUC: ',roc_auc_score(y_test.ravel(), pred_knn))
from xgboost import XGBClassifier



xg_clf = XGBClassifier()



parameters_xg = {"objective" : ["reg:linear"], "n_estimators" : [5, 10, 15, 20]}



grid_xg = GridSearchCV(xg_clf, parameters_xg, scoring='accuracy',cv=5,n_jobs=-1)

grid_xg.fit(x_train, y_train)



xg_clf = grid_xg.best_estimator_



xg_clf.fit(x_train, y_train.ravel())

pred_xg = xg_clf.predict(x_test)

acc_xg = accuracy_score(y_test.ravel(), pred_xg)



print("The accuracy of XGBoost is: " + str(acc_xg))

print('AUC: ',roc_auc_score(y_test.ravel(), pred_xg))
from sklearn.neural_network import MLPClassifier

nr.seed(1115)

nn_mod = MLPClassifier(hidden_layer_sizes = (100,100,), max_iter=1750, solver='sgd' )

nn_mod.fit(x_train, y_train.ravel())

scores = nn_mod.predict(x_test)

nnacc = accuracy_score(y_test.ravel(), scores)





print('The accuracy of MLP classifier: ',nnacc)

print('AUC: ',roc_auc_score(y_test.ravel(), scores))
from sklearn.ensemble import BaggingClassifier

Bagg_estimators = [10,25,50,75,100,150,250];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.33, random_state=15)



parameters = {'n_estimators':Bagg_estimators }

gridBG = GridSearchCV(BaggingClassifier(base_estimator= None, 

                                      bootstrap_features=False),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)





bg_mod=gridBG.fit(x_train,y_train.ravel())

bg_pred = bg_mod.predict(x_test)

bg_acc = accuracy_score(y_test.ravel(), bg_pred)

print("Bagging accuracy: ", bg_acc)

print('AUC: ',roc_auc_score(y_test.ravel(), bg_pred))
from sklearn.ensemble import AdaBoostClassifier

n_estimators = [100,140,145,150,160, 170,175,180,185];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

learning_rate = [0.1,1,0.01,0.5]



parameters = {'n_estimators':n_estimators,

              'learning_rate':learning_rate}

gridAda = GridSearchCV(AdaBoostClassifier(base_estimator= None, 

                                     ),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

ada_mod= gridAda.fit(x_train,y_train.ravel()) 

ada_pred = ada_mod.predict(x_test)

ada_acc = accuracy_score(y_test.ravel(), ada_pred)



print("Adaboost accuracy: ", ada_acc)

print('AUC: ',roc_auc_score(y_test.ravel(), ada_pred))
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gradient_boost = GradientBoostingClassifier()

gradient_boost.fit(x_train, y_train.ravel())

ygbc_pred = gradient_boost.predict(x_test)

gradient_accy = round(accuracy_score(ygbc_pred, y_test.ravel()), 3)

print('Gradient Boosting accuracy: ',gradient_accy)

print('AUC: ',roc_auc_score(y_test.ravel(), ygbc_pred))

from sklearn.ensemble import ExtraTreesClassifier

ExtraTreesClassifier = ExtraTreesClassifier()

ExtraTreesClassifier.fit(x_train, y_train.ravel())

y_pred = ExtraTreesClassifier.predict(x_test)

extraTree_accy = accuracy_score(y_pred, y_test.ravel())

print('ExtraTrees classifier accuracy: ',extraTree_accy)

print('AUC: ',roc_auc_score(y_test.ravel(), y_pred))
from sklearn.gaussian_process import GaussianProcessClassifier

GaussianProcessClassifier = GaussianProcessClassifier()

GaussianProcessClassifier.fit(x_train, y_train.ravel())

yg_pred = GaussianProcessClassifier.predict(x_test)

gau_pro_accy = accuracy_score(yg_pred, y_test.ravel())

print('Gaussian process classifier: ', gau_pro_accy)

print('AUC: ',roc_auc_score(y_test.ravel(), yg_pred))
from sklearn.ensemble import VotingClassifier



voting_classifier = VotingClassifier(estimators=[

    ('gradient_boosting', gradient_boost),

    ('bagging_classifier', bg_mod),

    ('ada_classifier',ada_mod),

    ('XGB_Classifier', xg_clf),

    ('gaussian_process_classifier', GaussianProcessClassifier)

],voting='hard')



#voting_classifier = voting_classifier.fit(train_x,train_y)

voting_classifier = voting_classifier.fit(x_train,y_train.ravel())
vote_pred = voting_classifier.predict(x_test)

voting_accy = round(accuracy_score(vote_pred, y_test.ravel()), 4)

print('Voting accuracy of the combined classifiers: ',voting_accy)

print('AUC: ',round(roc_auc_score(y_test.ravel(), vote_pred), 4))



print(classification_report(y_test, vote_pred))



#We fit the votingClasiffier on the test data

final_pred = voting_classifier.predict(test)
titanic_submission = pd.DataFrame({

        "PassengerId": b["PassengerId"],

        "Survived": final_pred

    })



titanic_submission.to_csv("titanic.csv", index=False)