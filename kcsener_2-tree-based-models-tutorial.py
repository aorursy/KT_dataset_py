# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
df.species = [2 if i == 'Iris-setosa' else 1 if i == 'Iris-versicolor' else 0 for i in df.species]
df.head()
X = df.drop('species', axis=1).values  

y = df['species'].values
#Import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier



#Import train_test_split

from sklearn.model_selection import train_test_split



#Import accuracy_score

from sklearn.metrics import accuracy_score
#Split dataset into 80% train, 20% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1) 

#stratify=y: train and test sets have the same proportion of class labels as the unsplit dataset
#Instantiate DecisionTree

dt = DecisionTreeClassifier(max_depth=2, random_state=1)



#random_state=1 for reproducability

#max_depth=2 2 seviye iniyor tree



#criterion='entropy' parametresi ile, decision region belirlemek için split yaparken hangi metodu kullanacağımızı seçiyoruz. default'u 'gini' dir.
#Fit dt to the training set

dt.fit(X_train, y_train)



#Predict test set labels

y_pred =dt.predict(X_test)



#Evaluate test-set accuracy

accuracy_score(y_test, y_pred)
df = pd.read_csv('../input/autompg-dataset/auto-mpg.csv')

df.drop(['car name', 'cylinders', 'model year'], axis=1, inplace=True)

df.replace('?','0', inplace=True)

df.horsepower = df.horsepower.astype('float')
df.head()
df.info()
plt.scatter(df.mpg, df.displacement) 



#aşağıdaki gibi nonlinear bir grafiği linear modeller ile çözümleyemeyiz.
X = df.drop('mpg', axis=1).values  

y = df['mpg'].values
#Import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor



#Import train_test_split

from sklearn.model_selection import train_test_split



#Import accuracy_score

from sklearn.metrics import mean_squared_error as MSE
#Split dataset into 80% train, 20% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3) 
#Instantiate DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=4,min_samples_leaf=0.1, random_state=3)



#random_state=3 for reproducability

#max_depth=4; 4 seviye iniyor tree

#min_samples_leaf: leaf dediğimiz şey decision regionların herbiri. 

#--bu parametre de, herbir leaf'e train datanın minimum 0.1'i gelecek diyor. 0.1'in altına düştüğünde duruyor. 
#Fit dt to the training set

dt.fit(X_train, y_train)



#Predict test set labels

y_pred =dt.predict(X_test)



#Compute test-set MSE

mse_dt = MSE(y_test, y_pred)



#Compute test-set RMSE

rmse_dt = mse_dt**(1/2)



#Print rmse_dt

print(rmse_dt) #bu performans ölçütünü bir de linear regression için yapıp sonuçlar arası farkı görebiliriz.
#Import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor



#Import train_test_split

from sklearn.model_selection import train_test_split



#Import accuracy_score

from sklearn.metrics import mean_squared_error as MSE



#Import cross validation score

from sklearn.model_selection import cross_val_score
#karışıklık çıkmasın diye df'yi baştan alıyoruz

df = pd.read_csv('../input/autompg-dataset/auto-mpg.csv')

df.drop(['car name', 'cylinders', 'model year'], axis=1, inplace=True)

df.replace('?','0', inplace=True)

df.horsepower = df.horsepower.astype('float')
df.head()
df.info()
X = df.drop('mpg', axis=1).values  

y = df['mpg'].values
#Split dataset into 70% train, 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) 
#Instantiate DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=4,min_samples_leaf=0.14, random_state=123)
#Evaluate the list of MSE obtained by 10-fold CV

#Set n_jobs to -1 in order to exploit all CPU cores in computation

# neg_mean_squared_error negative mse yap diyoruz metod olarak. bunun sebebi, cross val ile mse hesabının direkt oalrak yapılamaması

MSE_CV = - cross_val_score(dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1) 
#Fit dt to the training set

dt.fit(X_train, y_train)



#Predict the labels of the training set

y_predict_train = dt.predict(X_train)



#Predict the labels of the test set

y_predict_test = dt.predict(X_test)
# CV MSE:

print(MSE_CV.mean())
#Training Set MSE:

print(MSE(y_train, y_predict_train))

print()



#Test Set MSE:

print(MSE(y_test, y_predict_test))
# Training Set Error: 13.65

# Test Set Error    : 22.00

# CV Error          : 16.72

# TrainingSetError < CVError: high variance(overfit): model complexity'yi düşür (decrease max_depth, increase min_samples_leaf...)
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
df.diagnosis = [1 if i == 'M' else 0 for i in df.diagnosis]
df.head()
X = df.drop('diagnosis', axis=1).values  

y = df['diagnosis'].values
#import functions to compute accuracy and split data

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



#import models, including VotingClassifier as meta-model

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.ensemble import VotingClassifier
#set seed for reproducability

SEED = 1
#Split dataset into 70% train, 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
#Instantiate individual classifiers

lr = LogisticRegression(random_state=SEED)

knn = KNN()

dt = DecisionTreeClassifier(random_state=SEED)
#define a list called classifier that contains tupples (classifier_name, classifier)

classifiers = [('Logistic Regression', lr),

              ('K Nearest Neighbours', knn),

              ('Classification Tree', dt)]
#we can now write a for loop to iterate over the classifiers

for clf_name, clf in classifiers:

    #fit clf to the training set

    clf.fit(X_train, y_train)

    

    #predict the labels of the test set

    y_pred = clf.predict(X_test)

    

    #evaluate the accuracy of clf o the test set

    print(clf_name, ':', accuracy_score(y_test, y_pred))

    

#en iyi sonucu lr verdi.
#Instantiate a VotingClassifier 

vc = VotingClassifier(estimators=classifiers)



#fit vc to the training set and predict test set labels

vc.fit(X_train, y_train)

y_pred = vc.predict(X_test)



#evaluate the test-accuracy score of vc

print('Voting Classifier:', accuracy_score(y_test, y_pred))



#bu sonuç modelleri ayrı ayrı çözdüğümüzde çıkan sonuçlardan daha fazla.
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
df.diagnosis = [1 if i == 'M' else 0 for i in df.diagnosis]
X = df.drop('diagnosis', axis=1).values  

y = df['diagnosis'].values
#import functions to compute accuracy and split data

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



#import models, including BaggingClassifier as meta-model

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier
#set seed for reproducability

SEED = 1
#Split dataset into 70% train, 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
#Instantiate DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16 ,random_state=SEED)



#Instantiate BaggingClassifier

bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1) #300 tree var; n_jobs=-1 so that all CPU cores are used in calculation

#fit bc to the training set

bc.fit(X_train, y_train)



#predict test labels

y_pred = bc.predict(X_test)



#evaluate test-set accuracy

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))



#normalde dt'yi bagging yapmadan uyguladığımızda 0.88 gibi bişey çıkıyormuş.

#bagging ile dt'nin performansını artırmış olduk.
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
df.diagnosis = [1 if i == 'M' else 0 for i in df.diagnosis]
X = df.drop('diagnosis', axis=1).values  

y = df['diagnosis'].values
#import functions to compute accuracy and split data

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



#import models, including BaggingClassifier as meta-model

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier
#set seed for reproducability

SEED = 1
#Split dataset into 70% train, 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
#Instantiate DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16 ,random_state=SEED)



#Instantiate BaggingClassifier

bc = BaggingClassifier(base_estimator=dt, n_estimators=300, oob_score=True, n_jobs=-1) #300 tree var; n_jobs=-1 so that all CPU cores are used in calculation

#ayrıca oob_score=True -> oob score hesaplayabilmek için
#fit bc to the training set

bc.fit(X_train, y_train)



#predict test labels

y_pred = bc.predict(X_test)
#evaluate test-set accuracy

test_accuracy = accuracy_score(y_test, y_pred)



#evaluate OOB accuracy from bc

oob_accuracy = bc.oob_score_



print('Accuracy of Bagging Classifier: {:.3f}'.format(test_accuracy))

print('OOB Accuracy of Bagging Classifier: {:.3f}'.format(oob_accuracy))

#oob accuracy ile cross validation yapmadan bagging ensemble modeli performans tahmini yapabiliriz.
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
df.diagnosis = [1 if i == 'M' else 0 for i in df.diagnosis]
X = df.drop('diagnosis', axis=1).values  

y = df['diagnosis'].values
#import functions to compute accuracy and split data

from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import train_test_split



#import model as meta-model

from sklearn.ensemble import RandomForestRegressor
#set seed for reproducability

SEED = 1
#Split dataset into 70% train, 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
#Instantiate random forest regressor with 400 estimators

rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=SEED) #400 regression trees
#fit rf to the training set

rf.fit(X_train, y_train)



#predict test labels

y_pred = rf.predict(X_test)
#Evaluate RMSE

rmse_test = MSE(y_test, y_pred)**(1/2)



print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

#single regression tree'den daha düşük bir değere ulaşmışız.
#create a pd.series of features importances

importances_rf = pd.Series(rf.feature_importances_, index= df.drop('diagnosis', axis=1).columns) #index aslında X, ama array olmayacağı için values'İ çıkardık.



#sort importances_rf

sorted_importances_rf = importances_rf.sort_values()



#make horizontal plot

sorted_importances_rf.plot(kind='barh', color='lightgreen')

plt.show()
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
df.diagnosis = [1 if i == 'M' else 0 for i in df.diagnosis]
X = df.drop('diagnosis', axis=1).values  

y = df['diagnosis'].values
#import functions to compute accuracy and split data

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split



#import model as meta-model

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
#set seed for reproducability

SEED = 1
#Split dataset into 70% train, 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
#Instantiate DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)
#Instantiate AdaBoostClassifier

adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
#fit adb_clf to the training set

adb_clf.fit(X_train, y_train)



#predict test set probabilities of positive class

y_pred_proba = adb_clf.predict_proba(X_test)[:,1]
#evaluate test set roc auc score

adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba) #roc_auc_score için y_pred_proba gerekliydi
print('Test set roc auc score of dt: {:.2f}'.format(adb_clf_roc_auc_score))
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
df.diagnosis = [1 if i == 'M' else 0 for i in df.diagnosis]
X = df.drop('diagnosis', axis=1).values  

y = df['diagnosis'].values
#import functions to compute accuracy and split data

from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import train_test_split



#import model as meta-model

from sklearn.ensemble import GradientBoostingClassifier
#set seed for reproducability

SEED = 1
#Split dataset into 70% train, 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
#Instantiate GradientBoostingClassifier

gbt = GradientBoostingClassifier(n_estimators=300, max_depth=1, random_state=SEED)
#fit gbt to the training set

gbt.fit(X_train, y_train)



#predict the test set labels

y_pred = gbt.predict(X_test)
#evaluate test set

rmse_test = MSE(y_test, y_pred)**(1/2)



print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

df.diagnosis = [1 if i == 'M' else 0 for i in df.diagnosis]
X = df.drop('diagnosis', axis=1).values  

y = df['diagnosis'].values
#import functions to compute accuracy and split data

from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import train_test_split



#import model as meta-model

from sklearn.ensemble import GradientBoostingRegressor
#set seed for reproducability

SEED = 1
#Split dataset into 70% train, 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
#Instantiate STOCHASTiC GradientBoostingClassifier

sgbt = GradientBoostingRegressor(max_depth=1, subsample=0.8, max_features=0.2, n_estimators=300, random_state=SEED)

#subsample=0.8 -> each tree to sample 80% of the data for training

#max_features=0.2 -> each tree uses 20% of available features to perform the best-split
#fit sgbt to the training set

sgbt.fit(X_train, y_train)



#predict the test set labels

y_pred = sgbt.predict(X_test)
#evaluate test set

rmse_test = MSE(y_test, y_pred)**(1/2)



print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

df.diagnosis = [1 if i == 'M' else 0 for i in df.diagnosis]
X = df.drop('diagnosis', axis=1).values  

y = df['diagnosis'].values
from sklearn.model_selection import train_test_split

#Split dataset into 70% train, 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
#import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
#set seed for reproducability

SEED = 1
#Instantiate a DecisionClassifier dt

dt = DecisionTreeClassifier(random_state=SEED)
#print out dt's hyperparameters:

print(dt.get_params())

#biz sadece max_depth, max_features ve min_samples_leaf i optimize edelim.

#max_feature: nr of features to consider when looking for the best split

#Import GridSearchCV 

from sklearn.model_selection import GridSearchCV



#define the grid of hyperparameters 

params_dt = {'max_depth': [3,4,5,6],

             'min_samples_leaf': [0.04, 0.06, 0.08],

             'max_features': [0.2, 0.4, 0.6, 0.8]}



#Instantiate a 10-fold CV grid search object 

grid_dt = GridSearchCV(estimator=dt, 

                       param_grid=params_dt,

                       scoring='accuracy',

                       cv=10,

                       n_jobs=-1)



#fit grid_dt to the training set

grid_dt.fit(X_train, y_train)



#extract best hyperparameters from 'grid_dt'

best_hyperparams = grid_dt.best_params_

print('best hyperparameters:', best_hyperparams)



#extract best CV score from grid_dt

best_CV_score = grid_dt.best_score_

print('best CV score:', best_CV_score)



#extract best model from grid_dt

best_model = grid_dt.best_estimator_

print('best model:', best_model)



#evaluate test set accuracy

test_acc = best_model.score(X_test, y_test)

print('test set accuracy of best model:', test_acc)

df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

df.diagnosis = [1 if i == 'M' else 0 for i in df.diagnosis]
X = df.drop('diagnosis', axis=1).values  

y = df['diagnosis'].values
#Import RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
#set seed for reproducability

SEED = 1
from sklearn.model_selection import train_test_split

#Split dataset into 70% train, 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
#Instantiate RandomForestRegressor

rf = RandomForestRegressor(random_state=SEED)
#Inspect rf's hyperparameters

rf.get_params()

#we will optimize n_estimators, max_depth, min_samples_leaf, max_features
#Import GridSearchCV and metric MSE

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error as MSE



#define the grid of hyperparameters 

params_rf = { 'n_estimators': [300, 400, 500],

              'max_depth': [4,6,8],

              'min_samples_leaf': [0.1, 0.2],

              'max_features': ['log2', 'sqrt']}



#Instantiate a 3-fold CV grid search object 

grid_rf = GridSearchCV(estimator=rf, 

                       param_grid=params_rf,

                       scoring='neg_mean_squared_error', #negative mse

                       cv=3,

                       verbose=1, #verbose: gereksiz sözlerle dolu demek, verbosity'yi kontrol etmek içinmiş

                       n_jobs=-1)



#fit grid_dt to the training set

grid_rf.fit(X_train, y_train)
#extract best hyperparameters

best_hyperparams = grid_rf.best_params_

print('best parameters: \n',best_hyperparams)



#extract best model 

best_model = grid_rf.best_estimator_

print('\nbest model: \n',best_model)
#predict the test set labels

y_pred = best_model.predict(X_test)



#Evaluate the test set RSME

rsme_test = MSE(y_test, y_pred)**(1/2)

print('RSME of test set:{:.2f}'.format(rsme_test) )