# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import statements

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, KFold

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

import random

from sklearn.svm import SVC

import sklearn.metrics as sk

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
#change the dataset location

df = pd.read_csv('/kaggle/input/bank-marketing/bank-additional-full.csv', sep = ';')

df.shape
#viewing data

df.head()
#data info

df.info()

#No null values in the data
#Removing non-relevant variables

df1=df.drop(columns=['day_of_week','month','contact','poutcome','pdays'],axis=1)

df1
#Replacing all the binary variables to 0 and 1

df1.y.replace(('yes', 'no'), (1, 0), inplace=True)

df1.default.replace(('yes', 'no'), (1, 0), inplace=True)

df1.housing.replace(('yes', 'no'), (1, 0), inplace=True)

df1.loan.replace(('yes', 'no'), (1, 0), inplace=True)

df1
#creating Dummies for categorical variables

df2 = pd.get_dummies(df1)

df2.head()
#Removing extra dummy variables & checking descriptive stats

df3=df2.drop(columns=['job_unknown','marital_divorced','education_unknown'],axis=1)

df3.describe().T
#Correlation plot

plt.figure(figsize=(14,8))

df3.corr()['y'].sort_values(ascending = False).plot(kind='bar')
#Creating binary classification target variable

df_target=df3[['y']].values

df_features=df3.drop(columns=['y'],axis=1).values

x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.3, random_state = 0)
sc = StandardScaler()

x1_train = sc.fit_transform(x1_train)

x1_test = sc.transform(x1_test)
#Linear SVM

print('Linear Model',end='\n')

lsvclassifier = SVC(kernel='linear')

lsvclassifier.fit(x1_train, y1_train)



#Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = lsvclassifier, X = x1_train, y = y1_train, cv = 5)

mean_svm_linear=accuracies.mean()

std_svm_linear=accuracies.std()



#After using 5 fold cross validation

print('After 5 fold cross validation:')

print('Mean of Accuracies: ',mean_svm_linear*100,end='\n')

print('Standard deviation of Accuracies',std_svm_linear*100,end='\n')



#Predict SVM

y_predl = lsvclassifier.predict(x1_test)



#Confusion Matrix

print('Test Output:')

print('Confusion Matrix:')

print(sk.confusion_matrix(y1_test,y_predl))

print('Classification Report:')

print(sk.classification_report(y1_test,y_predl))

print('Accuracy: ',sk.accuracy_score(y1_test, y_predl, normalize=True, sample_weight=None))
#Polynomial SVM

print('Polynomial Model',end='\n')

psvclassifier = SVC(kernel='poly')

psvclassifier.fit(x1_train, y1_train)



#Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = psvclassifier, X = x1_train, y = y1_train, cv = 5)

mean_svm_poly=accuracies.mean()

std_svm_poly=accuracies.std()



#After using 5 fold cross validation

print('After 5 fold cross validation:')

print('Mean of Accuracies: ',mean_svm_poly*100,end='\n')

print('Standard deviation of Accuracies',std_svm_poly*100,end='\n')



#Predict SVM

y_predp = psvclassifier.predict(x1_test)



#Confusion Matrix

print('Test Output:')

print('Confusion Matrix:')

print(sk.confusion_matrix(y1_test,y_predp))

print('Classification Report:')

print(sk.classification_report(y1_test,y_predp))

print('Accuracy: ',sk.accuracy_score(y1_test, y_predp, normalize=True, sample_weight=None))
#RBF SVM

print('RBF Model',end='\n')

rsvclassifier = SVC(kernel='rbf')

rsvclassifier.fit(x1_train, y1_train)



#Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = rsvclassifier, X = x1_train, y = y1_train, cv = 5)

mean_svm_rbf=accuracies.mean()

std_svm_rbf=accuracies.std()



#After using 5 fold cross validation

print('After 5 fold cross validation:')

print('Mean of Accuracies: ',mean_svm_rbf*100,end='\n')

print('Standard deviation of Accuracies',std_svm_rbf*100,end='\n')



#Predict SVM

y_predr = rsvclassifier.predict(x1_test)



#Confusion Matrix

print('Test Output:')

print('Confusion Matrix:')

print(sk.confusion_matrix(y1_test,y_predr))

print('Classification Report:')

print(sk.classification_report(y1_test,y_predr))

print('Accuracy: ',sk.accuracy_score(y1_test, y_predr, normalize=True, sample_weight=None))
from sklearn.model_selection import ShuffleSplit

from sklearn.svm import SVC

from sklearn.model_selection import learning_curve

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

train_sizes, train_scores, test_scores = learning_curve(rsvclassifier, 

                                                        df_features, 

                                                        df_target,

                                                        # Number of folds in cross-validation

                                                        cv=cv,

                                                        # Evaluation metric

                                                        scoring='accuracy',

                                                        # Use all computer cores

                                                        n_jobs=-1, 

                                                        # 50 different sizes of the training set

                                                        train_sizes=np.linspace(0.01, 1.0, 50))



# Create means and standard deviations of training set scores

train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)



# Create means and standard deviations of test set scores

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



# Draw lines

plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")

plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")



# Draw bands

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")



# Create plot

plt.title("Learning Curve")

plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")

plt.tight_layout()

plt.show()

#Entropy Model

eclassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

eclassifier.fit(x1_train, y1_train)



#Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = eclassifier, X = x1_train, y = y1_train, cv = 5)

mean_dt_e=accuracies.mean()

std_dt_e=accuracies.std()



#After using 5 fold cross validation

print('After 5 fold cross validation:')

print('Mean of Accuracies: ',mean_dt_e*100,end='\n')

print('Standard deviation of Accuracies',std_dt_e*100,end='\n')



#predict y

y_pred = eclassifier.predict(x1_test)



#Confusion Matrix

print('Test Output:')

print('Confusion Matrix:')

print(sk.confusion_matrix(y1_test, y_pred))

print('Classification Report:')

print(sk.classification_report(y1_test, y_pred))

print('Accuracy: ',sk.accuracy_score(y1_test,y_pred))
#Gini Model

gclassifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)

gclassifier.fit(x1_train, y1_train)



#Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = gclassifier, X = x1_train, y = y1_train, cv = 5)

mean_dt_g=accuracies.mean()

std_dt_g=accuracies.std()



#After using 5 fold cross validation

print('After 5 fold cross validation:')

print('Mean of Accuracies: ',mean_dt_g*100,end='\n')

print('Standard deviation of Accuracies',std_dt_g*100,end='\n')



#predict y

y_pred = gclassifier.predict(x1_test)



#Confusion Matrix

print('Test Output:')

print('Confusion Matrix:')

print(sk.confusion_matrix(y1_test, y_pred))

print('Classification Report:')

print(sk.classification_report(y1_test, y_pred))

print('Accuracy: ',sk.accuracy_score(y1_test,y_pred))
#Pruning the better tree - Entropy Tree

parameters = [{'criterion': ['entropy'],'min_samples_leaf':[5,10,20,50,100],'max_depth':[5,10,20,50,100]}] 

grid_search = GridSearchCV(estimator = eclassifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(x1_train, y1_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



print('Accuracy: ',best_accuracy,end='\n')

print('Best Parameters: ',best_parameters,end='\n')
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

train_sizes, train_scores, test_scores = learning_curve(grid_search, 

                                                        df_features, 

                                                        df_target,

                                                        # Number of folds in cross-validation

                                                        cv=cv,

                                                        # Evaluation metric

                                                        scoring='accuracy',

                                                        # Use all computer cores

                                                        n_jobs=-1, 

                                                        # 50 different sizes of the training set

                                                        train_sizes=np.linspace(0.01, 1.0, 50))



# Create means and standard deviations of training set scores

train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)



# Create means and standard deviations of test set scores

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



# Draw lines

plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")

plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")



# Draw bands

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")



# Create plot

plt.title("Learning Curve")

plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")

plt.tight_layout()

plt.show()

# Boosting via Gradient Boost

from sklearn.ensemble import GradientBoostingClassifier

classifiergb = GradientBoostingClassifier(learning_rate=0.01,random_state=1)

classifiergb.fit(x1_train, y1_train)



# Applying k-Fold Cross Validation

accuracies = cross_val_score(estimator = classifiergb, X = x1_train, y = y1_train, cv = 10,n_jobs=-1)

mean_boosting=accuracies.mean()

std_boosting=accuracies.std()



#After using 5 fold cross validation

print('After 5 fold cross validation:')

print('Mean of Accuracies: ',mean_boosting*100,end='\n')

print('Standard deviation of Accuracies',std_boosting*100,end='\n')



# Predicting the Test set results

y_predgb = classifiergb.predict(x1_test)



#Confusion Matrix

print('Test Output:')

print('Confusion Matrix:')

print(sk.confusion_matrix(y1_test, y_predgb))

print('Classification Report:')

print(sk.classification_report(y1_test, y_predgb))

print('Accuracy: ',sk.accuracy_score(y1_test,y_predgb))

#playing around with the pruning to get the best boosting tree

# Applying Grid Search to find the best model and the best parameters

from sklearn.ensemble import AdaBoostClassifier

classifier_AdaBoost = AdaBoostClassifier(random_state=1)

classifier_AdaBoost.fit(x1_train, y1_train)

from sklearn.model_selection import GridSearchCV

parameters = [{'n_estimators': [50,100,200,300,500,1000,1500]}] 

grid_search = GridSearchCV(estimator = classifier_AdaBoost,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(x1_train, y1_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



print('Accuracy: ',best_accuracy,end='\n')

print('Best Parameters: ',best_parameters,end='\n')
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

train_sizes, train_scores, test_scores = learning_curve(classifier_AdaBoost, 

                                                        df_features, 

                                                        df_target,

                                                        # Number of folds in cross-validation

                                                        cv=cv,

                                                        # Evaluation metric

                                                        scoring='accuracy',

                                                        # Use all computer cores

                                                        n_jobs=-1, 

                                                        # 50 different sizes of the training set

                                                        train_sizes=np.linspace(0.01, 1.0, 50))



# Create means and standard deviations of training set scores

train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)



# Create means and standard deviations of test set scores

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



# Draw lines

plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")

plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")



# Draw bands

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")



# Create plot

plt.title("Learning Curve")

plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")

plt.tight_layout()

plt.show()
