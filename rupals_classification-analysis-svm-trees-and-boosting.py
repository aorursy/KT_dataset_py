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

df1 = pd.read_csv('/kaggle/input/gpu-runtime/sgemm_product.csv')

df = df1.sample(frac=0.4) #reducing data size for faster computation

df.shape
#creating Runtime, target variable by taking average of Run1, Run2, Run3, Run4

df['Runtime']=df[['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)']].mean(axis=1)
#viewing data

df.head()
#drop other Run time variables

df1=df.drop(columns =['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'], axis = 1)

df1.info()
#checking descriptive stats

df1.describe().T
#checking for NULL values

df1.isnull().sum() #no NULL values
#checking for outliers

plt.figure(figsize=(10,6))

sns.boxplot(df1['Runtime']);
#removing outliers

Q1=df1['Runtime'].quantile(0.25)

Q2=df1['Runtime'].quantile(0.75)

IQR = Q2 - Q1

LL=Q1-1.5*IQR

UL=Q2+1.5*IQR

df2 = df1[(df1.Runtime>LL) & (df1.Runtime<UL)]

df2.describe().T
plt.figure(figsize=(10,6))

sns.boxplot(df2['Runtime']);
#checking variable distribution

for index in range(10):

   df2.iloc[:,index] = (df2.iloc[:,index]-df2.iloc[:,index].mean()) / df2.iloc[:,index].std();

df2.hist(figsize= (14,16));
#plotting the distribution of Runtime

sns.distplot(df2['Runtime'])
df2['target']=np.log(df2.Runtime)

sns.distplot(df2['target'])
plt.figure(figsize=(14,14))

sns.set(font_scale=1)

sns.heatmap(df2.corr(),cmap='GnBu_r',annot=True, square = True ,linewidths=.5);

plt.title('Variable Correlation')
#Creating binary classification target variable

mean = df2['target'].mean()

df2.loc[df2['target'] <= mean, 'target'] = 0

df2.loc[df2['target'] > mean, 'target'] = 1

df_target=df2[['target']].values

df_features=df2.drop(columns=['target','Runtime'],axis=1).values

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
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):

  



    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs,scoring='accuracy', train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



from sklearn.model_selection import ShuffleSplit

from sklearn.svm import SVC

from sklearn.model_selection import learning_curve



title = r"Learning Curves (SVM, RBF kernel, $\gamma=auto$)"

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

#estimator = SVC(kernel = 'rbf', random_state = 0,gamma='auto')

plot_learning_curve(rsvclassifier, title, df_features, df_target, (0.8, 1.1), cv=cv)

plt.show()
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
#Pruning the better tree - Gini Tree

parameters = [{'criterion': ['gini'],'min_samples_leaf':[5,10,20,30,50,100],'max_depth':[1,5,10,20,50,100]}] 

grid_search = GridSearchCV(estimator = gclassifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(x1_train, y1_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



print('Accuracy: ',best_accuracy,end='\n')

print('Best Parameters: ',best_parameters,end='\n')
from sklearn.model_selection import ShuffleSplit

from sklearn.svm import SVC

from sklearn.model_selection import learning_curve

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

train_sizes, train_scores, test_scores = learning_curve(gclassifier, 

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
from sklearn.model_selection import learning_curve



train_sizes, train_scores, test_scores = learning_curve(classifier_AdaBoost, df_features, df_target,cv=3,n_jobs=-1)

train_sizes 

train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)

plt.grid()



plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



plt.legend(loc="best")

plt.show
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
