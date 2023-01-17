# import libraries

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import os

%matplotlib inline 



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        filevar = os.path.join(dirname, filename)



# import dataset

try: 

    df = pd.read_csv(filevar)

    print('File loading - Success!')

except:

    print('File loading - Failed!')
# sample dataframe content

df
df.info()
sepal = sns.FacetGrid(df,col='species')

sepal.map(plt.scatter,'sepal_width','sepal_length',alpha=0.7)



petal = sns.FacetGrid(df,col='species')

petal.map(plt.scatter,'petal_width','petal_length',alpha=0.7)
sns.pairplot(df,hue='species')
from sklearn.model_selection import train_test_split



X = df.drop(columns='species',axis=1)

y = df.species



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2) # 20% of samples reserved for test

print(X_train.shape)  # 120 records out of 150 will be the training data

print(X_test.shape)   #  30 records out of 150 will be the testing  data
from sklearn.model_selection import cross_val_score



def crossValScore(algorithm,X,y):

    """ Input: classifier algorithm, X features, y target

        Returns: mean accuracy of the classifier algorithm via cross validation scoring

    """

    clf = algorithm

    return cross_val_score(clf,X,y,cv=10,scoring='accuracy').mean() 
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



dec_tree_clf = DecisionTreeClassifier()



# fit and predict using train test data

dec_tree_clf.fit(X_train,y_train)

y_pred = dec_tree_clf.predict(X_test)



# evaluate prediction

print('CLF REPORT:\n',classification_report(y_test,y_pred))

print('\nCONFUSION MATRIX:\n',confusion_matrix(y_test,y_pred))

print('\nTTS ACCURACY:\n',accuracy_score(y_test,y_pred))



# fit, predict and evaluate using cross validation

dec_tree_score = crossValScore(dec_tree_clf,X,y)

print('\nCV ACCURACY:\n',dec_tree_score)
from sklearn.linear_model import LogisticRegression



log_reg_clf = LogisticRegression(solver='liblinear')



# fit and predict using train test data 

log_reg_clf.fit(X_train,y_train)

y_pred = log_reg_clf.predict(X_test)



# evaluate prediction

print('CLF REPORT:\n',classification_report(y_test,y_pred))

print('\nCONFUSION MATRIX:\n',confusion_matrix(y_test,y_pred))

print('\nTTS ACCURACY:\n',accuracy_score(y_test,y_pred))



# fit, predict and evaluate using cross validation

log_reg_score = crossValScore(log_reg_clf,X,y)

print('\nCV ACCURACY:\n',log_reg_score)
from sklearn.svm import SVC



svm_clf = SVC()



# fit and predict using train test data 

svm_clf.fit(X_train,y_train)

y_pred = svm_clf.predict(X_test)



# evaluate prediction

print('CLF REPORT:\n',classification_report(y_test,y_pred))

print('\nCONFUSION MATRIX:\n',confusion_matrix(y_test,y_pred))

print('\nTTS ACCURACY:\n',accuracy_score(y_test,y_pred))



# fit, predict and evaluate using cross validation

svm_score = crossValScore(svm_clf,X,y)

print('\nCV ACCURACY:\n',svm_score)
from sklearn.naive_bayes import GaussianNB



nb_clf = GaussianNB()



# fit and predict using train test data

nb_clf.fit(X_train,y_train)

y_pred = nb_clf.predict(X_test)



# evaluate prediction

print('CLF REPORT:\n',classification_report(y_test,y_pred))

print('\nCONFUSION MATRIX:\n',confusion_matrix(y_test,y_pred))

print('\nTTS ACCURACY:\n',accuracy_score(y_test,y_pred))



# fit, predict and evaluate using cross validation

nb_score = crossValScore(nb_clf,X,y)

print('\nCV ACCURACY:\n',nb_score)
X_new = pd.DataFrame(np.array([[5.8,3.5,1.4,0.4],[6.9,3.0,4.8,1.7],[7.6,3.0,6.6,2.5]]),columns=list(X.columns))



# new dataset to be predicted

X_new
# new species prediction result

X_new['species'] = svm_clf.predict(X_new)

X_new