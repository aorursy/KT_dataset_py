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
import pandas as pd

import numpy as np

import random as rnd3

from sklearn.impute import SimpleImputer



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score, cross_validate, ShuffleSplit, GridSearchCV

from sklearn.utils import resample

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import AdaBoostClassifier
df = pd.read_csv("/kaggle/input/titanic/train.csv")

df.info()

df.describe()

df.head()


# Title - retreving it from the name



df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')

pd.crosstab(df['Title'], df['Sex']) #some titles are rare - so bucket them

df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace('Ms', 'Miss')

df['Title'] = df['Title'].replace('Mme', 'Mrs')

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()  #survival rate by the title







# Age - impute the nulls (263 nulls), bucket



guess_age = df.groupby(['Pclass','Sex']).median()

df["Age"].fillna(df.groupby(['Pclass','Sex'])["Age"].transform("median"), inplace=True)

df["Age"].value_counts()

bins = [0,16,32,48,64,80] #to get the limits pd.cut(df['Age'], 5)

labels = ['0-16','17-32','33-48','49-64','64-80']

df['AgeBand'] =pd.cut(df['Age'], bins = bins, labels= labels) ## Age into band of 5

df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()  #survival rate by the title





# Fare  - impute the nulls (1 null), bucket

df["Fare"].fillna(df.groupby(['Pclass','Sex'])["Fare"].transform("median"), inplace=True)

bins_fare = [-1,8,14,31,513] #to get the limits pd.qcut(df['Fare'], 4)

labels_fare = ['0-8','9-14','15-31','32-513']

df['FareBand'] =pd.cut(df['Fare'], bins = bins_fare, labels= labels_fare) 

df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean()  #survival rate by the title





# Embarked  - impute the nulls (2 null), categorical so most frequent



df['Embarked'].value_counts()

imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

df["Embarked"] = imp.fit_transform(df[["Embarked"]]).ravel()

df['Embarked'].value_counts()





# New features



#Family Size

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()  #survival rate by the title



#is alone?





df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x==1 else 0 )

df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()  #survival rate by the title





df.info()

df.head()
## ENCODING

## Transforming categorical values to ordinal values

## Sex, AgeBand, Title, Embarked





encoding = {

        

            "Sex":     {"female": 0, "male": 1},

            "AgeBand":     {"0-16": 0, "17-32": 1,"33-48": 2,"49-64": 3, "64-80": 4},

            "FareBand":     {"0-8": 0, "9-14": 1,"15-31": 2,"32-513": 3},

            "Embarked":     {"S": 0, "C": 1,"Q": 2},

            "Title":     {"Mr": 0, "Miss": 1,"Mrs": 2,"Master": 4, "Rare":5}

        }





df.replace(encoding, inplace=True)
df.head()
df_train = df[(df['Type']=='Train')]

X = df_train[['Pclass','Sex','AgeBand','FareBand','Embarked','Title','IsAlone']]

np.any(np.isnan(X))## To check if anything is a NULL - if it is a null model will throw an error!



Y = df_train['Survived']

np.any(np.isnan(Y))## To check if anything is a NULL - if it is a null model will throw an error!





df_sumbission = df[(df['Type']=='Test')]

X_sumbission = df_sumbission[['Pclass','Sex','AgeBand','FareBand','Embarked','Title','IsAlone']]

np.any(np.isnan(X_sumbission))## To check if anything is a NULL - if it is a null model will throw an error!




## Logistic Regression



logreg = LogisticRegression()

'''

penalty = 'none', solver = 'newton-cg',C=1 



'''





logreg.fit(X, Y)

Y_pred = logreg.predict(X)

logreg.coef_

'''

array([[-0.98213819, -2.199304  , -0.0257264 ,  0.00286028,  0.27445946,

         0.3443201 ,  0.36350466]])

'''

logreg.intercept_

'''

array([ 3.07642252])

'''





    #Classification Report

print(classification_report(Y,Y_pred))

    #Accuracy

print(round(logreg.score(X, Y) * 100, 2)) #Get accuracy by placing training X and Y. It will predict the Y and then compare it with the training Y 

    #Confusion Matrix

print(confusion_matrix(Y, Y_pred))

    #Submission

    

logreg.get_params()    #all the parameters used for the LogisticRegression()

logreg.set_params()    #all the parameters used for the LogisticRegression()

logreg.sparsify()

    

Y_sumbission = logreg.predict(X_sumbission) ## This is what we are tested with, Y_pred needs to be uploaded to Kaggle 


X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    Y, test_size=0.30, 

                                                    random_state=101)

logreg = LogisticRegression(penalty = 'none', solver = 'newton-cg' )

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)



    #Classification Report

print(classification_report(y_test,Y_pred))

    #Accuracy

print(round(logreg.score(X_test, y_test) * 100, 2)) #Get accuracy by placing X and Y. It will predict the Y and then compare it with the training Y 

    #Confusion Matrix

print(confusion_matrix(y_test, Y_pred))



df_majority = df_train[df_train.Survived==0]

df_minority = df_train[df_train.Survived==1]



df_majority.describe()  #549 - not survived

df_minority.describe()  #342 - survived





#can do this using Scikit Learn as well...

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=549,    # to match majority class

                                 random_state=123) # reproducible results



df_upsampled = pd.concat([df_majority, df_minority_upsampled])



df_upsampled.Survived.value_counts()



X = df_upsampled[['Pclass','Sex','AgeBand','FareBand','Embarked','Title','IsAlone']]

Y = df_upsampled['Survived']





X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    Y, test_size=0.30, 

                                                    random_state=1011)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)



    #Classification Report

print(classification_report(y_test,Y_pred))

    #Accuracy

print(round(logreg.score(X_test, y_test) * 100, 2)) #Get accuracy by placing X and Y. It will predict the Y and then compare it with the training Y 

    #Confusion Matrix

print(confusion_matrix(y_test, Y_pred))

# primer on k-fold split - how the splitting happens



scores = []

logreg = LogisticRegression()



kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X):

    print("%s %s" % (train_index, test_index))

    X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[test_index], Y.loc[train_index], Y.loc[test_index]

    logreg.fit(X_train, y_train)

    scores.append(logreg.score(X_test, y_test))



df_score= pd.DataFrame(scores) 

scores


poly = PolynomialFeatures(2) #Modify this to change the order of the polynomial



'''

Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the 

specified degree. For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features 

are [1, a, b, a^2, ab, b^2].

'''

X = poly.fit_transform(X)



scores = []

logreg = LogisticRegression()

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(X):

    print("%s %s" % (train_index, test_index))

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

    logreg.fit(X_train, y_train)

    scores.append(logreg.score(X_test, y_test))







#Another way to get the cross validation score without doing the split manually (cv = integer - means it is k-fold)

print(cross_val_score(logreg, X, Y, cv=10)) #accuracy by deault

print(cross_val_score(logreg, X, Y, cv=10, scoring = 'f1')) #accuracy by deault

#Shuffle splitting - closest to Bootstrapping

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

cross_val_score(logreg, X, Y, cv=cv)  

import statsmodels.api as sm

from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df) #can't recollect why i did this

logit = sm.Logit(Y,X)

result = logit.fit()

result.summary()
'''

**For Pclass**





odds(survival/class+1)

______________________    =    e^-0.4110 = 0.662987   = odds ratio - see the table below

odds(survival/class)



'''
np.exp(result.params)
'''

Pclass      0.662988 - increasing a class by 1 (can be from 1 to 2 or 2 to 3), decreases the odds of survival by 24%

Sex         0.118074 - being a male (when compared to being a female), decreases the odd of survival by 89%

AgeBand     0.834307 - the more older you are the lesser odds of survival

FareBand    1.693142 - the more you paid, the higher odds of survival

Embarked    1.482920

Title       1.491672

IsAlone     2.627665 - 



'''
svc = SVC()

svc.fit(X, Y)

Y_pred = svc.predict(X)

acc_svc = round(svc.score(X, Y) * 100, 2)

print(classification_report(Y,Y_pred))

acc_svc
'''Random Forest - model 1'''

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=101)

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train,y_train)

Y_pred = random_forest.predict(X_test)

print(classification_report(y_test,Y_pred))

print(confusion_matrix(y_test, Y_pred))







# Feature importance

importances = list(random_forest.feature_importances_)

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(['Pclass','Sex','AgeBand','FareBand','Embarked','Title','IsAlone'], importances)]

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



'''Random Forest - model 2'''

random_forest = RandomForestClassifier(n_estimators=200, oob_score = True , n_jobs = -1,random_state =50,max_features = "auto", min_samples_leaf = 50)    

random_forest.fit(X_train,y_train)

Y_pred = random_forest.predict(X_test)

print(classification_report(y_test,Y_pred))

print(confusion_matrix(y_test, Y_pred))
'''Random Forest - model 3'''



random_forest = RandomForestClassifier(n_estimators=10)

param_grid = { 

    'n_estimators': [200, 700],

    'max_features': ['auto', 'sqrt', 'log2']

}

7865

CV_rfc = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv= 5, n_jobs = -1)

CV_rfc.fit(X_train,y_train)

Y_pred = CV_rfc.predict(X_test)

print(classification_report(y_test,Y_pred))

print(confusion_matrix(y_test, Y_pred))

CV_rfc.best_params_
abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)

model = abc.fit(X_train, y_train)

Y_pred = abc.predict(X_test)

print(classification_report(y_test,Y_pred))

print(confusion_matrix(y_test, Y_pred))
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X, Y)

Y_pred = knn.predict(X)



print(classification_report(Y,Y_pred))

gaussian = GaussianNB()

gaussian.fit(X, Y)

Y_pred = gaussian.predict(X)

print(classification_report(Y,Y_pred))
perceptron = Perceptron()

perceptron.fit(X, Y)

Y_pred = perceptron.predict(X)

    

print(classification_report(Y,Y_pred))
sgd = SGDClassifier()

sgd.fit(X,Y)

Y_pred = sgd.predict(X)

print(classification_report(Y,Y_pred))
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X, Y)

Y_pred = decision_tree.predict(X)

print(classification_report(Y,Y_pred))