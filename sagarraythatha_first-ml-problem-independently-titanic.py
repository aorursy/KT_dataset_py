# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.metrics import classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/titanic"))
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

result = pd.read_csv("../input/titanic/gender_submission.csv")
train_data.info()
train_data.head()
test_data.info()
# non imp fields which can be dropped because it adds no value in finding things

# passenger id, name, ticket

# drop column: cabin as there are more missing values then present values, in fact 25% values are present

# Do not drop Passengerid from test data set as we need to find who will be survive
train_data = train_data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

test_data = test_data.drop(['Name','Ticket','Cabin'], axis=1)
train_data['Age'].plot()
mean = train_data['Age'].mean()

median = train_data['Age'].median()

std = train_data['Age'].std()

mean, median, std
train_data['Age'].fillna(29, inplace = True)

train_data.isnull().sum()
train_data['Embarked'].value_counts()
train_data['Embarked'].fillna('S', inplace = True)

train_data.isnull().sum()
test_data.info()
test_mean = train_data['Age'].mean()

test_median = train_data['Age'].median()

test_std = train_data['Age'].std()

test_mean, test_median, test_std
test_data['Age'].fillna(29, inplace = True)

test_data.isnull().sum()
train_data['Age'] = train_data['Age'].astype(int)

test_data['Age'] = test_data['Age'].astype(int)
test_data.info()
mean = test_data['Fare'].mean()

median = test_data['Fare'].median()

std = test_data['Fare'].std()

mean, median, std
test_data['Fare'].fillna(36, inplace = True)

test_data.isnull().sum()
train_data['Fare'] = train_data['Fare'].astype(int)

test_data['Fare'] = test_data['Fare'].astype(int)
train_data.head()
train_data.info()
pclass_dummies_titanic_train  = pd.get_dummies(train_data['Pclass'])

pclass_dummies_titanic_train.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic_train.drop(['Class_3'], axis=1, inplace=True)

train_data = train_data.join(pclass_dummies_titanic_train)



pclass_dummies_titanic_test  = pd.get_dummies(test_data['Pclass'])

pclass_dummies_titanic_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic_test.drop(['Class_3'], axis=1, inplace=True)

test_data = test_data.join(pclass_dummies_titanic_test)
train_data.drop(['Pclass'],axis=1,inplace=True)

test_data.drop(['Pclass'],axis=1,inplace=True)
train_data.head()
person_dummies_titanic_train  = pd.get_dummies(train_data['Sex'])

person_dummies_titanic_train.columns = ['Female','Male']

person_dummies_titanic_train.drop(['Male'], axis=1, inplace=True)

train_data = train_data.join(person_dummies_titanic_train)

train_data.drop(['Sex'], axis=1, inplace=True)



person_dummies_titanic_test  = pd.get_dummies(test_data['Sex'])

person_dummies_titanic_test.columns = ['Female','Male']

person_dummies_titanic_test.drop(['Male'], axis=1, inplace=True)

test_data = test_data.join(person_dummies_titanic_test)

test_data.drop(['Sex'], axis=1, inplace=True)

test_data.info()
embarked_dummies_titanic_test  = pd.get_dummies(test_data['Embarked'])

embarked_dummies_titanic_test.columns = ['S','C', 'Q']

embarked_dummies_titanic_test.drop(['Q'], axis=1, inplace=True)

test_data = test_data.join(embarked_dummies_titanic_test)

test_data.drop(['Embarked'], axis=1, inplace=True)



embarked_dummies_titanic_train  = pd.get_dummies(train_data['Embarked'])

embarked_dummies_titanic_train.columns = ['S','C', 'Q']

embarked_dummies_titanic_train.drop(['Q'], axis=1, inplace=True)

train_data = train_data.join(embarked_dummies_titanic_train)

train_data.drop(['Embarked'], axis=1, inplace=True)



test_data.info()

train_data.info()
train_data['S'] = train_data['S'].astype(int)

test_data['S'] = test_data['S'].astype(int)



train_data['C'] = train_data['C'].astype(int)

test_data['C'] = test_data['C'].astype(int)
test_data.info()
x_train = train_data.drop('Survived',axis=1)

y_train = train_data['Survived']

x_test  = test_data.drop("PassengerId",axis=1)

y_test = result['Survived']

x_test.head()
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot

from sklearn.metrics import precision_recall_curve



def draw_graphs(y_pred):

    auc = roc_auc_score(y_test, y_pred)



    # calculate roc curve

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # plot no skill

    pyplot.title('ROC Curve')

    pyplot.plot([0, 1], [0, 1], linestyle='--')

    # plot the roc curve for the model

    pyplot.plot(fpr, tpr, marker='.')

    # show the plot

    pyplot.show()

    

    #----------------------------------

    # Precision Recall curve

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

    pyplot.title('Precison Recall Curve')

    pyplot.plot([0, 1], [0.1, 0.1], linestyle='--')

    # plot the precision-recall curve for the model

    pyplot.plot(recall, precision, marker='.')

    # show the plot

    pyplot.show()





from sklearn.metrics import confusion_matrix

def evaluate_algo(algo, x_train, y_train, y_pred):

    matrices = {'score': algo.score(x_test, y_test),

                'confusion matrix':confusion_matrix(y_test, y_pred), 

                'F1 Score': f1_score(y_test, y_pred, average="macro"), 

                'Precision Score': precision_score(y_test, y_pred, average="macro"),

                'Recall score': recall_score(y_test, y_pred, average="macro"),

                'ROC AUC Score': '%.3f' % (roc_auc_score(y_test, y_pred))

               }

    

    return matrices

    



logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, y_train)

y_pred = logisticRegr.predict(x_test)

print(classification_report(y_test,y_pred))



#draw_graphs(y_pred)

#evaluate_algo(logisticRegr,x_train, y_train, y_pred)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

# feature extraction

test = SelectKBest(score_func=chi2, k=4)

fit = test.fit(x_train, y_train)

# summarize scores

np.set_printoptions(precision=3)

print(fit.scores_)

features = fit.transform(x_train)

# summarize selected features

print(features)
# For feature selection

from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import f_regression

Selector_f = SelectPercentile(f_regression, percentile=25)

Selector_f.fit(x_train,y_train)

Selector_f.scores_

#for n,s in zip(boston.feature_names,Selector_f.scores_):

#    print ('F-score: %3.2ft for feature %s ' % (s,n))
# Recursive Feature Elimination 

# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8



from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, 20)

rfe = rfe.fit(x_train, y_train)

print(rfe.support_)

print(rfe.ranking_)
# Get Correlation Coefficient for each feature using Logistic Regression

coeff_df = pd.DataFrame(train_data.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logisticRegr.coef_[0])



# preview

coeff_df
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)

y_pred = nb.predict(x_test)

print(classification_report(y_test,y_pred))

draw_graphs(y_pred)

evaluate_algo(nb,x_train, y_train, y_pred)
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='squared_hinge', shuffle=True, random_state=101, max_iter=150, early_stopping=False, fit_intercept=True )

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_test)



draw_graphs(y_pred)

evaluate_algo(logisticRegr,x_train, y_train, y_pred)



#try with diff loss function

# ‘hinge’(75.86), ‘log’(73.62), ‘modified_huber’(default - 74.41), ‘squared_hinge’(80.13), ‘perceptron’(74.63)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)



draw_graphs(y_pred)

evaluate_algo(knn,x_train, y_train, y_pred)

#try with different values with neighbors and try to find the optimum value
from sklearn.ensemble import RandomForestClassifier

rfm = RandomForestClassifier(n_estimators = 25, oob_score=True, n_jobs=3, random_state=101, max_features=None, min_samples_leaf=15)

rfm.fit(x_train, y_train)

y_pred = rfm.predict(x_test)



draw_graphs(y_pred)

evaluate_algo(rfm,x_train, y_train, y_pred)
from sklearn.svm import SVC

svm = SVC(kernel='rbf', gamma=0.01, C=10, random_state=101)

svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)



draw_graphs(y_pred)

evaluate_algo(svm,x_train, y_train, y_pred)
