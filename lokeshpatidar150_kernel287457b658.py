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
train_data = pd.read_csv(r'/kaggle/input/titanic/train.csv')

test_data = pd.read_csv(r'/kaggle/input/titanic/test.csv')

data = [train_data, test_data]

type(data)
train_data = train_data.drop(['Name','Ticket', 'Cabin'], axis = 1)

test_data = test_data.drop(['Name','Ticket', 'Cabin'], axis = 1)
print("Train Data Null :\n",train_data.isnull().any())

print("Test Data Null :\n",test_data.isnull().any())
#In Training data, embarked feature has some missing value, fill those with the most occurred value ( 'S' )

train_data['Embarked'].fillna('S', inplace = True)



#Age featunre has some missing value in both the data, fill those with mean value. (As this would be most appropriate.)

train_data['Age'].fillna(train_data['Age'].mean(), inplace = True)

test_data['Age'].fillna(test_data['Age'].mean(), inplace = True)



#In Testing data, Fare feature has one missing value, fill it with the mean fare value.

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace = True)



test_data.isnull().values.sum(axis=0)
from sklearn.preprocessing import LabelEncoder,StandardScaler

label = LabelEncoder()

train_data['Sex'] = label.fit_transform(train_data['Sex'])

test_data['Sex'] = label.fit_transform(test_data['Sex'])

train_data['Embarked'] = label.fit_transform(train_data['Embarked'])

test_data['Embarked'] = label.fit_transform(test_data['Embarked'])
#Adding "Family_Size" Column

train_data["Family_Size"] = train_data["SibSp"] + train_data["Parch"]

test_data["Family_Size"] = test_data["SibSp"] + test_data["Parch"]



#Adding "Fare_Per_Person" Column

#train_data["Fare_Per_Person"] = train_data["Fare"]/(train_data["Family_Size"]+1) #Adding 1 to avoid division by Zero

#test_data["Fare_Per_Person"] = test_data["Fare"]/(test_data["Family_Size"]+1)



#Converting Age & Fare feature values into categorical values.

train_data['Age'] = pd.cut(train_data['Age'], 5, labels=False)

test_data['Age'] = pd.cut(test_data['Age'], 5, labels=False)

train_data['Fare'] = pd.qcut(train_data['Fare'], 4, labels=False)

test_data['Fare'] = pd.qcut(test_data['Fare'], 4, labels=False)

PassengerId = test_data['PassengerId']



train_data.head()
# Reorganize headers

train_data = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Family_Size', 'Fare', 'Embarked', 'Survived']]

test_data = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Family_Size', 'Fare', 'Embarked']] # Test data doesn't have target variable 'Survived'.
Y = train_data['Survived']

X = train_data.drop('Survived', axis = 1)

X.head()
#Norm_X = pd.DataFrame(StandardScaler().fit_transform(X))

#Norm_X.head()
Y.head()
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25)

X_test.shape

Y_test.shape
def plot_lc(lbl,c):

    # Create CV training and test scores for various training set sizes

    train_sizes, train_scores, test_scores = learning_curve(c, 

                                                            X_train, 

                                                            Y_train,

                                                            # Number of folds in cross-validation

                                                            cv=10,

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

    plt.title(lbl)

    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")

    plt.tight_layout()

    print(plt.show())
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt



#clf = DecisionTreeClassifier(criterion='gini',

#                            splitter='best',

#                            max_depth=20,

#                            min_samples_split=0.1,

#                            min_samples_leaf= 0.1,

#                            max_features=None).fit(X_train,Y_train)

clf = AdaBoostClassifier().fit(X_train,Y_train)

classifier = [LogisticRegression(), GaussianNB(), RandomForestClassifier(), AdaBoostClassifier(), DecisionTreeClassifier(), SVC(), LinearSVC() ]

lbl = ['LogisticRegression', 'GaussianNB', 'RandomForestClassifier', 'AdaBoostClassifier', 'DecisionTreeClassifier', 'SVC', 'LinearSVC' ]



#for i,c in enumerate(classifier):

#    plot_lc(lbl[i],c)



print("score on training data:", clf.score(X_train,Y_train))

print("Cross Validation Score Analysis :")

y_pred = clf.predict(X_test)

print(classification_report(Y_test, y_pred, labels=[0,1]))

print("Accuracy:",accuracy_score(Y_test, y_pred))



# Plotting ROC Curve

from sklearn import metrics

probs = clf.predict_proba(X_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)



# This is the ROC curve

plt.plot(fpr,tpr)

plt.show() 



# This is the AUC

auc = np.trapz(tpr,fpr)

print('AUC:', auc)


print(test_data.head())

print(test_data.isnull().any())
prediction = clf.predict(test_data)

submission_data = pd.DataFrame({'PassengerId' : PassengerId, 'Survived' : prediction})

submission_data.to_csv(r'submission.csv', index = False)

submission_data.head()
sex_group = train_data.groupby("Survived")

sex_group.mean()