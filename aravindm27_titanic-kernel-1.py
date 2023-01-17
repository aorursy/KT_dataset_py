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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
import pandas_profiling
pandas_profiling.ProfileReport(train_data)
import seaborn as sb
train_data.dtypes
train_data_filtered = train_data.drop(columns=['PassengerId','Ticket','Cabin','Name'])
train_data_filtered["Survived"] = train_data_filtered["Survived"].astype('category').cat.codes
train_data_filtered["Embarked"] = train_data_filtered["Embarked"].astype('category').cat.codes

train_data_filtered["Family"] = train_data_filtered["Parch"] +train_data_filtered["SibSp"]

train_data_filtered['Family'].loc[train_data_filtered['Family'] > 0] = 1

train_data_filtered['Family'].loc[train_data_filtered['Family'] == 0] = 0
train_data_filtered = train_data_filtered.drop(columns = ["Parch","SibSp"])
train_data_filtered
test_data
test_data_filtered = test_data.drop(columns=['PassengerId','Ticket','Cabin','Name'])

test_data_filtered["Embarked"] = test_data_filtered["Embarked"].astype('category').cat.codes

test_data_filtered["Family"] = test_data_filtered["Parch"] +test_data_filtered["SibSp"]

test_data_filtered['Family'].loc[test_data_filtered['Family'] > 0] = 1

test_data_filtered['Family'].loc[test_data_filtered['Family'] == 0] = 0
test_data_filtered = test_data_filtered.drop(columns = ["Parch","SibSp"])
average_age_train   = train_data_filtered["Age"].mean()

std_age_train       = train_data_filtered["Age"].std()

count_nan_age_train = train_data_filtered["Age"].isnull().sum()



# get average, std, and number of NaN values in test_df

average_age_test   = test_data_filtered["Age"].mean()

std_age_test       = test_data_filtered["Age"].std()

count_nan_age_test = test_data_filtered["Age"].isnull().sum()



# generate random numbers between (mean - std) & (mean + std)

rand_1 = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)



# fill NaN values in Age column with random values generated

train_data_filtered["Age"][np.isnan(train_data_filtered["Age"])] = rand_1

test_data_filtered["Age"][np.isnan(test_data_filtered["Age"])] = rand_2



# convert from float to int

train_data_filtered['Age'] = train_data_filtered['Age'].astype(int)

test_data_filtered['Age']    = test_data_filtered['Age'].astype(int)
test_data_filtered
def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

train_data_filtered['Person'] = train_data_filtered[['Age','Sex']].apply(get_person,axis=1)

test_data_filtered['Person']    = test_data_filtered[['Age','Sex']].apply(get_person,axis=1)



# No need to use Sex column since we created Person column

train_data_filtered.drop(['Sex'],axis=1,inplace=True)

test_data_filtered.drop(['Sex'],axis=1,inplace=True)



# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers

person_dummies_titanic  = pd.get_dummies(train_data_filtered['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']



person_dummies_test  = pd.get_dummies(test_data_filtered['Person'])

person_dummies_test.columns = ['Child','Female','Male']



train_data_filtered = train_data_filtered.join(person_dummies_titanic)

test_data_filtered    = test_data_filtered.join(person_dummies_test)
pclass_dummies_titanic  = pd.get_dummies(train_data_filtered['Pclass'])

pclass_dummies_titanic.columns = [1,2,3]



pclass_dummies_test  = pd.get_dummies(test_data_filtered['Pclass'])

pclass_dummies_test.columns = [1,2,3]



train_data_filtered.drop(['Pclass'],axis=1,inplace=True)

test_data_filtered.drop(['Pclass'],axis=1,inplace=True)



train_data_filtered = train_data_filtered.join(pclass_dummies_titanic)

test_data_filtered  = test_data_filtered.join(pclass_dummies_test)
train_data_filtered = train_data_filtered.drop(columns=["Person"])

test_data_filtered = test_data_filtered.drop(columns=["Person"])
train_data_filtered.dropna(inplace=True)

train_data_filtered.reset_index(drop=True, inplace=True)
train_data_filtered
print("The shape of the training data is: " + str(train_data_filtered.shape))
test_data_filtered.dropna(inplace=True)

test_data_filtered.reset_index(drop=True, inplace=True)
test_data_filtered
print("The shape of the test data is: " + str(test_data_filtered.shape))
Q1 = train_data_filtered.quantile(0.25)

Q3 = train_data_filtered.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
train_data_filtered_IQR = train_data_filtered[~((train_data_filtered < (Q1-1.5 * IQR)) |(train_data_filtered > (Q3 + 1.5 * IQR))).any(axis=1)]

train_data_filtered_IQR.shape
train_data_filtered.boxplot(column="Age",by = "Survived",figsize=(5,5))

plt.ylabel("Age")
train_data_filtered
train_data_filtered.boxplot(column="Fare",by = "Survived",figsize=(5,5))

plt.ylabel("Fare")
train_data_filtered.boxplot(column="Embarked",by = "Survived",figsize=(5,5))

plt.ylabel("Embarked")
train_data_filtered_x = train_data_filtered.loc[:,train_data_filtered.columns != "Survived"]
train_data_filtered_y = train_data_filtered.loc[:,train_data_filtered.columns == "Survived"]
import numpy as np

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data_filtered_x, train_data_filtered_y, test_size=0.30, random_state=42)
from sklearn import linear_model

from sklearn.utils.validation import column_or_1d

reg = linear_model.RidgeClassifier()

reg.fit(X_train,column_or_1d(y_train, warn=True))

reg.coef_
y_pred = reg.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=["Dead","Survived"]))
reg_alpha = linear_model.LogisticRegression()

reg_alpha.fit(X_train,column_or_1d(y_train, warn=True))

y_pred_alpha = reg_alpha.predict(X_test)

print(classification_report(y_test, y_pred_alpha, target_names=["Dead","Survived"]))
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import classification_report



h = .02  # step size in the mesh



names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",

         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

         "Naive Bayes", "QDA"]



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="poly", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=50),

    RandomForestClassifier(n_estimators=100),

    MLPClassifier(alpha=1, max_iter=1000),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()]



for name, clf in zip(names, classifiers):

    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    if hasattr(clf, "decision_function"):

        Z = clf.predict(X_test)

    else:

        Z = clf.predict(X_test)



    # Put the result into a color plot

    Z = Z.reshape(y_test.shape)

    print("=====================================================")

    print(name)

    print(classification_report(y_test,Z))

final_model = AdaBoostClassifier()

final_model.fit(X_train, y_train)
test_data = test_data.drop(columns=["Name","Ticket","Cabin"])
def code_embarked(embarked):

    if embarked =="Q":

        embarked = 1

    elif embarked=="C":

        embarked = 0

    elif embarked =="S":

        embarked = 2

    return embarked
test_data["Embarked"] = test_data["Embarked"].apply(code_embarked)


average_age_test_real   = test_data["Age"].mean()

std_age_test_real       = test_data["Age"].std()

count_nan_age_test_real = test_data["Age"].isnull().sum()



# generate random numbers between (mean - std) & (mean + std)

rand_2 = np.random.randint(average_age_test_real - std_age_test_real, average_age_test_real + std_age_test_real, size = count_nan_age_test_real)



test_data["Age"][np.isnan(test_data["Age"])] = rand_2



test_data['Age']    = test_data['Age'].astype(int)
def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex



test_data['Person']    = test_data[['Age','Sex']].apply(get_person,axis=1)



test_data.drop(['Sex'],axis=1,inplace=True)



person_dummies_test_real  = pd.get_dummies(test_data['Person'])

person_dummies_test_real.columns = ['Child','Female','Male']



test_data    = test_data.join(person_dummies_test_real)
pclass_dummies_test_real  = pd.get_dummies(test_data['Pclass'])

pclass_dummies_test_real.columns = [1,2,3]



test_data.drop(['Pclass'],axis=1,inplace=True)



test_data  = test_data.join(pclass_dummies_test_real)
test_data
test_data["Family"] = test_data["Parch"] +test_data["SibSp"]

test_data['Family'].loc[test_data['Family'] > 0] = 1

test_data['Family'].loc[test_data['Family'] == 0] = 0

test_data = test_data.drop(columns = ["Parch","SibSp"])
test_data = test_data.drop(columns=['Person'])
test_data
test_data = test_data.fillna(0)
y_pred = final_model.predict(test_data.drop(columns=["PassengerId"]))
y_pred
result = pd.DataFrame(columns=["PassengerId","Survived"])

result["PassengerId"] = test_data["PassengerId"]

result["Survived"] = y_pred
result.to_csv("result_12_04_2020.csv", index=False)