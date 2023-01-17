# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib as mpl

import matplotlib.pyplot as plt

import os

import sys

import seaborn as sns

%matplotlib inline



np.random.seed(42)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Getting de training set

Train_set = pd.read_csv("/kaggle/input/titanic/train.csv")

Train_set.head()
# Getting de testing set

Test_set = pd.read_csv("/kaggle/input/titanic/test.csv")

Test_set.head()
Train_set.isnull().sum()
Train_set.groupby("Embarked").count()
mean_age = round(Train_set["Age"].mean())

Train_set["Age"].fillna(mean_age, inplace = True)



Train_set["Embarked"].fillna("S", inplace = True)



Train_set = Train_set.dropna(axis = 1)



Train_set.isnull().sum()
len(Train_set.groupby("Ticket").count())
dummy_sex = pd.get_dummies(Train_set.Sex)

dummy_embarked = pd.get_dummies(Train_set.Embarked)

Train_set_num = pd.concat([Train_set, dummy_sex["male"], dummy_embarked[['C', 'Q']]], axis = 1)



pd.pivot_table(Train_set_num, index = "male", values = "Survived", aggfunc = np.mean)

# Very high correlation between being woman and surviving or being man and dying
pd.pivot_table(Train_set_num, index = ['C', 'Q'], values = "Survived", aggfunc = np.mean)

# Exists correlation between the port and surviving
Train_set_num.drop(["Ticket", "Name", "Sex", "Embarked"], inplace = True, axis = 1)

Train_set_num.head()
plt.figure(figsize=(12,10))

cor = Train_set_num.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
sns.boxplot(data = Train_set)
sns.boxplot(data = Train_set.Fare)
Train_set_Fare100 = Train_set[Train_set.Fare > 100]

Train_set_Fare100[["Fare", "Survived"]].groupby("Fare").mean()
fare_classes = []

for fare in Train_set["Fare"]:

    if fare < 10:

        fare_classes.append('0')

    elif fare < 25:

        fare_classes.append('1')

    elif fare < 55:

        fare_classes.append('2')

    elif fare < 70:

        fare_classes.append('3')

    elif fare < 100:

        fare_classes.append('4')

    else:

        fare_classes.append('5')

        

Train_set["Fare_class"] = fare_classes

Train_set.head()
Train_set[["Fare_class", "Survived"]].groupby("Fare_class").mean()
sns.boxplot(data = Train_set.Age)
age_classes = []

for age in Train_set["Age"]:

    if age < 16:

        age_classes.append('0')

    elif age < 32:

        age_classes.append('1')

    elif age < 48:

        age_classes.append('2')

    elif age < 70:

        age_classes.append('3')

    else:

        age_classes.append('4')

        

Train_set["Age_class"] = age_classes

Train_set.head()
Train_set[["Age_class", "Survived"]].groupby("Age_class").mean()
Train_set[["SibSp", "Survived"]].groupby("SibSp").mean()
Train_set[["Parch", "Survived"]].groupby("Parch").mean()
Train_set["Family_members"] = Train_set["SibSp"] + Train_set["Parch"]
family_members = []

for number in Train_set["Family_members"]:

    if number < 3:

        family_members.append('a')

    elif number < 6:

        family_members.append('b')

    elif number < 9:

        family_members.append('c')

    elif number < 11:

        family_members.append('d')

    else:

        family_members.append('e')

        

Train_set["Family_class"] = family_members

Train_set.head()
Train_set[["Family_class", "Survived"]].groupby("Family_class").mean()
dummy_fare = pd.get_dummies(Train_set.Fare_class, prefix = "fare")

dummy_age = pd.get_dummies(Train_set.Age_class, prefix = "age")

dummy_family = pd.get_dummies(Train_set.Family_class, prefix = "family")



Train_set_num = pd.concat([Train_set_num, dummy_fare, dummy_age, dummy_family], axis = 1)



Train_set_num.columns
Train_set_num.drop(["Age", "Fare", "SibSp", "Parch", "PassengerId"], inplace = True, axis = 1)

Train_set_num.head()
Cols = Train_set_num.columns

feature_cols = Cols.drop("Survived")



X_train = Train_set_num[feature_cols] 

y_train = Train_set_num.Survived
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



max_score = -1

best_degree = -1

    

for degree in range(1,6):



    

    poly_features = PolynomialFeatures(degree = degree)

    X_poly = poly_features.fit_transform(X_train)

    lrc = LogisticRegression(C = 5, solver = 'liblinear', max_iter = 1000)



    lrc.fit(X_poly, y_train)

    

    train_score = lrc.score(X_poly, y_train)

    

    print("Polynomial of grade: {}".format(degree))

    print("Score on the test set: {}".format(train_score))

    

    if train_score >= max_score:

        max_score = train_score

        best_degree = degree

        

print()

print("The polynomial degree with which the score is the highest is: {}".format(best_degree))
from sklearn.svm import LinearSVC



linear_svc = Pipeline([

    ("scaler", StandardScaler()),

    ("svc", LinearSVC(C = 1, loss = "hinge")),

])

linear_svc.fit(X_train.astype("float64"), y_train)



print("Score on the test set: {}".format(linear_svc.score(X_train.astype("float64"), y_train)))
scaler = StandardScaler()



scaler.fit(X_train.astype("float64"))

X_train_scaled = X_train

X_train_scaled = scaler.transform(X_train_scaled.astype("float64"))
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC



# Polynomial kernel

params = {"degree": [1,2,3], "coef0": [0,0.25,0.5,0.75,1], 'C':[1, 10, 100, 1000]}

svc = SVC(kernel = "poly")



poly_svc = GridSearchCV(svc, params, cv = 4, n_jobs = -1)



poly_svc.fit(X_train_scaled.astype("float64"), y_train)



print("Best parameters: {}".format(poly_svc.best_params_))

print("Score on the test set: {}".format(poly_svc.score(X_train_scaled.astype("float64"), y_train)))
# Gaussian RBF Kernel

params = {"gamma": [0,0.12,0.24,0.48,0.60,0.72,0.84,1], 'C':[1, 10, 100, 1000]}

svc = SVC(kernel = "rbf")



rbf_svc = GridSearchCV(svc, params, cv = 4, n_jobs = -1, iid = True)



rbf_svc.fit(X_train_scaled.astype("float64"), y_train)



print("Best parameters: {}".format(rbf_svc.best_params_))

print("Score on the test set: {}".format(rbf_svc.score(X_train_scaled.astype("float64"), y_train)))
from sklearn.neighbors import KNeighborsClassifier



params = {"n_neighbors": [2,4,6,8,10,12,14,16,18,20,25,50]}

knn = KNeighborsClassifier()



knnc = GridSearchCV(knn, params, cv = 4, n_jobs = -1)



knnc.fit(X_train_scaled.astype("float64"), y_train)



print("Best parameters: {}".format(knnc.best_params_))

print("Score on the test set: {}".format(knnc.score(X_train_scaled.astype("float64"), y_train)))
from sklearn.ensemble import RandomForestClassifier



rnd = RandomForestClassifier()



params = {"n_estimators": [2,4,8,20,50,100,200], 'max_depth':[2,4,6,8,10], 'max_features': [2,4,6,8]}





rndc = GridSearchCV(rnd, params, cv = 4, n_jobs = -1, iid = True)



rndc.fit(X_train_scaled.astype("float64"), y_train)



print("Best parameters: {}".format(rndc.best_params_))

print("Score on the test set: {}".format(rndc.score(X_train_scaled.astype("float64"), y_train)))
from sklearn.neural_network import MLPClassifier



mlpc = MLPClassifier(random_state=42, max_iter = 5000)

mlpc.fit(X_train_scaled, y_train)



print("Score on the test set: {}".format(mlpc.score(X_train_scaled.astype("float64"), y_train)))
from sklearn.ensemble import VotingClassifier



lr_c = Pipeline([

    ("poly_features", PolynomialFeatures(degree = 2)),

    ("lr", LogisticRegression(C = 5, solver = 'liblinear'))

])



svcp_c = SVC(kernel = "poly", degree = 2, C = 1, coef0 = 0.5, probability = True)



svc_c = SVC(kernel = "rbf", gamma = 0.12, C = 1, probability = True)



knn_c = KNeighborsClassifier(weights = "uniform", n_neighbors = 6)



rnd_c = RandomForestClassifier(max_depth = 8, n_estimators = 20, max_features = 2)



mlp_c = MLPClassifier(random_state=42, max_iter = 5000)



voting_clf = VotingClassifier(

    estimators = [("lr", lr_c), ("svcp", svcp_c), ("svc", svc_c),("knn", knn_c),("rnd", rnd_c),("mlp", mlp_c)], 

    voting = "soft")



voting_clf.fit(X_train_scaled, y_train)



train_score = voting_clf.score(X_train_scaled, y_train)



print("Train score: {}".format(train_score))
Test_set.isnull().sum()
mean_age_test = round(Test_set["Age"].mean())

mean_fare_test = round(Test_set["Fare"].mean())



Test_set["Age"].fillna(mean_age_test, inplace = True)



Test_set["Fare"].fillna(mean_fare_test, inplace = True)



Test_set = Test_set.dropna(axis = 1)



Test_set.isnull().sum()
dummy_sex_test = pd.get_dummies(Test_set.Sex)

dummy_embarked_test = pd.get_dummies(Test_set.Embarked)

Test_set_num = pd.concat([Test_set, dummy_sex_test["male"], dummy_embarked_test[['C', 'Q']]], axis = 1)



Test_set_num.drop(["Ticket", "Name", "Sex", "Embarked"], inplace = True, axis = 1)

Test_set_num.head()
fare_classes = []

for fare in Test_set["Fare"]:

    if fare < 10:

        fare_classes.append('0')

    elif fare < 25:

        fare_classes.append('1')

    elif fare < 55:

        fare_classes.append('2')

    elif fare < 70:

        fare_classes.append('3')

    elif fare < 100:

        fare_classes.append('4')

    else:

        fare_classes.append('5')

        

Test_set["Fare_class"] = fare_classes

Test_set.head()
age_classes = []

for age in Test_set["Age"]:

    if age < 16:

        age_classes.append('0')

    elif age < 32:

        age_classes.append('1')

    elif age < 48:

        age_classes.append('2')

    elif age < 70:

        age_classes.append('3')

    else:

        age_classes.append('4')

        

Test_set["Age_class"] = age_classes

Test_set.head()
Test_set["Family_members"] = Test_set["SibSp"] + Test_set["Parch"]



family_members = []

for number in Test_set["Family_members"]:

    if number < 3:

        family_members.append('a')

    elif number < 6:

        family_members.append('b')

    elif number < 9:

        family_members.append('c')

    elif number < 11:

        family_members.append('d')

    else:

        family_members.append('e')

        

Test_set["Family_class"] = family_members

Test_set.head()
dummy_fare = pd.get_dummies(Test_set.Fare_class, prefix = "fare")

dummy_age = pd.get_dummies(Test_set.Age_class, prefix = "age")

dummy_family = pd.get_dummies(Test_set.Family_class, prefix = "family")



Test_set_num = pd.concat([Test_set_num, dummy_fare, dummy_age, dummy_family], axis = 1)



Test_set_num.drop(["Age", "Fare", "SibSp", "Parch", "PassengerId"], inplace = True, axis = 1)

Test_set_num.head()
Cols = Test_set_num.columns

X_test = Test_set_num[Cols] 



scaler = StandardScaler()



scaler.fit(X_test.astype("float64"))

X_test_scaled = X_test

X_test_scaled = scaler.transform(X_test_scaled.astype("float64"))
X_test_scaled.shape

X_train.columns
predictions = voting_clf.predict(X_test_scaled)

predictions
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':Test_set['PassengerId'],'Survived':predictions})



submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

file_name = 'Titanic Predictions2.csv'



df = submission

# import the modules we'll need

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = file_name):  

    csv = df.to_csv(index = False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(df)
submission.to_csv(filename,index=False)



print('Saved file: ' + filename)