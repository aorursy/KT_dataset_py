



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
datasets = [train_data, test_data]

#drop columns with too many missing values and columns with no impact on the survival status

drop_columns = ["PassengerId", "Ticket", "Cabin"]

train_data.drop(drop_columns, axis = 1, inplace = True)

#check to see if there is any missing values in each data set

print("sum of missing values in train data is \n",train_data.isnull().sum() )

print("sum of missing values in test data is \n", test_data.isnull().sum())
#replace missing values in 'Age', 'Embarked' and 'Fare'

for dataset in datasets:

    dataset.Age.fillna(dataset.Age.median(), inplace = True)

    dataset.Embarked.fillna(dataset.Embarked.mode()[0], inplace = True)

    dataset.Fare.fillna(dataset.Fare.median(),inplace = True)

    

#check to see if there is any missing values left in each data set

print("sum of missing values in train data is \n",train_data.isnull().sum())

print("sum of missing values in test data is \n", test_data.isnull().sum())





#feature engineering for train and test data

for dataset in datasets:

    #add a feature as 'Title' from each person in the shipinto the dataset  

    dataset["Title"] = dataset.Name.str.split(',',expand = True)[1].str.split('.',expand = True)[0]

    #add a feature as 'Family size' into the dataset

    dataset["Family_size"] = dataset["Parch"]+dataset["SibSp"]+1

    #add a feature as 'is along' = 0 if family size greater than 1, else equals to 1

    dataset["Is_along"] = 1

    dataset["Is_along"].loc[dataset["Family_size"]>1] = 0

    #add a bin for Fare

    dataset["Farebin"] = pd.qcut(dataset["Fare"],4)

    #add a bin for Age

    dataset["Agebin"] = pd.cut(dataset["Age"].astype(int), 5)

    

    

#cleanup rare title names(with count value less than 10)

title_names = train_data["Title"].value_counts() < 10 #this will create a true false series with title name as index

train_data["Title"] = train_data["Title"].apply(lambda x : 'Misc' if (title_names).loc[x] == True else x)

print(train_data["Title"].value_counts())

train_data.head()
import sklearn

print(train_data.columns)

#encoding the categorical value

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

encoder = LabelEncoder()

for dataset in datasets:

    dataset["Embarked_encode"] = encoder.fit_transform(dataset["Embarked"])

    dataset["Sex_encode"]= encoder.fit_transform(dataset["Sex"]) 

    dataset["Title_encode"] = encoder.fit_transform(dataset["Title"])

    dataset["Agebin_encode"] = encoder.fit_transform(dataset["Agebin"])

    dataset["Farebin_encode"] = encoder.fit_transform(dataset["Farebin"])



train_data.head()






train_data.head()

train_data.columns

test_data.columns


feature_cols = ['Sex_encode', 'Embarked_encode',"Title_encode","Agebin_encode","Farebin_encode","Is_along", "Pclass"]

X_train = train_data[feature_cols]

y_train = train_data['Survived']

X_test = test_data[feature_cols]





MLA = [

    ensemble.RandomForestClassifier(),

    ensemble.GradientBoostingClassifier(),

    gaussian_process.GaussianProcessClassifier(),

    linear_model.LogisticRegressionCV(),

    tree.DecisionTreeClassifier(),

    XGBClassifier()

]



table_column = ['Model_name', 'Score']

table = pd.DataFrame(columns = table_column)

row_index = 0

for alg in MLA:

    table.loc[row_index, 'Model_name'] = alg.__class__.__name__

    fit_model = alg.fit(X_train, y_train)

    table.loc[row_index, 'Score'] = round(alg.score(X_train,y_train)*100,2)

    row_index +=1

table.sort_values(by = "Score", ascending = False, inplace = True)



table

    
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

print(acc_random_forest)

submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv("my_submission.csv",index=False)

#model = DecisionTreeClassifier()

#model.fit(X_train,y_train)

#predictions = model.predict(X_test)

#output = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predictions})

#output.to_csv('my_submission.csv',index=False)

#print("Your submission was successfully saved!")
