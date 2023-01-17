# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



print("Prepared input data set:")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print("Read the training data") 



train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

print("Pre-process the training data") 



print("--Blank data make filled with their median in case of numeric data")

train_data = train_data.fillna(train_data.median())

print("--Categorize Cabin data by initials(In case of blank, assign O)")

train_data["Cabin"] = train_data["Cabin"].str[:1]

train_data["Cabin"] = train_data["Cabin"].fillna("O")

print("--Change data type of [Pclass] to string as this data is categorical")

train_data["Pclass"] = train_data["Pclass"].astype(str)



train_data.head()
print("Read the test data") 



test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
print("Pre-process the test data") 



print("--Blank data make filled with their median in case of numeric data")

test_data = test_data.fillna(test_data.median())

print("--Categorize Cabin data by initials(In case of blank, assign O)")

test_data["Cabin"] = test_data["Cabin"].str[:1]

test_data["Cabin"] = test_data["Cabin"].fillna("O")

print("--Change data type of [Pclass] to string as this data is categorical")

test_data["Pclass"] = test_data["Pclass"].astype(str)



test_data.head()
print("Feature setting:")

print("--Use one-hot column for categorical data\n")

y = train_data["Survived"]



features = ["Pclass", "Sex", "Age", "SibSp", "Parch","Fare", "Cabin","Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

X_test["Cabin_T"] = 0



features_update = X.columns

X_test = X_test[features_update]



print("train data columns:\n ",X.columns)

print("test data columns:\n ",X_test.columns)
# Cross validation for some training models.



print("Cross validation for some training models.")

print("  1. Training some models by cross validation.")

print("  2. Predict results of test data by mean of the cross validations.")

print("  3. Finally, make majority of all models\n\n")



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold



models = []

models.append(("LogisticRegression",LogisticRegression(max_iter=1000, random_state=1)))

models.append(("k-Nearest Neighbors",KNeighborsClassifier()))

models.append(("Decision Tree",DecisionTreeClassifier(random_state=1)))

models.append(("Support Vector Machine(linear)",SVC(kernel='linear',random_state=1)))

models.append(("Support Vector Machine(rbf)",SVC(kernel='rbf',random_state=1)))

models.append(("Random Forest",RandomForestClassifier(n_estimators=120, max_depth=7, random_state=1)))

models.append(("Perceptron",Perceptron(random_state=1)))

models.append(("Multilayer perceptron",MLPClassifier(max_iter=1000, random_state=1)))



names_en = []

results = []

predictions_all = np.zeros(len(test_data))

n_splits = 3

kfold = KFold(n_splits=n_splits)

for name_en,model in models:

    

    print(name_en, ": Start...")

    cv_result = cross_validate(model, X, y, cv=kfold, return_estimator=True)

    

    names_en.append(name_en)

    results.append(np.round(cv_result["test_score"],3))

    predictions = np.round(sum(cv_result["estimator"][i].predict(X_test) for i in range(n_splits))/n_splits)

    predictions = np.array(predictions, dtype = 'int32')

    

    np.set_printoptions(threshold=10)

    print("--Prediction (average of each cross validation): ", predictions)

    predictions_all = predictions_all + predictions

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

    output.to_csv("my_submission_"+name_en+".csv", index=False)

    print("--Submission file were successfully saved!", "(my_submission_"+name_en+".csv)\n")



print("Result of each model")



list_df = pd.DataFrame( columns=["Identifier","Score of each cross val"] )

 

for i in range(len(names_en)):

    list_df = list_df.append( pd.Series( [names_en[i],results[i]], index=list_df.columns ), ignore_index=True)

 

list_df
print("Majority decision start...")

predictions_all = np.round(predictions_all/len(models))

predictions_all =  np.array(predictions_all, dtype = 'int32')

print("Predicted by all models majority: ", predictions_all)

output_all = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_all})

output_all.to_csv('my_submission_all.csv', index=False)



print("Submission file were successfully saved! (my_submission_all.csv): Majority decision of all models")

print("Submit your favorite result!")
