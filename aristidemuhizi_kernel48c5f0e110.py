# linear algebra

import numpy as np 

# data processing

import pandas as pd



from sklearn.model_selection import train_test_split

# Algorithms

from sklearn.ensemble import RandomForestClassifier
test_clean = pd.read_csv("../input/titanic-cleaned-data/test_clean.csv").replace(to_replace =["female", "male"], value =[0, 1])

train_clean = pd.read_csv("../input/titanic-cleaned-data/train_clean.csv").replace(to_replace =["female", "male"], value =[0, 1])
# X_train = train_clean.drop("Survived", axis=1)

# y_train = train_clean["Survived"]

# X_test  = test_clean.drop("PassengerId", axis=1).copy()

# X_train



# titanic = titanic.replace(to_replace =["female", "male"], value =[0, 1])

_titanic = pd.DataFrame({"Survived": train_clean.Survived.values, "Pclass": train_clean.Pclass.values, "Sex": train_clean.Sex.values, "Age": train_clean.Age.values}) #.dropna()

X_train = _titanic.drop("Survived", axis=1)

y_train = _titanic.Survived.values

X_test = pd.DataFrame({"Pclass": test_clean.Pclass.values, "Sex": test_clean.Sex.values, "Age": test_clean.Age.values}) #.dropna()

# X_train, X_test, y_train, y_test = train_test_split(x_titanic, y_titanic, train_size=0.75,test_size=0.25, random_state=101)
X_test
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest
predict_y = random_forest.predict(X_test)

predict_y
X_test
xcell = pd.DataFrame({"PassengerId":test_clean.PassengerId.values, "Survived" : predict_y})

xcell_csv = xcell.to_csv (r'../input/titanic-cleaned-data/export_dataframe.csv', index = None, header=True)