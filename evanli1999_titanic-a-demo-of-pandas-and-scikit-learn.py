import numpy as np
import re
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


original = pandas.read_csv("../input/train.csv")
print(original.columns.values)
original['Age'] = original['Age'].replace(np.nan, original['Age'].mean(), regex=True)
original['Embarked'] = original['Embarked'].replace(np.nan, "M", regex=True)
#We fill in missing values with M

original.sample(5)
name = original['Name']
titles = []

for i in range(len(name)):
	s = name[i]
	title = re.search(', (.*)\.', s)
	title = title.group(1)
	titles.append(title)

original['Titles'] = titles
original.sample(5)
original['Titles'].replace(['Sir', 'Rev', 'Major', 'Lady', 'Jonkheer', 'Dr', 'Don', 'Countess', 'Col', 'Capt'], 'Name')

original['Titles'].replace(['Ms', 'Mme', 'Mlle'], ['Miss', 'Mrs', 'Miss'])
Sex_binarized = pandas.DataFrame(preprocessing.LabelBinarizer().fit_transform(original.Sex))
Ticket_binarized = pandas.DataFrame(preprocessing.LabelBinarizer().fit_transform(original.Ticket))
Embarked_binarized = pandas.DataFrame(preprocessing.LabelBinarizer().fit_transform(original.Embarked))
Titles_binarized = pandas.DataFrame(preprocessing.LabelBinarizer().fit_transform(original.Titles))
prediction_params = pandas.concat([Sex_binarized, Ticket_binarized, Embarked_binarized, Titles_binarized, original.Pclass, original.Age, original.SibSp, original.Fare, original.Parch], axis=1)
prediction_result = original.Survived

x_train, x_test, y_train, y_test = train_test_split(prediction_params, prediction_result, test_size = 0.15, random_state = 10)
logistic_model = linear_model.LogisticRegression().fit(x_train, y_train.values.ravel())
logistic_prediction = logistic_model.predict(x_test)

accuracy_score(logistic_prediction, y_test)
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_prediction = dt.predict(x_test)

accuracy_score(dt_prediction, y_test)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)

accuracy_score(rf_prediction, y_test)
clf = MLPClassifier(activation = 'relu', solver='lbfgs', hidden_layer_sizes=(150), random_state=10)
clf.fit(x_train, y_train)
neural_prediction = clf.predict(x_test)

accuracy_score(neural_prediction, y_test)