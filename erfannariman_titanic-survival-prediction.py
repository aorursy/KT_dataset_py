from IPython.display import HTML

HTML('<center><iframe width="800" height="450" src="https://www.youtube.com/embed/YhZXU5zUnO0" frameborder="0" allowfullscreen></iframe></center>')
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('seaborn')
#change directory to 01_Input
os.chdir('../input/titanic')
df = pd.read_csv('train.csv')

df.head()
#version 1 with def and .apply
def survived_to_words(row):
    if row.Survived == 1:
        value = "yes"
    elif row.Survived == 0:
        value = "no"
    else:
        value = "unkown"
    
    return value

df["Survived_word"] = df.apply(survived_to_words, axis=1)
#version 2 with lambda
df["Survived_word"] = df.Survived.apply(lambda row: "yes" if row == 1 else "no")
#version 3 with np.where
df["Survived_word"] = np.where(df.Survived == 1, "yes", "no")

df.head()
print('shape is',df.shape, '\n')
print(df.count())
fig=plt.figure(figsize=(8, 5), dpi= 80, facecolor='w', edgecolor='k')

df.Survived.value_counts(normalize=True).plot(kind="bar")
plt.title("Survived")

plt.show()
fig=plt.figure(figsize=(8, 5), dpi= 80, facecolor='w', edgecolor='k')

plt.scatter(df.Survived, df.Age, alpha=0.1)
plt.title("Age wrt Survived")

plt.show()
fig=plt.figure(figsize=(22, 13), dpi= 80, facecolor='w', edgecolor='k')

plt.subplot2grid((2,3), (0,0))
df.Pclass.value_counts(normalize=True).plot(kind="pie")
plt.title("Distribution of class")

plt.subplot2grid((2,3), (0,1))
df.Pclass.value_counts(normalize=True).plot(kind="bar")
plt.title("Distribution of class")

plt.show()
fig=plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')

for x in [1,2,3]:
    df.Age[df.Pclass == x].plot(kind="kde")
plt.title("Class wrt Age")
plt.legend(("Class 1", "Class 2", "Class 3"))

plt.show()
fig=plt.figure(figsize=(8, 5), dpi= 80, facecolor='w', edgecolor='k')

df.Embarked.value_counts(normalize=True).plot(kind="bar")
plt.title("Embarked location")

plt.show()
fig = plt.figure(figsize=(22,15))

plt.subplot2grid((3,4), (0,0))
df.Survived_word[df.Sex == "male"].value_counts(normalize=True).plot(kind="bar", color="#fc8320")
plt.title("Men survived")

plt.subplot2grid((3,4), (0,1))
df.Survived_word[df.Sex == "female"].value_counts(normalize=True).plot(kind="bar", color="#006418")
plt.title("Women survived")

plt.subplot2grid((3,4), (0,2))
df.Sex[df.Survived_word == "yes"].value_counts(normalize=True).plot(kind="bar", color="#065183")
plt.title("Sex of survivors")

plt.show()
fig=plt.figure(figsize=(10, 5), dpi= 80, facecolor='w', edgecolor='k')

for x in [1,2,3]:
    df.Survived[df.Pclass == x].plot(kind="kde")
plt.title("Survived wrt Class")
plt.legend(("Class 1", "Class 2", "Class 3"))

plt.show()
fig = plt.figure(figsize=(22,15))

plt.subplot2grid((3,4), (0,0))
df.Survived_word[(df.Sex == "male")  &  (df.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", color="#fc8320")
plt.title("Rich men survived")

plt.subplot2grid((3,4), (0,1))
df.Survived_word[(df.Sex == "male")  &  (df.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", color="#006418")
plt.title("Poor men survived")

plt.subplot2grid((3,4), (1,0))
df.Survived_word[(df.Sex == "female")  &  (df.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", color="#065183")
plt.title("Rich women survived")

plt.subplot2grid((3,4), (1,1))
df.Survived_word[(df.Sex == "female")  &  (df.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", color="#b30000", alpha=0.75)
plt.title("Poor women survived")
plt.show()
train = df.copy()

train["Hypothesis"] = np.where(train.Sex == 'female', 1, 0)
train['Result'] = np.where(train.Hypothesis == train.Survived, 1, 0)

train.Result.value_counts(normalize=True)
def clean_data(data):
    # Fill the empty rows of Fare and Age with their median
    data['Fare'] = data.Fare.fillna(data.Fare.dropna().median())
    data['Age'] = data.Age.fillna(data.Age.dropna().median())
    
    # Convert Sex column from words "Female" and "Male" to 0 and 1
    data["Sex"] = np.where(data.Sex == "male", 0, 1)
    
    # Fill NaNs of Embarked column and convert to integers
    data["Embarked"] = data["Embarked"].fillna("S")
    conditions = [data['Embarked'] == "S", 
                  data['Embarked'] == "C",
                  data['Embarked'] == "Q"]
    choices = [0, 1, 2]
    data["Embarked"] = np.select(conditions, choices, default = 0)

clean_data(train)

train.head()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
feature_columns = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch']

target = train.Survived.values
features = train[feature_columns].values

classifier = LogisticRegression(solver='lbfgs')
classifier_ = classifier.fit(features, target)

print("logistic regression accuracy =", classifier_.score(features, target))

print("logistic regression accuracy with cross validation = ", np.mean(cross_val_score(classifier, 
                                                                                        features, 
                                                                                        target, 
                                                                                        cv=10)))
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)

classifier = LogisticRegression()
classifier_ = classifier.fit(poly_features, target)
print("accuracy with polynomial features =", classifier_.score(poly_features, target))

print("accuracy with polynomial features combined with cross validation =", 
      np.mean(cross_val_score(classifier, 
                               poly_features, 
                               target, 
                               cv=10)))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf_ = rf.fit(features, target)

print("accuracy with random forrest regressor =", rf_.score(features, target))

print("accuracy with random forrest regressor and cross validation =",
      np.mean(cross_val_score(rf, 
                               features, 
                               target, 
                               cv=10)))
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=1,
                                      max_depth=7,
                                      min_samples_split=2)

decision_tree_ = decision_tree.fit(features, target)

print("accuracy with decision tree =", decision_tree_.score(features, target))
print("accuracy with decision tree and cross validation =",
     np.mean(cross_val_score(decision_tree, features, target, cv=50)))