# data analysis and wrangling
import pandas as pd
import numpy as np

# visualisation
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# read in the files provided by Kaggle
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.columns
train_df.head()
train_df.info()
test_df.info()
# check floats and ints stats
train_df.describe()
# check objects stats
train_df.describe(include=['O'])
# check 'Pclass' feature
class_pivot = train_df.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar()
plt.show()
sex_pivot = train_df.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()
survived = train_df[train_df["Survived"] == 1]
died = train_df[train_df["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='green',bins=50)
died["Age"].plot.hist(alpha=0.5,color='red',bins=50)
plt.legend(['Survived','Died'])
plt.show()
sib_pivot = train_df.pivot_table(index="SibSp",values="Survived")
sib_pivot.plot.bar()
plt.show()
parch_pivot = train_df.pivot_table(index="Parch",values="Survived")
parch_pivot.plot.bar()
plt.show()
survived = train_df[train_df["Survived"] == 1]
died = train_df[train_df["Survived"] == 0]
survived["Fare"].plot.hist(alpha=0.5,color='green',bins=50)
died["Fare"].plot.hist(alpha=0.5,color='red',bins=50)
plt.legend(['Survived','Died'])
plt.show()
embark_pivot = train_df.pivot_table(index="Embarked",values="Survived")
embark_pivot.plot.bar()
plt.show()
def divide_age(df, cut_points, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)
    return df

# declare ranges and their labels
age_cut_points = [-1,0,3,12,18,35,60,100]
age_label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

# change both train and test datasets
train_df = divide_age(train_df, age_cut_points, age_label_names)
test_df = divide_age(test_df, age_cut_points, age_label_names)

#plot the results
age_pivot = train_df.pivot_table(index="Age_categories",values='Survived')
age_pivot.plot.bar()
plt.show()
def divide_fare(df, cut_points, label_names):
    df["Fare_categories"] = pd.cut(df["Fare"], cut_points, labels=label_names)
    return df

# declare ranges and their labels
fare_cut_points = [0,25,50,75,100,520]
fare_label_names = ["Cheap","Low","Medium","High","Premium"]

# change both train and test datasets
train_df = divide_fare(train_df, fare_cut_points, fare_label_names)
test_df = divide_fare(test_df, fare_cut_points, fare_label_names)

#plot the results
fares_pivot = train_df.pivot_table(index="Fare_categories",values='Survived')
fares_pivot.plot.bar()
plt.show()
def is_alone(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    return df

# change both train and test datasets
train_df = is_alone(train_df)
test_df = is_alone(test_df)

#plot the results
is_alone = train_df.pivot_table(index="IsAlone",values='Survived')
is_alone.plot.bar()
plt.show()
train_df['IsAlone'].value_counts(normalize=True)
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df,dummies], axis=1)
    return df

for column in ["Pclass","Sex","Embarked", "Age_categories", "Fare_categories"]:
    train_df = create_dummies(train_df, column)
    test_df = create_dummies(test_df, column)
test_df.head()
# change the 'real' test data to holdout
holdout = test_df

# select the columns we are interested about making predictions
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'Age_categories_Missing','Age_categories_Infant','Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult', 'Age_categories_Senior', 'Fare_categories_Cheap',
          'Fare_categories_Low', 'Fare_categories_Medium', 'Fare_categories_High', 'Fare_categories_Premium']

# chooose xs and y
all_x = train_df[columns]
all_y = train_df['Survived']

train_x, test_x, train_y, test_y = train_test_split(all_x, all_y, test_size=0.20,random_state=0)
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(train_x, train_y)
y_pred = logreg.predict(test_x)
acc_log = round(logreg.score(train_x, train_y) * 100, 2)
print(acc_log)
# Support Vector Machines
svc = SVC()
svc.fit(train_x, train_y)
y_pred = svc.predict(test_x)
acc_svc = round(svc.score(train_x, train_y) * 100, 2)
print(acc_svc)
# K-nearest neighbours
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_x, train_y)
y_pred = knn.predict(test_x)
acc_knn = round(knn.score(train_x, train_y) * 100, 2)
print(acc_knn)
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(train_x, train_y)
y_pred = gaussian.predict(test_x)
acc_gaussian = round(gaussian.score(train_x, train_y) * 100, 2)
print(acc_gaussian)
# Perceptron
perceptron = Perceptron()
perceptron.fit(train_x, train_y)
y_pred = perceptron.predict(test_x)
acc_perceptron = round(perceptron.score(train_x, train_y) * 100, 2)
print(acc_perceptron)
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(train_x, train_y)
y_pred = linear_svc.predict(test_x)
acc_linear_svc = round(linear_svc.score(train_x, train_y) * 100, 2)
print(acc_linear_svc)
# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(train_x, train_y)
y_pred = sgd.predict(test_x)
acc_sgd = round(sgd.score(train_x, train_y) * 100, 2)
print(acc_sgd)
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_x, train_y)
y_pred = decision_tree.predict(test_x)
acc_decision_tree = round(decision_tree.score(train_x, train_y) * 100, 2)
print(acc_decision_tree)
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_x, train_y)
y_pred = random_forest.predict(test_x)
acc_random_forest = round(random_forest.score(train_x, train_y) * 100, 2)
print(acc_random_forest)
names = ['Logistic Regression', 'Support Vector Machines', 'KNN', ' Gaussian Naive Bayes', 'Perceptron', 'Linear SVC', 
         'Stochastic Gradient Decent', 'Decision Tree', 'Random Forest']

estimators = [logreg, svc, knn, gaussian, perceptron, linear_svc, sgd, decision_tree, random_forest]

scores = [acc_log, acc_svc, acc_knn, acc_gaussian, acc_perceptron, acc_linear_svc, acc_sgd, acc_decision_tree, 
          acc_random_forest]

models_summary = pd.DataFrame({'Name':names, 'Score':scores})
test_list = []
for est in estimators:
    accuracy = cross_val_score(est, all_x, all_y, cv=10).mean()
    test_list.append(accuracy)
print(test_list)

models_summary.loc[:, 'Accuracy'] = pd.Series(test_list, index=models_summary.index)
models_summary.sort_values(by='Accuracy', ascending=False)
svc.fit(all_x,all_y)
holdout_predictions = svc.predict(holdout[columns])
holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv("submission4.csv",index=False)