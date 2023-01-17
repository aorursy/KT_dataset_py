import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
os.listdir('../')

# Importing train dataset
train = pd.read_csv('../input/train.csv')

# To see the no of rows and columns in the dataframe
print(train.shape)

# Describe dataframe to see more information about the data and missing values
train.describe() # train_df.info() also can be used

#  We can notice DF total have 891 rows and Age is missing in some rows
# Importing testing dataset
test = pd.read_csv('../input/test.csv')

print(test.shape)
print(test.info())
# Printig some rows from train
train.head()
sex_plt = train.pivot_table(index='Sex', values='Survived')
sex_plt.plot.bar()
sex_plt = train.pivot_table(index='Pclass', values='Survived')
sex_plt.plot.bar()
# As you can see, If passenger is women or passenger belongs to 1st class has better chance of surviving

# Data missing for Age in the train dataset
print(train.shape)
display(train.describe())

# Using pandas.fillna method to fill the NaN values using Age mean
train['Age'].fillna(train.Age.mean(), inplace=True)

print(test.shape)
display(test.describe())

# Using pandas.fillna method to fill the NaN values using Fare median
test['Fare'].fillna(test.Fare.median(), inplace=True)
test['Age'].fillna(test.Age.mean(), inplace=True)
# For Train data
drop_features = ['Name', 'PassengerId', 'Ticket', 'Cabin']
train = train.drop(drop_features, axis = 1)
train = pd.get_dummies(train, drop_first=True)

display(train.head())
# For Test data
test = test.drop(drop_features, axis = 1)
test = pd.get_dummies(test, drop_first=True)
display(test.head())

train.describe()
train.head()
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()
]
log_cols = ['Classifier', 'Accuracy']
log = pd.DataFrame(columns=log_cols)
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

X = train.iloc[:, 1:].values
y = train.iloc[:, 0].values

print("X ", X.shape)
print("y ", y.shape)

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        
        y_prediction = clf.predict(X_test)
        acc = accuracy_score(y_test, y_prediction)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc
for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)
    
plt.xlabel = "Accuracy"
plt.title = "Clasifier Accurcy"

sns.set_color_codes("muted")
sns.barplot(x="Accuracy", y="Classifier", data=log, color="b")
output_cols = ['PassengerId', 'Survived']

high_acc_clf = GradientBoostingClassifier()
high_acc_clf.fit(X, y)

y_pred = high_acc_clf.predict(test.values)
y_pred

test_data = pd.read_csv('../input/test.csv')
data = {
    "PassengerId": test_data.PassengerId,
    "Survived": y_pred
}
data
new_df = pd.DataFrame(data)
new_df.to_csv('./prediction.csv', index=False)

