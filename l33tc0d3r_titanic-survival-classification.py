# importing necessary libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# importing dataset 

train_df = pd.DataFrame(pd.read_csv('../input/titanic/train.csv'))
test_df = pd.DataFrame(pd.read_csv('../input/titanic/test.csv'))
class_df =  pd.DataFrame(pd.read_csv('../input/titanic/gender_submission.csv'))
train_df.head()
# showing column wise %ge of NaN values they contains 

for i in train_df.columns:
  print(i,"\t-\t", train_df[i].isna().mean()*100)

train_df = train_df.drop(["Cabin"], axis=1)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)#filling Nan values of Age
train_df['Embarked'].fillna(train_df['Embarked'].mode(), inplace=True)
train_df.info()
train_df = train_df.drop(["PassengerId", "Fare", "Ticket", "Name"], axis = 1)   #Since PassengerId, Fare, Name, Ticket does not has any role in price prediction
from sklearn.preprocessing import LabelEncoder

cat_col= train_df.drop(train_df.select_dtypes(exclude=['object']), axis=1).columns
print(cat_col)

enc1 = LabelEncoder()
train_df[cat_col[0]] = enc1.fit_transform(train_df[cat_col[0]].astype('str'))

enc2 = LabelEncoder()
train_df[cat_col[1]] = enc2.fit_transform(train_df[cat_col[1]].astype('str'))
train_df.head()
train_df.info()
# Pclass

sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'Pclass')
# Sex

sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'Sex')
# Age

sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'Age')
# SibSp

sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'SibSp')
# Parch

sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'Parch')
# Embarked

sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'Embarked')
X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']
#now lets split data in test train pairs

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# model training 

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()   # Here kernel used is RBF (Radial Basis Function)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
pred_df.head()
#plot

plt.scatter([i for i in range(len(X_test["Age"]))], y_test, color='black')
plt.plot([i for i in range(len(X_test["Age"]))], y_pred, color='red')

plt.ylabel('Survived')
plt.xlabel('Passenger')

plt.show()
# To check Accuracy

from sklearn import metrics

# Generate the roc curve using scikit-learn.
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

# Measure the area under the curve.  The closer to 1, the "better" the predictions.
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))

# Measure the Accuracy Score
print("Accuracy score of the predictions: {0}".format(metrics.accuracy_score(y_pred, y_test)))
test_df.head()
# showing column wise %ge of NaN values they contains 

for i in train_df.columns:
  print(i,"\t-\t", train_df[i].isna().mean()*100)

test_df = test_df.drop(["Cabin"], axis=1)

test_df['Age'].fillna(test_df['Age'].median(), inplace=True) #filling Nan values of Age
train_df['Embarked'].fillna(train_df['Embarked'].mode(), inplace=True)
test_df.info()
PassengerId = test_df["PassengerId"]

test_df = test_df.drop(["PassengerId", "Fare", "Ticket", "Name"], axis = 1)   #Since PassengerId, Fare, Name, Ticket does not has any role in price prediction
test_df[cat_col[0]] = enc1.transform(test_df[cat_col[0]].astype('str'))

test_df[cat_col[1]] = enc2.transform(test_df[cat_col[1]].astype('str'))
test_df.head()
y_test_pred = model.predict(test_df)
#plot

plt.scatter([i for i in range(len(test_df["Age"]))], y_test_pred, color='black')

plt.ylabel('Survived')
plt.xlabel('Passenger')

plt.show()
submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_test_pred
    })

submission.to_csv('./submission.csv', index=False)
