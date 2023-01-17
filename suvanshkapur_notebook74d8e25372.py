# Importing required libraries



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

# Reading the dataset 



df = pd.DataFrame(pd.read_csv('../input/titanic-dataset-with-logistic-regression/train (1).csv'))

test_data = pd.DataFrame(pd.read_csv('../input/titanic-dataset-with-logistic-regression/test.csv'))

gender_df =  pd.DataFrame(pd.read_csv('../input/titanic-dataset-with-logistic-regression/gender_submission.csv'))
df.head()
# Checking for the NaN values



for i in df.columns:

  print(i,"\t-\t", df[i].isna().mean()*100)

df = df.drop(["Cabin"], axis=1)
#Filling the Nan values for Age

df['Age'].fillna(df['Age'].median(), inplace=True)

df['Embarked'].fillna(df['Embarked'].mode(), inplace=True)
df.info()
#We can drop PassengerId, Fare, Name, Ticket values because they are not affecting our prediction.



df = df.drop(["PassengerId", "Fare", "Ticket", "Name"], axis = 1)   
from sklearn.preprocessing import LabelEncoder



cat_col= df.drop(df.select_dtypes(exclude=['object']), axis=1).columns

print(cat_col)



enc1 = LabelEncoder()

df[cat_col[0]] = enc1.fit_transform(df[cat_col[0]].astype('str'))



enc2 = LabelEncoder()

df[cat_col[1]] = enc2.fit_transform(df[cat_col[1]].astype('str'))
df.head()
df.info()
# Pclass



sns.FacetGrid(df, col= 'Survived').map(plt.hist,'Pclass')
# Sex



sns.FacetGrid(df, col='Survived').map(plt.hist, 'Sex')
# Age



sns.FacetGrid(df, col='Survived').map(plt.hist, 'Age')
# SibSp



sns.FacetGrid(df, col='Survived').map(plt.hist, 'SibSp')
# Parch



sns.FacetGrid(df, col='Survived').map(plt.hist, 'Parch')
# Embarked



sns.FacetGrid(df, col='Survived').map(plt.hist, 'Embarked')
X = df.drop(['Survived'], axis=1)

y = df['Survived']
#now lets split data in test train pairs



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# model training 



from sklearn.linear_model import LogisticRegression



model = LogisticRegression()   

model.fit(X_train, y_train)
y_pred = model.predict(X_test)



pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

pred_df.head()
#Data visualisation



plt.scatter([i for i in range(len(X_test["Age"]))], y_test, color='red')

plt.plot([i for i in range(len(X_test["Age"]))], y_pred, color='green')



plt.ylabel('Survived')

plt.xlabel('Passenger')



plt.show()
# Accuracy check



from sklearn import metrics



# Generating the roc curve using scikit-learn.

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

plt.plot(fpr, tpr)

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.show()



# Measuring the area under the curve  

print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))



# Measuring the Accuracy Score

print("Accuracy score of the predictions: {0}".format(metrics.accuracy_score(y_pred, y_test)))
test_data.head()
# Displaying the column wise %ge of NaN values 



for i in df.columns:

  print(i,"\t-\t", df[i].isna().mean()*100)

test_data = test_data.drop(["Cabin"], axis=1)



test_data['Age'].fillna(test_data['Age'].median(), inplace=True) #filling Nan values of Age

df['Embarked'].fillna(df['Embarked'].mode(), inplace=True)
test_data.info()
PassengerId = test_data["PassengerId"]



test_data = test_data.drop(["PassengerId", "Fare", "Ticket", "Name"], axis = 1)   #Since PassengerId, Fare, Name, Ticket does not has any role in price prediction
test_data[cat_col[0]] = enc1.transform(test_data[cat_col[0]].astype('str'))



test_data[cat_col[1]] = enc2.transform(test_data[cat_col[1]].astype('str'))
test_data.head()
y_test_pred = model.predict(test_data)
#Data Visualization



plt.scatter([i for i in range(len(test_data["Age"]))], y_test_pred, color='red')



plt.ylabel('Survived')

plt.xlabel('Passenger')



plt.show()
submission = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": y_test_pred

    })



submission.to_csv('./submission.csv', index=False)