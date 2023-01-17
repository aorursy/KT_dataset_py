# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data=pd.read_csv("/kaggle/input/titanic/train.csv")



women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



features = ["Pclass", "Sex", "SibSp", "Parch","Fare","Age"]

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(abs(train_data[features].corr()), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
train_data.plot(kind='scatter', x='Fare', y='Pclass',alpha = 0.5,color = 'red')

plt.xlabel('Fare')              # label = name of label

plt.ylabel('Passenger class')

plt.title('Fare/class Scatter Plot')

plt.show()
train_data.plot(kind='scatter', x='Parch', y='SibSp',alpha = 0.5,color = 'red')

plt.xlabel('Number of parents/children on boat')              # label = name of label

plt.ylabel('Number of Siblings/spouses on boat')

plt.title('Parch/SibSp Scatter Plot')

plt.show()
train_data.plot(kind='scatter', x='Age', y='Pclass',alpha = 0.5,color = 'red')

plt.xlabel('Age')              # label = name of label

plt.ylabel('Class')

plt.title('Age/Pclass Scatter Plot')

plt.show()
train_data.plot(kind='scatter', x='Parch', y='SibSp',alpha = 0.5,color = 'red')

plt.xlabel('Number of parents/children on boat')              # label = name of label

plt.ylabel('Number of Siblings/spouses on boat')

plt.title('Parch/SibSp Scatter Plot')

plt.show()
#from pandas.tools.plotting import scatter_matrix

pd.plotting.scatter_matrix(train_data.get(features), alpha=0.2,

               figsize=(8, 8), diagonal='kde');
data_plot = train_data.assign(LogFare=lambda x : np.log(x.Fare + 1.))



log_features = ["Pclass", "Sex", "SibSp", "Parch","LogFare","Age"]



data_plot.head(5)
pd.plotting.scatter_matrix(data_plot.get(log_features), alpha=0.2, figsize=(8, 8), diagonal='kde');



f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(abs(data_plot.get(log_features).corr()), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
log_features = ["Sex", "SibSp", "Parch","LogFare","Age","Survived"]

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(abs(data_plot.get(log_features).corr()), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
log_features = ["Sex", "SibSp", "Parch","LogFare","Age"]

type(log_features)



data_plot[log_features+['Survived']].groupby('Survived').count()
train_data['Age'].describe()
data_plot.fillna(train_data['Age'].mean(),inplace=True)



log_features = ["Sex", "SibSp", "Parch","LogFare","Age"]

type(log_features)



data_plot[log_features+['Survived']].groupby('Survived').count()
test_data = test_data.assign(LogFare=lambda x : np.log(x.Fare + 1.))



test_data[log_features].count()
test_data['Age'].fillna(train_data['Age'].mean(),inplace=True)



data_plot['LogFare'].describe()
test_data['LogFare'].describe()
test_data[log_features].count()
test_data['LogFare'].fillna(data_plot['LogFare'].median(),inplace=True)
y = train_data["Survived"]



X = pd.get_dummies(data_plot[log_features])

X_test = pd.get_dummies(test_data[log_features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission2.csv', index=False)

print("Your submission was successfully saved!")
from sklearn import svm



clf = svm.SVC(C=100.0, kernel='rbf', gamma='scale') #C = regularization parameter (big for no reg)

clf.fit(X, y)



predictions=clf.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission2_svm.csv', index=False)
from sklearn.neural_network import MLPClassifier



Network = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic',solver='adam', alpha=0.0001)



from sklearn.preprocessing import StandardScaler  



scaler = StandardScaler()  

scaler.fit(X)

X_scaled = scaler.transform(X)

# apply same transformation to test data

X_test_scaled = scaler.transform(X_test)  # doctest: +SKIP



Network.fit(X_scaled, y)

predictions=Network.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission2_NN.csv', index=False)
from sklearn.linear_model import LogisticRegression



Logreg=LogisticRegression()



Logreg.fit(X_scaled, y)

predictions = Logreg.predict(X_test_scaled)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission2_logreg.csv', index=False)