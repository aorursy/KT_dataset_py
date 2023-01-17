# Import libraries for data munging

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix



# Import machine learning libraries from scikit-learn

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier



# split data into training and testing set

from sklearn.model_selection import train_test_split



from sklearn import metrics
data = pd.read_csv('../input/train.csv')

data.head(10)
labels = 'Not Survived', 'Survived',

f, ax = plt.subplots(figsize=(12, 6))

data['Survived'].value_counts().plot.pie(autopct='%1.2f%%',ax = ax, labels = labels)

ax.set_ylabel('')

ax.axis('equal')

plt.show()
data.info()
data[['Name', 'Age']].loc[data['Age'].isnull()]
data['Title'] = data.Name.map(lambda x: x.split(',')[1].split('.')[0].strip())
data.groupby(data['Title'])['Title'].count()

data.groupby(['Title', 'Sex'])['Sex'].count()
cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(data.Title, data.Sex).T.style.background_gradient(cmap=cm)
data['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Mlle', 'Mme', 'Ms', 'the Countess', 'Rev', 'Sir'],

                      ['Mr', 'Mr', 'Mr', 'Mr', 'Mr', 'Mrs', 'Mr', 'Mrs', 'Mrs', 'Miss', 'Mrs', 'Mr', 'Mrs'], inplace = True)

data.groupby(['Title'])['Title'].count()
data.loc[(data.Age.isnull()) & (data.Title == 'Master'), 'Age'] = data[data['Title'] == 'Master']['Age'].mean()

data.loc[(data.Age.isnull()) & (data.Title == 'Miss'), 'Age'] = data[data['Title'] == 'Miss']['Age'].mean()

data.loc[(data.Age.isnull()) & (data.Title == 'Mr'), 'Age'] = data[data['Title'] == 'Mr']['Age'].mean()

data.loc[(data.Age.isnull()) & (data.Title == 'Mrs'), 'Age'] = data[data['Title'] == 'Mrs']['Age'].mean()
data.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)
data[data['Embarked'].isnull()]
data.dropna(subset=['Embarked'], how = 'all', inplace = True)
data.isnull().sum()
# Encode 'Sex' to 1 or 0 label

data['Sex'].replace(['male', 'female'], [1, 0], inplace = True)



# Apply One Hot Encoding to 'Title'

titleDf = pd.DataFrame()

titleDf = pd.get_dummies(data['Title'], drop_first = True)

data = pd.concat([data, titleDf], axis = 1, join='inner')



# Apply One Hot Encoding to 'Embarked'

embarkedDf = pd.DataFrame()

embarkedDf = pd.get_dummies(data['Embarked'], drop_first = True)

data = pd.concat([data, embarkedDf], axis = 1, join='inner')



# Drop 'Title' and 'Embarked' since we already have encoded them

data.drop(['Title', 'Embarked'], axis = 1, inplace = True)

data.head()
train_data, test_data = train_test_split(data, test_size = 0.2, random_state = 0)



X_train = train_data.iloc[:,1:]

y_train = train_data.iloc[:,:1]



X_test = test_data.iloc[:,1:]

y_test = test_data.iloc[:,:1]
svm = SVC(kernel='linear', gamma = 0.1, C = 1)

svm.fit(X_train, y_train)

prediction = svm.predict(X_test)

print('Linear SVM Accuracy : ', metrics.accuracy_score(prediction, y_test))



logisticRegression = LogisticRegression()

logisticRegression.fit(X_train,y_train)

prediction = logisticRegression.predict(X_test)

print('Logistic Regression Accuracy : ', metrics.accuracy_score(prediction, y_test))



decisionTree = DecisionTreeClassifier()

decisionTree.fit(X_train,y_train)

prediction = decisionTree.predict(X_test)

print('Decision Tree Accuracy : ', metrics.accuracy_score(prediction, y_test))



knn = KNeighborsClassifier() 

knn.fit(X_train,y_train)

prediction = knn.predict(X_test)

print('KNN Accuracy : ', metrics.accuracy_score(prediction, y_test))



randomForest = RandomForestClassifier(n_estimators=100)

randomForest.fit(X_train,y_train)

prediction = randomForest.predict(X_test)

print('Random Forest Accuracy : ', metrics.accuracy_score(prediction, y_test))