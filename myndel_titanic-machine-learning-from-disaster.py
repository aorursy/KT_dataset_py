import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.model_selection import train_test_split



TRAIN_DIR = '../input/titanic/train.csv'

TEST_DIR = '../input/titanic/test.csv'

predict = 'Survived'
train = pd.read_csv(TRAIN_DIR, index_col='PassengerId')

test = pd.read_csv(TEST_DIR, index_col='PassengerId')



#    Survived    Survived                   0 = No, 1 = Yes

#    Pclass      Ticket class               1 = 1st, 2 = 2nd, 3 = 3rd

#    Name        Name

#    Sex         Sex

#    Age         Age in years

#    SibSp       # of siblings / spouses aboard the Titanic

#    Parch       # of parents / children aboard the Titanic

#    Ticket      Ticket number

#    Fare        Passenger fare

#    Cabin       Cabin number

#    Embarked    Port of Embarkation        C = Cherbourg, Q = Queenstown, S = Southampton
# Columns with missing values

train.columns[train.isnull().any()]
def prepare_data(data):

    data = data.drop(['Ticket', 'Name', 'Cabin'], axis=1)



    # 72% of Embarked values are 'S' which stands for Southampton

    # so in my opinion we can fill missing data with this value.

    # Age we will fill with mean values for age



    data['Embarked'] = data['Embarked'].fillna('S')

    data['Age'] = data['Age'].fillna(data['Age'].mean())

    

    # Now we have to transform alphabetic values to numberic

    le = preprocessing.LabelEncoder()

    sex = le.fit_transform(list(data['Sex']))

    embarked = le.fit_transform(list(data['Embarked']))



    data['Sex'] = pd.Series(list(sex), dtype='uint8', index=data.index)

    data['Embarked'] = pd.Series(list(embarked), dtype='uint8', index=data.index)

    

    return data





train = prepare_data(train)
# Let's see how our data looks like

train.head()
X = train.drop(predict, axis=1)

Y = train[predict]



x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
# Train the best model

knn_model = KNeighborsClassifier(n_neighbors=7)



knn_model.fit(x_train, y_train)

knn_acc = knn_model.score(x_val, y_val)



print(f'K-Nearest Neighbours accuracy: {knn_acc}')
linear_model = LinearRegression()



linear_model.fit(x_train, y_train)

linear_acc = linear_model.score(x_val, y_val)



print(f'Linear Regression accuracy: {linear_acc}')
logistic_model = LogisticRegression()



logistic_model.fit(x_train, y_train)

logistic_acc = logistic_model.score(x_val, y_val)



print(f'Logistic Regression accuracy: {logistic_acc}')
dtc_model = DecisionTreeClassifier()



dtc_model.fit(x_train, y_train)

dtc_acc = dtc_model.score(x_val, y_val)



print(f'Decision Tree Classifier accuracy: {dtc_acc}')
test = prepare_data(test)



# Check for empty values

test.columns[test.isnull().any()]



# I will fill NaN values with mean

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
model = dtc_model

predictions = model.predict(test)



series = pd.Series(i for i in predictions)

series = pd.Series(series, name='Survived')



submission = pd.concat([pd.Series(test.index), series], axis=1)

submission

submission.to_csv("submission.csv", index=False)