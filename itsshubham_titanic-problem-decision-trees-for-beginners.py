import pandas as pd

import numpy as np

train_file  = '../input/train.csv'

dataset = pd.read_csv(train_file)

dataset.head()
def parameter_preprocess(X):

    X['Sex'] = pd.get_dummies(X['Sex'])

    X['Embarked'] = pd.get_dummies(X['Embarked'])

    X['Age > 10'] = X['Age'] >= 10

    return X._get_numeric_data()



def remove_params(X):

    del X['Survived'], X['Age'], X['Fare'], X['PassengerId']

    return X



def cross_val(X, y, clf):

    from sklearn import cross_validation

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size = 0.35)

    clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)

def predict_values(X,y, clf, X_test):

    clf.fit(X,y)

    return clf.predict(X_test)



def save2file(Pass_Id, pred):

    df2 = list(zip(Pass_Id, pred))

    df2 = pd.DataFrame(data = df2, columns = ['PassengerId','Survived'])

    df2.to_csv('pred2.csv', index = False)

    return df2
X = parameter_preprocess(dataset)

y = X['Survived']

X = remove_params(X)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 100, random_state=47)

print (cross_val(X,y,clf))
test_file = '../input/test.csv'

test_data = pd.read_csv(test_file)

Pass_Id = test_data['PassengerId']

X_test = parameter_preprocess(test_data)

del X_test['Age'], X_test['Fare'], X_test['PassengerId']

from sklearn.tree import DecisionTreeClassifier

clf1 = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 100)

pred = predict_values(X, y, clf1, X_test)

save2file(Pass_Id, pred)