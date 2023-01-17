import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
%matplotlib inline
train_raw = pd.read_csv('../input/train.csv')
train_raw.head()
# Replace Sex column with digit, male=1, female=0
train = train_raw
# check the keys
#train.keys()

def gender_digit(x):
    if x == 'male':
        return 1
    else:
        return 0
train['Sex'] = train['Sex'].apply(gender_digit)
# Reasonable effective features: Pclass, Sex, Age, SibSp, Fare
train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Fare']]
train.keys()
# define features X and target y
X = train.drop('Survived', axis=1)
y = train['Survived']
# fill non-valid value with mean of the column
X.head()
# To check if any column has invalid values
X.isnull().sum()
# to fill nan-values with the mean
X['Age'] = X['Age'].fillna(np.mean(X['Age']))
mean_sex = np.mean(train['Survived'][train['Sex']==1])
print('Male suvival rate: ', mean_sex)
mean_Sib = np.mean(train['Survived'][train['SibSp']!=0])
print('Survival rate for people with sibling: ', mean_Sib)
print('len of X: ',len(X), 'len of y:', len(y))
from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size=0.33, 
                                                    random_state=42)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
# Accuracy of training
tr_acc = np.mean(y_train == classifier.predict(X_train))*100
print('Training accuracy: ', tr_acc)
te_acc = np.mean(y_test==classifier.predict(X_test))*100
print('Validation accuracy: ', te_acc)
classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(X_train, y_train)
# Training accuracy 
tr_acc = np.mean(classifier.predict(X_train)==y_train)*100
print("Depth=3, training accuracy :", tr_acc)
# Validation accuracy
te_acc = np.mean(classifier.predict(X_test)==y_test)*100
print("Depth=3, training accuracy :", te_acc)

test = pd.read_csv('../input/test.csv')
test.head()
# Check for Nan entries
print(test.isnull().sum())
# There are some Nan values in 'Age' column, to Fix this
test['Age'] = test['Age'].fillna(np.mean(test['Age']))
test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))
X_to_predict = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']]
# replace the gender with 0,1 values
X_to_predict['Sex'] = X_to_predict['Sex'].apply(gender_digit)
y_predicted = classifier.predict(X_to_predict)
submission = pd.DataFrame({'PassengerId':test['PassengerId'],
                           'Survived':y_predicted})
submission.head()
# Convert DataFrame to a csv file that can be uploaded
filename = 'Titanic_predict1.csv'
submission.to_csv(filename, index=False)
print('Saved file '+ filename)