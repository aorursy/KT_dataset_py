import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('../input/titanic/train.csv')
# PassengerId column is not something to learn on
data.drop(columns=['PassengerId'], inplace=True)
test = pd.read_csv('../input/titanic/test.csv')
test.drop(columns=['PassengerId'], inplace=True)
X = data.drop(columns=['Survived'])
# Survived column is what will be predicting
Y = data['Survived']
for i in range(len(X.columns)):
    print(X.iloc[:, i].value_counts(), end='\n\n')
X['Ticket'] = X['Ticket'].apply(lambda s:  str(s.split()[-1][0]))
test['Ticket'] = test['Ticket'].apply(lambda s:  str(s.split()[-1][0]))
X['Name'] = X['Name'].apply(lambda s: s.split(',')[0])
test['Name'] = test['Name'].apply(lambda s: s.split(',')[0])
XNamevalue_counts = X['Name'].value_counts()
testNamevalue_counts = test['Name'].value_counts()
def GetFamilySize(name):
    family_size = 0
    try:
        family_size += XNamevalue_counts[name]
    finally:
        try:
            family_size += testNamevalue_counts[name]
        finally:
            return family_size
X['Family_size'] = X['Name'].apply(lambda s: GetFamilySize(s))
test['Family_size'] = test['Name'].apply(lambda s: GetFamilySize(s))
X['Cabin'] = X['Cabin'].apply(lambda s: str(s)[0])
test['Cabin'] = test['Cabin'].apply(lambda s: str(s)[0])
X_size_age = X.shape[0] - X['Age'].isna().sum()
test_size_age = test.shape[0] - test['Age'].isna().sum()
mean_age = (X_size_age * X['Age'].mean() + 
            test_size_age * test['Age'].mean()) / \
            (X_size_age + test_size_age)
X['Age'].fillna(mean_age, inplace=True)
test['Age'].fillna(mean_age, inplace=True)
X_size_fare = X.shape[0] - X['Fare'].isna().sum()
test_size_fare = test.shape[0] - test['Fare'].isna().sum()
mean_fare = (X_size_fare * X['Fare'].mean() + 
            test_size_fare * test['Fare'].mean()) / \
            (X_size_fare + test_size_fare)
test['Fare'].fillna(mean_fare, inplace=True)
X_dum = pd.get_dummies(X, drop_first=True)
X_test_dum = pd.get_dummies(test, drop_first=True)
X_dum.shape , X_test_dum.shape
# symetric difference:
# the columns that are not contained either in training or in testing datasets
unnecesarry_columns = X_test_dum.columns ^ X_dum.columns
# this are the columns that exist in testing dataset, but not in training
unnecesarry_columns_X_dum = unnecesarry_columns & X_dum.columns 
# this are the columns that exist in training dataset, but not in testing
unnecesarry_columns_X_test_dum = unnecesarry_columns & X_test_dum.columns
X_dum = X_dum.drop(columns=unnecesarry_columns_X_dum)
X_test_dum = X_test_dum.drop(columns=unnecesarry_columns_X_test_dum)
# after this training and testing datasets should have the same columns
all(X_dum.columns == X_test_dum.columns)
ros = RandomOverSampler()
A, B = ros.fit_resample(X_dum, Y)
X_ros = pd.DataFrame(A, columns=X_dum.columns)
Y_ros = pd.Series(B)
X_train, X_val, y_train, y_val = train_test_split(X_ros,
                                                  Y_ros,
                                                  test_size=0.25)
model_rf = RandomForestClassifier(criterion='gini', n_estimators=121, 
                                  max_features=80, random_state=18)
model_rf.fit(X_ros, Y_ros)
print('accuracy: ', accuracy_score(model_rf.predict(X_ros), Y_ros))
pred_rf = pd.DataFrame({'PassengerId' : np.arange(len(X_test_dum)) + 892,
                        'Survived': model_rf.predict(X_test_dum)})
pred_rf.to_csv('pred_rf.csv', index=False)
pred_rf