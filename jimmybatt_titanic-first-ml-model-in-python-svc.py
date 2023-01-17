import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/train.csv', quotechar="\"")

df.head(5)
# create a function 

def familyAggregate(x, y):

    if x >= 1 or y >= 1:

        return 1

    else:

        return 0



df['familyAgg'] = list(map(familyAggregate, df['SibSp'], df['Parch']))
y = df.Survived

predictors = ['Pclass', 'Sex', 'Age', 'familyAgg', 'Fare', 'Embarked']

X = df[predictors]



# Create dummies for categorical field when possible !

X_dummies = pd.get_dummies(X)

X_dummies.head(3)
from sklearn.preprocessing import Imputer

my_imputer = Imputer()

data_with_imputed_values = my_imputer.fit_transform(X_dummies)
from sklearn import svm

from sklearn.model_selection import cross_val_score, train_test_split



X_train, X_test, y_train, y_test = train_test_split(data_with_imputed_values, 

                                                    y,

                                                    train_size=0.7, 

                                                    test_size=0.3, 

                                                    random_state=0)



# SVC linear (sadly poly SVC is taking too long to run!)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

scores = cross_val_score(clf, data_with_imputed_values, y, cv=5)

print(scores)

print(scores.mean())
from sklearn import svm



# building model with the entire training set

model = svm.SVC(kernel='linear', C=1).fit(data_with_imputed_values, y)



# ==================== IMPORT AND TREAT TEST DATA =======================

# Read the test data

df = pd.read_csv('../input/test.csv', quotechar="\"")



df['familyAgg'] = list(map(familyAggregate, df['SibSp'], df['Parch']))



# Create X

predictors = ['Pclass', 'Sex', 'Age', 'familyAgg', 'Fare', 'Embarked']

X = df[predictors]



# create dummies

X_dummies = pd.get_dummies(X)



# fill in missing value

my_imputer = Imputer()

test_X = my_imputer.fit_transform(X_dummies)



# ====================== MAKE PREDICTIONS ================

predicted_survivors = model.predict(test_X)



# ======================= PREPARE SUBMISSION FILES ===============

my_submission = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': predicted_survivors})

# you could use any filename. We choose submission here

my_submission.to_csv('../input/submission.csv', index=False)