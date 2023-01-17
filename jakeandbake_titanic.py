import pandas as pd



training_data = pd.read_csv('../input/train.csv')

testing_data = pd.read_csv('../input/test.csv')

all_data = pd.concat([training_data, testing_data],sort=True)

print('done')

print(all_data.Embarked.unique())

print(all_data.Parch.unique())

print(all_data.Sex.unique())

print(all_data.SibSp.unique())

print(all_data.columns)

all_data.head(15)
training_data = training_data[['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Survived']]

testing_data_labels = testing_data['PassengerId']

testing_data = testing_data[['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']]



all_data = all_data[['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Survived']]

print('done')




def fill_age(df):

    med = df['Age'].median()

    df.Age = df.Age.fillna(med)

    return df



def fill_emb(df):

    df.Embarked = df.Embarked.fillna('NA')

    return df



def fill_fare(df):

    med = df['Fare'].median()

    df.Fare = df.Fare.fillna(med)

    return df



def prep_data(df):

    df = fill_age(df)

    df = fill_emb(df)

    df = fill_fare(df)



prep_data(training_data)

prep_data(testing_data)
print(training_data.Embarked.unique())
from sklearn import preprocessing



def encode_features(df_train, df_test):

    features = ['Embarked', 'Parch', 'Pclass', 'Sex']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test



training_data, testing_data = encode_features(training_data, testing_data)







testing_data.isnull().any(axis=0)

from sklearn.model_selection import train_test_split



testNum = .2

features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']



X_all = training_data[features]

Y_all = training_data['Survived']



x_train, x_test, y_train, y_test = train_test_split(training_data[features], training_data['Survived'], test_size=testNum, random_state=23)
from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



clf = RandomForestClassifier()



parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(x_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

print(accuracy_score(y_test, predictions))



 
from sklearn.model_selection import KFold

import numpy as np



def run_kfold(clf):

    kf = KFold(n_splits=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf.split(X_all):

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = Y_all.values[train_index], Y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome))

    

run_kfold(clf)
predictions = clf.predict(testing_data)



output = pd.DataFrame({ 'PassengerId' : testing_data_labels, 'Survived': predictions })



display(output)
output.to_csv('submission.csv', index=False)