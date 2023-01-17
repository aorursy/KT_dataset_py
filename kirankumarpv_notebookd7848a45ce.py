# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.cross_validation import KFold

 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



titanic_pass = pd.read_csv('../input/ChanDarren_RaiTaran_Lab2a.csv')

titanic_pass_collist =  list(titanic_pass)

for i in titanic_pass_collist:

    print(i)

#titanic_pass.head(30)



plt.figure()

sb.barplot(x="Embarked", y="Survived", hue="Sex", data = titanic_pass)

plt.figure()

sb.pointplot(x="Pclass", y="Survived", hue = "Sex", data = titanic_pass, 

            palette = {"male":"Blue", "female":"pink"}, 

                        markers = ["*", "o"], linestyles = ["-", "--"]);

plt.figure()

sb.barplot(x="Embarked",y="Survived", hue="Sex", data=titanic_pass);



#print

def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels = group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df



def format_embarked(df):

    df.Embarked = df.Fare.fillna('0')

    return df



def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df  



def drop_features(df):

    return df.drop(['Ticket', 'Name'], axis=1)





def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_name(df)

    df = format_embarked(df)

    df = drop_features(df)

    return df



titanic_pass = transform_features(titanic_pass)

#data_test = transform_features(data_test)

#titanic_pass.head()

#titanic_pass.Age.fillna(-0.5)

#titanic_pass.head(30)

plt.figure()

sb.barplot(x="Age", y="Survived", hue="Sex", data=titanic_pass);

plt.figure()

sb.barplot(x="Cabin", y="Survived", hue="Sex", data=titanic_pass);

plt.figure()

sb.barplot(x="Fare", y="Survived", hue="Sex", data=titanic_pass);

plt.figure()





def encode_features(df_train):

    features = ['Fare','Cabin','Age','Sex','Lname','NamePrefix','Embarked']

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_train[feature])

        df_train[feature]=le.transform(df_train[feature])

    return df_train



#titanic_pass.head(30)

       

titanic_pass = encode_features(titanic_pass)

#titanic_pass.head(30)

titanic_pass.sample(30)





X_all = titanic_pass.drop(['Survived','PassengerId'],axis=1)

y_all = titanic_pass['Survived']

num_test = 0.10

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)



# Choose the type of classifier. 

clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)



predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))





def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 



run_kfold(clf)





ids = titanic_pass['PassengerId']

predictions = clf.predict(X_all)





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

# output.to_csv('titanic-predictions.csv', index = False)

output.head(n=20)

output.to_csv('predictions.csv', sep='\t')






