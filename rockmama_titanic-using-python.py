import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

"""

Created on Sun Aug 20 10:40:51 2017

"""



data_train = pd.read_csv('train.csv')

data_test = pd.read_csv('test.csv')



data_train.sample(3)

data_train.Fare.describe()

data_train.Age.describe()





data_train.shape

data_test.shape



pd.value_counts(data_train.Embarked)

pd.value_counts(data_train.Cabin)



sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)

sns.boxplot(x="Parch", y="Survived", hue="Sex", data=data_train, palette="PRGn")

sns.despine(offset=10, trim=True)

sns.pointplot(x="Pclass",y="Survived", hue="Sex", data= data_train)



def simplify_ages(df):

    df.Age=df.Age.fillna(-0.5)

    bins=(-1,0,5,12,18,25,35,60,120)

    group_names = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']

    categories = pd.cut(df.Age,bins,labels=group_names)

    df.Age=categories

    df.Age.cat.categories = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x:x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins= (-1,0,8,15,31,1000)

    group_names= ['Unknown','1_Quartile','2_Quartile','3_Quartile','4_Quartile']

    categories = pd.cut(df.Fare, bins, labels = group_names)

    df.Fare = categories

    return df



def format_names(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['Fname'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df



def drop_features(df):

    return df.drop(['Ticket','Name','Embarked'],axis=1)





def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_names(df)

    df = drop_features(df)

    return df

    

data_train = transform_features(data_train)   

data_test = transform_features(data_test) 

data_train.head() 

    

data_train.Age.cat.categories = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']

sns.barplot(x="Age",y="Survived",hue="Sex", data=data_train)

sns.barplot(x="Age",y="Survived",hue="Fname", data=data_train)

sns.barplot(x="Cabin",y="Survived",hue="Sex", data=data_train)

sns.barplot(x="Fare", y="Survived", hue="Sex", data=data_train)



    

pd.crosstab(data_train['Fname'], data_train['Survived'])

pd.crosstab(data['Age'], data['Sex'])

data_train.info()

data_test.info()









from sklearn import preprocessing

def encode_features(df_train, df_test):

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'Fname']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test

    

data_train, data_test = encode_features(data_train, data_test)

data_train.head()



from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)

y_all = data_train['Survived']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

clf = RandomForestClassifier()



#==============================================================================

# # Choose some parameter combinations to try

# parameters = {'n_estimators': [4, 6, 9], 

#               'max_features': ['log2', 'sqrt','auto'], 

#               'criterion': ['entropy', 'gini'],

#               'max_depth': [2, 3, 5, 10], 

#               'min_samples_split': [2, 3, 5],

#               'min_samples_leaf': [1,5,8]

#              }

# 

# # Type of scoring used to compare parameter combinations

# acc_scorer = make_scorer(accuracy_score)

# 

# # Run the grid search

# grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

# grid_obj = grid_obj.fit(X_train, y_train)

# 

# # Set the clf to the best combination of parameters

# clf = grid_obj.best_estimator_

#==============================================================================



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)



predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))





ids = data_test['PassengerId']

predictions = clf.predict(data_test.drop('PassengerId', axis=1))

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.head()

output.to_csv('titanic-predictions.csv', index = False)





data_train.isnull().sum()