# Import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



from sklearn import preprocessing

from sklearn import metrics
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



full_data = [train, test]

#print (df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())



# train['Fare'].fillna(train['Fare'].median(), inplace = True)

# train['FareBin'] = pd.qcut(train['Fare'], 4)

# train['FareBin']
#Pre-processing



# Create new features

for dataset in full_data:

    # dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    #-----

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 

                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    #---

# stat_min = 10

# title_names = (train['Title'].value_counts() < stat_min)

# train['Title'] = train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)



print(train['Title'].value_counts())

    

for dataset in full_data:



        

    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})

    dataset['Title'] = dataset['Title'].fillna(0)

    

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

    dataset['IsAlone'] = 0

    #dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    dataset['IsAlone'].loc[dataset['FamilySize'] == 1] = 1

    

    # Complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
full_data[0] = full_data[0].drop(['Name', 'PassengerId'], axis=1)

full_data[1] = full_data[1].drop('Name', axis=1)
# Convert categorical data into numerical data, and something else 

guess_ages = np.zeros((2,3))





for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C': 1, 'Q':2}).astype(int)

    

    dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)

    #dataset['FareBand'] = pd.qcut(dataset['Fare'], 4)

    

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1}).astype(int)



    #Note: train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)    

    #Fix nan age

    #row_index = dataset.Age.isna()

    #dataset['Age'] = dataset['Age'].fillna(train.Age.mean()).astype(int)

   

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

     

    dataset['HasCabin'] = 0

    dataset.loc[dataset['Cabin'].isna() == False, 'HasCabin'] = 1

    dataset['Age*Class'] = dataset.Age * dataset.Pclass

    

    print(dataset.columns)



full_data[0] = full_data[0].drop(['Parch', 'SibSp', 'Ticket', 'Cabin'], axis=1)

full_data[1] = full_data[1].drop(['Parch', 'SibSp', 'Ticket', 'Cabin'], axis=1)
train = full_data[0]

test = full_data[1]

train.head()
X = train.drop("Survived", axis=1)

y = train["Survived"]

XforTest  = test.drop("PassengerId", axis=1)

X.head()
# train.dtypes

# train.describe()

# train.info()

# type(train)

# train.isnull().sum()
# Normalize Data

# X = preprocessing.StandardScaler().fit(X).transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

print('Train set: ', X_train.shape, y_train.shape)

print('Test set: ', X_test.shape, y_test.shape)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.03, solver='liblinear').fit(X_train,y_train)

LR0 = LogisticRegression(C=0.03, solver='liblinear').fit(X,y)



print('Train set accuracy: ', metrics.accuracy_score(y_train, LR.predict(X_train)))

print('Test set accuracy: ', metrics.accuracy_score(y_test, LR.predict(X_test)))
# KNN

from sklearn.neighbors import KNeighborsClassifier

k = 4

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)

neigh0 = KNeighborsClassifier(n_neighbors = k).fit(X,y)

print('Train set accuracy: ', metrics.accuracy_score(y_train, neigh.predict(X_train)))

print('Test set accuracy: ', metrics.accuracy_score(y_test, neigh.predict(X_test)))
# SVM

from sklearn import svm

clf = svm.SVC(gamma = 'auto')

clf.fit(X_train, y_train)

clf0 = svm.SVC(gamma = 'auto').fit(X, y)

print('Train set accuracy: ', metrics.accuracy_score(y_train, clf.predict(X_train)))

print('Test set accuracy: ', metrics.accuracy_score(y_test, clf.predict(X_test)))
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

acc_knn = round(knn.score(X_test, y_test) * 100, 2)

acc_knn
# Random Forest

from sklearn.ensemble import RandomForestClassifier

cla = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

cla0 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0).fit(X, y)





# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = cla, X=X , y=y , cv = 10)

print("Random Forest:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std())
def mytree(df):

    

    #initialize table to store predictions

    Model = pd.DataFrame(data = {'Predict':[]})

    male_title = [4] #survived titles



    for index, row in df.iterrows():



        #Question 1: Were you on the Titanic; majority died

        Model.loc[index, 'Predict'] = 0



        #Question 2: Are you female; majority survived

        if (df.loc[index, 'Sex'] == 1):

                  Model.loc[index, 'Predict'] = 1



        #Question 3A Female - Class and Question 4 Embarked gain minimum information



        #Question 5B Female - FareBin; set anything less than .5 in female node decision tree back to 0       

        if ((df.loc[index, 'Sex'] == 1) & 

            (df.loc[index, 'Pclass'] == 3) & 

            (df.loc[index, 'Embarked'] == 0)  &

            (df.loc[index, 'Fare'] > 0)):

                  Model.loc[index, 'Predict'] = 0



        #Question 3B Male: Title; set anything greater than .5 to 1 for majority survived

        if ((df.loc[index, 'Sex'] == 0 & (df.loc[index, 'Title'] in male_title))):

            Model.loc[index, 'Predict'] = 1

        

        

    return Model





#model data

Tree_Predict = mytree(train)

print('Decision Tree Model Accuracy/Precision Score: {:.2f}%\n'.format(metrics.accuracy_score(train['Survived'], Tree_Predict)*100))
#test_X = np.asanyarray(X_test)

#test_X = preprocessing.StandardScaler().fit(test_X).transform(test_X)

pred_y = neigh0.predict(XforTest)

len(pred_y)
# BaggingClassifier

from sklearn.ensemble import BaggingClassifier

BR = BaggingClassifier().fit(X,y)

BR_y_pred = BR.predict(XforTest)

my_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': BR_y_pred})

my_submission.to_csv('submission.csv', index=False)