# Importing necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os



%matplotlib inline
# Loading data files

train_df = pd.read_csv('../input/titanic/train.csv', index_col = 'PassengerId')

test_df = pd.read_csv('../input/titanic/test.csv', index_col = 'PassengerId')
train_df.head()
test_df.head()
train_df.info()
test_df.info()
# Concatinating train_df and test_df

test_df['Survived'] = -1

df = pd.concat([train_df, test_df], sort = True)
df.head()
df.info()
# Visualizing Features vs Survived plots

for col in ['Embarked', 'Sex', 'Pclass', 'Parch', 'SibSp']:

    pd.crosstab(df.loc[df['Survived'] != -1, col], df.loc[df['Survived'] != -1, 'Survived']).plot(kind = 'bar', rot = 1)

    
# Boxplots features

for col in ['Age', 'Fare']:

    df[df[col].notnull() & (df['Survived'] != -1)].boxplot(col, 'Survived');
# Histogram on numerical features to see the skewness of their data

for col, c , b in zip(['Fare', 'Age', 'Parch', 'SibSp'], ['darkorange', 'k', 'c', 'b'], [20, 30, 8, 8]):

    skewness = df[col].skew()

    print(f"Skewness of {col} : ", skewness)

    plt.hist(df.loc[df[col].notnull(), col], color = c, bins = b)

    plt.title(f'Frequency of {col} feature')

    plt.xlabel(col)

    plt.ylabel('Frequency')

    plt.show()
df.describe()
# Valuecounts of fare feature

df['Fare'].value_counts()
# Histogram of 'Fare'

plt.hist(df.loc[df['Fare'].notnull(), 'Fare'], color = 'c', bins = 30)

plt.title("Frequency of Fare feature")

plt.xlabel("Fare")

plt.ylabel("Frequency")

plt.show()
# KDE of 'Fare'

df['Fare'].plot(kind = 'kde',  title = "Fare", color = 'c');

print("Skewness of Fare feature :", df.Fare.skew())
# Passenger with missing Fare feature

df[df.Fare.isnull()]
# Distribution of fares for different embarkment locations

plt.scatter(df.loc[df.Fare.notnull() & df.Embarked.notnull(), 'Embarked'], df.loc[df.Fare.notnull() & df.Embarked.notnull(), 'Fare'], color = 'c', alpha = 0.2);

plt.title("Distribution of fares for different embarkment points")

plt.xlabel("Embarked")

plt.ylabel("Fare")

plt.show()
# Distribution of fares for different classes

plt.scatter(df.loc[df.Fare.notnull() & df.Pclass.notnull(), 'Pclass'], df.loc[df.Fare.notnull() & df.Pclass.notnull(), 'Fare'], color = 'c', alpha = 0.2);

plt.title("Distribution of fares for different classes")

plt.xlabel("Pclass")

plt.ylabel("Fare")

plt.show()
# Plotting fair distribution among people with Pclass 3

plt.hist(df.loc[(df.Pclass == 3) & df.Fare.notnull(), 'Fare'], color = 'c', bins = 12);
df[(df.Pclass == 3) & (df.Embarked == 'S')]
print("Median Fare :", df.loc[(df.Pclass == 3) & (df.Embarked == 'S'), 'Fare'].median())
# Setting value to the median

df.Fare.fillna(df.loc[(df.Pclass == 3) & (df.Embarked == 'S'), 'Fare'].median(), inplace = True)
df.info()
df.describe()
# Embarked Feature

df[df.Age.notnull()]
# Histogram of 'Age'

plt.hist(df.loc[df['Age'].notnull(), 'Age'], color = 'c', bins = 30)

plt.title("Frequency of Age feature")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.show()
# KDE of 'Age'

df['Age'].plot(kind = 'kde',  title = "Age", color = 'c');

print("Skewness of Age feature :", df.Age.skew())
# Function to extract the title from the name

def GetTitle(name):

    title_group = {

        'mr' : 'Mr',

        'mrs' : 'Mrs',

        'miss': 'Miss',

        'master' : 'Master',

        'don' : 'Sir',

        'rev' : 'Sir',

        'dr': 'Officer',

        'mme' : 'Mrs',

        'ms' : 'Mrs',

        'major' : 'officer',

        'lady' : 'Lady',

        'sir' : 'Sir',

        'mlle' : 'Miss',

        'col' : 'Officer',

        'capt' : 'Officer',

        'the countess': 'Lady',

        'jonkheer' : 'Sir',

        'dona' : 'Lady'

    }

    

    first_name_with_title = name.split(',')[1]

    title = first_name_with_title.split('.')[0]

    title = title.strip().lower()

    return title_group[title]

df.Name.map(lambda x : GetTitle(x)).unique()
# create Title feature

df['Title'] = df.Name.map(lambda x : GetTitle(x))
df['Title'].value_counts()
# Box plot of Age with title

df[df.Age.notnull()].boxplot('Age', 'Title');
# replace missing values

title_age_median = df.groupby('Title').Age.transform('median')

df.Age.fillna(title_age_median, inplace = True)
# KDE of 'Age'

df['Age'].plot(kind = 'kde',  title = "Age", color = 'c');

print("Skewness of Age feature :", df.Age.skew())
df['Cabin'].unique()
# extract first character of Cabin string to the deck

def get_deck(cabin):

    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')

df['Deck'] = df['Cabin'].map(lambda x : get_deck(x))
df['Deck'].value_counts()
df.info()
# Dropping cabin feature

df.drop('Cabin',axis = 1, inplace = True)
df.info()
df[df['Embarked'].isnull()]
df[df['Ticket'] == '113572']
df.groupby(df['Pclass']).Embarked.value_counts()
df.groupby(df['Deck']).Embarked.value_counts()
df['Embarked'].fillna('S', inplace = True)
df.info()
# Checking skewness of numrical features

for col, types in zip(df.columns, df.dtypes):

    if types != 'object':

        print(f"The skewness of {col} : ", df[col].skew())

# Using log transformation to reduce skewness of Fare feature

LogFare = np.log(df.Fare + 1.0) 

LogFare.plot(kind = 'hist', color = 'c', bins = 20);

print("Skewness of LogFare Feature : ", LogFare.skew())
df['LogFare'] = LogFare

df.head()
df.describe(include = 'all')
# AgeGroup feature splits the data based on age

"""

0 - 15 childern

16 - 21 young adults

21 - 40 adults

40 - 55 Middle age

55 -> Senior citizen

"""



df['AgeGroup'] = 'child'

df.loc[df.Age >= 16,'AgeGroup'] = 'y_adult'

df.loc[df.Age >= 21, 'AgeGroup'] = 'adult'

df.loc[df.Age >= 40, 'AgeGroup'] = 'm_age'

df.loc[df.Age >= 55, 'AgeGroup'] = 's_citizen'

df.head()
# Adding is Mother Feature

df['IsMother'] = np.where((df['Title'] != 'Miss') & (df['Age'] > 21) & (df['Parch'] != 0), 1, 0)

df.head()
df["IsMother"].value_counts()
df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

df.FamilySize.value_counts()
df.FamilySize.plot(kind = 'hist', color = 'c');
df.FamilySize.skew()
df['LogFamilySize'] = np.log(df.FamilySize + 1)

df['LogParch'] = np.log(df.Parch + 1)

df['LogSibSp'] = np.log(df.SibSp + 1)

df.LogFamilySize.plot(kind = 'hist', color = 'c');

print('Skewness of LogFamily Size :', df.LogFamilySize.skew())
# Since there features are very skewed, I will try feature binning

df['FamilyType'] = 'Single'

df.loc[df.FamilySize > 1, 'FamilyType'] = 'Couple'

df.loc[df.FamilySize > 2, 'FamilyType'] = 'SmallFamily'

df.loc[df.FamilySize > 4, 'FamilyType'] = 'LargeFamily'

df.head()
df.FamilyType.value_counts()
df.drop(['FamilySize'], axis = 1, inplace = True)
df.head()
df['IsAdult'] = np.where(df.Age >= 18, 1, 0)
df = pd.get_dummies(df, columns = ['Pclass', 'Title', 'AgeGroup', 'Embarked', 'Sex', 'FamilyType', "Deck"])

df.info()
df.drop(['Name', 'Ticket', 'Fare', 'Parch', 'SibSp'], axis = 1, inplace = True)
df.info()
train_df = df[df['Survived'] != -1]

test_df = df[df['Survived'] == -1]
train_df
test_df
test_df.drop('Survived', axis = 1, inplace = True)

test_df.info()
columns = [column for column in df.columns if column != 'Survived']

columns.append('Survived')

train_df = df.loc[df['Survived'] != -1, columns]
train_df
test_df
def get_submission_file(model, filename):

    # model.fit(X, y)

    # converting to the matrix

    test_X = test_df.to_numpy().astype('float')

    # get predictions

    predictions = model.predict(test_X)

    # submission dataframe

    df_submission = pd.DataFrame({'PassengerId' : test_df.index, 'Survived' : predictions})

    # submission file

    submission_data_path = os.path.join(os.path.pardir, 'data', 'predictions')

    submission_file_path = os.path.join(submission_data_path, filename)

    # write to the file

    df_submission.to_csv(submission_file_path, index = False)
X = train_df.iloc[:,0:-1].to_numpy().astype('float')

y = train_df.loc[:,'Survived'].to_numpy()

print(X.shape, y.shape)
from sklearn.model_selection import train_test_split



def test_model(model, trials):

    total_score = 0

    for trial in range(trials):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        model.fit(X_train,  y_train)

        total_score += model.score(X_test, y_test)

    print(f"Average Accuracy of the model : {round(total_score / trials, 3)}")  
from sklearn import preprocessing

X = preprocessing.scale(X)
from sklearn.dummy import DummyClassifier

dummy_model = DummyClassifier(strategy = 'most_frequent')



test_model(dummy_model, 10)
from sklearn.linear_model import LogisticRegression

lr_1 = LogisticRegression(max_iter = 500)



test_model(lr_1, 10)
from sklearn.linear_model import RidgeClassifierCV

ridge_classifier_model = RidgeClassifierCV(alphas = (0.01, 0.1, 1, 10), cv = 5)



test_model(ridge_classifier_model, 10)
ridge_classifier_model.fit(X, y)

print("Prefered alpha :", ridge_classifier_model.alpha_)
ridge_classifier_model = RidgeClassifierCV(alphas = (1, 5, 10, 15, 20), cv = 5)



test_model(ridge_classifier_model, 10)
ridge_classifier_model.fit(X, y)

print("Prefered alpha :", ridge_classifier_model.alpha_)
# Narrowing down alpha

ridge_classifier_model = RidgeClassifierCV(alphas = (10, 11, 12, 13, 14, 15, 16, 17, 18 ,19, 20), cv = 5)



test_model(ridge_classifier_model, 10)
ridge_classifier_model.fit(X, y)

print("Prefered alpha :", ridge_classifier_model.alpha_)
# Narrowing down alpha

ridge_classifier_model = RidgeClassifierCV(alphas = (11.7, 12, 12.5), cv = 5)



test_model(ridge_classifier_model, 10)
ridge_classifier_model.fit(X, y)

print("Prefered alpha :", ridge_classifier_model.alpha_)
# Narrowing down alpha

ridge_classifier_model = RidgeClassifierCV(alphas = (12.3, 12.5, 12.7), cv = 22)



test_model(ridge_classifier_model, 10)
ridge_classifier_model.fit(X, y)

print("Prefered alpha :", ridge_classifier_model.alpha_)
# Narrowing down alpha

ridge_classifier_model = RidgeClassifierCV(alphas = (10, 11, 12, 13, 14, 15, 16, 17, 18 ,19, 20), cv = 20)



test_model(ridge_classifier_model, 10)
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()



test_model(RFC, 10)
RFC_1 = RandomForestClassifier(ccp_alpha = 0.01, n_estimators = 1000)



test_model(RFC_1, 10)
RFC_2 = RandomForestClassifier(max_features = 0.4, n_estimators = 1000, min_samples_leaf = 20)



test_model(RFC_2, 10)
RFC_3 = RandomForestClassifier(max_features = None, n_estimators = 1000, min_samples_leaf = 100)



test_model(RFC_3, 10)
from sklearn.neural_network import MLPClassifier

nn_1 = MLPClassifier(max_iter = 1000, alpha = 0.3)



test_model(nn_1, 10)