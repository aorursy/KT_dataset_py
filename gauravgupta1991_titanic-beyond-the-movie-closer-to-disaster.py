# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
RawData = pd.read_csv('../input/train.csv');

#X = RawData.copy().drop('Survived', 1);

X = RawData.copy();

Y = RawData['Survived'];

print(X.head())

print(list(X))
print(X[X['Survived'] == 1].describe())
#print(X[X['Survived'] == 0].describe())
#print(pd.DataFrame({'count' : X[['Fare', 'Pclass']].groupby( [ "Pclass", 'Fare'] ).size()}))

X_groupBy_Class = X.groupby( [ "Pclass"] );

#print(X_groupBy_Class.mean()) #head(10))
X_groupBy_Class_And_Gender = X.groupby( [ "Pclass", "Sex"] );

print(X_groupBy_Class_And_Gender.mean()) #head(10))
X_categorized_by_age = pd.cut(X['Age'], np.arange(0, 90, 10)); # return array of half open bins to which `age` belongs.

#print(X.groupby([X_categorized_by_age, 'Pclass', 'Sex'])['Survived', 'SibSp', 'Parch'].mean())
#X.groupby([X_categorized_by_age]).mean()['Survived'].plot.bar()
X.corr()
print(list(X))

Tickets_df = X[['Ticket', 'Pclass']]

X = X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
Tickets_df.groupby('Pclass').describe()
def print_missing_age(df, targetCol = 'Age', checkAgainstCol= 'Pclass'):

    print('=== Analysis of missing Age by pClass ===')

    Missing_Age_Df = df[[checkAgainstCol, targetCol]]

    #Missing_Df = df[df.isnull().any(axis=1)]

    Missing_Age_Df = Missing_Age_Df[Missing_Age_Df.isnull().any(axis=1)]

    print('Description of %s when %s was missing', checkAgainstCol, targetCol, Missing_Age_Df[checkAgainstCol].describe())

    print('Most frequently occuring class when age was missing', Missing_Age_Df[checkAgainstCol].mode())



def print_not_missing_age(df):

    print('Check distribution of age by Pclass...')

    Not_Missing_Df = df[df[['Age']].notnull().any(axis=1)]

    print(Not_Missing_Df[['Pclass', 'Age']].groupby(['Pclass']).describe())



print('Statistics of age,', X['Age'].describe())

print_missing_age(X)

print_not_missing_age(X)

# handle missing Age



from sklearn.preprocessing import Imputer

mean_imputer = Imputer(missing_values=X['Age'], strategy='mean',axis=0)

X['Age'] = X['Age'].fillna(25)
#X = X.dropna(subset=['Age'])

#X.count()

Y = X['Survived']

Y.head()

X = X.drop(['Survived'], axis=1)

#X.count()
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#print(X.columns)

# Since label encoder accept only 1-d array. so we need to create 1 LE per categorical col

X['Sex'].replace(['female', 'male'], [0, 1], inplace=True)

X = pd.get_dummies(X)

X = X.drop(['Embarked_S'], axis=1)

print(X.head())
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Normalized_X = scaler.fit_transform(X)
from sklearn import svm

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Normalized_X, Y, test_size=0.33, random_state=42)

scores = []

C_Vals = []

#print(Normalized_X.columns)

# Hyper parameter tuning 

for idx in np.arange(1,10, 1):

    C_Val = .01*idx**10

    C_Vals.append(C_Val)

    clf = svm.SVC(C_Val)

    clf.fit(X_train, y_train)

    scores.append(clf.score(X_test, y_test))

    

print(pd.DataFrame({'scores': scores, 'C': C_Vals}))
# Train best performant model...

from sklearn.model_selection import ShuffleSplit



final_clf = svm.SVC(C=0.4)

final_clf.fit(X_train, y_train)

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(final_clf, 'Learning Curve (SVM)', Normalized_X, Y, (0.7, 1.01), cv=cv, n_jobs=4)

#plot_learning_curve(final_clf)
# print(final_clf.predict(X_train))



RawTestData = pd.read_csv('../input/test.csv');

TestX = RawTestData.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

print_missing_age(TestX)

print_not_missing_age(TestX)

TestX['Age'] = TestX['Age'].fillna(24)

TestX['Fare'] = TestX['Fare'].fillna(16)

TestX['Sex'].replace(['female', 'male'], [0, 1], inplace=True)

TestX = pd.get_dummies(TestX)

TestX = TestX.drop(['Embarked_S'], axis=1)

scaler = StandardScaler()

Normalized_Test_X = scaler.fit_transform(TestX)

#print(TestX.columns)

Final_Prediction = final_clf.predict(Normalized_Test_X)

Output = pd.DataFrame({

    'PassengerId': RawTestData['PassengerId'],

    'Survived': Final_Prediction

})
print('PassengerId',',','Survived')

for index, row in Output.iterrows():

    print(row['PassengerId'], ',', row['Survived'])