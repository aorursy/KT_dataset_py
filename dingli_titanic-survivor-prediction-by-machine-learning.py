# data process
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# system
import os
import sys

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Any results you write to the current directory are saved as output.
# Input data files are available in the "../input/" directory.
print(os.listdir("../input"))

# read data into pandas' data frame
train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')

# make a deep copy to keep the original input
train_df = train_raw.copy(deep=True)
test_df = test_raw.copy(deep=True)

combine = [train_df, test_df]
# train data sample
train_df.head()

# test data sample
test_df.head()
# train data info
train_df.info()
# test data info
train_df.info()
#distribution of survived
train_df[['PassengerId', 'Survived']].groupby('Survived').count()
#distribution of Pclass
train_df[['Pclass', 'Survived']].groupby(['Pclass']).agg(['count', 'mean'])
#get title from name
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
#Aggregate the Titles
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
#distribution of Title
train_df[['Title', 'Survived']].groupby(['Title']).agg(['count', 'mean']).sort_values(by=[('Survived','mean')], ascending=False)
#convert the categorical titles to ordinal
title_mapping = {"Mrs": 5, "Miss": 4, "Master": 3, "Rare": 2, "Mr": 1}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train_df.head()
# Gives the length of the name
for dataset in combine:
    dataset['Name_length'] = dataset['Name'].apply(len)
train_df.head()
#distribution of Title
train_df[['Name_length', 'Survived']].groupby(['Name_length']).agg(['count', 'mean'])
#drop Name
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
#distribution of Sex
train_df[['Sex', 'Survived']].groupby(['Sex']).agg(['count', 'mean'])
#convert the categorical Sex to ordinal
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
#distribution of sibling or spouse
train_df[['SibSp', 'Survived']].groupby(['SibSp']).agg(['count', 'mean'])
#distribution of parent or childen
train_df[['Parch', 'Survived']].groupby(['Parch']).agg(['count', 'mean'])
#distribution of ticket
train_df[['Ticket', 'Survived']].groupby(['Ticket']).agg(['count', 'mean']).sort_values(by=[('Survived','mean')], ascending=False)
#distribution of Cabin
train_df[['Cabin', 'Survived']].groupby(['Cabin']).agg(['count', 'mean']).sort_values(by=[('Survived','mean')], ascending=False)
#Get the first letter of Cabin
for dataset in combine:
    dataset['CabinLetter'] = dataset.Cabin.str[0]
    
train_df.head()


#distribution of the first letter of Cabin
train_df[['CabinLetter', 'Survived']].groupby(['CabinLetter']).agg(['count', 'mean']).sort_values(by=[('Survived','mean')], ascending=False)
#group cabins by suviving rate
title_mapping = {"T": 0, "A": 1, "G": 1, "C": 2, "F": 2, "B":3, "E":3, "D":3}
for dataset in combine:
    dataset['Cabin'] = dataset['CabinLetter'].map(title_mapping)
    dataset['Cabin'] = dataset['Cabin'].fillna(0)
    dataset['Cabin'] = dataset['Cabin'].astype(int)
train_df[['Cabin', 'Survived']].groupby(['Cabin']).agg(['count', 'mean']).sort_values(by=[('Survived','mean')], ascending=False)
#drop CabinLetter and Ticket
train_df = train_df.drop(['Ticket', 'CabinLetter'], axis=1)
test_df = test_df.drop(['Ticket', 'CabinLetter'], axis=1)
combine = [train_df, test_df]

train_df.head()
train_df.head()
train_df[['Embarked', 'Survived']].groupby(['Embarked']).agg(['count', 'mean']).sort_values(by=[('Survived','mean')], ascending=False)
#fill the null Embarked with most frequent port
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
#convert the categorical Embarked to ordinal
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'C': 2, 'Q': 1, 'S':0} ).astype(int)

train_df.head()
#distribution of test data
test_df.describe()
#fill the null Fare in test with the median value
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
#distribution of Fare
train_df['FareBand'] = pd.qcut(train_df['Fare'], 30)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 9.5, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 9.5) & (dataset['Fare'] <= 52.4), 'Fare'] = 1
    dataset.loc[dataset['Fare'] > 52.4, 'Fare']   = 2
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)
train_df[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Fare', ascending=True)
#distribution of age
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
#set Passenger id as index
for dataset in combine:
    dataset.set_index('PassengerId', inplace=True)
dataset.head(10)


#get train data having age value
train_with_age_df = train_df[train_df['Age'].notnull()]
train_with_age_df.shape
#distribution of age
train_with_age_df['AgeBand'] = pd.qcut(train_with_age_df['Age'], 30)
train_with_age_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
#the survive rate is obvious high when age <7, set it as one band, others and missing value to another band. 
for dataset in combine:
    dataset['Age'] = np.where((dataset['Age']> 0) & (dataset['Age']<7), 1, 0)

train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
X = train_df.drop(['Survived'], axis=1)
y= train_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape, X_test.shape
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_train_log = round(logreg.score(X_train, y_train) * 100, 2)
acc_test_log = round(logreg.score(X_test, y_test) * 100, 2)
print('logistic regression train accurary: ',acc_train_log)
print('logistic regression test accurary: ',acc_test_log)
#coefficients of each factors
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines

svc = SVC()
svc.fit(X_train, y_train)
acc_train_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_test_svc = round(svc.score(X_test, y_test) * 100, 2)
print('Support Vector Machine train accurary: ',acc_train_svc)
print('Support Vector Machine test accurary: ',acc_test_svc)
# K-Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
acc_train_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_test_knn = round(knn.score(X_test, y_test) * 100, 2)
print('K-Nearest Neighbors train accurary: ',acc_train_knn)
print('K-Nearest Neighbors test accurary: ',acc_test_knn)
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
acc_train_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_test_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
print('Decision Tree train accurary: ',acc_train_decision_tree)
print('Decision Tree test accurary: ',acc_test_decision_tree)
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
#Y_pred = random_forest.predict(X_test)
acc_train_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_test_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
print('Random Forest train accurary: ',acc_train_random_forest)
print('Random Forest test accurary: ',acc_test_random_forest)
# Pick best model, Random Forest. Then use all the training data to gegerate the new model

random_forest_final = RandomForestClassifier(n_estimators=100)
random_forest_final.fit(X, y)
Y_pred = random_forest_final.predict(test_df)
acc_random_forest = round(random_forest_final.score(X, y) * 100, 2)
print('Random Forest train accurary: ',acc_random_forest)

# get result file
submission_random_forest = pd.DataFrame({
        "PassengerId": test_df.index,
        "Survived": Y_pred
    })
#submission.to_csv('../output/submission.csv', index=False)
submission_random_forest.head()
#save the submission file
#the final score is 0.7703, rank 6,547/10,683
submission_random_forest.to_csv('submission_random_forest.csv', index=False)
