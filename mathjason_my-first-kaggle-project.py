from IPython.display import display, Markdown



# data tools

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.head()
test_df.head()
train_df.info()

print("_" * 40)

test_df.info()
train_df.describe() # This is the numeric columns only
train_df.describe(include=['O']) # This is the object columns
for col in ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']:

    display(train_df[[col, 'Survived']].

            groupby([col], as_index=False).

            mean().sort_values(by='Survived', ascending=False)

           )
g = sns.FacetGrid(train_df, row='Survived', aspect=2.5)

g.map(plt.hist, 'Age', alpha=.5, bins=range(81))
# Warning the colors/legend are wrong for 

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', legend_out=False)

grid.add_legend()
#train_df[['Embarked', 'Pclass', 'Sex', 'Survived']].groupby(['Embarked', 'Pclass', 'Sex'], as_index=False).count()

#pd.pivot_table(train_df, values='Survived', index=['Embarked', 'Pclass'], columns=['Sex'], aggfunc=lambda a:str(np.sum(a)) + "/" + str(len(a)))

pd.pivot_table(train_df, values='Survived', index=['Embarked', 'Pclass'], columns=['Sex'], aggfunc=np.mean)
# Every 10th ticket after sorting alphabetically

np.sort(train_df['Ticket'].values)[np.arange(0,len(train_df['Ticket']),10)]
original_train_df = train_df

original_test_df = test_df
train_df = original_train_df.copy()

test_df = original_test_df.copy()
for df in [train_df, test_df]:

    split_ticket = df['Ticket'].str.split()

    df['TicketNum'] = split_ticket.map(lambda l:l[-1])

    df['TicketLabel'] = split_ticket.map(lambda l:" ".join(l[0:len(l)-1]))

    # fix data

    df.ix[df['TicketNum']=="LINE", 'TicketLabel'] = 'LINE'

    df.ix[df['TicketNum']=="LINE", 'TicketNum'] = '0'

    df['TicketNum'] = train_df['TicketNum'].astype(int);

    display(df[['TicketLabel', 'TicketNum']].info())
combined_df = pd.concat([train_df, test_df])

combined_df[['TicketLabel', 'TicketNum', 'Pclass', 'Fare', 'Embarked', 'Cabin']].sort_values(by=['TicketNum','TicketLabel'], ascending=True)
combined_df[combined_df['TicketLabel']!=''][['TicketLabel', 'TicketNum', 'Pclass', 'Fare', 'Embarked']].sort_values(by=['TicketLabel','TicketNum'], ascending=True)
pd.crosstab(combined_df['TicketLabel'], combined_df['Pclass'])
pd.crosstab(combined_df['TicketLabel'], combined_df['Embarked'])
# bins based on number of digits

bins = [10**i for i in range(8)]

plt.hist(train_df['TicketNum'], bins=bins)

plt.gca().set_xscale("log")
for df in [train_df, test_df, combined_df]:

    df['TicketLabel'] = df['TicketLabel'].str.replace('.', '')

    df['TicketLabel'] = df['TicketLabel'].str.upper()

    df['TicketLabel'] = df['TicketLabel'].str.replace(' ', '')

    df['TicketLabel'] = df['TicketLabel'].replace('A/4', 'A4') 

    df['TicketLabel'] = df['TicketLabel'].replace(['A/S', 'A/5'], 'A5')

    df['TicketLabel'] = df['TicketLabel'].replace('WE/P', 'WEP')

    df['TicketLabel'] = df['TicketLabel'].replace('W/C', 'WC')

    df['TicketLabel'] = df['TicketLabel'].replace('SO/C', 'SOC')



pd.crosstab(train_df['TicketLabel'], train_df['Survived'])
train_df[['TicketLabel', 'Survived']].groupby(['TicketLabel'], as_index=False).mean().sort_values(by='Survived', ascending=False)
bins = sorted([(j+10) * 10**i for i in range(1, 6) for j in range(90)])

#plt.hist(train_df['TicketNum'], bins=bins)

#plt.gca().set_xscale("log")

#plt.gca().set_ylim([0,50])

g = sns.FacetGrid(combined_df, aspect=2.5)

g.map(plt.hist, 'TicketNum', alpha=1, bins=bins)

plt.gca().set_xscale("log")

plt.gca().set_ylim([0,30])
bins2 = [100, 1000, 5000, 10000, 20000, 100000, 200000, 300000, 10**6, 10**7]

#plt.hist(train_df['TicketNum'], alpha=.25, bins=bins2)

#plt.hist(train_df['TicketNum'], bins=bins)

#plt.gca().set_xscale("log")

#plt.gca().set_ylim([0,30])

g = sns.FacetGrid(combined_df, aspect=2.5)

g.map(plt.hist, 'TicketNum', alpha=.25, bins=bins2)

g.map(plt.hist, 'TicketNum', alpha=1, bins=bins)

plt.gca().set_xscale("log")

#plt.gca().set_ylim([0,50])
bins = sorted([(j+10) * 10**i for i in range(1, 6) for j in range(90)])

g = sns.FacetGrid(combined_df, row='Survived', aspect=2.5)

g.map(plt.hist, 'TicketNum', alpha=.25, bins=bins2)

g.map(plt.hist, 'TicketNum', alpha=.5, bins=bins)

plt.gca().set_xscale("log")

#plt.gca().set_ylim([0,20])
# Automatic groups

#omag      = lambda x: 10**np.floor(np.log10(np.abs(x)))

#signifFig = lambda x, n: (10**(n-1) * x)//omag(x) * omag(x) // 10**(n-1)

#train_df.ix[train_df['TicketNum'] == 0, 'TicketGroup'] = 0

#train_df['TicketGroup'][train_df['TicketNum'] != 0] = signifFig(train_df['TicketNum'][train_df['TicketNum'] != 0], 2)

#train_df['TicketGroup'] = train_df['TicketGroup'].astype(int)



# Manual groups

ticket_bins = [-1, 100, 1000, 5000, 10000, 20000, 100000, 200000, 300000, 10**6, 10**7]

train_df['TicketGroup'] = pd.cut(train_df['TicketNum'], bins=ticket_bins)

test_df['TicketGroup'] = pd.cut(test_df['TicketNum'], bins=ticket_bins)

combined_df['TicketGroup'] = pd.cut(combined_df['TicketNum'], bins=ticket_bins)
#many_df = train_df[['TicketGroup', 'Survived']].groupby(['TicketGroup'], as_index=False).filter(lambda x: len(x) > 40)

#many_df.groupby(['TicketGroup'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['TicketGroup', 'Survived']].groupby(['TicketGroup'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(combined_df['TicketLabel'], combined_df['TicketGroup'])
for df in [train_df, test_df, combined_df]:

    df['HasTicketLabel'] = (df['TicketLabel'] != "")



pd.crosstab(combined_df['TicketGroup'], combined_df['HasTicketLabel'])
pd.crosstab(combined_df['Pclass'], combined_df['HasTicketLabel'])
pd.crosstab(combined_df['TicketGroup'], combined_df['Pclass'])
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

train_df.shape, test_df.shape
for dataset in [test_df, train_df, combined_df]:

    dataset['TicketGroup'] = -1 # just to set a value

    for i in range(len(ticket_bins)-1):

        dataset.loc[(dataset['TicketNum'] > ticket_bins[i]) &

                    (dataset['TicketNum'] <= ticket_bins[i+1]), 'TicketGroup'] = i

train_df.head()
for dataset in [test_df, train_df, combined_df]:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')



pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in [test_df, train_df, combined_df]:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in [test_df, train_df, combined_df]:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1) # Keep PassengerId for prediction

train_df.shape, test_df.shape
for dataset in [test_df, train_df, combined_df]:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in [test_df, train_df, combined_df]:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                               (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            guess_ages[i,j] = age_guess

        

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & 

                         (dataset.Sex == i) & 

                         (dataset.Pclass == j+1),\

                         'Age'] = guess_ages[i,j]

            

    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()            
age_bins = [-1, 1.5, 6, 12, 20, 40, 60, 90]

train_df['AgeBand'] = pd.cut(train_df['Age'], bins=age_bins)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in [test_df, train_df, combined_df]:

    for i in range(len(age_bins)-1):

        dataset.loc[(dataset['Age'] > age_bins[i]) &

                    (dataset['Age'] <= age_bins[i+1]), 'Age'] = i

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

train_df.head()
for dataset in [test_df, train_df, combined_df]:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in [test_df, train_df, combined_df]:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
for dataset in [test_df, train_df, combined_df]:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df[['Age*Class', 'Survived']].groupby(['Age*Class'], as_index=False).mean().sort_values(by='Survived', ascending=False)
freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in [test_df, train_df, combined_df]:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in [test_df, train_df, combined_df]:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
for dataframe in [test_df, train_df, combined_df]:

    dataframe['Fare'].fillna(dataframe['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in [test_df, train_df, combined_df]:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

    

train_df.head(10)
test_df.head(10)
test_df = test_df.drop(['TicketNum'], axis=1)

train_df = train_df.drop(['TicketNum'], axis=1)

test_df = test_df.drop(['TicketLabel'], axis=1)

train_df = train_df.drop(['TicketLabel'], axis=1)

test_df = test_df.drop(['HasTicketLabel'], axis=1)

train_df = train_df.drop(['HasTicketLabel'], axis=1)



train_df.head(10)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred # using random forest

    })



#submission.to_csv('../output/submission.csv', index=False)