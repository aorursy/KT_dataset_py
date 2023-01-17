import pandas as pd

import random as rnd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint

import seaborn as sns

%matplotlib inline

sns.set()



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

random_state = 3
y_values = [75.598, 75.598, 73.205, 72.727, 74.641, 72.727, 75.119, 76.555, 78.947]

fig = plt.figure(figsize=(14,5), dpi=80)

_ =  sns.lineplot(x=range(len(y_values)),y=y_values,linestyle='--', marker='o', color='b')

plt.xlabel('Submission # (Not including errors)')

plt.ylabel('Percent Accuracy')

arrowprops=dict(color='black',headwidth=2,headlength=2,width=.5)

plt.annotate(xy=(7,76.555),xytext=(5,76.555),arrowprops=arrowprops,

             s="Added VotingPredictor (661 places)")

plt.annotate(xy=(8,78.947),xytext=(5,78.947),arrowprops=arrowprops,

             s="Added KFolds (5,142 places)")
#'../input/train.csv'

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

both_df = [train_df, test_df]
train_df.head()
print(train_df.columns.values)
print(train_df.dtypes)
train_df.describe()
train_df.describe(include=['O'])
#Look for null values to deal with

print('Null Values (train data):\n', train_df.isnull().sum())

print('Null Values (test data):\n', test_df.isnull().sum())
#Remove both Ticket, Cabin from the datasets

for dataset in both_df:

    dataset = dataset.drop(['Ticket','Cabin'],axis=1,inplace=True)

train_df.head()
#Compare PClass, Survival

train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#COmpare gender, Survival

train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#Compare SibSp, Survival

train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#Compare Parch, Survival

train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)

#FacetGrid can be used to compare multiple values in a histogram

#col is the column you are dividing the counted value by

#map uses the "row"

_ = sns.FacetGrid(train_df,col='Survived')

_.map(plt.hist,'Age',bins=5)
_ = sns.FacetGrid(train_df,col='Survived',row='Pclass')

_.map(plt.hist,'Age',bins=5)

_.add_legend()
_ = sns.FacetGrid(train_df,row='Embarked')

_.map(sns.barplot,'Pclass','Survived','Sex',palette='viridis',ci=None)

_.add_legend()
_ = sns.FacetGrid(train_df,col='Survived',row='Embarked')

_.map(sns.barplot,'Sex','Fare',palette='viridis_r',ci=None)

_.add_legend()
#Find the titles of people that may indicate age, rank

for dataset in both_df:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    dataset = dataset.drop('Name',axis=1,inplace=True)



pd.crosstab(train_df['Title'], train_df['Sex'])
#percents by title/gender pertaining to survival rate

train_df.groupby(['Title','Sex'])['Survived'].value_counts(normalize=True)
#Simplify titles by creating group names

for dataset in both_df:

    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Ms')

    dataset['Title'] = dataset['Title'].replace('Miss','Ms')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#Categorical values for gender

for dataset in both_df:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
#adding row to FacetGrid creates rows based on 'row'; map is just x-axis

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', bins=5)

grid.add_legend()
#Convert titles to numbers

title_mapping = {"Mr": 1, "Ms": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in both_df:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
#Confirming that everything is working fine

train_df['Age'][train_df['Title'] == 4]
#sibsp Number of Siblings/Spouses Aboard

#parch Number of Parents/Children Aboard 

for dataset in both_df:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in both_df:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

both_df = [train_df, test_df]



train_df.head()
#Find most common Embarked

freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in both_df:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Map Embarked to integers

for dataset in both_df:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
#Bin fares based on a qcut of four

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



fare_bins = pd.qcut(train_df['Fare'],4,duplicates='drop').unique()

print(fare_bins)



for dataset in both_df:

    for i in range(0,len(fare_bins)):

        dataset.loc[(dataset['Fare'] > pd.IntervalIndex(fare_bins).left[i]) & (dataset['Fare'] <= pd.IntervalIndex(fare_bins).right[i]), 'Fare'] = i

    

train_df.head(10)
train_df.dtypes
for dataset in both_df:

    #Alternative way of mapping based on cut/qcut

    dataset['Age'] = pd.qcut(dataset['Age'],4, labels=range(0,4))

dataset['Age'].head(10)
'''

rf = RandomForestClassifier(random_state = random_state)

rf.fit(train_df.dropna().drop(['Age','Fare','Survived','PassengerId'],axis=1), train_df.dropna()['Age'].astype('int'))

acc_random_forest = round(rf.score(train_df.dropna().drop(['Age','Fare','Survived','PassengerId'],axis=1), train_df.dropna()['Age'].astype('int')) * 100, 20)

print(acc_random_forest)'''
#ML model to predict age bin number

results = []

results_train = []

train_agetest = train_df.dropna().drop(['Age','Fare','Survived','PassengerId'],axis=1)

train_agetest_y = train_df.dropna()['Age']

train_agetest_nulls = train_df[train_df['Age'].isnull()].drop(['Age','Fare','Survived','PassengerId'],axis=1)

test_agetest = test_df[test_df['Age'].isnull()].drop(['Age','Fare','PassengerId'],axis=1)

for x in range(0,10):

    X_train, X_test, y_train, y_test = train_test_split(train_agetest,train_agetest_y,random_state=random_state)

    rf = RandomForestClassifier()

    rf.fit(X_train,y_train)

    results.append(rf.predict(test_agetest))

    results_train.append(rf.predict(train_agetest_nulls))

print(results)
#Find the most often age value from the model

ages = pd.DataFrame()

for result in range(len(results)):

    ages['Age ' + str(result)] = results[result]

ages = ages.mode(axis=1)[0]

ages.head()



ages_train = pd.DataFrame()

for result in range(len(results_train)):

    ages_train['Age ' + str(result)] = results_train[result]

ages_train = ages_train.mode(axis=1)[0]
test_df['Age'].where(test_df['Age'].notnull(), other=list(ages),inplace=True)

train_df['Age'].where(train_df['Age'].notnull(), other=list(ages_train),inplace=True)

train_df.head()
for dataset in both_df:

    dataset['Age*Class'] = dataset.Age.astype('int') * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
test_df.head(10)
print(list(train_df.columns.values))

print(list(test_df.columns.values))

passengers = test_df['PassengerId']

test_df = test_df.drop("PassengerId", axis=1)

train_df_y = train_df["Survived"]

train_df = train_df.drop(["Survived","PassengerId"], axis=1)



X_train, X_test_dump, Y_train, Y_test_dump = train_test_split(train_df,train_df_y, random_state=random_state)



X_train.shape, Y_train.shape, X_test.shape
# Support Vector Machines



svc = SVC(probability=True)

svc.fit(X_train, Y_train)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn


# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN',

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
'''

PassengerId, Survived

1,0

2,0

3,0

etc.'''

trials = []

for trial in range(10):

    X_train, X_test_dump, Y_train, Y_test_dump = train_test_split(train_df,train_df_y, random_state=random_state)

    rf = RandomForestClassifier()

    rf.fit(X_train, Y_train)

    trials.append(list(rf.predict(test_df)))

print(trials[0:2])
ans = pd.DataFrame()

for trial in range(len(trials)):

    ans['Trial ' + str(trial)] = trials[trial]

ans['Survived'] = ans.mode(axis=1)[0]

ans = ans['Survived'].astype('int')

ans.head()
ans = pd.DataFrame()

for trial in range(len(trials)):

    ans['Trial ' + str(trial)] = trials[trial]

ans['Survived'] = ans.mode(axis=1)[0]

ans = ans['Survived'].astype('int')

ans.head()
#passengers remains same

test = test_df

y = train_df_y

train = train_df
def model_fitter(model,folds, X=train,y=y,test=test):

    kfolds = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    cv = cross_validate(model, X, y,cv=kfolds,scoring='accuracy',

                        return_train_score=False, return_estimator=True)

    print("Best Score is: %s, located at %s"%(max(cv['test_score']),list(cv['test_score']).index(max(cv['test_score']))))

    best_rfc = cv['estimator'][list(cv['test_score']).index(max(cv['test_score']))]

    print(best_rfc)

    return(best_rfc)
#mass fitting of models with the model_fitter

svc = SVC(probability=True)

f_svc = model_fitter(svc,3,train,y,test)

knn = KNeighborsClassifier()

f_knn = model_fitter(knn,3,train,y,test)

gaussian = GaussianNB()

f_gaussian = model_fitter(gaussian,3,train,y,test)

perceptron = Perceptron()

f_perceptron = model_fitter(perceptron,3,train,y,test)

linear_svc = LinearSVC()

f_linear_svc = model_fitter(linear_svc,3,train,y,test)

sgd = SGDClassifier()

f_sgd = model_fitter(sgd,3,train,y,test)

decision_tree = DecisionTreeClassifier()

f_decision_tree = model_fitter(decision_tree,3,train,y,test)

random_forest = RandomForestClassifier()

f_random_forest = model_fitter(random_forest,3,train,y,test)
estimators=[('SVC', f_svc),('KNN', f_knn),('Gaussian', f_gaussian),('Perceptron', f_perceptron),('Linear_SVC',f_linear_svc),('SGD',f_sgd),('Decision_Tree',f_decision_tree),('Random_Forest',f_random_forest)]

VotingPredictor = VotingClassifier(estimators=estimators,voting='hard', n_jobs=5)

f_VotingPredictor = model_fitter(VotingPredictor,3,train,y,test)

final_Predictions = f_VotingPredictor.predict(test)

#added VotingPredictor- 0.76555, 661 places

#added Kfolds- 0.78947, 5,142 places
ans = pd.DataFrame()

ans['Survived'] = final_Predictions
ans.index.name='PassengerId'

ans = ans.reset_index()

ans['PassengerId'] = passengers
ans.head()
ans.to_csv('ans.csv',index=False)