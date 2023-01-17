# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # basic plotting stuff

import seaborn as sns # cooler plotting stuff



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load and take a look at the data



train_df = pd.read_csv("../input/titanic/train.csv", header=0)

print(train_df.head(10))

print(train_df.describe())

plt.figure(figsize=(8,8))

sns.heatmap(train_df.corr(),cmap='coolwarm',annot=True, linewidth=0.5)

print(train_df.shape)

print(train_df.columns)

print(train_df.isnull().sum())

print(train_df.Cabin.unique()[:10])
train_df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

print(train_df.describe())

print(train_df.columns)
most_common_embarked = train_df.Embarked.value_counts().index[0]

print(most_common_embarked)
train_df.Embarked.fillna(most_common_embarked, inplace=True)

print(train_df.Embarked.value_counts())

print(train_df.isnull().sum())
median_age = train_df.Age.median()

train_df.Age.fillna(median_age, inplace=True)

print(train_df.describe())

train_class_dummies = pd.get_dummies(train_df.Pclass, prefix='class')

train2 = pd.get_dummies(train_df)

train3 = pd.concat([train2,train_class_dummies],axis=1)

train3.drop('Pclass', axis=1)

print(train3.head())

# split the data for training and get to it

from sklearn.model_selection import train_test_split



y = train3['Survived']

X= train3.copy()

X.drop('Survived', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=112358, stratify=y)

print(X_train.shape)

print(X_test.shape)
from sklearn.metrics import accuracy_score



# calculate percentage survived

pct_survived = train3.Survived.sum() / len(train3.Survived)

print("{0:2.1f} percent survival rate".format(pct_survived * 100))



replicates = 10000
#fair coin modeling. Do some replicates and choose a random float such that a float <0.5 means they survived (1)

fair_y_coinflips= []

fair_accuracies = []



for i in range(replicates):

    rands = np.random.random_sample(len(y_train))

    fair_y_coinflips = []

    for r in rands:

        if r < 0.5:

            fair_y_coinflips.append(1)

        else:

            fair_y_coinflips.append(0)

            

    fair_accuracies.append(accuracy_score(y_train, fair_y_coinflips))



fair_coinflip_mean = np.mean(fair_accuracies)

fair_coinflip_stdev = np.std(fair_accuracies)

fair_coinflip_accuracy_target = fair_coinflip_mean + (2 * fair_coinflip_stdev)

print("{0:.3f} mean accuracy, stdev {1:.3f}, {2} replicates".format(fair_coinflip_mean, fair_coinflip_stdev, replicates))

print("Score for an ML model to beat (avg + 3SD) = {0:.3f} + 2 * {1:.3f} = {2:.3f}".format(fair_coinflip_mean, fair_coinflip_stdev, fair_coinflip_accuracy_target))

plt.ylim((np.min(fair_accuracies)-0.1,np.max(fair_accuracies)+0.1))

plt.title('Distribution of accuracies of fair coin flipping')

plt.boxplot(fair_accuracies)

plt.show()



plt.hist(fair_accuracies, bins=20)

plt.title('fair coin flip accuracy outcomes')

plt.xlabel('accuracy')

plt.show()


# biased coin modeling. Do some replicates and choose a random float such that a float <= pct_survived means they survived (1)



y_coinflips= []

accuracies = []

for i in range(replicates):

    rands = np.random.random_sample(len(y_train))

    y_coinflips = []

    for r in rands:

        if r <= pct_survived:

            y_coinflips.append(1)

        else:

            y_coinflips.append(0)

            

    accuracies.append(accuracy_score(y_train, y_coinflips))



coinflip_mean = np.mean(accuracies)

coinflip_stdev = np.std(accuracies)

coinflip_accuracy_target = coinflip_mean + (2 * coinflip_stdev)

print("{0:.3f} mean accuracy, stdev {1:.3f}, {2} replicates".format(coinflip_mean, coinflip_stdev, replicates))

print("Score for an ML model to beat (avg + 3SD) = {0:.3f} + 2 * {1:.3f} = {2:.3f}".format(coinflip_mean, coinflip_stdev, coinflip_accuracy_target))

plt.ylim((np.min(accuracies)-0.1,np.max(accuracies)+0.1))

plt.title('Distribution of accuracies of biased coin flipping')

plt.boxplot(accuracies)

plt.show()
# just say "everyone dies" in this model and test with unstratified train-test splits

# add random state sequence for reproducability



accuracies_alldead = []

for rep in range(replicates):

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=rep)

    y_dead = [0 for x in y_test]

    accuracies_alldead.append(accuracy_score(y_test, y_dead))

    

alldead_mean = np.mean(accuracies_alldead)

alldead_stdev = np.std(accuracies_alldead)

alldead_accuracy_target = alldead_mean + (2 * alldead_stdev)

print("{0:.3f} mean accuracy, stdev {1:.3f}, {2} replicates".format(alldead_mean, alldead_stdev, replicates))

print("Score for an ML model to beat (avg + 3SD) = {0:.3f} + 2 * {1:.3f} = {2:.3f}".format(alldead_mean, alldead_stdev, alldead_accuracy_target))

plt.ylim((np.min(accuracies_alldead)-0.1,np.max(accuracies_alldead)+0.1))

plt.title('Distribution of accuracies based on saying "everyone died"')

plt.boxplot(accuracies_alldead)

plt.show()
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, y_train)

print("Training accuracy: {0:.3f}".format(accuracy_score(y_train, knn.predict(X_train))))

y_pred = knn.predict(X_test)

print("Test accuracy: {0:.3f}".format(accuracy_score(y_test, y_pred)))

neighbors = list(range(2,42,2))

train_accuracies = []

test_accuracies = []



for n in neighbors:

    knn=KNeighborsClassifier(n_neighbors=n)

    knn.fit(X_train, y_train)

    train_pred = knn.predict(X_train)

    y_pred = knn.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)

    test_acc = accuracy_score(y_pred, y_test)

    train_accuracies.append(train_acc)

    test_accuracies.append(test_acc)



plt.plot(neighbors, train_accuracies, color='red', label='train')

plt.plot(neighbors, test_accuracies, color='blue', label='test')

plt.xticks(neighbors)

plt.xlabel('number neighbors')

plt.ylabel('accuracy')

plt.legend(loc='best')

plt.show
from sklearn.metrics import confusion_matrix



knn=KNeighborsClassifier(n_neighbors=20)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Training accuracy: {0:.3f}".format(accuracy_score(y_train, knn.predict(X_train))))

print("Testing accuracy: {0:.3f}".format(accuracy_score(y_pred, y_test)))



print(confusion_matrix(y_pred, y_test))


# start from scratch and pipeline it together to make sure we get an accuracy that matches the above



from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer



rand_seed = 112358



# make lists of columns for processing

first_drop = ['Cabin','Name','Ticket']

final_drop = ['Pclass','Sex','Embarked']

numerics_for_mean = ['Age', 'Fare']

numerics_for_mode = ['SibSp','Parch']

string_for_most_common = ['Embarked']



# this imputer imputes with the mean

si_numeric = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),

])



# this imputer imputes with most common value

si_category = Pipeline(

    steps=[('imputer',

            SimpleImputer(strategy='most_frequent'))])



# then we put the features list and the transformers together using the column transformer

preprocessor = ColumnTransformer(transformers=[('si_numeric',

                                                si_numeric,

                                                numerics_for_mean),

                                               ('si_most_common',

                                                si_category,

                                                string_for_most_common)])



df = pd.read_csv("../input/titanic/train.csv", header=0)



y = df['Survived']

df.drop(['Survived'], axis=1, inplace=True) # this is broken out individually so the test data can use the same process exactly

df.drop(first_drop, axis=1,inplace=True)



preprocessor.fit(df)

df_imputed = pd.DataFrame(preprocessor.transform(df),columns=['Age','Fare','Embarked'])

df['Age'] = df_imputed['Age']

df['Embarked'] = df_imputed['Embarked']

X = pd.get_dummies(df, columns=['Embarked','Sex','Pclass'], prefix=['emb','is','class'])

X.drop(['PassengerId'], axis=1, inplace=True)



pipeline = Pipeline(steps=[('knn',KNeighborsClassifier(n_neighbors=20))])



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=rand_seed, stratify=y)



pipeline.fit(X_train,y_train)

accuracy = accuracy_score(pipeline.predict(X_test), y_test)

print("Accuracy score of pipeline: {0:.3f}".format(accuracy))

test_df = pd.read_csv("../input/titanic/test.csv",header=0)



test_df.drop(first_drop, axis=1, inplace=True)

test_imputed = pd.DataFrame(preprocessor.transform(test_df), columns=['Age','Fare','Embarked'])

test_df['Age'] = test_imputed['Age']

test_df['Fare'] = test_imputed['Fare']

test_df['Embarked'] = test_imputed['Embarked']



testX = pd.get_dummies(test_df, columns=['Embarked','Sex', 'Pclass'], prefix=['emb','is','class'])

submit_ids = testX['PassengerId']

testX.drop(['PassengerId'], inplace=True, axis=1)



result_pred = pipeline.predict(testX)



result=pd.DataFrame({'PassengerId' : submit_ids, 'Survived': result_pred})

result.to_csv('titanic_V1_baseline_submit.csv', index=False, header=True)