# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load and take a look at the data



train_df = pd.read_csv("../input/titanic/train.csv", header=0)

print(train_df.columns)



# save out our label

y = train_df['Survived']



# let's engineer some features

train_df['Group_size'] = train_df['SibSp'] + train_df['Parch'] + 1



def get_group_size_category(n):

    """return a category based on the size of the traveling group"""

    if n == 1:

        return 'alone'

    elif n < 5:

        return 'small'

    else:

        return 'large'



train_df['Group_class'] = train_df.apply(lambda x: get_group_size_category(x['Group_size']), axis=1)



for col in ['Sex', 'Pclass', 'Embarked', 'Group_size','Group_class']:

    print("Survival by {0}".format(col))

    print(train_df.groupby(col)['Survived'].mean())

    print(train_df[col].value_counts())

    print("-----")    



train_df.drop(['PassengerId', 'Ticket', 'Cabin', 'Name', 'SibSp', 'Parch'], axis=1, inplace=True)



# check for missing data

print(train_df.isnull().sum())

# let's look at some histograms

train_df.hist('Age', bins=20)

train_df.hist('Fare', bins=50)

train_df['Fare_individual'] = train_df['Fare'] / train_df['Group_size']

sns.pairplot(train_df[['Age','Group_size','Fare_individual']])
sorted_fares = train_df['Fare_individual'].sort_values()

fare_percentiles = np.percentile(sorted_fares,[x for x in range(10,110,10)])

print("Percentiles (n={0}): ".format(len(fare_percentiles)))

print(fare_percentiles)



def fare_to_percentile(fare, percentiles):

    """return the passed-in amount as its decile category"""

    for i in range(len(percentiles)):

        if fare <= percentiles[i]:

            return i+1

        

    return len(percentiles)

    

train_df['Fare_percentile'] = train_df.apply(lambda x: fare_to_percentile(x['Fare'], fare_percentiles), axis=1)

print(train_df['Fare_percentile'].value_counts())

print(train_df.groupby('Fare_percentile')['Survived'].mean())

sns.pairplot(train_df[['Fare_percentile', 'Age', 'Group_size']])
# Make a mean age grouping dictionary by things everyone has: a fare decile, Passenger class, and a sex

# and fill in any missing values by the mean age



fare_pclass_sex_mean_age = train_df.groupby(['Fare_percentile','Pclass','Sex'])['Age'].mean()

fdict = pd.DataFrame(fare_pclass_sex_mean_age).fillna(train_df['Age'].mean()).to_dict()



def impute_age_by_fare_class_sex(fpsa_dict, total_mean_age, fd, pc, sex):

    """Use a dictionary to impute missing ages based on fare decile, passenger class, and sex. Failing that, use mean age."""

    #print("fpsa is a: ", fpsa_dict.dtype)

    

    if fpsa_dict['Age'][(fd, pc, sex)] == np.nan:

        return total_mean_age

    else:

        return fpsa_dict['Age'][(fd, pc, sex)]

    

grand_mean_age = train_df['Age'].mean()

train_df['Age'] = train_df.apply(lambda x: impute_age_by_fare_class_sex(fdict, grand_mean_age, x['Fare_percentile'], x['Pclass'], x['Sex']) if pd.isnull(x['Age']) else x['Age'], axis=1)

# Let's bucket ages into their decade: age 0-9 = 1, 10-19 = 2, etc.



train_df['Age_decade'] = np.round((train_df['Age'] // 10) +1, decimals=0)



print("Counts by age decade")

decade_dict = train_df['Age_decade'].value_counts()

print(decade_dict)

print("-----")

print("Survival rate by age decade")

print(train_df.groupby('Age_decade')['Survived'].mean())

print("-----")



plt.figure(figsize=(8,8))

sns.violinplot(x='Age_decade', y='Age', data=train_df, order=[1,2,3,4,5,6,7,8,9])

# Let's also give people an age category



def age_category(age):

    if age < 2:

        return 'infant'

    elif age < 5:

        return 'toddler'

    elif age < 13:

        return 'child'

    elif age < 20:

        return 'teen'

    elif age < 40:

        return 'yadult'

    elif age < 60:

        return 'adult'

    else:        

        return 'aged'

    

train_df['Age_category'] = train_df.apply(lambda x: age_category(x['Age']), axis=1)

print(train_df['Age_category'].value_counts())

print(train_df.groupby('Age_category')['Survived'].mean())

print(train_df.isnull().sum())
# There are two missing Embarkeds, so let's just fill those with the most common value. That is both defensible as a choice 

# and more sensible than removing them.



train_df['Embarked'].fillna(train_df['Embarked'].mode()[0],inplace=True)

print(train_df.isnull().sum())

print(train_df.shape)
# all right! Now let's encode the string categories and drop any redundant columns

    

train_df = pd.get_dummies(train_df, columns=['Group_class', 'Age_category','Sex', 'Pclass','Embarked'], prefix=['Group_size', 'is','is', 'Class','Emb'])



# we need to drop one of the columns that was just created from each of these five categorical columns 

# because the value of the nth column is implied by columns 1 to (n-1). Let's just drop the first one 

# we come across.



dummy_drop = ['Group_size_alone', 'is_toddler', 'is_female', 'Class_1', 'Emb_C']

train_df.drop(dummy_drop, axis=1, inplace=True)



# now get rid of the rest of the unneeded columns



train_df.drop(['Age','Fare','Survived'], axis=1, inplace=True)

print(train_df.shape)

print(train_df.head())

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

rand_seed = 112358



X_train, X_test, y_train, y_test = train_test_split(train_df, y, train_size=0.75, stratify=y, random_state=rand_seed)

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import to_categorical

y_train_binary = to_categorical(y_train)

y_test_binary = to_categorical(y_test)



n_features = X_train.shape[1]

n_epochs = 50

model = Sequential()



model.add(Dense(n_features, activation='relu', input_shape=(n_features,)))

model.add(Dense(16,activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(4,activation='relu'))

model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(X_train, y_train_binary, epochs=n_epochs, verbose=1)

_, accuracy = model.evaluate(X_test, y_test_binary, verbose=1)

print("Test accuracy: {0:.3f}".format(accuracy))

# We're going to assess a bunch of default classifiers and see how each one does



from sklearn.model_selection import StratifiedKFold

from sklearn.dummy import DummyClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix



clfs = [

        ('rfc', RandomForestClassifier(random_state=rand_seed)),

        ('logreg', LogisticRegression(random_state=rand_seed, max_iter=500)),

        ('dtree',DecisionTreeClassifier(random_state=rand_seed)),

        ('gb', GradientBoostingClassifier()),

        ('ada',AdaBoostClassifier(random_state=rand_seed))

       ]



n_folds = 5



print("Voting Classifier")

print("-----")

vclf = VotingClassifier(estimators=clfs)

vclf.fit(X_train, y_train)

vclf_train_pred = vclf.predict(X_train)

vclf_pred = vclf.predict(X_test)

print("Training accuracy: {0:.3f}".format(accuracy_score(vclf_train_pred, y_train)))

print(" Testing accuracy: {0:.3f}".format(accuracy_score(vclf_pred, y_test)))

print(confusion_matrix(y_test, vclf_pred))

print("=====")

print("Stacking Classifier (cv={0})".format(n_folds))

print("-----")

sc_clf = StackingClassifier(clfs, cv=n_folds)

sc_clf.fit(X_train, y_train)

sc_clf_train_pred = sc_clf.predict(X_train)

sc_pred = sc_clf.predict(X_test)

print("Training accuracy: {0:.3f}".format(accuracy_score(sc_clf_train_pred, y_train)))

print(" Testing accuracy: {0:.3f}".format(accuracy_score(sc_pred, y_test)))

print(confusion_matrix(y_test, sc_pred))

check = pd.DataFrame(X_test)

check['Survived'] = y_test

check['vclf'] = vclf_pred

check['sc'] = sc_pred

print("both model TNs")

print(check[(check['vclf'] == check['sc']) & (check['Survived'] == check['sc']) & (check['Survived'] == 0)].shape[0])

print("both model FPs")

print(check[(check['vclf'] == check['sc']) & (check['Survived'] != check['sc']) & (check['Survived'] == 0)].shape[0])

print("both model FNs")

print(check[(check['vclf'] == check['sc']) & (check['Survived'] != check['sc']) & (check['Survived'] == 1)].shape[0])

print("both model TPs")

print(check[(check['vclf'] == check['sc']) & (check['Survived'] == check['sc']) & (check['Survived'] == 1)].shape[0])

def vote_balance(sc_vote, vclf_vote):

    """privilege a 1 from se over a zero from vclf, and a 0 from vclf over a 1"""

    if sc_vote == vclf_vote:

        return sc_vote

    else:

        if sc_vote == 1:

            return sc_vote

        else:

            return vclf_vote



check['preferred_vote'] = check.apply(lambda x: vote_balance(x['sc'], x['vclf']), axis=1)

print(accuracy_score(y_test, check['preferred_vote']))

print(confusion_matrix(y_test, check['preferred_vote']))

print(check[check['Survived'] != check['preferred_vote']])
test_df = pd.read_csv("../input/titanic/test.csv", header=0)



test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)



submission_ids = test_df['PassengerId']



test_df['Group_size'] = test_df['SibSp'] + test_df['Parch'] + 1

test_df['Group_class'] = test_df.apply(lambda x: get_group_size_category(x['Group_size']), axis=1)



test_df.drop(['PassengerId', 'Ticket', 'Cabin', 'Name', 'SibSp', 'Parch'], axis=1, inplace=True)



test_df['Fare_individual'] = test_df['Fare'] / test_df['Group_size']

test_df['Fare_percentile'] = test_df.apply(lambda x: fare_to_percentile(x['Fare'], fare_percentiles), axis=1)



test_df['Age'] = test_df.apply(lambda x: impute_age_by_fare_class_sex(fdict, grand_mean_age, x['Fare_percentile'], x['Pclass'], x['Sex']) if pd.isnull(x['Age']) else x['Age'], axis=1)

test_df['Age_decade'] = np.round((test_df['Age'] // 10) + 1, decimals=0)

test_df['Age_category'] = test_df.apply(lambda x: age_category(x['Age']), axis=1)



print(test_df['Age_category'].value_counts())

test_df = pd.get_dummies(test_df, columns=['Group_class', 'Age_category','Sex', 'Pclass','Embarked'], prefix=['Group_size', 'is', 'is', 'Class','Emb'])

test_df.drop(dummy_drop, axis=1, inplace=True)

test_df.drop(['Age','Fare'], inplace=True, axis=1)

# Let's do some predictin'!



print(test_df.columns)

print(X_test.columns)

test_pred = vclf.predict(test_df)



print("Predicted survival rate: {0:.3f}".format(sum(test_pred)/len(test_pred)))

sub_df = pd.DataFrame({'PassengerId': submission_ids, 'Survived': test_pred})



sub_df.to_csv('titanic_better_processing_features_v3_submit.csv', index=False, header=True)