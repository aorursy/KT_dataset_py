# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load the datasets



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
# look at the data and see if we have any missing values



print(train.dtypes)

train.describe()

train.head(5)
train_missing = train.isna().mean()

train_missing
women = train.loc[train['Sex'] == 'female']

women_survived = women.loc[women['Survived'] == 1]

rate_women = len(women_survived)/len(women)

print("% of women who survived:", "{:.2%}".format(rate_women))
men = train.loc[train['Sex'] == 'male']

men_survived = men.loc[men['Survived'] == 1]

rate_men = len(men_survived)/len(men)

print('% of men survivors:', "{:.2%}".format(rate_men))
train = train.replace({'Survived': {0: 'No', 1: 'Yes'}})

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))

fig.suptitle('Titanic Passenger Class Distribution', fontsize=16)



pclass_data = train['Pclass'].value_counts()

pclass_data.plot.bar(ax=axes[0], alpha=0.7)

axes[0].set_title('Passenger Class Distribution')

axes[0].set_xlabel('Passenger Class')

axes[0].set_ylabel('Passenger Class Count')



class_sex = pd.crosstab(train['Pclass'], train['Sex'])

class_sex.plot.bar(ax=axes[1], alpha=0.7)

axes[1].set_xlabel('Passenger Class')

axes[1].set_title('Gender by Pclass')



class_survived = pd.crosstab(train['Pclass'], train['Survived'])

class_survived.plot.bar(ax=axes[2], alpha=0.7)

axes[2].set_xlabel('Passenger Class')

axes[2].set_title('Survival by Passenger Class')



plt.show()

train = train.replace({'Survived': {'No': 0, 'Yes': 1}})
men = men.replace({'Survived': {0: 'No', 1: 'Yes'}})



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

fig.suptitle('Men Class Distribution', fontsize=16)



men_pclass= men['Pclass'].value_counts()

men_pclass.plot.bar(ax=axes[0], alpha=0.7)

axes[0].set_title('Passenger Class Distribution by Men')

axes[0].set_xlabel('Passenger Class by Men')

axes[0].set_ylabel('Passenger Class Count')



class_survived_men = pd.crosstab(men['Pclass'], men['Survived'])

class_survived_men.plot.bar(ax=axes[1], alpha=0.7)

axes[1].set_xlabel('Passenger Class by Men')

axes[1].set_ylabel('Passenger Class Count')

axes[1].set_title('Men Survival by Passenger Class')
women = women.replace({'Survived': {0: 'No', 1: 'Yes'}})



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

fig.suptitle('Women Class Distribution', fontsize=16)



women_pclass= women['Pclass'].value_counts()

women_pclass.plot.bar(ax=axes[0], alpha=0.7)

axes[0].set_title('Passenger Class Distribution by Women')

axes[0].set_xlabel('Passenger Class by Women')

axes[0].set_ylabel('Passenger Class Count')



class_survived_women = pd.crosstab(women['Pclass'], women['Survived'])

class_survived_women.plot.bar(ax=axes[1], alpha=0.7)

axes[1].set_xlabel('Passenger Class by Women')

axes[1].set_ylabel('Passenger Class Count')

axes[1].set_title('Women Survival by Passenger Class')
def titanic_children(passenger):

    """Finding children among titanic passengers"""

    age , sex = passenger

    if age < 16:

        return 'child'

    else:

        return sex



train['person'] = train[['Age','Sex']].apply(titanic_children,axis=1)

child = train.loc[train['person'] == 'child']
child = child.replace({'Survived': {0: 'No', 1: 'Yes'}})



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

fig.suptitle('Child Passenger Class Distribution', fontsize=16)



child_pclass= child['Pclass'].value_counts()

child_pclass.plot.bar(ax=axes[0], alpha=0.7)

axes[0].set_title('Passenger Class Distribution by Children')

axes[0].set_xlabel('Passenger Class by Children')

axes[0].set_ylabel('Passenger Class Count')



class_survived_child = pd.crosstab(child['Pclass'], child['Survived'])

class_survived_child.plot.bar(ax=axes[1], alpha=0.7)

axes[1].set_xlabel('Passenger Class by Children')

axes[1].set_ylabel('Passenger Class Count')

axes[1].set_title('Children Survival by Passenger Class')
first_class = train.loc[train['Pclass'] == 1]

second_class = train.loc[train['Pclass'] == 2]

third_class = train.loc[train['Pclass'] == 3]



print("% of 1st class who survived: ", "{:.2%}".format(first_class['Survived'].mean()))

print("% of 2nd class who survived: ", "{:.2%}".format(second_class['Survived'].mean()))

print("% of 3rd class who survived: ", "{:.2%}".format(third_class['Survived'].mean()))
first_class_male = first_class.loc[first_class['Sex'] == 'male']

first_class_female = first_class.loc[first_class['Sex'] == 'female']

second_class_male = second_class.loc[second_class['Sex'] == 'male']

second_class_female = second_class.loc[second_class['Sex'] == 'female']

third_class_male = third_class.loc[third_class['Sex'] == 'male']

third_class_female = third_class.loc[third_class['Sex'] == 'female']
survival_rate = {'Male':[first_class_male.Survived.mean(),

                       second_class_male.Survived.mean(),

                       third_class_male.Survived.mean()],

               'Female':[first_class_female.Survived.mean(),

                         second_class_female.Survived.mean(),

                         third_class_female.Survived.mean()]}



survival_df = pd.DataFrame(data=survival_rate,

                           index=['First Class', 'Second Class','Third Class'])
fig = plt.figure(figsize=(14, 8))

ax = fig.add_subplot(1, 1, 1)



ax.plot(survival_df['Male'], marker='o', label='Male Survival Rate')

ax.plot(survival_df['Female'], marker='o', label='Female Surival Rate')



plt.title("Survival Rates by Passenger Class for Men/Women")

ax.set_xlabel("Titanic Passenger Classes")

ax.set_ylabel("Survival Rate")

ax.set_yticklabels(['20%','30%','40%','50%','60%','70%'])

plt.legend()
train = train.replace({'Survived': {0: 'No', 1: 'Yes'}})



fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))

fig.suptitle('Titanic Passenger Class Distribution', fontsize=16)



embark_data = train['Embarked'].value_counts()

embark_data.plot.bar(ax=axes[0], alpha=0.7)

axes[0].set_title('Passenger Embarkement Distribution')

axes[0].set_xlabel('Passenger Embarkement')

axes[0].set_ylabel('Passenger Embarkement Count')



embark_class = pd.crosstab(train['Pclass'], train['Embarked'])

embark_class.plot.bar(ax=axes[1], alpha=0.7)

axes[1].set_xlabel('Passenger Class')

axes[1].set_title('Embarkment by Pclass')



embark_survived = pd.crosstab(train['Embarked'], train['Survived'])

embark_survived.plot.bar(ax=axes[2], alpha=0.7)

axes[2].set_xlabel('Embarkment')

axes[2].set_title('Survival by Embarkment')



plt.show()



train = train.replace({'Survived': {'No' : 0, 'Yes' : 1}})
from sklearn.model_selection import train_test_split



model_features = ['Pclass', 'Sex', 'SibSp', 'Parch']



X_train, X_test, y_train, y_test = train_test_split(pd.get_dummies(train[model_features]), train['Survived'],

                                                    random_state = 0) 



print("X_train shape: {}".format(X_train.shape)) 

print("y_train shape: {}".format(y_train.shape))



print("X_test shape: {}".format(X_test.shape)) 

print("y_test shape: {}".format(y_test.shape))
from sklearn.linear_model import LogisticRegression



logreg_model = LogisticRegression(C=1)

logreg_model.fit(X_train, y_train)

logreg_predictions = logreg_model.predict(X_test)



print("Training set score: {:.2%}".format(logreg_model.score(X_train, y_train))) 

print("Test set score: {:.2%}".format(logreg_model.score(X_test, y_test)))
logreg100_model = LogisticRegression(C=100)

logreg100_model.fit(X_train, y_train)

logreg100_predictions = logreg_model.predict(X_test)



print("Training set score: {:.2%}".format(logreg100_model.score(X_train, y_train))) 

print("Test set score: {:.2%}".format(logreg100_model.score(X_test, y_test)))
logreg001_model = LogisticRegression(C=.01)

logreg001_model.fit(X_train, y_train)

logreg001_predictions = logreg_model.predict(X_test)



print("Training set score: {:.2%}".format(logreg001_model.score(X_train, y_train))) 

print("Test set score: {:.2%}".format(logreg001_model.score(X_test, y_test)))
plt.plot(logreg_model.coef_.T, 'o', label="C=1") 

plt.plot(logreg100_model.coef_.T, '^', label="C=100") 

plt.plot(logreg001_model.coef_.T, 'v', label="C=0.01") 

plt.xticks(range(train.shape[1]), list(X_train.columns.values), rotation=90) 

plt.hlines(0, -1, train.shape[1]) 



plt.xlim(-1, 5)

plt.ylim(-5, 5)

plt.title("Coefficients learned by logistic regression on the dataset for different values of C")

plt.xlabel("Coefficient index") 

plt.ylabel("Coefficient magnitude") 

plt.legend()
from sklearn.svm import LinearSVC



linearSVC_model = LinearSVC()

linearSVC_model.fit(X_train, y_train)

linearSVC_predictions = linearSVC_model.predict(X_test)



print("Training set score: {:.2%}".format(linearSVC_model.score(X_train, y_train))) 

print("Test set score: {:.2%}".format(np.mean(linearSVC_predictions == y_test)))
linearSVC100_model = LinearSVC(C=100)

linearSVC100_model.fit(X_train, y_train)

linearSVC100_predictions = linearSVC100_model.predict(X_test)



print("Training set score: {:.2%}".format(linearSVC100_model.score(X_train, y_train))) 

print("Test set score: {:.2%}".format(np.mean(linearSVC100_predictions == y_test)))
linearSVC001_model = LinearSVC(C=.01)

linearSVC001_model.fit(X_train, y_train)

linearSVC001_predictions = linearSVC001_model.predict(X_test)



print("Training set score: {:.2%}".format(linearSVC001_model.score(X_train, y_train))) 

print("Test set score: {:.2%}".format(np.mean(linearSVC001_predictions == y_test)))
plt.plot(linearSVC_model.coef_.T, 'o', label="C=1") 

plt.plot(linearSVC100_model.coef_.T, '^', label="C=100") 

plt.plot(linearSVC001_model.coef_.T, 'v', label="C=0.01") 

plt.xticks(range(train.shape[1]), list(X_train.columns.values), rotation=90) 

plt.hlines(0, -1, train.shape[1]) 



plt.xlim(-1, 5)

plt.ylim(-5, 5)

plt.title("Coefficients learned by logistic regression on the dataset for different values of C")

plt.xlabel("Coefficient index") 

plt.ylabel("Coefficient magnitude") 

plt.legend()
from sklearn.ensemble import RandomForestClassifier



rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)



print("Training set score: {:.2%}".format(rf_model.score(X_train, y_train))) 

print("Test set score: {:.2%}".format(np.mean(rf_predictions == y_test)))
from sklearn.neighbors import KNeighborsClassifier



KNN_model = KNeighborsClassifier(n_neighbors=3)

KNN_model.fit(X_train, y_train)

KNN_predictions = KNN_model.predict(X_test)



print("Training set score: {:.2%}".format(KNN_model.score(X_train, y_train))) 

print("Test set score: {:.2%}".format(np.mean(KNN_predictions == y_test)))
training_accuracy = [] 

test_accuracy = [] 

# try n_neighbors from 1 to 6 

neighbors_settings = range(1, 6)



for n_neighbors in neighbors_settings:

    # build the model

    KNN_model = KNeighborsClassifier(n_neighbors=n_neighbors)

    KNN_model.fit(X_train, y_train)

    # record training set accuracy

    training_accuracy.append(KNN_model.score(X_train, y_train))

    # record generalization accuracy

    test_accuracy.append(KNN_model.score(X_test, y_test))



fig = plt.figure(figsize=(14, 8))

plt.title("KNN Model Accuracy across n_neighbors")

plt.plot(neighbors_settings, training_accuracy, 

         label="training accuracy",marker='o') 

plt.plot(neighbors_settings, test_accuracy, 

         label="test accuracy",marker='o') 

plt.ylabel("Accuracy") 

plt.xlabel("n_neighbors") 

plt.yticks(np.arange(.74, .84, .01))

plt.legend()
# reset X_test to the test dataset features (was previously from train dataset)

X_test = pd.get_dummies(test[model_features])



# fit logreg_model, rf_model, and KNN_model to test dataset features

logreg_test_predictions = logreg_model.predict(X_test)

linearSVC_test_predictions = linearSVC_model.predict(X_test)

rf_test_predictions = rf_model.predict(X_test)

KNN_test_predictions = KNN_model.predict(X_test)
logreg_output = pd.DataFrame({'PassengerId': test.PassengerId,

                       'Survived': logreg_test_predictions})

logreg_output.to_csv('logreg_submission.csv', index=False)
linearSVC = pd.DataFrame({'PassengerId': test.PassengerId,

                       'Survived': linearSVC_test_predictions})

linearSVC.to_csv('linearSVC_submission.csv', index=False)
rf_output = pd.DataFrame({'PassengerId': test.PassengerId,

                       'Survived': rf_test_predictions})

rf_output.to_csv('rf_submission.csv', index=False)
KNN_output = pd.DataFrame({'PassengerId': test.PassengerId,

                       'Survived': KNN_test_predictions})

KNN_output.to_csv('KNN_submission.csv', index=False)