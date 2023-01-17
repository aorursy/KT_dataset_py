import pandas as pd

import numpy as np

import random as rnd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from sklearn.metrics import accuracy_score

from sklearn.utils import shuffle



import tensorflow as tf

from tensorflow import keras
if_group_family = True   # if group family size feature or not

if_group_age = False     # if group age feature or not

if_group_fare = False    # if group fare feature or not

if_norm_age = True       # if normalize age feature or not

if_norm_fare = True      # if normalize fare feature or not
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]
train_df.info()

print('_'*40) # add a line

test_df.info()



# Note 

# a) Age has a lot missing values

# b) Passenger Id and Cabin num are useless features

# c) Sex, Embarked features need to be converted to numerics

# d) Name cannot be directly used but some info (title) can be extraced as a new feature
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



# Drop Name and Passenger ID info

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]
# Replace missing values

guess_ages = np.zeros((2,3))



for dataset in combine:

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



# Group age

if if_group_age:

    

   for dataset in combine:    

       dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

       dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

       dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

       dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

       dataset.loc[ dataset['Age'] > 64, 'Age']

       train_df.head()

 

# Normalize age

if if_norm_age:

    age_all=train_df['Age']

    age_mean=age_all.mean()

    age_std=age_all.std()

    for dataset in combine:  

        dataset['Age'] = dataset['Age']/age_mean
if if_group_family:

   for dataset in combine:

       dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

   for dataset in combine:

       dataset['IsAlone'] = 0

       dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

   train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

   #Drop Parch, SibSp, and FamilySize features

   train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

   test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

   combine = [train_df, test_df]
freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# Missing value = median

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



# Group fare

if if_group_fare:

   for dataset in combine:

       dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

       dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

       dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

       dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

       dataset['Fare'] = dataset['Fare'].astype(int)

  

# Normalize fare

if if_norm_fare:

    fare_all=train_df['Fare']

    fare_mean=fare_all.mean()

    fare_std=fare_all.std()

    for dataset in combine:  

        dataset['Fare'] = dataset['Fare']/fare_mean 
Y_train = train_df["Survived"]



X_train = train_df.drop("Survived",  axis=1)

X_sub  = test_df.drop("PassengerId", axis=1)
def evaluate(y_true, y_pred):

    #logloss = log_loss(y_true, y_pred)

    score = accuracy_score(y_true, y_pred)

    return round(score,2)



def models_CV_train(models, X, y, X_sub, n_folds=5):

    

    summary = {}



    skf = list(StratifiedKFold(n_folds, random_state=0).split(X, y))

    

    # contain predicted labels after n_fold

    stack_train = np.zeros((X.shape[0], len(models)))

    stack_sub = np.zeros((X_sub.shape[0], len(models))) # corresponds to X_submission

    

    for i, model in enumerate(models):

    

        print('_'*40) # add a line

        print('Model', i+1)

        

        metric_avg = 0

        

        stack_sub_model_i = np.zeros((X_sub.shape[0], len(skf)))

        #print('shape of stack_test_model_i',np.shape(stack_test_model_i))

        

        for j, (train_idx, test_idx) in enumerate(skf):

            

            print('Fold', j)

            

            # i) split data 

            X_train = np.array(X)[train_idx]

            y_train = y[train_idx]

            X_test = np.array(X)[test_idx]

            y_test = y[test_idx]



            # ii) train model

            model.fit(X_train, y_train)

             

            # iii) make prediction on test subset    

            y_test_pred = model.predict(X_test) 

            stack_train[test_idx, i] = y_test_pred

            

            # iv) evaluate model based on test subset

            metric = evaluate(y_test, y_test_pred)

            metric_avg += metric

            print('one-fold metric', metric)

            

            # v) predict labels for submission dataset

            y_sub_pred = model.predict(X_sub) 

            stack_sub_model_i[:, j] = y_sub_pred

        

        metric_avg = metric_avg / n_folds

        print('n-fold average metric:', round(metric_avg,2))

        summary[i] = metric_avg

        

        stack_sub[:, i] = stack_sub_model_i.mean(axis=1) # corresponds to X_submission



    return stack_train, stack_sub, summary
models = []



# Logistic Regression 

logreg1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.43, class_weight=None)

logreg2 = LogisticRegression(penalty='l2', solver='liblinear', C=0.01, class_weight=None, multi_class='ovr')

    

# KNN

knn1 = KNeighborsClassifier(n_neighbors=3)

knn2 = KNeighborsClassifier(n_neighbors=6)

knn3 = KNeighborsClassifier(n_neighbors=18)           

knn4 = KNeighborsClassifier(n_neighbors=54)

knn5 = KNeighborsClassifier(n_neighbors=162)



# Decision Tree

decision_tree = DecisionTreeClassifier()



# Random Forest

rf1 = RandomForestClassifier(n_estimators=100)

rf2 = RandomForestClassifier(criterion='gini', n_estimators=250)

rf3 = ExtraTreesClassifier(criterion='gini', n_estimators=500)



models += [logreg1]

models += [logreg2]

models += [knn1]

models += [knn2]

models += [knn3]

models += [knn4]

models += [knn5]

models += [decision_tree]

models += [rf1]

models += [rf2]

models += [rf3]



num_models = len(models)



n_folds = 5

Y_train_pred, Y_sub_pred, summary = models_CV_train(models, X_train, Y_train, X_sub, n_folds)
model = keras.Sequential([

    keras.layers.Dense(32, input_dim=num_models, activation='relu'),

    #keras.layers.Dropout(0.2),

    keras.layers.Dense(16, activation='relu'),

    #keras.layers.Dropout(0.2),

    keras.layers.Dense(1, activation='sigmoid') 

])



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
n_epoch = 100



history=model.fit(Y_train_pred, np.array(Y_train), epochs=n_epoch, verbose=0)



# Plot loss vs epochs

loss = history.history['loss']

epochs = range(len(loss))

plt.plot(epochs, loss, label='log loss')

plt.ylim([0.1, 1.0])

plt.grid()

plt.legend()

plt.show()



acc = history.history['accuracy']

plt.plot(epochs, acc, label='accuracy')

plt.ylim([0.5, 1.0])

plt.grid()

plt.legend()

plt.show()
predict = model.predict(Y_sub_pred)

predict = np.squeeze(np.array(predict))

predict = (predict > 0.5).astype(int)



print('Writing results into csv ...')

test_Id = test_df["PassengerId"]

test_Survived = pd.Series(predict.astype(int), name="Survived")

results = pd.concat([test_Id, test_Survived],axis=1)

results.to_csv("titanic_submission.csv",index=False)

print('All Done!')