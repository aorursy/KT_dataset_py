import numpy as np 

import pandas as pd 

import csv as csv

from subprocess import check_output



training_data = pd.read_csv('../input/train.csv', header=0)

test_data = pd.read_csv('../input/test.csv', header=0)
training_data['Gender'] = training_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

test_data['Gender'] = test_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
print(training_data[["Gender", "Survived"]].groupby(['Gender']).mean())
training_data.Embarked[training_data.Embarked.isnull()] = 'U'

test_data.Embarked[test_data.Embarked.isnull()] = 'U'





Ports = list(enumerate(np.unique(training_data['Embarked'])))   

Ports_dict = {name : i for i, name in Ports}   

training_data['Embarked'] = training_data['Embarked'].map(lambda x: Ports_dict[x]).astype(int)



Ports = list(enumerate(np.unique(test_data['Embarked'])))    

Ports_dict = { name : i for i, name in Ports }    

test_data['Embarked'] = test_data['Embarked'].map(lambda x: Ports_dict[x]).astype(int)     
print(training_data[['Embarked','Survived']].groupby(['Embarked']).mean())
training_data.loc[ (training_data.Age.isnull()), 'Age'] = -1

test_data.loc[ (test_data.Age.isnull()), 'Age'] = -1
import matplotlib.pyplot as plt



plt.plot(np.unique(training_data['Age']), training_data[['Age', 'Survived']].groupby(['Age']).mean())

plt.show()
print(training_data[['Fare', 'Pclass']].dropna().groupby(['Pclass']).max())

print(training_data[['Fare', 'Pclass']].dropna().groupby(['Pclass']).mean())

print(training_data[['Fare', 'Pclass']].dropna().groupby(['Pclass']).median())



print(training_data[['Pclass', 'Survived']].dropna().groupby(['Pclass']).mean())
training_data.loc[(training_data.Fare.isnull()), 'Fare'] = -1

test_data.loc[(test_data.Fare.isnull()), 'Fare'] = -1
classMedians = training_data.groupby(['Pclass'])['Fare'].median()

classMeans = training_data.groupby(['Pclass'])['Fare'].mean()

print(classMedians)

print(classMeans)



training_data.loc[(training_data.Fare.isnull()) & (training_data['Pclass'] == 1), 'Fare'] = classMedians[1]

training_data.loc[(training_data.Fare.isnull()) & (training_data['Pclass'] == 2), 'Fare'] = classMedians[2]

training_data.loc[(training_data.Fare.isnull()) & (training_data['Pclass'] == 3), 'Fare'] = classMedians[3]



test_data.loc[(test_data.Fare.isnull()) & (test_data['Pclass'] == 1), 'Fare'] = classMedians[1]

test_data.loc[(test_data.Fare.isnull()) & (test_data['Pclass'] == 2), 'Fare'] = classMedians[2]

test_data.loc[(test_data.Fare.isnull()) & (test_data['Pclass'] == 3), 'Fare'] = 85
training_data['FareCats3'] = pd.qcut(training_data['Fare'], 3)



print(training_data[['FareCats3', 'Survived']].dropna().groupby(['FareCats3']).mean())
training_data['FareCats4'] = pd.qcut(training_data['Fare'], 4)



print(training_data[['FareCats4', 'Survived']].dropna().groupby(['FareCats4']).mean())
training_data['FareCats5'] = pd.qcut(training_data['Fare'], 5)



print(training_data[['FareCats5', 'Survived']].dropna().groupby(['FareCats5']).mean())
training_data = training_data.drop(['FareCats3', 'FareCats4', 'FareCats5'], axis=1)
training_data['Family'] = training_data['SibSp'] + training_data['Parch'] + 1 # add 1 for 'self'

test_data['Family'] = test_data['SibSp'] + test_data['Parch'] + 1 # add 1 for 'self'   
print(training_data[['Family', 'Survived']].groupby(['Family']).sum())
print(training_data[['Family', 'Survived']].groupby(['Family']).mean())

print(training_data[['SibSp', 'Survived']].groupby(['SibSp']).mean())

print(training_data[['Parch', 'Survived']].groupby(['Parch']).mean())
import re as re



def title_from_name(name):

    title = re.search(' ([A-Za-z]+)\.', name)

    if title:

        return title.group(1)

    return ""



training_data['Title'] = training_data['Name'].apply(title_from_name)

test_data['Title'] = training_data['Name'].apply(title_from_name)



print(training_data[['Title', 'Survived']].groupby(['Title']).mean())

print(training_data[['Title', 'Survived']].groupby(['Title']).count())
#training_data['Title'] = training_data['Title'].replace(['Capt', 'Col', 'Countess', 

#                                                        'Don', 'Jonkheer', 'Lady', 'Sir', 'Dr', 'Rev',

#                                                        'Major'], 'Other')



#test_data['Title'] = test_data['Title'].replace(['Capt', 'Col', 'Countess', 

#                                                        'Don', 'Jonkheer', 'Lady', 'Sir', 'Dr', 'Rev',

#                                                        'Major'], 'Other')



#test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')

#training_data['Title'] = training_data['Title'].replace('Mlle', 'Miss')



#training_data['Title'] = training_data['Title'].replace('Ms', 'Miss')

#test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')



#training_data['Title'] = training_data['Title'].replace('Mme', 'Mrs')

#test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')







#dummy_titles1 = pd.get_dummies(training_data['Title'],prefix='Title')

#dummy_titles2 = pd.get_dummies(test_data['Title'],prefix='Title')



#training_data = pd.concat([training_data,dummy_titles1],axis=1)

#test_data = pd.concat([test_data,dummy_titles2],axis=1)

#test_data['Title_Major'] = 0

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score, accuracy_score, log_loss



# drop unused data columns

training_data = training_data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Title'], axis=1) 

test_data = test_data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Title'], axis=1) 
# Normalize the data

min_age = np.asarray(training_data['Age'].min(), test_data['Age'].min()).min()

max_age = np.asarray(training_data['Age'].max(), test_data['Age'].max()).max()

training_data['Age'] = (training_data['Age'] - min_age )/ (max_age - min_age)

test_data['Age'] = test_data['Age'] / max_age



min_fare = np.asarray(training_data['Fare'].min(), test_data['Fare'].min()).min()

max_fare = np.asarray(training_data['Fare'].max(), test_data['Fare'].max()).max()

training_data['Fare'] = (training_data['Fare'] - min_fare) / (max_fare - min_fare)

test_data['Fare'] = (test_data['Fare'] - min_fare) / (max_fare - min_fare)



min_fam_size = np.asarray(training_data['Family'].min(), test_data['Family'].min()).min()

max_fam_size = np.asarray(training_data['Family'].max(), test_data['Family'].max()).max()

training_data['Family'] = (training_data['Family'] - min_fam_size)/ (max_fam_size - min_fam_size)

test_data['Family'] = (test_data['Family'] - min_fam_size) / (max_fam_size - min_fam_size)



min_emb = np.asarray(training_data['Embarked'].min(), test_data['Embarked'].min()).min()

max_emb = np.asarray(training_data['Embarked'].max(), test_data['Embarked'].max()).max()

training_data['Embarked'] = (training_data['Embarked'] - min_emb)/ (max_emb - min_emb)

test_data['Embarked'] = (test_data['Embarked'] - min_emb) / (max_emb - min_emb)



min_class = np.asarray(training_data['Pclass'].min(), test_data['Pclass'].min()).min()

max_class = np.asarray(training_data['Pclass'].max(), test_data['Pclass'].max()).max()

training_data['Pclass'] = (training_data['Pclass'] - min_class)/ (max_class - min_class)

test_data['Pclass'] = (test_data['Pclass'] - min_class) / (max_class - min_class)



min_class = np.asarray(training_data['Pclass'].min(), test_data['Pclass'].min()).min()

max_class = np.asarray(training_data['Pclass'].max(), test_data['Pclass'].max()).max()

training_data['Pclass'] = (training_data['Pclass'] - min_class)/ (max_class - min_class)

test_data['Pclass'] = (test_data['Pclass'] - min_class) / (max_class - min_class)
features = training_data.columns[2:]

train_true = training_data.columns[1:2]



X = training_data[features]

y = training_data[train_true]

y = np.asarray(y).reshape(-1)



# these are the features we'll use to classify

print(features)
import seaborn as sns

import matplotlib.pyplot as plt



classes = [KNeighborsClassifier(5),SVC(probability=True),DecisionTreeClassifier(),RandomForestClassifier(),

          AdaBoostClassifier(),GradientBoostingClassifier(),GaussianNB(),LogisticRegression()]



splits = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)





log_cs = ['Classifier', 'F1', 'Acc']

log = pd.DataFrame(columns=log_cs)

f1s = {}

acc = {}



X = X.values



# intialize f1 and accuracy values

for cls in classes:

    clf = cls.__class__.__name__

    f1s[clf] = 0

    acc[clf] = 0



for trn_idx, tst_idx in splits.split(X, y):

    X_train, X_test = X[trn_idx], X[tst_idx]

    Y_train, Y_test = y[trn_idx], y[tst_idx]

    

    for cls in classes:

        name = cls.__class__.__name__

        cls.fit(X_train, Y_train)

        train_preds = cls.predict(X_test)

        

        f1 = f1_score(Y_test, train_preds)

        acc_sc = accuracy_score(Y_test, train_preds)



        # add splits f1, acc

        f1s[name] += f1

        acc[name] += acc_sc



for cls in f1s:

    f1s[cls] = f1s[cls] / 10.0 # average over ten runs

    acc[cls] = acc[cls] / 10.0 # average over ten runs

    log_ = pd.DataFrame([[cls, f1s[cls], acc[cls]]], columns=log_cs)

    log = log.append(log_)



print(log)
outfileName = 'mosleylm_titanicGBC.csv'



splits = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

cls = GradientBoostingClassifier()



for trn_idx, tst_idx in splits.split(X, y):

    X_train, X_test = X[trn_idx], X[tst_idx]

    Y_train, Y_test = y[trn_idx], y[tst_idx]

    



    name = cls.__class__.__name__

    cls.fit(X_train, Y_train)

    train_preds = cls.predict(X_test)

res = cls.predict(test_data[features])



submit = test_data['PassengerId']

submit = np.vstack((submit,res)).T
import csv

submit = pd.DataFrame(submit)

submit.columns = ["PassengerId", "Survived"]

submit.to_csv(outfileName, sep=',', index=False)
objects = ('Random Forest', 'SVC')

y_pos = np.arange(len(objects))

performance = [0.73684, 0.61244]

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)



plt.show()
objects = ('R Forest', 'R Forest (+Fam)', 'SVC', 'SVC (+Fam)')

y_pos = np.arange(len(objects))

performance = [0.73684, 0.74162, 0.61244, 0.60287]

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylim(0.60,0.78)



plt.show()
objects = ('Non-Cat', 'Categorized(5)', 'Categorized(4)', 'Categorized(3)')

y_pos = np.arange(len(objects))

performance = [0.74162, 0.67464, 0.70813, 0.69813]

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)



plt.show()

print("Comparison of Fare Categories")
objects = ('Benchmark', 'Add Titles', 'Groups')

y_pos = np.arange(len(objects))

performance = [0.74162, 0.69377, 0.68899] 

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)



plt.show()
objects = ('R Forest', 'Gradient Boost')

y_pos = np.arange(len(objects))

performance = [0.74162, 0.75598] 

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylim(0.7,0.8)



plt.show()
objects = ('Age Mean', 'Age Median', 'Age -1')

y_pos = np.arange(len(objects))

performance = [0.75598, 0.75598, 0.76076] 

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylim(0.7,0.8)



plt.show()
objects = ('Benchmark', 'Dummy', 'Normalized', 'Unknown')

y_pos = np.arange(len(objects))

performance = [0.76076, 0.71770, 0.59330, 0.60287] 

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylim(0.55,0.8)



plt.show()
objects = ('Benchmark', 'Normalized')

y_pos = np.arange(len(objects))

performance = [0.76076, 0.77511] 

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylim(0.55,0.8)



plt.show()
print(submit)