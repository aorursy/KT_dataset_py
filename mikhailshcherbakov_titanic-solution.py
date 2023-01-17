import keras

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from keras import layers, models

from keras.utils import to_categorical

import tensorflow as tf

import datetime, os

from matplotlib.pyplot import figure

import missingno

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

import seaborn as seb

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.gaussian_process.kernels import DotProduct

from sklearn.gaussian_process.kernels import Matern



seed = 42

np.random.seed(seed)
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")
train_data.head(3)
test_data.head(3)
id = test_data["PassengerId"]



train_data.drop(["PassengerId", "Ticket"], axis =1, inplace=True)

test_data.drop(["PassengerId", "Ticket"], axis =1, inplace=True)
for df in [train_data, test_data]:

    missingno.matrix(df)
#fill in the NaN using the most common value

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace = True)



#fill in the NaN using the median value

test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())



#similar to the previous one

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())



#and here let the absence of the cabin number be characterized by the letter 'O'

train_data['Cabin'] = train_data['Cabin'].fillna('O')

test_data['Cabin'] = test_data['Cabin'].fillna('O')
for df in [train_data, test_data]:

    missingno.matrix(df)
figure(num=None, figsize=(15, 4), dpi=100)

seb.boxplot(x="Fare",  data=train_data)
train_data = train_data[train_data['Fare'] <= 300]
train_data = train_data.reset_index(drop = True)
train_data['Sex'].value_counts()
test_data['Sex'].value_counts()
train_data['Sex'] = train_data['Sex'].replace({'male': 0, 'female': 1})

test_data['Sex'] = test_data['Sex'].replace({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].replace({'S': 1, 'C': 2, 'Q': 3, 'O' : 0})

test_data['Embarked'] = test_data['Embarked'].replace({'S': 1, 'C': 2, 'Q': 3, 'O' : 0})
train_data = pd.get_dummies(train_data, columns = ['Pclass', 'Sex', 'Embarked'])

test_data = pd.get_dummies(test_data, columns = ['Pclass', 'Sex', 'Embarked'])
train_data['Family_members'] = train_data['Parch'] + train_data['SibSp']

test_data['Family_members'] = test_data['Parch'] + test_data['SibSp']
train_data['Cabin'] = train_data['Cabin'].apply(lambda x: x[0])

test_data['Cabin'] = test_data['Cabin'].apply(lambda x: x[0])
le = LabelEncoder()

train_data['Cabin'] = le.fit_transform(train_data['Cabin'])

test_data['Cabin'] = le.fit_transform(test_data['Cabin'])
train_data['Cabin'].value_counts()
#divide the set of different ages into 4 parts

age_bins_train = np.linspace(np.min(train_data['Age']), np.max(train_data['Age']), 4)



#set the names of these parts (classes)

age_classes = [1,2,3]



#creating a new column

train_data['Age_classes'] = pd.cut(train_data['Age'], bins = age_bins_train, labels = age_classes, include_lowest = True)



#same with the test dataset

age_bins_test = np.linspace(np.min(test_data['Age']), np.max(test_data['Age']), 4)

test_data['Age_classes'] = pd.cut(test_data['Age'], bins = age_bins_test, labels = age_classes, include_lowest = True)
fare_bins_train = np.linspace(np.min(train_data['Fare']), np.max(train_data['Fare']), 5)

fare_classes = [1,2,3,4]

train_data['Fare_classes'] = pd.cut(train_data['Fare'], bins = fare_bins_train, labels =fare_classes, include_lowest = True)



fare_bins_test = np.linspace(np.min(test_data['Fare']), np.max(test_data['Fare']), 5)

test_data['Fare_classes'] = pd.cut(test_data['Fare'], bins = fare_bins_test, labels =fare_classes, include_lowest = True)
#extracting the title from the name

train_data['Name'] = train_data['Name'].apply(lambda x: x.split('.')[0].split(',')[1][1:])



train_data['Name'].value_counts()
#creating a separate class for rare titles

train_data['Name'] = train_data['Name'].replace(['Lady', 'Ms', 'Mme', 'the Countess', 'Mlle', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



#encoding titles

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

train_data['Name'] = train_data['Name'].map(title_mapping)



#just in case there are missing values, we will replace them with zeros

train_data['Name'] = train_data['Name'].fillna(0)
train_data['Name'].value_counts()
test_data['Name'] = test_data['Name'].apply(lambda x: x.split('.')[0].split(',')[1][1:])

test_data['Name'] = test_data['Name'].replace(['Lady', 'Ms', 'Mme', 'the Countess', 'Mlle', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test_data['Name'] = test_data['Name'].map(title_mapping)

test_data['Name'] = test_data['Name'].fillna(0)
train_data.head()
scaler = StandardScaler()



#selecting specific columns from the dataset

unscaled_train = train_data[["Age", "Fare", "Cabin", "SibSp", "Parch", "Family_members", "Name", "Age_classes", "Fare_classes"]]



#scaling

scaled_train = scaler.fit_transform(unscaled_train)



#in the previous step, we got a numpy array, so we convert it back to a DataFrame

scaled_train = pd.DataFrame(scaled_train, columns = ["Age", "Fare", "Cabin", "SibSp", "Parch", "Family_members", "Name", "Age_classes", "Fare_classes"])



#creating a dataset without the attributes that we scaled

train_data_scaling = train_data.drop(["Age", "Fare", "Cabin", "SibSp", "Parch", "Family_members", "Name", "Age_classes", "Fare_classes"], axis = 1)



#putting it all together

train_data_scaled = pd.concat([scaled_train, train_data_scaling], axis=1, sort=False)

train_data = train_data_scaled
train_data.head(3)
scaler = StandardScaler()

unscaled_test = test_data[["Age", "Fare", "Cabin", "SibSp", "Parch", "Family_members", "Name", "Age_classes", "Fare_classes"]]

scaled_test = scaler.fit_transform(unscaled_test)

scaled_test = pd.DataFrame(scaled_test, columns = ["Age", "Fare", "Cabin", "SibSp", "Parch", "Family_members", "Name", "Age_classes", "Fare_classes"])

test_data_scaling = test_data.drop(["Age", "Fare", "Cabin", "SibSp", "Parch", "Family_members", "Name", "Age_classes", "Fare_classes"], axis = 1)

test_data_scaled = pd.concat([scaled_test, test_data_scaling], axis=1, sort=False)

test_data = test_data_scaled
#throwing overboard doppelgangers

train_data.drop_duplicates(inplace = True)



#restoring the order of the indexes

train_data = train_data.reset_index(drop = True)
train_labels = train_data['Survived']

train_data.drop('Survived', axis = 1, inplace = True)
train_data.head(3)
test_data.head(3)
#type coercion

train_data = train_data.astype('float32')



#just in case, we'll leave DataFrame-copy

train_data_df = train_data



#DataFrame to numpy array

train_data = train_data.to_numpy()



#similarly

test_data = test_data.astype('float32')

test_data_df = test_data

test_data = test_data.to_numpy()



train_labels = train_labels.to_numpy()
#initializing the classifier

forest = ExtraTreesClassifier(n_estimators=250,

                              random_state=seed)

forest.fit(train_data, train_labels.reshape(-1,))



#output a sorted list of important parameters and build a graph

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



print("Feature ranking:")



for f in range(train_data.shape[1]):

    print("%d. '%s' (%f)" % (f + 1, train_data_df.columns[indices[f]], importances[indices[f]]))



figure(num=None, figsize=(8, 6), dpi=100)

plt.title("Feature importances")

plt.bar(range(train_data.shape[1]), importances[indices],

        color="r", yerr=std[indices], align="center")

plt.xticks(range(train_data.shape[1]), indices)

plt.xlim([-1, train_data.shape[1]])

plt.show()
#removing unnecessary columns from both sets

train_data_df.drop(["Embarked_1", "Embarked_2", "Embarked_3", "Fare_classes", 'Parch'], axis =1, inplace=True)

test_data_df.drop(["Embarked_1", "Embarked_2", "Embarked_3", "Fare_classes", 'Parch'], axis =1, inplace=True)



#getting the final datasets

train_data = train_data_df.to_numpy()

test_data = test_data_df.to_numpy()
#model names

names = ["Nearest Neighbors", 

         "Linear SVM", 

         "RBF SVM", 

         "Gaussian Process",

         "Decision Tree", 

         "Random Forest", 

         "AdaBoost",

         "Naive Bayes"

         ]



#initialize models

classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    AdaBoostClassifier(),

    GaussianNB()

    ]



#output the accuracy of each model on cross-validation with 5 folds

for name, clf in zip(names, classifiers):

    print("%s : %.2f%%" % (name, np.mean(cross_val_score(clf, train_data, train_labels.reshape(-1,), cv=5))*100))
gp_param = {

    'kernel':[DotProduct(i) for i in [2,3,5]] + [Matern(i) for i in [2,3,5]]  + [RBF(i) for i in [2,3,5]],

    'max_iter_predict' : [30, 50, 80]

    }





gp = GaussianProcessClassifier(random_state=seed)

gp_clf = GridSearchCV(gp, gp_param, n_jobs=-1, verbose=1)

gp_clf.fit(train_data, train_labels)
#show best parameters

gp_best_params = gp_clf.best_params_    

print('Best params : ', gp_best_params)

gp_best = gp_clf.best_estimator_
print(np.mean(cross_val_score(gp_best, train_data, train_labels.reshape(-1,), cv=5)))
#a simple function that returns a network trained on N epochs

def get_network(N):

    all_accuracy_history = []

    all_val_accuracy_history = []

    num_epochs = N



    #perform cross-validation with five folds

    skf = StratifiedKFold(n_splits = 5, random_state = seed, shuffle = True)



    for train_index, val_index in skf.split(train_data, train_labels):

        X_train, X_val = train_data[train_index], train_data[val_index]

        y_train, y_val = train_labels[train_index], train_labels[val_index]



        network = models.Sequential()

        network.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))

        network.add(layers.Dense(64, activation='relu'))

        network.add(layers.Dense(1, activation='sigmoid'))

        network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



        history = network.fit(X_train, y_train, epochs = num_epochs, validation_data = (X_val, y_val), batch_size = 4, verbose = 0)

        history_dict = history.history

        all_accuracy_history.append(history_dict['accuracy'])

        all_val_accuracy_history.append(history_dict['val_accuracy'])



    avg_accuracy_per_epoch = [np.mean([x[i] for x in all_accuracy_history]) for i in range(num_epochs)]

    avg_val_accuracy_per_epoch = [np.mean([x[i] for x in all_val_accuracy_history]) for i in range(num_epochs)]



    #plotting how the accuracy of our model has changed over time to track overfitting

    figure(num=None, figsize=(8, 6), dpi=100)

    plt.plot(range(1, num_epochs + 1), avg_accuracy_per_epoch, 'b', label='train')

    plt.plot(range(1, num_epochs + 1), avg_val_accuracy_per_epoch, 'r', label='validation')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()



    print("Average accuracy: %.2f%%" % (np.mean([np.mean(fold) for fold in all_val_accuracy_history])*100))



    return network
network = get_network(20)
network = get_network(10)
#getting predictions on the test set

result = network.predict(test_data)



#let's take a look at the predictions

result[:5]
#convert to binary form

result =(result > 0.5)



#we create the final answer and output it to the "answer.csv" file

answer = np.c_[id, result]

answer = answer.astype('int64')

answer = pd.DataFrame(answer, columns=["PassengerId", "Survived"])

answer.to_csv("answer.csv", index=False)