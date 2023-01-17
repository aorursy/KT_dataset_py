import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from scipy.stats import chi2_contingency



from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score



import warnings

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.info()
test.head()
test.info()
train.isnull().sum()[train.isnull().sum() > 0]
test.isnull().sum()[test.isnull().sum() > 0]
def fill_missing(data):

    data.Age      = data.Age.fillna(data.Age.mean())

    data.Fare     = data.Fare.fillna(data.Fare.mean())

    data.Cabin    = data.Cabin.fillna("unknown")

    data.Embarked = data.Embarked.fillna("unknown")

    

    return data
train = fill_missing(train)

test  = fill_missing(test)
plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)

plt.bar(0, len(train[(train.Pclass == 1) & (train.Survived == 0)]), label="Not survived")

plt.bar(1, len(train[(train.Pclass == 1) & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("Lower SES")

plt.legend()



plt.subplot(1, 3, 2)

plt.bar(0, len(train[(train.Pclass == 2) & (train.Survived == 0)]), label="Not survived")

plt.bar(1, len(train[(train.Pclass == 2) & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("Middle SES")

plt.legend()



plt.subplot(1, 3, 3)

plt.bar(0, len(train[(train.Pclass == 3) & (train.Survived == 0)]), label="Not survived")

plt.bar(1, len(train[(train.Pclass == 3) & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("Upper SES")

plt.legend()



plt.show();
# Filter title

def split_name(data):

    res = data.split(".")

    res = res[0].split(", ")

    return res[1]



titles = pd.unique(train['Name'].apply(split_name))

print(titles)
def name2role(data):

    titles = {

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "the Countess":"Royalty",

        "Lady" :      "Royalty",

        "Dona":       "Royalty",

        "Master" :    "Master",

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "Mme":        "Normal",

        "Mlle":       "Normal",

        "Ms":         "Normal",

        "Mr" :        "Normal",

        "Mrs" :       "Normal",

        "Miss" :      "Normal"

    }

    encoded_titles = {

        "Royalty": 1,

        "Master": 2,

        "Officer": 3,

        "Normal": 4,

    }

    

    res = data.apply(split_name).map(titles).map(encoded_titles)

    

    return res
train_Name = name2role(train['Name'])
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)

plt.bar(0, len(train[(train_Name == 1) & (train.Survived == 0)]), label="Not survived")

plt.bar(1, len(train[(train_Name == 1) & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("Royalty")

plt.legend()



plt.subplot(1, 4, 2)

plt.bar(0, len(train[(train_Name == 2) & (train.Survived == 0)]), label="Not survived")

plt.bar(1, len(train[(train_Name == 2) & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("Master")

plt.legend()



plt.subplot(1, 4, 3)

plt.bar(0, len(train[(train_Name == 3) & (train.Survived == 0)]), label="Not survived")

plt.bar(1, len(train[(train_Name == 3) & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("Officer")

plt.legend()



plt.subplot(1, 4, 4)

plt.bar(0, len(train[(train_Name == 4) & (train.Survived == 0)]), label="Not survived")

plt.bar(1, len(train[(train_Name == 4) & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("Normal")

plt.legend()



plt.show();

train['Sex'] = train['Sex'].apply(lambda x: 0 if x == 'female' else 1)

test['Sex'] = test['Sex'].apply(lambda x: 0 if x == 'female' else 1)
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

plt.bar(0, len(train[(train.Sex == 0) & (train.Survived == 0)]), label="Not survived")

plt.bar(1, len(train[(train.Sex == 0) & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("Female")

plt.legend()



plt.subplot(1, 2, 2)

plt.bar(0,len(train[(train.Sex == 1) & (train.Survived == 0)]), label="Not survived")

plt.bar(1,len(train[(train.Sex == 1) & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("Male")

plt.legend()



plt.show();
print("The descriptive statistics of Age in train set:")

display(train.Age.describe())

print("The descriptive statistics of Age in test set:")

display(test.Age.describe())
warnings.filterwarnings(action="ignore")



def visualize_group_age(data):

    plt.figure(figsize=(20,5))

    plt.subplot(1, 3, 1)

    plt.bar(0, len(data[(data.Age < 15) & (data.Survived == 0)]), label="Not survived")

    plt.bar(1, len(data[(data.Age < 15) & (data.Survived == 1)]), label="Survived")

    plt.xticks([0, 1])

    plt.xlabel("Survived")

    plt.ylabel("Number of people")

    plt.title("People aged less than 15")

    plt.legend()



    plt.subplot(1, 3, 2)

    plt.bar(0,len(data[(data.Age >= 15) & (data.Age <= 43) & (data.Survived == 0)]), label="Not survived")

    plt.bar(1,len(data[(data.Age >= 15) & (data.Age <= 43) & (data.Survived == 1)]), label="Survived")

    plt.xticks([0, 1])

    plt.xlabel("Survived")

    plt.ylabel("Number of people")

    plt.title("People aged from 15 to 43")

    plt.legend()





    plt.subplot(1, 3, 3)

    plt.bar(0, len(data[(data.Age > 43) & (data.Survived == 0)]), label="Not survived")

    plt.bar(1, len(data[(data.Age > 43) & (data.Survived == 1)]), label="Survived")

    plt.xticks([0, 1])

    plt.xlabel("Survived")

    plt.ylabel("Number of people")

    plt.title("People aged greater than 43")

    plt.legend()



    plt.show();
visualize_group_age(train)
print("The descriptive statistics of Fare column in the train set:")

display(train.Fare.describe())

print("The descriptive statistics of Fare column in the test set:")

display(test.Fare.describe())
plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)

plt.boxplot(train.Fare)

plt.yticks(range(0,550, 50))

plt.title("A boxplot of Fare column in train set")

plt.subplot(1, 2, 2)

plt.boxplot(test.Fare)

plt.yticks(range(0,550, 50))

plt.title("A boxplot of Fare column in test set")



plt.show();
plt.figure(figsize=(20,5))

plt.subplot(1, 2, 1)

plt.bar(0, len(train[(train.Fare > 40) & (train.Survived == 0)]), label="Not survived")

plt.bar(1, len(train[(train.Age > 40) & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("People who paid more than the normal price")

plt.legend()



plt.subplot(1, 2, 2)

plt.bar(0,len(train[(train.Fare >= 0) & (train.Fare <= 40) & (train.Survived == 0)]), label="Not survived")

plt.bar(1,len(train[(train.Fare >= 0) & (train.Fare <= 40) & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("People who paid within the normal price")

plt.legend()



plt.show();
def process_cabin(data):

    # Unique cabin

    uni_ca = list(set(x[0] for x in pd.unique(data) if x != 'unknown'))

    uni_ca = sorted(uni_ca)

    uni_ca = {ca: i+1 for i, ca in enumerate(uni_ca)}

    

    # Map value of each cabin

    res = data.apply(lambda x: uni_ca[x[0]] if x[0] in uni_ca and x != 'unknown' else 0)

    

    return res
train['Cabin'] = process_cabin(train.Cabin)

test['Cabin'] = process_cabin(test.Cabin)
cabins = ['unknown', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']



cabin = 0

fig, ax = plt.subplots(3, 3, figsize=(20, 15))

for row in range(3):

    for col in range(3):

        fig.tight_layout()

        ax[row, col].bar(0, len(train[(train.Cabin == cabin) & (train.Survived == 0)]), label="Not survived")

        ax[row, col].bar(1, len(train[(train.Cabin == cabin) & (train.Survived == 1)]), label="Survived")

        ax[row, col].set_xticks([0, 1])

        ax[row, col].set_xlabel("Survived")

        ax[row, col].set_xlabel("Number of people")

        ax[row, col].set_title("Number of people who are located in cabin {}".format(cabins[cabin]))

        ax[row, col].legend()

        cabin += 1
plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)

plt.bar(0, len(train[(train.Embarked == 'C') & (train.Survived == 0)]), label="Not survived")

plt.bar(1, len(train[(train.Embarked == 'C') & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("People got on the ship in Cherbourg")

plt.legend()



plt.subplot(1, 3, 2)

plt.bar(0, len(train[(train.Embarked == 'S') & (train.Survived == 0)]), label="Not survived")

plt.bar(1, len(train[(train.Embarked == 'S') & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("People got on the ship in Southanmpton")

plt.legend()



plt.subplot(1, 3, 3)

plt.bar(0, len(train[(train.Embarked == 'Q') & (train.Survived == 0)]), label="Not survived")

plt.bar(1, len(train[(train.Embarked == 'Q') & (train.Survived == 1)]), label="Survived")

plt.xticks([0, 1])

plt.xlabel("Survived")

plt.ylabel("Number of people")

plt.title("People got on the ship in Queenstown")

plt.legend()



plt.show();
cates = {

    "gender" : {

        0: "Females",

        1: "Males"

    },

    "class" : {

        1: "Lower SES",

        2: "Middle SES",

        3: "Upper SES"

    }

}



socio_class = 1

fig, ax = plt.subplots(3, 2, figsize=(20, 12))

for row in range(3):

    gender = 0

    for col in range(2):

        fig.tight_layout()

        ax[row, col].bar(0, len(train[(train.Pclass == socio_class) & (train.Sex == gender) & (train.Survived == 0)]), label="Not survived")

        ax[row, col].bar(1, len(train[(train.Pclass == socio_class) & (train.Sex == gender) & (train.Survived == 1)]), label="Survived")

        ax[row, col].set_xticks([0, 1])

        ax[row, col].set_xlabel("Survived")

        ax[row, col].set_ylabel("Number of people")

        ax[row, col].set_title("{} who are in {}".format(cates["gender"][gender], cates["class"][socio_class]))

        ax[row, col].legend()

        gender += 1

    socio_class += 1
cates = {

    "paid" : {

        0: "within the usual price",

        1: "above the usual price"

    },

    "class" : {

        1: "Lower SES",

        2: "Middle SES",

        3: "Upper SES"

    }

}



paid = [(train.Fare >= 0) & (train.Fare <= 40), (train.Fare >= 40)]



fig, ax = plt.subplots(2, 3, figsize=(20, 10))

for row in range(2):

    socio_class = 1

    for col in range(3):

        fig.tight_layout()

        ax[row, col].bar(0, len(train[(train.Pclass == socio_class) & paid[row] & (train.Survived == 0)]), label="Not survived")

        ax[row, col].bar(1, len(train[(train.Pclass == socio_class) & paid[row] & (train.Survived == 1)]), label="Survived")

        ax[row, col].set_xticks([0, 1])

        ax[row, col].set_xlabel("Survived")

        ax[row, col].set_ylabel("Number of people")

        ax[row, col].set_title("People in {} who paid {}".format(cates["class"][socio_class], cates["paid"][row]))

        ax[row, col].legend()

        

        socio_class += 1
cates = {

    "gender" : {

        0: "Females",

        1: "Males"

    },

    "age" : {

        0: "less than 15",

        1: "from 15 to 43",

        2: "greater than 43"

    }

}



ages = [(train.Age < 15), (train.Age >= 15) & (train.Age <= 43), (train.Age > 43)]



fig, ax = plt.subplots(2, 3, figsize=(20, 10))

for row in range(2):

    age = 0

    for col in range(3):

        fig.tight_layout()

        ax[row, col].bar(0, len(train[(train.Sex == row) & ages[age] & (train.Survived == 0)]), label="Not survived")

        ax[row, col].bar(1, len(train[(train.Sex == row) & ages[age] & (train.Survived == 1)]), label="Survived")

        ax[row, col].set_xticks([0, 1])

        ax[row, col].set_xlabel("Survived")

        ax[row, col].set_ylabel("Number of people")

        ax[row, col].set_title("{} aged {}".format(cates["gender"][row], cates["age"][age]))

        ax[row, col].legend()

        

        age += 1
# Code Name

train.Name = name2role(train['Name'])

test.Name  = name2role(test['Name'])



# Recode Sex

# 1 -> Female | 2 -> Male

train.Sex = train.Sex.apply(lambda x: 0 if x == 0 else 1)

test.Sex = test.Sex.apply(lambda x: 0 if x == 0 else 1)



# Code Fare

train.Fare = train.Fare.apply(lambda x: 1 if x > 40 else 0)

test.Fare = test.Fare.apply(lambda x: 1 if x > 40 else 0)



# Codde Embarked

dict_embarked = {"unknown": 0,"C": 1, "S": 2, "Q": 3}

train.Embarked = train.Embarked.map(dict_embarked)

test.Embarked = test.Embarked.map(dict_embarked)



# Group Age

def filter_age(age):

    if age < 15:

        return 0

    elif age > 43:

        return 2

    else:

        return 1

    

train.Age = train.Age.apply(filter_age)

test.Age  = test.Age.apply(filter_age)
# Sibsp + Parch -> Detect who were in 1 family and who traveled alone

# Status: 1 -> Family | 2 -> Alone

train['Status'] = train.SibSp + train.Parch

test['Status'] = test.SibSp + test.Parch

train.Status = train.Status.apply(lambda x: 0 if x > 0 else 1)

test.Status = test.Status.apply(lambda x: 0 if x > 0 else 1)



# Pclass * Fare

train['Pclass_Fare'] = train.Pclass * train.Fare

test['Pclass_Fare'] = test.Pclass * test.Fare



# Pclass * Sex

train['Pclass_Sex'] = train.Pclass * train.Sex

test['Pclass_Sex'] = test.Pclass * test.Sex



# Drop Ticket, Cabin, SibSp and Parch

train.drop(['Ticket', 'SibSp', 'Parch','Cabin'], axis=1, inplace=True)

test.drop(['Ticket', 'SibSp', 'Parch','Cabin'], axis=1, inplace=True)
# Build model

def build_model(model, X_train, X_test, y_train, y_test):

    ''' A wrapper of building classification model

    

    Parameters

    ----------

    

    Return

    ------

    clf: Model

    cm: Confusion matrix

    result: A dictionary of evalation metrics

    

    Notes

    -----

    LR = Logistic Regression

    RF = Random Forest

    NB = Naive Bayes

    '''

    if model['name'] not in ["LR", "RF", "NB", "SVM"]:

        raise ValueError("There is no model, called {}".format(model['name']))

        

    clf = None

    cm  = None

    result = dict()

    

    # Model selection

    if model['name'] is "RF":

        clf = RandomForestClassifier(n_estimators=model['n_estimators'], criterion=model['criterion'],\

                                     random_state=model['random_state'])

    elif model['name'] is "LR":

        clf = LogisticRegression(penalty=model['penalty'], random_state=model['random_state'],\

                                 solver=model['solver'])

    elif model['name'] is "NB":

        clf = GaussianNB()

    elif model['name'] is "SVM":

        clf = SVC(C=model['C'], kernel=model['kernel'], tol=model['tol'], probability = model['probability'], random_state=model['random_state'])

    

    # Fit the model

    clf.fit(X_train, y_train)

    

    # Predict based X_test

    y_pred = clf.predict(X_test)

    y_scores = clf.predict_proba(X_test)

    

    # Evaluation

    cm = confusion_matrix(y_test, y_pred)

    result['Accuracy'] = accuracy_score(y_test, y_pred)

    result['ROC_AUC']  = roc_auc_score(y_test, y_scores[:,1])

    

    return clf, cm, result
# Split train and test set

from sklearn.model_selection import train_test_split



X = train.iloc[:, 2:11]

y = train.iloc[:, 1:2]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
# Avoid warning

import warnings

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)



# Run model

models = [

    {

        "name": "RF",

        "n_estimators": 100,

        "criterion": "gini",

        "random_state": 0

    },

    {

        "name": "LR",

        "penalty": "none",

        "random_state": 0,

        "solver": "newton-cg"

    },

    {

        "name": "NB"

    },

    {

        "name": "SVM",

        "C": 50,

        "kernel": "rbf",

        "tol": 10**-5,

        "probability": True,

        "random_state": 0

    }

]



for model in models:

    clf, cm, res = build_model(model, X_train, X_test, y_train, y_test)

    print("Model: {}".format(model['name']))

    print("Accuracy: {}".format(res['Accuracy']))

    print("ROC_AUC: {}".format(res['ROC_AUC']))

    print("Confusion matrix:")

    display(

        pd.DataFrame(cm, columns=['True', 'False'], index=['True', 'False'])

    )
# Select the model

selected_model = {

    "name": "SVM",

    "C": 50,

    "kernel": "rbf",

    "tol": 10**-5,

    "probability": True,

    "random_state": 0

}



# Fit and Predict data

clf, _, _ = build_model(selected_model, X_train, X_test, y_train, y_test)

predictions = clf.predict(test.iloc[:, 1:])



# Create result dataframe

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})

submission.head()
# Create csv file to upload to Kaggle

submission.to_csv('titanic_output.csv', index=False)