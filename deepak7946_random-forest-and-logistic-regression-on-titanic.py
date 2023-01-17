#importing required packages and data

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt



data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")
print("Training data has {} rows and {} columns".format(data_train.shape[0], data_train.shape[1]))

data_train.head()
print("Test data has {} rows and {} columns".format(data_test.shape[0], data_test.shape[1]))

data_test.head()
data_train.isnull().sum()
data_test.isnull().sum()
drop_cols = ["PassengerId", "Ticket", "Cabin"]

data_train.drop(drop_cols, inplace=True, axis=1, errors="ignore")

data_test.drop(drop_cols, inplace=True, axis=1, errors="ignore")

print(data_train.info())

print("==================================")

print(data_test.info())
#imputing embarked missing values in training data

data_train.loc[data_train["Embarked"].isnull(),"Embarked"] = data_train["Embarked"].mode()[0]

data_test.loc[data_test["Fare"].isnull(),"Fare"] = data_test["Fare"].median()

print("The 2 missing values in Embarked feature has been filled with {}".format(data_train["Embarked"].mode()[0]))

print("The 1 missing values in Fare feature has been filled with {}".format(data_test["Fare"].median()))
#Creating derived feature "Title"

combined = [data_train, data_test]

for dataset in combined:

    dataset["Title"] = dataset["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    

pd.concat([data_train["Title"], data_test["Title"]]).value_counts()
valid_title = ["Mr","Miss","Mrs","Master"]

data_train["Title"] = data_train.Title.apply(lambda x: x if x in valid_title else "Other")

data_test["Title"] = data_test.Title.apply(lambda x: x if x in valid_title else "Other")

pd.concat([data_train["Title"], data_test["Title"]]).value_counts()
#Drop the name column

data_train.drop("Name", inplace=True, axis=1, errors="ignore")

data_test.drop("Name", inplace=True, axis=1, errors="ignore")
f,axes= plt.subplots(1,2, figsize=(10,5))

p1 = sns.boxplot(x="Title", y="Age", data=data_train, ax=axes[0])

p2 = sns.boxplot(x="Title", y="Age", data=data_test, ax=axes[1])

p1.set(title="Training data")

p2.set(title="Test data")
for dataset in combined:

    dataset.loc[(dataset["Age"].isnull()) & (dataset["Title"]=="Mr"),"Age"] = dataset.loc[dataset["Title"]=="Mr", "Age"].median()

    dataset.loc[(dataset["Age"].isnull()) & (dataset["Title"]=="Mrs"), "Age"] = dataset.loc[dataset["Title"]=="Mrs", "Age"].median()

    dataset.loc[(dataset["Age"].isnull()) & (dataset["Title"]=="Miss"), "Age"] = dataset.loc[dataset["Title"]=="Miss", "Age"].median()

    dataset.loc[(dataset["Age"].isnull()) & (dataset["Title"]=="Master"),"Age"] = dataset.loc[dataset["Title"]=="Master", "Age"].median()

    dataset.loc[(dataset["Age"].isnull()) & (dataset["Title"]=="Other"), "Age"] = dataset.loc[dataset["Title"]=="Other", "Age"].median()



print(data_train.isnull().sum())

print("================================")

print(data_test.isnull().sum())
data_train.head()
f,axes = plt.subplots(1,2, figsize=(20,5))

p1=sns.boxplot(data=data_train, x="Age", ax=axes[0])

p2=sns.boxplot(data=data_train, x="Fare", ax=axes[1])

p2.set_xlim(right=100)
def bin_age(data):

    labels = ("0-5","5-15","15-24","24-35","35-45","45-55","55-65","65-90")

    bins = (0,5,15,24,35,45,55,65,90)

    data["Age"] = pd.cut(data.Age, bins, labels=labels)

    

def bin_fare(data):

    labels = ("very_low", "low","moderate","high","very_high")

    bins=(-1,10,15,30,50,700)

    data["Fare"] = pd.cut(data.Fare, bins, labels=labels)
datasets = [data_train, data_test]

for dataset in datasets:

    bin_age(dataset)

    bin_fare(dataset)
data_train.head()
def plot_graph(data, x, y, hue=None):

    if hue==None:

        f,axes = plt.subplots(1,2, figsize=(15,5))

        sns.barplot(data=data, x=x, y=y, ax=axes[0])

        sns.countplot(data=data, x=x, ax=axes[1])

    else:

        f,axes = plt.subplots(1,2,figsize=(15,5))

        sns.barplot(data=data, x=x, y=y, hue=hue, ax=axes[0])

        sns.countplot(data=data, x=x, hue=hue, ax=axes[1])
plot_graph(data_train, "Sex","Survived")
plot_graph(data_train,"Age","Survived", "Sex")
plot_graph(data_train, x="Pclass", y="Survived", hue="Sex")
plot_graph(data=data_train, x="Title",y="Survived")
plot_graph(data_train, "Embarked", "Survived")
plot_graph(data_train,"Fare","Survived")
plot_graph(data_train,"Fare","Survived","Pclass")
#shuffling the data

data_train = data_train.reindex(np.random.permutation(data_train.index))



features = ['Pclass','Sex','Age','SibSp','Parch','Embarked','Title']

data_X = data_train[features]

test_X = data_test[features]

data_Y = data_train["Survived"]

data_X.head()
from sklearn import preprocessing



def encode_features(data_X,test_X):

    features_to_label = ["Sex","Age","Embarked","Title"]

    combined_X = pd.concat([data_X[features_to_label],test_X[features_to_label]])

    for feature in features_to_label:

        encoder = preprocessing.LabelEncoder()

        encoder = encoder.fit(combined_X[feature])

        data_X[feature] = encoder.transform(data_train[feature])

        test_X[feature] = encoder.transform(test_X[feature])

    return (data_X, test_X)



data_X,test_X = encode_features(data_X,test_X)

data_X.head()
from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size=0.15, shuffle=False)

print("Training set shape : {}".format(x_train.shape))

print("Validation set shape : {}".format(x_val.shape))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, make_scorer



model = RandomForestClassifier()



#choosing a set of parameters to try on

params = {

    'n_estimators':[4,5,9,64,100,250],

    'criterion':['gini','entropy'],

    'max_features':['sqrt','log2',None],

    'max_depth':[4,8,16,32,None],

    'max_depth':[3,5,7,9,11]

}



accuracy_scorer = make_scorer(accuracy_score)



search_params = GridSearchCV(model, params, scoring=accuracy_scorer)

search_params = search_params.fit(x_train, y_train)



model = search_params.best_estimator_

print("The model has an accuracy of {}".format(search_params.best_score_))

model.fit(x_train, y_train)
predict = model.predict(x_val)

print("The validation set accuracy is {}".format(accuracy_score(y_val, predict)))
features = ['Pclass','Sex','Age','SibSp','Parch','Embarked','Title']

lr_data_X = data_train[features]

lr_test_X = data_test[features]

lr_data_Y = data_train["Survived"]

lr_data_X.head()
pd.set_option('display.max_columns',100)

def lr_encoding(data_X,test_X):

    len_train = len(data_X)

    features_to_label = ["Pclass","Sex","Age","Embarked","Title"]

    combined_X = pd.concat([data_X[features_to_label],test_X[features_to_label]])

    combined_X = pd.get_dummies(combined_X, columns=features_to_label)

    data_X = combined_X.iloc[0:len_train,:]

    test_X = combined_X.iloc[len_train:,:]

    return(data_X, test_X)



lr_data_X,lr_test_X = lr_encoding(data_X,test_X)

lr_data_X.head()
x_train, x_val, y_train, y_val = train_test_split(lr_data_X, lr_data_Y, test_size=0.15, shuffle=False)

print("Training set shape : {}".format(x_train.shape))

print("Validation set shape : {}".format(x_val.shape))
from sklearn.linear_model import LogisticRegression



lr_model = LogisticRegression(solver='liblinear')

params = {

    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],

    'penalty':["l1","l2"],

    'max_iter':[100,150,200,300,400,500]

}



accuracy_scorer = make_scorer(accuracy_score)



search_params = GridSearchCV(lr_model, params, scoring=accuracy_scorer)

search_params = search_params.fit(x_train, y_train)



lr_model = search_params.best_estimator_

print("The training set accuracy is {}".format(search_params.best_score_))

lr_model.fit(x_train, y_train)
predict = lr_model.predict(x_val)

print("The validation set accuracy is {}".format(accuracy_score(y_val, predict)))
p_id = pd.read_csv("../input/test.csv")['PassengerId']

predict = model.predict(test_X)



out = pd.DataFrame({'PassengerId' : p_id, 'Survived': predict})



out.head()