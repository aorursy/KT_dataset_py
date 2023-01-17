# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import classification_report 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/titanic/train.csv')

#print(df.head())

useful_df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

useful_df.dropna(how = 'any', inplace = True)

#print(useful_df)
train_data, val_data = train_test_split(useful_df, test_size = 0.3, random_state = 1)

print(train_data.head())
def extract_train_features_and_labels(train_data):

    train_labels = list(train_data['Survived'])

    train_features = []

    pclass = list(train_data['Pclass'])

    sex = list(train_data['Sex'])

    age = list(train_data['Age'])

    sibsp = list(train_data['SibSp'])

    parch = list(train_data['Parch'])

    embarked = list(train_data['Embarked'])

    for i in range(len(train_data)):

        feature_dict = {'pclass': pclass[i], 'sex':sex[i], 'age': age[i], 'sibsp': sibsp[i], 'parch': parch[i], 'embarked': embarked[i]}

        train_features.append(feature_dict)

    return train_features, train_labels 



#extract_train_features_and_labels(train_data)
def extract_val_features_and_labels(val_data):

    val_labels = list(val_data['Survived'])

    val_features = []

    pclass = list(val_data['Pclass'])

    sex = list(val_data['Sex'])

    age = list(val_data['Age'])

    sibsp = list(val_data['SibSp'])

    parch = list(val_data['Parch'])

    embarked = list(val_data['Embarked'])

    for i in range(len(val_data)):

        feature_dict = {'pclass': pclass[i], 'sex':sex[i], 'age': age[i], 'sibsp': sibsp[i], 'parch': parch[i], 'embarked': embarked[i]}

        val_features.append(feature_dict)

    return val_features, val_labels 



#extract_val_features_and_labels(val_data)
def create_classifier(train_features, train_labels):

    vec = DictVectorizer()

    vec_train_features = vec.fit_transform(train_features)

    model = LogisticRegression()

    fitted_model = model.fit(vec_train_features, train_labels)

    return vec, fitted_model
def classify_data(vec, fitted_model, val_features):

    vec_val_features = vec.transform(val_features)

    prediction = fitted_model.predict(vec_val_features)

    return list(prediction)
def evaluation(val_labels, prediction):

    report = classification_report(val_labels, prediction)

    print(report)

    return report 
#test the method, f1-score about 72

train_features, train_labels = extract_train_features_and_labels(train_data)

val_features, val_labels = extract_val_features_and_labels(val_data)

vec, fitted_model = create_classifier(train_features, train_labels)

prediction = classify_data(vec, fitted_model, val_features)

evaluation(val_labels, prediction)
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

useful_test_df = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

test = useful_test_df.fillna(method = 'ffill')

print(len(test))
def extract_test_features(test_data):

    test_features = []

    pclass = list(test_data['Pclass'])

    sex = list(test_data['Sex'])

    age = list(test_data['Age'])

    sibsp = list(test_data['SibSp'])

    parch = list(test_data['Parch'])

    embarked = list(test_data['Embarked'])

    for i in range(len(test_data)):

        feature_dict = {'pclass': pclass[i], 'sex':sex[i], 'age': age[i], 'sibsp': sibsp[i], 'parch': parch[i], 'embarked': embarked[i]}

        test_features.append(feature_dict)

    return test_features 
test_features = extract_test_features(test)

train_features, train_labels = extract_train_features_and_labels(useful_df)

vec, fitted_model = create_classifier(train_features, train_labels)

result = classify_data(vec, fitted_model, test_features)
submission = pd.DataFrame({'PassengerId': list(test_df['PassengerId']), "Survived": list(result)})

submission.to_csv("submission.csv", index = None)



#pd.read_csv('submission.csv')

#pd.read_csv('/kaggle/input/titanic/gender_submission.csv')