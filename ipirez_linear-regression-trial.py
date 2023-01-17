import os
import numpy as np 
import pandas as pd 

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder

path = "/kaggle/input/titanic/"
df_train = pd.read_csv(os.path.join(path, "train.csv"))
df_test = pd.read_csv(os.path.join(path, "test.csv"))
df_train
df_train.columns
df_train.Age.describe()
df_train.Age[df_train.Survived == 0].plot.hist(bins=30, alpha=0.5, color="red")
df_train.Age[df_train.Survived == 1].plot.hist(bins=30, alpha=0.5, color="blue")

# Get the titles of people (not pretty)

# We take only the most uncommon
special_people = df_train.Name.apply(lambda x: [i for i in x.split() if i[-1] == "."][0]).value_counts().keys()[3:5]
xx = df_train.Name.apply(lambda x: [i for i in x.split() if i in special_people])
df_train["title"] = xx.apply(lambda x: x[0] if len(x) > 0 else "")
df_test["title"] = xx.apply(lambda x: x[0] if len(x) > 0 else "")
print(special_people)

def define_new_sex(passenger):
    """ Instead """
    age, sex = passenger
    if age < 16:
        return 'child_{}'.format(sex)
    elif age > 50:
        return 'old_{}'.format(sex)
    else:
        return sex

df_train['person'] = df_train[['Age','Sex']].apply(define_new_sex, axis=1)
df_test['person'] = df_train[['Age','Sex']].apply(define_new_sex, axis=1)

drop_these_column = ["title", "PassengerId", "Name", "Ticket", "Cabin", "Pclass", "Embarked", "Sex", "person"]
x = df_train.drop(columns=drop_these_column)
x = x.drop(columns=["Survived"])

hot_encoder = ["person", "Embarked", "Pclass", 'title']
for column in hot_encoder:
    x = x.join(pd.get_dummies(df_train[column]))

y = df_train["Survived"]
pipeline = Pipeline([('scale', StandardScaler()),
                     ('knn_Imputer', KNNImputer(n_neighbors=2, weights='distance')),
                     ('logistic', LogisticRegression(random_state=1))])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
pipeline.fit(x_train, y_train)
prediction = pipeline.predict(x_test)

print(metrics.accuracy_score(y_test, prediction))
print(metrics.precision_score(y_test, prediction))
x = df_test.drop(columns=drop_these_column)
for column in hot_encoder:
    x = x.join(pd.get_dummies(df_test[column]))

pred = pipeline.predict(x)

df_sub = pd.DataFrame(data=df_test["PassengerId"])
df_sub["Survived"] = pred
df_sub.to_csv("gender_submission.csv",index=False)
