import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.model_selection import cross_val_score
train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")
def preprocess_titanic(df):

    newdf = df.drop(['Survived','Name', 'Ticket', 'Cabin'],axis=1)   

    newdf['Sex'] = pd.Series([1 if s == 'male' else 0 for s in df.Sex], name = 'Sex' )

    newdf['Age'] = df.Age.fillna( df.Age.mean() )

    newdf['Fare'] = df.Fare.fillna( df.Fare.mean() )

    newdf['Embarked'] = df.Embarked.fillna('S').map({'S':1,'C':2, 'Q':3})    

    return newdf
train_features = preprocess_titanic(train)

train_labels = train.Survived
models = [GaussianNB(), DecisionTreeClassifier(), LogisticRegression(), KNeighborsClassifier(), SVC(), LinearSVC(), RandomForestClassifier (), GradientBoostingClassifier()]

for model in models:

    score = cross_val_score(model, train_features,train_labels, cv=10).mean()

    print ("{}\t{}".format(score, str(model)[:22]))