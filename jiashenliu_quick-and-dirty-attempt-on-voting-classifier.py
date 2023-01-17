import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import numpy as np

from sklearn.cross_validation import train_test_split



## Read Data

df=pd.read_csv('../input/prescriber-info.csv')

df.head()
print(df.shape)
opioids=pd.read_csv('../input/opioids.csv')

name=opioids['Drug Name']

import re

new_name=name.apply(lambda x:re.sub("\ |-",".",str(x)))

columns=df.columns

Abandoned_variables = set(columns).intersection(set(new_name))

Kept_variable=[]

for each in columns:

    if each in Abandoned_variables:

        pass

    else:

        Kept_variable.append(each)
df=df[Kept_variable]

print(df.shape)
train,test = train_test_split(df,test_size=0.2,random_state=42)

print(train.shape)

print(test.shape)
Categorical_columns=['Gender','State','Credentials','Specialty']

for col in Categorical_columns:

    train[col]=pd.factorize(train[col], sort=True)[0]

    test[col] =pd.factorize(test[col],sort=True)[0]
features=train.iloc[:,1:245]

features.head()
import sklearn

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import VotingClassifier
features=train.iloc[:,1:244]

target = train['Opioid.Prescriber']

Name=[]

Accuracy=[]

model1=LogisticRegression(random_state=22,C=0.000000001,solver='liblinear',max_iter=200)

model2=GaussianNB()

model3=RandomForestClassifier(n_estimators=200,random_state=22)

model4=GradientBoostingClassifier(n_estimators=200)

model5=KNeighborsClassifier()

model6=DecisionTreeClassifier()

model7=LinearDiscriminantAnalysis()

Ensembled_model=VotingClassifier(estimators=[('lr', model1), ('gn', model2), ('rf', model3),('gb',model4),('kn',model5),('dt',model6),('lda',model7)], voting='hard')

for model, label in zip([model1, model2, model3, model4,model5,model6,model7,Ensembled_model], ['Logistic Regression','Naive Bayes','Random Forest', 'Gradient Boosting','KNN','Decision Tree','LDA', 'Ensemble']):

    scores = cross_val_score(model, features, target, cv=5, scoring='accuracy')

    Accuracy.append(scores.mean())

    Name.append(model.__class__.__name__)

    print("Accuracy: %f of model %s" % (scores.mean(),label))
Name_2=[]

Accuracy_2=[]

Ensembled_model_2=VotingClassifier(estimators=[('rf', model3),('gb',model4)], voting='hard')

for model, label in zip([model3, model4,Ensembled_model_2], ['Random Forest', 'Gradient Boosting','Ensemble']):

    scores = cross_val_score(model, features, target, cv=5, scoring='accuracy')

    Accuracy_2.append(scores.mean())

    Name_2.append(model.__class__.__name__)

    print("Accuracy: %f of model %s" % (scores.mean(),label))
from sklearn.metrics import accuracy_score

classifers=[model3,model4,Ensembled_model_2]

out_sample_accuracy=[]

Name_2=[]

for each in classifers:

    fit=each.fit(features,target)

    pred=fit.predict(test.iloc[:,1:244])

    accuracy=accuracy_score(test['Opioid.Prescriber'],pred)

    Name_2.append(each.__class__.__name__)

    out_sample_accuracy.append(accuracy)
in_sample_accuracy=Accuracy_2
Index = [1,2,3]

plt.bar(Index,in_sample_accuracy)

plt.xticks(Index, Name_2,rotation=45)

plt.ylabel('Accuracy')

plt.xlabel('Model')

plt.title('In sample accuracy of models')

plt.show()
plt.bar(Index,out_sample_accuracy)

plt.xticks(Index, Name_2,rotation=45)

plt.ylabel('Accuracy')

plt.xlabel('Model')

plt.title('Out sample accuracies of models')

plt.show()