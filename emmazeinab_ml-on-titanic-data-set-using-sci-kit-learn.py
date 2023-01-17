import pandas as pd

import seaborn as sns

%matplotlib inline
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')





data_train.sample(5)
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=data_train);
sns.barplot(x="Parch", y="Survived", hue="Sex", data=data_train);
sns.barplot(x="SibSp", y="Survived", hue="Sex", data=data_train);
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);
sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train);
#Creating new family_size column

data_train['Family_Size']=data_train['SibSp']+data_train['Parch']
data_train.sample(5)
sns.barplot(x="Family_Size", y="Survived", hue="Sex", data=data_train);
train_x = data_train[['Pclass','Sex','Family_Size']]

train_x.sample(5)
train_y = data_train['Survived']

train_y.sample(4)
from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

train_x["Sex"] = lb_make.fit_transform(train_x["Sex"])
from sklearn.cross_validation import train_test_split
##split the data into two 

xtrain, xtest, ytrain, ytest = train_test_split(train_x, train_y,

random_state=1)
from sklearn.naive_bayes import GaussianNB 

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

model = GaussianNB()

model1= SVC()

model2 = RandomForestClassifier()

model3 = GradientBoostingClassifier()
from sklearn.metrics import accuracy_score  ###checks the accuracy of the model
models = [model, model1, model2, model3]

for i in models:

    i.fit(xtrain, ytrain)

    ypred = i.predict(xtest) 

    print(i ,accuracy_score(ytest, ypred))
model.fit(train_x, train_y)
data_test['Family_Size']=data_test['SibSp']+data_test['Parch']

test_x = data_test[['Pclass','Sex','Family_Size']]

test_x["Sex"] = lb_make.fit_transform(test_x["Sex"])

test_x.sample(5)
y_model = model.predict(test_x)
Pa_id = data_test['PassengerId']
results = pd.DataFrame({ 'PassengerId' : Pa_id, 'Survived': y_model})

results.sample(5)
##Export as csv file

results.to_csv('titanic_pred_results.csv', index = False)