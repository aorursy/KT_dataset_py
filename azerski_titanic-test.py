# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()



print(train_data.columns)

train_data['Survived'].describe()

sns.distplot(train_data['Survived'])
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
print(type(train_data['Sex']))

sex=train_data.groupby('Sex').size()

print(sex)
#correlation matrix

corrmat = train_data.corr()

#plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)



men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")



y=train_data['Survived']

x=pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])





from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3)

#x_test = pd.get_dummies(test_data[features])



from sklearn.neighbors import KNeighborsClassifier

my_classifier=KNeighborsClassifier()



my_classifier.fit(x_train,y_train)

predictions=my_classifier.predict(x_test)



print(predictions)



from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predictions))



final_predict = my_classifier.predict(X_test)



print(final_predict)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': final_predict})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

import matplotlib.pyplot as plt

%matplotlib inline



k_range = list(range(2, 10))

scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    scores.append(metrics.accuracy_score(y_test, y_pred))



plt.plot(k_range, scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Testing Accuracy')
y=train_data['Survived']

x=pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])





from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3)

#x_test = pd.get_dummies(test_data[features])



from sklearn.ensemble import GradientBoostingClassifier



gcb=GradientBoostingClassifier()



gcb.fit(x_train,y_train)

predictions=gcb.predict(x_test)



#print(predictions)



from sklearn.metrics import accuracy_score

print('GCB Accuracy Score:',accuracy_score(y_test,predictions))



final_predict = gcb.predict(X_test)



#print(final_predict)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': final_predict})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

    
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

print("LogisticRegression Accuracy", metrics.accuracy_score(y_test, y_pred))



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': final_predict})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")