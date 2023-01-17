# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Load Train data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

#Load test data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data.head()
test_data.head()
train_data.dtypes
train_data.columns
train_data.shape
test_data.shape
#Check for the number of data values that are NaN in each column

train_data.isnull().sum()

import seaborn as sns

sns.heatmap(train_data.corr())
train_data.plot.hist(x='Age',y='Survived',bins=5)

plt.show()
train_data['Age'].fillna(train_data.mean()['Age'],inplace=True)

train_data.isnull().sum()
test_data['Age'].fillna(test_data.mean()['Age'],inplace=True)

test_data['Fare'].fillna(test_data.mean()['Fare'],inplace=True)

test_data.isnull().sum()
train_data['Pclass'].unique()
one=train_data.loc[train_data.Pclass == 1]['Survived']

rate_one = sum(one)/len(one)



print("% of people who survived with Class 1:", rate_one)



two=train_data.loc[train_data.Pclass == 2]['Survived']

rate_two = sum(two)/len(two)



print("% of people who survived with Class 1:", rate_two)



three=train_data.loc[train_data.Pclass == 3]['Survived']

rate_three = sum(three)/len(three)



print("% of people who survived with Class 1:", rate_three)

train_data['Embarked'].unique()
temp=train_data

temp.dropna(axis=0,inplace=True)



S=temp.loc[temp.Embarked == 'S']['Survived']

rate_S = sum(S)/len(S)



print("% of people who survived with Embarked : S -", rate_S)



C=temp.loc[temp.Embarked == 'C']['Survived']

rate_C = sum(C)/len(C)



print("% of people who survived with Embarked : C -", rate_C)



Q=temp.loc[temp.Embarked == 'Q']['Survived']

rate_Q = sum(Q)/len(Q)



print("% of people who survived with Embarked : Q -", rate_Q)







train_data.plot.hist(x='Sex',y='Survived',bins=2)

plt.xlabel("Sex")
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
#SibSp

train_data['SibSp'].unique()
train_data.plot.hist(x='SibSp',y='Survived',bins=8)

plt.xlabel("SibSp")

plt.ylabel("Survived")
#Parch

train_data['Parch'].unique()
train_data.plot.hist(x='Parch',y='Survived',bins=6)

plt.xlabel("Parch")

plt.ylabel("Survived")
y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch","Age"]

X = pd.get_dummies(train_data[features])

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=20,test_size=0.2)


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=11, max_depth=5, random_state=1)





rf.fit(x_train,y_train)

accuracy=rf.score(x_test,y_test)



print("Random Forest accuracy is :{}".format(accuracy))

from sklearn.naive_bayes import MultinomialNB

# Instantiate the classifier

mnb = MultinomialNB()



# Train classifier

mnb.fit( x_train,y_train)

accuracy=mnb.score(x_test,y_test)

print("Naive bayes accuracy is :{}".format(accuracy))


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=29)            #n_neighbors optimal value should be suqare root of n

knn.fit(x_train,y_train)

y_pred_knn=knn.predict(x_test)



#finding accuracy and confusion matrix

accuracy=accuracy_score(y_pred_knn,y_test)

print("KNN accuracy is :{}".format(accuracy))
#SVM

from sklearn import svm    			

C = 0.6  # SVM regularization parameter

svc = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)

#svc = svm.LinearSVC(C=C).fit(X, y)

rbf_svc = svm.SVC(kernel='rbf', gamma=0.5, C=C).fit(x_train, y_train)

# SVC with polynomial (degree 3) kernel

poly_svc = svm.SVC(kernel='poly', degree=2, C=C).fit(x_train, y_train)

accuracy1=svc.score(x_test,y_test)

print("SVM accuracy is :{}".format(accuracy1))

accuracy2=rbf_svc.score(x_test,y_test)

print("SVM rbf accuracy is :{}".format(accuracy2))

accuracy3=poly_svc.score(x_test,y_test)

print("SVM poly accuracy is :{}".format(accuracy3))
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(max_depth = 6,random_state = 99, max_features = None, min_samples_leaf = 5)

dtree.fit(x_train,y_train)

accuracy=dtree.score(x_test,y_test)

print("Decision tree accuracy is :{}".format(accuracy))
#Choose best value for n_estimators

from sklearn.model_selection import validation_curve

param_range = np.arange(1, 250, 2)

train_scoreNum, test_scoreNum = validation_curve(RandomForestClassifier(),X = x_train, y = y_train, 

                                param_name = 'n_estimators', scoring="accuracy",

                                param_range = param_range, cv = 3,n_jobs=-1)

plt.plot(train_scoreNum)
#Choose best value for max_depth

from sklearn.model_selection import validation_curve

param_range = np.arange(5, 30, 1)

train_scoreNum, test_scoreNum = validation_curve(RandomForestClassifier(),X = x_train, y = y_train, 

                                param_name = 'max_depth', scoring="accuracy",

                                param_range = param_range, cv = 3,n_jobs=-1)

plt.plot(train_scoreNum)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=20, max_depth=8, random_state=1)



rf.fit(x_train,y_train)

rf.score(x_test,y_test)
model = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=1)

model.fit(X, y)

X_test = pd.get_dummies(test_data[features])

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('titan_predictions_1.csv', index=False)

print("Your submission was successfully saved!")