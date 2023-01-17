# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline
# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.info()
train_data[train_data.Survived == 1]
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(16, 4))
df_pclass = train_data.groupby(['Pclass', 'Survived'], as_index=False).count()
sns.barplot(x = 'Pclass', y='PassengerId', hue='Survived', data=df_pclass, ci="sd", ax=ax1)

# For sex
df_sex = train_data.groupby(['Sex', 'Survived'], as_index=False).count()
sns.barplot(x = 'Sex', y='PassengerId', hue='Survived', data=df_sex, ci="sd", ax=ax2)

#SibSp
df_sib = train_data.groupby(['SibSp', 'Survived'], as_index=False).count()
sns.barplot(x = 'SibSp', y='PassengerId', hue='Survived', data=df_sib, ci="sd", ax=ax3)

#Parch
df_parch = train_data.groupby(['Parch', 'Survived'], as_index=False).count()
sns.barplot(x = 'Parch', y='PassengerId', hue='Survived', data=df_parch, ci="sd", ax=ax4)
result = pd.DataFrame(columns=['PassengerId', 'Survived'])
result.PassengerId = test_data.PassengerId
train_data = train_data.drop('Cabin', axis=1)
test_data = test_data.drop('Cabin', axis=1)
train_data = train_data.drop(['PassengerId','Name','Ticket'], axis=1)
test_data = test_data.drop(['PassengerId','Name','Ticket'], axis=1)
train_data.Sex = train_data.Sex.map({'female': 0, 'male': 1})
test_data.Sex = test_data.Sex.map({'female': 0, 'male': 1})
avg_train_age = train_data.Age.mean()
train_data.Age = np.where(train_data.Age != train_data.Age, avg_train_age, train_data.Age)
avg_test_age = test_data.Age.mean()
test_data.Age = np.where(test_data.Age != test_data.Age, avg_test_age, test_data.Age)
avg_train_fare = train_data.Fare.mean()
train_data.Fare = np.where(train_data.Fare != train_data.Fare, avg_train_fare, train_data.Fare)
avg_test_fare = test_data.Fare.mean()
test_data.Fare = np.where(test_data.Fare != test_data.Fare, avg_test_fare, test_data.Fare)
train_data.Embarked = np.where(train_data.Embarked != train_data.Embarked, pd.value_counts(train_data.Embarked, sort=True, ascending=False).index[0], train_data.Embarked)
test_data.Embarked = np.where(test_data.Embarked != test_data.Embarked, pd.value_counts(test_data.Embarked, sort=True, ascending=False).index[0], test_data.Embarked)
train_data.Embarked = train_data.Embarked.map({'S': 1, 'C': 2, 'Q':3})
test_data.Embarked = test_data.Embarked.map({'S': 1, 'C': 2, 'Q':3})
trainY = train_data.Survived
trainX = train_data.drop('Survived', axis=1)
#C = [10**-3,10**-2,10**-1,1,10,10**2,10**3]
C=[]
for i in range(len(C)):
    for j in range(len(C)):
        svc = SVC(C=C[i], gamma=C[j])
        svc.fit(trainX, trainY)
        scores = cross_val_predict(svc, trainX, trainY, cv=10)
        print("C=%s Gamma=%s Score=%s" % (C[i], C[j],accuracy_score(trainY, scores)))
svc = SVC(C=10, gamma=0.01)
svc.fit(trainX, trainY)
predY = svc.predict(trainX)
print("SVC Score = " + str(accuracy_score(trainY, predY)))
result.Survived = svc.predict(test_data)
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import cross_val_score 
scaler = StandardScaler()  
scaler.fit(trainX)  
trainXNN = scaler.transform(trainX)  
test_dataNN = scaler.transform(test_data) 
max_hidden_layer = 5
activations = ['relu']
clfs = []
df_scores = pd.DataFrame(columns=["Layers", "Activation", "Score", "CVScore"])
for i in np.arange(1, max_hidden_layer):
    scores = []
    for a in activations:
        clf = MLPClassifier(activation=a, alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
           epsilon=1e-08, hidden_layer_sizes=(i,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)
        clf.fit(trainXNN, trainY)
        df_scores = df_scores.append({'Layers': i, 'Activation' : a, 'Score': clf.score(trainXNN, trainY), 'CVScore': np.mean(cross_val_score(clf, trainXNN, trainY, cv=2))}, ignore_index=True)
        clfs.append(clf)

bestIndex = df_scores["Score"].argmax()  
bestCVIndex = df_scores["CVScore"].argmax()
clf = clfs[bestCVIndex]
print("NN - The best score is : " + str(df_scores["Score"][bestIndex]))
print("NN - The best cross validation score is : " + str(df_scores["CVScore"][bestCVIndex]))
print("NN - The best hidden layer is : " + str(df_scores["Layers"][bestIndex]))
print("NN - The best activation is : " + str(df_scores["Activation"][bestIndex]))
print("NN - The best (CV) hidden layer is : " + str(df_scores["Layers"][bestCVIndex]))
print("NN - The best (CV) activation is : " + str(df_scores["Activation"][bestCVIndex]))
    
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(16, 8), sharey=True)
ax1.set_ylim(bottom=0.6)
sns.barplot(x = 'Layers', y='Score', hue='Activation', data=df_scores, ax=ax1)
sns.barplot(x = 'Layers', y='CVScore', hue='Activation', data=df_scores, ax=ax2)

plt.show()
result.Survived = clf.predict(test_dataNN)
submission = pd.DataFrame({
        "PassengerId": result["PassengerId"],
        "Survived": result.Survived
})
submission.to_csv('submission.csv', index=False)