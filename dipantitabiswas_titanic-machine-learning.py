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
# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#load data
train_df=pd.read_csv("../input/titanic/train.csv")
test_df=pd.read_csv("../input/titanic/test.csv")
#printing 5 rows of data
train_df.head()
train_df.info() #to get a quick description of data
train_df.describe()
#Handling the missing value
train_df.isnull().sum()
test_df.isnull().sum()
sns.heatmap(train_df.isnull(),yticklabels=False,cmap='viridis')
train_df.isnull().mean()
test_df.isnull().mean()
sns.countplot(x="Survived",data=train_df)#Analysing wheather the data is imbalance or not
sns.countplot(x='Survived',hue='Sex',data=train_df)
sns.countplot(x='Survived',hue='Pclass',data=train_df)
sns.countplot(x='Survived',hue='Embarked',data=train_df)
train_df = train_df.drop("PassengerId", axis=1)
#filling the missing value with random sample imputation
for df in [train_df, test_df]:
    df["Age_random"]=df["Age"]
    random_sample=df["Age"].dropna().sample(df["Age"].isnull().sum())
    random_sample.index=df[df["Age"].isnull()].index
    df.loc[df["Age"].isnull(),"Age_random"]=random_sample
data=[train_df,test_df]
for dataset in data:
    dataset["Embarked"].fillna(dataset['Embarked'].value_counts().index[0],inplace=True)#replacing the missing values with most frequent value(mode)
data=[train_df,test_df]
for dataset in data:
    dataset["Cabin_val"]=np.where(dataset["Cabin"].isnull(),1,0)
    dataset["Cabin_cap"]=dataset["Cabin"].fillna(dataset['Cabin'].value_counts().index[0])
data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
sns.heatmap(train_df.isnull(),yticklabels=False,cmap='viridis')
#Creating a new feature "Fare_Per_Person" which can be useful

data = [train_df, test_df]
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['SibSp'] + dataset['Parch']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
train_df.corr()
sns.heatmap(train_df.corr())
#Feature scaling
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
colum_to_scale=['Age_random','Fare_Per_Person','Parch','SibSp']
data = [train_df, test_df]
for dataset in data:
    dataset[colum_to_scale]=std.fit_transform(dataset[colum_to_scale])

data = [train_df, test_df]
for dataset in data:
    dataset.drop(['Name','Ticket','Age','Cabin','Fare'],axis=1,inplace=True)
#mean encoding
train_df['Cabin_cap']=train_df["Cabin_cap"].astype(str).str[0]
test_df['Cabin_cap']=test_df["Cabin_cap"].astype(str).str[0]
mean=train_df.groupby(['Cabin_cap'])['Survived'].mean().to_dict()
train_df["Cabin_cap"]=train_df["Cabin_cap"].map(mean)
test_df["Cabin_cap"]=test_df["Cabin_cap"].map(mean)

#one hot encoding
gender = pd.get_dummies(train_df['Sex'],drop_first=True)
embarked= pd.get_dummies(train_df['Embarked'],drop_first=True)
train_df = pd.concat([train_df,gender,embarked],axis=1)
train_df.drop(['Sex','Embarked'],axis=1,inplace=True)

gender1 = pd.get_dummies(test_df['Sex'],drop_first=True)
embarked1= pd.get_dummies(test_df['Embarked'],drop_first=True)
test_df = pd.concat([test_df,gender1,embarked1],axis=1)
test_df.drop(['Sex','Embarked'],axis=1,inplace=True)
#separating the dependent and independent features
X = train_df.drop("Survived", axis=1)
Y = train_df["Survived"]
test_X  = test_df.drop("PassengerId", axis=1).copy()
#split the data into test and training
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25,random_state=42)
#Stochastic Gradient Descent (SGD):
SGD = SGDClassifier(max_iter=5, tol=None)
SGD.fit(X_train, Y_train)
Y_pred1 = SGD.predict(X_test)

SGD.score(X_train, Y_train)

accuracy_SGD = round(SGD.score(X_train, Y_train) * 100, 2)

test_accuracy_SGD = round(accuracy_score(Y_test,Y_pred1)*100,2)
#RANDOM fOREST CLASSIFIER
random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)

Y_pred2 = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
accuracy_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
test_accuracy_random = round(accuracy_score(Y_test,Y_pred2)*100,2)
#LOGISTIC REGRESSION
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred3 = logreg.predict(X_test)

accuracy_logreg = round(logreg.score(X_train, Y_train) * 100, 2)
test_accuracy_log = round(accuracy_score(Y_test,Y_pred3)*100,2)
#K NEAREST NEIGHBOUR
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X_train, Y_train)
Y_pred4 = knn.predict(X_test)
accuracy_knn = round(knn.score(X_train, Y_train) * 100, 2)
test_accuracy_knn = round(accuracy_score(Y_test,Y_pred4)*100,2)
#GAUSSIAN NEIVE BAYS
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred5 = gaussian.predict(X_test)
accuracy_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
test_accuracy_gaussian = round(accuracy_score(Y_test,Y_pred5)*100,2)
#LINEAR SUPPORT VECTOR MACHINE
SVC = LinearSVC()
SVC.fit(X_train, Y_train)

Y_pred6 = SVC.predict(X_test)

accuracy_svc = round(SVC.score(X_train, Y_train) * 100, 2)
test_accuracy_svc = round(accuracy_score(Y_test,Y_pred6)*100,2)
#DECISION TREE
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
Y_pred7 = dtree.predict(X_test)
accuracy_dtree = round(dtree.score(X_train, Y_train) * 100, 2)
test_accuracy_dtree = round(accuracy_score(Y_test,Y_pred7)*100,2)
train_result = pd.DataFrame({
    'ML Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [accuracy_SGD, accuracy_knn, accuracy_logreg, 
              accuracy_random_forest, accuracy_gaussian, accuracy_svc, test_accuracy_dtree]})
train_result = train_result.sort_values(by='Score', ascending=False)
train_result.head(9)
test_result = pd.DataFrame({
    'ML Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [test_accuracy_SGD, test_accuracy_knn, test_accuracy_log, 
              test_accuracy_random, test_accuracy_gaussian, test_accuracy_svc, accuracy_dtree]})
test_result = test_result.sort_values(by='Score', ascending=False)
test_result.head(9)
from sklearn.model_selection import cross_val_score
rd = RandomForestClassifier()
scores = cross_val_score(rd, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
Imp= pd.DataFrame({'Feature':X_train.columns,'Importance':np.round(random_forest.feature_importances_,3)})
Imp = Imp.sort_values('Importance',ascending=False)
Imp.head(8)
rd.get_params().keys()
from sklearn.model_selection import RandomizedSearchCV
param_test1 = {
    'n_estimators': [100,200,300,350,500,750, 1000],
    'criterion' : ['gini', 'entropy'], 
    'max_depth':[5,10,15,20,25,30],    
    'min_samples_leaf' : [1,2,5, 10], 
    'min_samples_split' : [1,2, 10,25]
}
gsearch1 = RandomizedSearchCV(estimator = rd,param_distributions = param_test1,scoring='roc_auc', cv=5)

gsearch1.fit(X_train, Y_train)
gsearch1.best_params_
gsearch1.best_estimator_
rd1=gsearch1.best_estimator_ #RandomForestClassifier(min_samples_leaf=2,min_samples_split=10, n_estimators=500)
rd1.fit(X_train, Y_train)
predictions = rd1.predict(X_test)
print(predictions)
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(rd1, X_train, Y_train, cv=10)
confusion_matrix(Y_train, y_train_pred)
from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(Y_train, y_train_pred))
print("Recall:",recall_score(Y_train, y_train_pred))
from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = rd1.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()
from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()
from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)
#predicting the test data
pred_y=rd1.predict(test_X).astype(int)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": pred_y})
submission.to_csv('submission.csv', index=False)