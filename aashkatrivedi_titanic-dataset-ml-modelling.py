# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np #Linear Algebra
import pandas as pd #Data Processing

%matplotlib inline
from matplotlib import pyplot as plt #Visualisation
from matplotlib import style
import seaborn as sns


#ML Algorithms

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

#References:
#https://towardsdatascience.com/comparative-study-on-classic-machine-learning-algorithms-24f9ff6ab222
#https://medium.com/@dannymvarghese/comparative-study-on-classic-machine-learning-algorithms-part-2-5ab58b683ec0
#https://towardsdatascience.com/what-is-a-perceptron-210a50190c3b
train_df= pd.read_csv('/kaggle/input/titanic/train.csv')
test_df= pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.info()
train_df.head()
train_df.describe()
#Determining Missing Data

total_null= train_df.isnull().sum().sort_values(ascending=False)
percentage_null= (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending=False)
missing_data= pd.concat([total_null,percentage_null], axis=1, keys=['Total','Percentage']) #axiz=1: concatenate along columns

missing_data.head()
#Analyse Features
train_df.columns.values
#Age and Sex
sns.set_palette("Pastel1", 6)

fig, axes= plt.subplots(nrows=1, ncols=2, figsize=(10,4))

women= train_df[train_df['Sex']=='female']
men= train_df[train_df['Sex']=='male']

ax=sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label='Survived', ax=axes[0], kde=False)
ax=sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label='Not Survived', ax=axes[0], kde=False) 
#bins(not survived)>bins(survived) as number of females who survived are greater
ax.legend()
ax.set_title('Female Data')

ax=sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label='Survived', ax=axes[1], kde=False)
ax=sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label='Not Survived', ax=axes[1], kde=False) 
#bins(not survived)>bins(survived) as number of females who survived are greater
ax.legend()
ax.set_title('Male Data')
#Embarked, Class and Sex

grid= sns.FacetGrid(train_df, row='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived','Sex',palette="Pastel2", order=None, hue_order=None )
grid.add_legend()
#Pclass
sns.set_palette("winter", 3)

sns.barplot(x='Pclass', y='Survived',data=train_df)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6, palette='Pastel1')
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
#Sibsp and Parch: making combined features

data=[train_df,test_df]

for dataset in data:
    dataset['Relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['Relatives']>0, 'Not_Alone']=0
    dataset.loc[dataset['Relatives']==0, 'Not_Alone']=1
    dataset['Not_Alone'] = dataset['Not_Alone'].astype(int)

train_df['Not_Alone'].value_counts()
axes= sns.factorplot('Relatives','Survived', data=train_df, color='lightsteelblue')
#drop passenger ID column, not useful as a feature
train_df=train_df.drop(['PassengerId'], axis=1)
#Cabin
import re
import string

deck= dict(zip(string.ascii_uppercase, range(1,8)))
deck["U"]=8

data=[train_df,test_df]

for dataset in data:
    dataset['Cabin']= dataset['Cabin'].fillna('U0')
    dataset['Deck']= dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck']= dataset['Deck'].map(deck)
    dataset['Deck']= dataset['Deck'].fillna(0)
    dataset['Deck']= dataset['Deck'].astype(int)

train_df=train_df.drop(['Cabin'], axis=1)
test_df= test_df.drop(['Cabin'], axis=1)
#Age
data=[train_df, test_df]

for dataset in data:
    mean= train_df['Age'].mean()
    std= test_df['Age'].std()
    is_null= dataset['Age'].isnull().sum()
    
    rand_age= np.random.randint(mean-std, mean+std, size=is_null)
    age_slice= dataset['Age'].copy()
    age_slice[np.isnan(age_slice)]= rand_age
    dataset['Age']=age_slice
    dataset['Age']= dataset['Age'].astype(int)

train_df['Age'].isnull().sum()   
common_value= train_df['Embarked'].describe().top
data=[train_df, test_df]

for dataset in data:
    dataset['Embarked']= dataset['Embarked'].fillna(common_value)
train_df.info()
#convert fare to integer

data=[train_df, test_df]

for dataset in data:
    dataset['Fare']= dataset['Fare'].fillna(0)
    dataset['Fare']= dataset['Fare'].astype(int)
#Extracting titles from names

data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
#Map sex to numeric
genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
train_df['Ticket'].describe()
#Instead of creating 681 categories of Ticket, just drop it
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)
#Convert Embarked to a numeric feature

ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

train_df['Age'].value_counts()
train_df.head()
#To find how to make categories in Fare
pd.qcut(train_df['Fare'],6) 
train_df['Fare'].describe()
data = [train_df, test_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.00, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.00) & (dataset['Fare'] <= 8.00), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 8.00) & (dataset['Fare'] <= 14.00), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 14) & (dataset['Fare'] <= 26), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 26) & (dataset['Fare'] <= 52), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 52, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df['Fare'].value_counts()
#Age_class= product of age and class
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
#Fare per person
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['Relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
train_df.head(10)
X_train= train_df.drop('Survived', axis=1)
Y_train= train_df['Survived']
X_test= test_df.drop('PassengerId', axis=1).copy()
sgd= linear_model.SGDClassifier(max_iter=10, tol=None)
#max_iter: Number of Epochs, tol: stopping criterion 
sgd.fit(X_train, Y_train)
Y_pred= sgd.predict(X_test)


acc_sgd= round(sgd.score(X_train, Y_train)*100, 2)
sgd.score(X_train, Y_train)
random_forest= RandomForestClassifier(n_estimators=100) #number of trees in the forest
random_forest.fit(X_train, Y_train)
Y_pred= random_forest.predict(X_test)


acc_random_forest= round(random_forest.score(X_train, Y_train)*100, 2)
random_forest.score(X_train, Y_train)
logreg= LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred= logreg.predict(X_test)

acc_logreg= round(logreg.score(X_train, Y_train)*100,2)
logreg.score(X_train, Y_train)
knn= KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred= knn.predict(X_test)

acc_knn= round(knn.score(X_train, Y_train)*100,2)
knn.score(X_train, Y_train)
gaussian= GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred= gaussian.predict(X_test)

acc_gaussian= round(gaussian.score(X_train, Y_train)*100,2)
gaussian.score(X_train, Y_train)

perceptron= Perceptron(max_iter=15)
perceptron.fit(X_train, Y_train)
Y_pred= perceptron.predict(X_test)

acc_perceptron= round(perceptron.score(X_train, Y_train)*100,2)
perceptron.score(X_train, Y_train)
svc= LinearSVC(max_iter=100000)
svc.fit(X_train, Y_train)
Y_pred= svc.predict(X_test)

acc_svc= round(svc.score(X_train, Y_train)*100,2)
svc.score(X_train, Y_train)
decision_tree= DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred= decision_tree.predict(X_test)

acc_decision_tree= round(decision_tree.score(X_train, Y_train)*100,2)
decision_tree.score(X_train, Y_train)
result= pd.DataFrame({
    'Model': ['Stochaistic Gradient Descent', 'Random Forest', 'Logistic Regression','K Nearest Neighbors', 'Gaussian Naive Bayes', 'Perceptron', 'Linear SVM', 'Decision Tree'],
    'Accuracy': [acc_sgd, acc_random_forest, acc_logreg, acc_knn, acc_gaussian, acc_perceptron, acc_svc, acc_decision_tree]})
result= result.sort_values(by='Accuracy', ascending=False)
result=result.set_index('Accuracy')
result
from sklearn.model_selection import cross_val_score

random_forest= RandomForestClassifier(n_estimators=100)

scores= cross_val_score(random_forest, X_train, Y_train, cv=5, scoring='accuracy')
print("Scores: ", scores)
print("Mean: ", scores.mean())
print("Standard Deviation: ", scores.std())
random_forest= RandomForestClassifier(n_estimators=100) #number of trees in the forest
random_forest.fit(X_train, Y_train)
Y_pred= random_forest.predict(X_test)
importances= pd.DataFrame({'Feature': X_train.columns, 'Importance': np.round(random_forest.feature_importances_,3)})
importances= importances.sort_values('Importance', ascending=False).set_index('Feature')

importances
importances.plot.bar()
train_df= train_df.drop('Not_Alone', axis=1)
test_df= test_df.drop('Not_Alone', axis=1)

train_df= train_df.drop('Parch', axis=1)
test_df= test_df.drop('Parch', axis=1)
random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
random_forest.score(X_train, Y_train)
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
#Confusion Matrix

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)

#Precision and Recall
from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))
#f1 Score
from sklearn.metrics import f1_score
f1_score(Y_train, predictions)
#Precision Recall curve

from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5, color='peachpuff')
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5, color='steelblue')
    plt.xlabel("Threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()
#ROC Curve

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
