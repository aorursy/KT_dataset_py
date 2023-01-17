import pandas as pd
import numpy as np
import matplotlib 

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#How many rows (observations) and columns(features) in train dataset and in test dataset
train.shape, test.shape

train.head(5)
test.head(5)
#visualize with bar charts
#"Women first": the probability to survive rises for women, as they were given priority for seats in lifeboats

def chart1(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(6,3), color =("tab:pink","tab:cyan"))
chart1('Sex')
def chart2(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(6,3))
#According to graph, class1 had higher probability to survive than class3
chart2('Pclass')

# if a passenger had parents or children onboard, probability to survive was higher
chart2('Parch')
## Let's define how many missing value (NaN) in train.csv dataset
train.isnull().sum() 
## How many missing value (NaN) in test.csv dataset
test.isnull().sum()
## unite two files into one dataset in order to clean all data at once
data_cleaner = [test, train]
      
for dataset in data_cleaner:    
    #fill missing values of Embarked with mode (if there is no info about port city, fill in with the most frequent one )
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    #fill missing value of Fare with median of each class's fare.
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True) 
    #fill missing value of Age with median for each Sex
    dataset["Age"].fillna(dataset.groupby("Sex")["Age"].transform("median"), inplace=True)
   
    #bucket Age into 6 Age bins 
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 6)
    
    #Add FamilySize feature to test if existence of parents and children impacts
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    dataset["Single"] = dataset["FamilySize"].apply(lambda r: 1 if r == 1 else 0)
    
    decks = {"U": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
    dataset["Deck"] = dataset["Cabin"].fillna("U").apply(lambda c: decks.get(c[0], -1))

train.head(5)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

label = LabelEncoder()
for dataset in data_cleaner:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    
train.sample(5)
## Let's look at correlation
pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending = False))
df = train
df_train = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'AgeBin', 'PassengerId'], axis = 1)

df_train.sample(5)
df2=test
df_test = df2.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'AgeBin'], axis = 1)
df_test.sample(5)
#After we converted all categorical features into numerical and deleted all features that we are not going to use, 
#we save our final dataset as df_test and df_train 
#to train our model we need to show our model which features are independent and which feature we try to predict. 
#Thus, we use df_train2 to designate independent features.
inputs = df_train.drop(['Survived'], axis =1)
target = train['Survived']
inputs.shape, target.shape
#import modeling packages
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import mean_absolute_error, accuracy_score

## k-fold cross-validation,the original train.csv sample is randomly partitioned into k equal sized subsamples. 
#Of the k subsamples a single subsample is retained as the validation data for testing the model, 
#and the remaining k − 1 subsamples are used as training data

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
###kNN, n_neighbors should be odd in case if it is a draw

clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, inputs, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# kNN Score
round(np.mean(score)*100,0)
### Decision Tree

clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, inputs, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# Decision Tree Score
round(np.mean(score)*100, 0)
### Random Forest
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, inputs, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# Random Forest Score
round(np.mean(score)*100, 0)
###Naive Bayes
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, inputs, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
### Naive Bayes Score
round(np.mean(score)*100, 0)
### SVM
clf = SVC(gamma='auto')
scoring = 'accuracy'
score = cross_val_score(clf, inputs, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# SVM score
round(np.mean(score)*100, 0)

#Logistic Regression

clf = LogisticRegression(solver = "liblinear")
scoring = 'accuracy'
score = cross_val_score(clf, inputs, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100, 0)
##Bagging Classifier

clf = BaggingClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, inputs, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100, 0)
##After we tried 7 different models, we should choose the one with the highest score. 
#I chose Random Forest model

clf = RandomForestClassifier(n_estimators=13)
clf.fit(inputs, target)

test_data = df_test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)
prediction = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": prediction
    })

prediction.to_csv('prediction.csv', index=False)
prediction = pd.read_csv('prediction.csv')
prediction.head(20)
