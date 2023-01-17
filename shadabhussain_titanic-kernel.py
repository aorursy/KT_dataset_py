# linear algebra

import numpy as np

# data processing

import pandas as pd



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Algortithmic packages

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier



import os

import warnings

warnings.filterwarnings('ignore')
print(os.listdir("../input/titanic/"))
# loading Datasets

train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')



#printing first 5 rows of data

train_df.head()
# let's display the columns names

train_df.columns
# printing shape of Train Dataframe

train_df.shape
train_df.info()
test_df.info()
train_df.describe()
train_df.head()
# count of null values in each column

count  = train_df.isnull().sum().sort_values(ascending = False)



# percentage of Null Values in each column

percent = train_df.isnull().sum()/len(train_df)*100



# rounding and arranging the percentage

percent = round(percent,2).sort_values(ascending = False)



# concatenating count and percentage into one

missing_data = pd.concat([count,percent], axis = 1)

missing_data.columns = ['Count', 'Percent']

# printing top 5 rows

missing_data.head()
sns.barplot(x = 'Pclass', y = 'Survived' , data = train_df)

plt.show()
f, ax = plt.subplots(1,2, figsize=(20,8))



colors = ["#FA5858", "#64FE2E"]

labels ="Not Survived", "Survived"



plt.suptitle('Information on Survival', fontsize=20)



train_df["Survived"].value_counts().plot.pie(explode=[0,0.05], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 

                                             labels=labels, fontsize=15, startangle=45)







ax[0].set_xlabel('Survived vs Non Survived percentage', fontsize=14)

ax[0].set_ylabel('')



palette = ["#64FE2E", "#FA5858"]



sns.barplot(x = 'Pclass', y = 'Survived' , hue = 'Sex' , data = train_df, palette=palette)

ax[1].set(ylabel="(Percentage of Passengers)")

ax[1].set_xticklabels(train_df["Pclass"].unique(), rotation=0, rotation_mode="anchor")

plt.show()
train_df[(train_df.Pclass == 3) & (train_df.Survived == 1)].Sex.value_counts()/len(train_df[(train_df.Pclass == 3) & (train_df.Survived == 1)].Sex)*100
survived = 'survided'

not_survived = 'not_survived'

fig, axes = plt.subplots(nrows = 1 , ncols = 2 , figsize = (18,8))

women = train_df[train_df['Sex'] == 'female']

men = train_df[train_df['Sex'] == 'male']



ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=8, label = survived , ax=axes[0], kde = False)

ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins = 40, label=not_survived, ax = axes[0], kde =False)

ax.set_title('Female')



ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=8, label = survived , ax=axes[1], kde = False)

ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins = 40, label=not_survived, ax = axes[1], kde =False)

_ = ax.set_title('Male')

ax.legend()

plt.show()
facetgrid = sns.FacetGrid(train_df , row = 'Embarked', height = 4.5 , aspect =1.8)

facetgrid.map(sns.pointplot, 'Pclass','Survived', 'Sex', order=None, hue_order=None)

facetgrid.add_legend()

plt.show()
grid = sns.FacetGrid(train_df, row = 'Pclass', col='Survived', hue_order=None, height = 3, aspect=2)

grid.map(plt.hist, 'Age', alpha=0.7, bins = 20)

plt.show()

data = [train_df , test_df]

for dataset in data:

  dataset['Relatives']=dataset['Parch']+dataset['SibSp']

  dataset.loc[dataset['Relatives']>0,'Alone']=0

  dataset.loc[dataset['Relatives']==0,'Alone']=1

  dataset['Alone']=dataset['Alone'].astype(int)
train_df.head()
train_df.Alone.value_counts()
f, ax = plt.subplots(1,3, figsize=(20,8 ))

plt.suptitle('Information on Survival for Alone vs With Family', fontsize=20)



train_df["Alone"].value_counts().plot.pie(explode=[0,0.05], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 

                                             labels=['Alone', 'With Family'], fontsize=15, startangle=45)

sns.barplot(x = 'Alone', y = 'Survived',data = train_df , ax = ax[1])

sns.barplot(x = 'Alone', y = 'Survived', hue = 'Sex',data = train_df,  ax = ax[2])

plt.show()
plt.figure(figsize=(16,7))

sns.pointplot(x='Relatives', y = 'Survived',hue = 'Sex', data= train_df )

plt.show()
train_df = train_df.drop(['PassengerId'], axis = 1)

train_df.head()
import re



deck =  {'A':1  , 'B': 2 , 'C': 3, 'D':4 ,'E' : 5 , 'F':6 , 'G':7 , 'U':8}

data = [train_df , test_df]

for dataset in data:

  dataset['Cabin']=dataset['Cabin'].fillna('U0')

  dataset['Deck']=dataset['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())

  dataset['Deck']=dataset['Deck'].map(deck)

  dataset['Deck']=dataset['Deck'].fillna(0)

  dataset['Deck']=dataset['Deck'].astype(int)
train_df.Deck.value_counts()
# Same code as above, Regular Expression Simplified



import re



deck =  {'A':1  , 'B': 2 , 'C': 3, 'D':4 ,'E' : 5 , 'F':6 , 'G':7 , 'U':8}

data = [train_df , test_df]



for dataset in data:

  dataset['Cabin']=dataset['Cabin'].fillna('U0')

  dataset['Deck']=dataset['Cabin'].map(lambda x : x[0])

  dataset['Deck']=dataset['Deck'].map(deck)

  dataset['Deck']=dataset['Deck'].fillna(0)

  dataset['Deck']=dataset['Deck'].astype(int)



  

  

train_df.Deck.value_counts()
data = [train_df,test_df]

for dataset in data:

  dataset=dataset.drop(['Cabin'], axis = 1)
train_df=train_df.drop('Cabin', axis = 1)

test_df=test_df.drop('Cabin', axis = 1)
data = [train_df , test_df]

mean = train_df['Age'].mean()

std  = test_df['Age'].std()





for dataset in data:

  count_of_null = dataset['Age'].isnull().sum()

  

  rand_age = np.random.randint(mean-std,mean+std, size = count_of_null)

  

  age_slice = dataset['Age'].copy()

  age_slice[np.isnan(age_slice)]= rand_age

  

  dataset['Age']=age_slice

  dataset['Age']=dataset['Age'].astype(int)
train_df['Embarked'].describe()
train_df['Embarked'] = train_df['Embarked'].fillna('S')

test_df['Embarked'] = test_df['Embarked'].fillna('S')
train_df.info()
test_df.info()
test_df.Fare = test_df.Fare.fillna(mean)

test_df.Fare.isna().sum()
test_df.info()
train_df.Name.head()
data = [train_df,test_df]



for dataset in data:

  dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand = False)
train_df.Title.value_counts()
data = [train_df,test_df]



for dataset in data:

  dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don','Dr',\

                                              'Major','Rev','Sir','Jonkheer','Dona'], 'Rare')

  dataset['Title'] = dataset['Title'].replace('Mlle' , 'Miss')

  dataset['Title'] = dataset['Title'].replace('Ms' , 'Miss')

  dataset['Title'] = dataset['Title'].replace('Mme' , 'Mrs')

  
titles={'Mr':1,'Miss':2,'Mrs':3, 'Master':4, 'Rare':5}



for dataset in data:

  dataset['Title']=dataset['Title'].map(titles)
print(train_df.Title.isna().sum())

print(test_df.Title.isna().sum())
train_df = train_df.drop(['Name'], axis = 1)

test_df = test_df.drop(['Name'], axis = 1)
train_df.Sex.value_counts()
gender = {'male':0 , 'female':1}

data = [train_df,test_df]



for dataset in data:

  dataset['Sex']=dataset['Sex'].map(gender)
train_df.Ticket.head()
train_df=train_df.drop('Ticket', axis = 1)

test_df=test_df.drop('Ticket', axis = 1)
ports = {'S':0,'C':1, 'Q':2}

data = [train_df,test_df]



for dataset in data:

  dataset['Embarked']=dataset['Embarked'].map(ports)
data = [train_df,test_df]

for dataset in data:

  dataset['Age']=dataset['Age'].astype(int)

  dataset.loc[dataset['Age']<=11, 'Age']=0

  dataset.loc[(dataset['Age']>11) & (dataset['Age']<=18), 'Age']=1

  dataset.loc[(dataset['Age']>18) & (dataset['Age']<=22), 'Age']=2

  dataset.loc[(dataset['Age']>22) & (dataset['Age']<=27), 'Age']=3

  dataset.loc[(dataset['Age']>27) & (dataset['Age']<=33), 'Age']=4

  dataset.loc[(dataset['Age']>33) & (dataset['Age']<=40), 'Age']=5

  dataset.loc[(dataset['Age']>40) & (dataset['Age']<=66), 'Age']=6

  dataset.loc[(dataset['Age']>66), 'Age']=6
train_df.Age.value_counts()
train_df.head()
data = [train_df,test_df]



for dataset in data:

  dataset.loc[dataset['Fare']<=7.91, 'Fare']=0

  dataset.loc[(dataset['Fare']>7.91) & (dataset['Fare']<=14.454), 'Fare']=1

  dataset.loc[(dataset['Fare']>14.454) & (dataset['Fare']<=31), 'Fare']=2

  dataset.loc[(dataset['Fare']>31) & (dataset['Fare']<=99), 'Fare']=3

  dataset.loc[(dataset['Fare']>99) & (dataset['Fare']<=250), 'Fare']=4

  dataset.loc[(dataset['Fare']>250) , 'Fare']=5

  dataset['Fare']=dataset['Fare'].fillna(0)

  dataset['Fare']=dataset['Fare'].astype(int)



train_df.Fare.value_counts()
# test_df[test_df.Fare.isna()==True]
data = [train_df, test_df]





for dataset in data:

  dataset['age_class'] = dataset['Age']*dataset['Pclass']

  

  

train_df.head()
# Let's see the top 5 row of processed dataset

train_df.head()
X_train = train_df.drop('Survived', axis = 1)

y_train = train_df['Survived']



X_test = test_df.drop('PassengerId' ,  axis = 1)
# creating model object

sgd = linear_model.SGDClassifier(max_iter = 5, tol = None)



# Fitting model on Data

sgd.fit(X_train,y_train)



#using model to predict

y_pred = sgd.predict(X_test)



# Storing prediction accuracy

acc_sgd = round(sgd.score(X_train,y_train)*100,2)

print(acc_sgd)
logreg = LogisticRegression()

logreg.fit(X_train,y_train)





acc_log=round(logreg.score(X_train,y_train)*100,2)

print(acc_log)
# Creating a dictionary with Name of models as keys and Model Objects as values

dict_model = {'sgd':linear_model.SGDClassifier(max_iter = 5, tol = None), 

             'log_reg':LogisticRegression(),

             'decision_tree':DecisionTreeClassifier(),

             'random_forest':RandomForestClassifier(n_estimators = 100),

             'knn_classifier': KNeighborsClassifier(n_neighbors= 3),

             'gaussian':GaussianNB(),

             'perceptron':Perceptron(max_iter=5),

             'linear_svc':LinearSVC()

             }
# dictionary to store the name of model as key and respective accuracy as value

dict_accuracies={}



for name,classifier in dict_model.items():

  dict_model[name].fit(X_train,y_train)

  score = dict_model[name].score(X_train,y_train)

  dict_accuracies[name]=round(score*100,2)

  

result_df=pd.DataFrame.from_dict(dict_accuracies,orient = 'index',columns = ['Score'])

result_df= result_df.sort_values(by = 'Score', ascending = False)

result_df
rf_final = RandomForestClassifier(n_estimators = 100 , oob_score = True)

rf_final.fit(X_train , y_train)



rf_final_score = rf_final.score(X_train,y_train)*100

print(round(rf_final_score,2,), "%")
print("OOB Score: ", round(rf_final.oob_score_ *100, 2))
#Importing cross_val_score

from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators =  100, random_state = 42)



# Passing our RF model, training data, number of folds and evaluation metric to cross_val_score object

cv_score = cross_val_score(rf, X_train, y_train, cv = 5,  scoring = 'accuracy')
print("The results of cross validation are:")

print("Scores on each fold:",cv_score)

print("Mean:",cv_score.mean())

print("Standard Deviation:", cv_score.std())
# Parameter Grid to look for best Parameters

param_grid = { "n_estimators": [100, 200,500,1000],

              "criterion" : ["gini", "entropy"], 

              "min_samples_leaf" : [1, 5, 10], 

              "min_samples_split" : [2, 5, 10]

              }
from sklearn.model_selection import GridSearchCV, cross_val_score

rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state = 42, n_jobs=-1)



gs_clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1,

                           scoring = 'accuracy')



gs_clf.fit(X_train, y_train)



# Displaying the best parameters found by gridSearch

print(gs_clf.best_params_)
# Passing the dictionary of best params 

random_forest_final = RandomForestClassifier(**gs_clf.best_params_, oob_score = True , n_jobs = -1, random_state = 42)

random_forest_final.fit(X_train,y_train)



# Let's see the training scores now

print("Training Score: ", round(random_forest_final.score(X_train,y_train)*100,2),"%")
print("OOB Score for baseline Model: ", round(rf_final.oob_score_ *100, 2))

print("OOB Score for tuned Model: ", round(random_forest_final.oob_score_ *100, 2))
from sklearn.metrics import confusion_matrix



# Calulating predicted values for train data

y_pred = random_forest_final.predict(X_train)



# Calculating and displaying the confusion matrix

confusion_matrix(y_train, y_pred)
from sklearn.metrics import precision_score , recall_score

print("Precision Score:" , round(precision_score(y_train,y_pred)*100,2),"%" )

print("Recall Score:", round(recall_score(y_train,y_pred)*100,2),"%")

from sklearn.metrics import f1_score

f1_score(y_train,y_pred)
## TODO
## TODO
feature_importance = pd.DataFrame({'Feature':X_train.columns, 'Importance Score':random_forest_final.feature_importances_})

feature_importance = feature_importance.sort_values(by='Importance Score',  ascending = False).set_index('Feature')



#printing feature importance score of all the features

feature_importance
plt.figure(figsize = (16,8))

sns.barplot(x=feature_importance.index , y = 'Importance Score' ,data  = feature_importance)

plt.show()
from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(random_forest_final, threshold=0.05 )



# Train the selector

sfm.fit(X_train, y_train)



# Printing the names of the important features

for feature_list_index in sfm.get_support(indices=True):

    print(X_train.columns[feature_list_index])
# Transform the data to create a new dataset containing only the most important features

# Note: We have to apply the transform to both the training X and test X data.

X_important_train = sfm.transform(X_train)

X_important_test = sfm.transform(X_test)
# Create a new random forest classifier for the most important features

rf_important = RandomForestClassifier( random_state = 42)

rf = RandomForestClassifier(random_state = 42)



# Train the new classifier on the new dataset containing the most important features

rf_important.fit(X_important_train, y_train)

rf.fit(X_train, y_train)

rf.score(X_train,y_train), rf_important.score(X_important_train,y_train)
# To-Do