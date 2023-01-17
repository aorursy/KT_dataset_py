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

import seaborn as sn

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style

import warnings

warnings.filterwarnings("ignore")

# Algorithms

from sklearn import linear_model

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve

from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, KFold, cross_validate

from sklearn.metrics import accuracy_score

from sklearn.linear_model import RidgeClassifier, LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding
titanic_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

sample_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

# separately store ID of test datasets, 

# this will be using at the end of the task to predict.

TestPassengerID = test['PassengerId']

titanic_df.shape
titanic_df.info()
titanic_df.describe()
titanic_df.head(4)
titanic_df.columns.values
#Training Data Set

AmitPandey = titanic_df.isnull().sum().sort_values(ascending=False)

percent_1 = titanic_df.isnull().sum()/titanic_df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([AmitPandey, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
#Test Data Set

APandey = test.isnull().sum().sort_values(ascending=False)

percent_1 = test.isnull().sum()/test.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([APandey, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
###Imputing missing values in train and test dataset

Cleaning = [titanic_df,test]

for dataset in Cleaning:    

    #complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
#Checking Again for Missing Values 

titanic_df.isnull().sum()
#Checking Again for Missing Values 

test.isnull().sum()
drop_column = ["PassengerId","Cabin"]

titanic_df.drop(drop_column, axis=1, inplace = True)

test.drop(drop_column, axis=1, inplace = True)
print("Dimension of train:",titanic_df.shape)

print("Dimension of test:",test.shape)
sn.barplot(x='Sex', y='Survived', data=titanic_df,palette="rocket")

plt.ylabel("Survival Rate")

plt.title("Visualizing Survival by bifurcation of Sex", fontsize=20)

plt.show()

titanic_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sn.barplot(x='Pclass', y='Survived', data=titanic_df,palette="deep")

plt.ylabel("Survival Rate")

plt.title("Visualizing Survival by bifurcation of Pclass", fontsize=16)

plt.show()

titanic_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sn.barplot(x='Embarked', y='Survived', data=titanic_df)

plt.ylabel("Survival Rate")

plt.title("Visualizing Survival by bifurcation of Embarked", fontsize=16)

plt.show()

titanic_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sn.barplot(x='Sex', y='Survived', hue='Pclass', data=titanic_df)

plt.ylabel("Survival Rate")

plt.title("Visualizing Survival by Pclass and Sex", fontsize=15)

plt.show()
titanic_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic_df.describe()
#Create a new Column "Title"

Total = [titanic_df, test]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Unknown": 5}

for dataset in Total:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
titanic_df.Title.value_counts()
sn.barplot(x='Title', y='Survived', data=titanic_df)

plt.ylabel("Survival Rate")

plt.title("Visualizing Survival by Title", fontsize=15)

plt.show()
#Create a new Column "Title"

#Grouped together Miss /Mrs & Master together as there is high survival chance

Cleaning = [titanic_df, test]

titles = {"Mr": 1, "Miss": 2, "Mrs": 2, "Master": 2, "Unknown": 3}

for dataset in Total:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Jonkheer', 'Dona'], 'Unknown')

    dataset['Title'] = dataset['Title'].replace('Sir', 'Mr')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Lady', 'Mrs')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)
#Merged Miss/ Mrs & Master based on my research & movie watching that they were main survivor [ You can say this as domain knowledge]

sn.barplot(x='Title', y='Survived', data=titanic_df)

plt.ylabel("Survival Rate")

plt.title("Visualizing Survival by Title", fontsize=15)

plt.show()
#Introducing a new column as "FamilySize"

titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] 
titanic_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).agg('mean')
#Introducing a new column "IsAlone"

titanic_df['IsAlone'] = 0

titanic_df.loc[titanic_df['FamilySize'] == 0, 'IsAlone'] = 1

titanic_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
cols = ['Survived', 'Parch', 'SibSp', 'Embarked','IsAlone', 'FamilySize']

nr_rows = 2

nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        

        i = r*nr_cols+c       

        ax = axs[r][c]

        sn.countplot(titanic_df[cols[i]], hue=titanic_df["Survived"], ax=ax)

        ax.set_title(cols[i], fontsize=18, fontweight='bold')

        ax.legend(title="survived", loc='upper center') 

        

plt.tight_layout()
#Fare Column

titanic_df[["Fare", "Survived"]].groupby(['Survived'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Premium Fare means better chance of Survival

titanic_df.groupby(['Sex','Survived'])[['Fare']].agg(['min','mean','max'])
#Based on my own understanding of data, created 4 category in "Fare" column

titanic_df.loc[ titanic_df['Fare'] <= 7.89, 'Fare'] = 0

titanic_df.loc[(titanic_df['Fare'] > 7.89) & (titanic_df['Fare'] <= 14.45), 'Fare'] = 1

titanic_df.loc[(titanic_df['Fare'] > 14.45) & (titanic_df['Fare'] <= 31.47), 'Fare'] = 2

titanic_df.loc[ titanic_df['Fare'] > 31.47, 'Fare'] = 3

titanic_df['Fare'] = titanic_df['Fare'].astype(int)
sn.barplot(x='Fare', y='Survived', data=titanic_df)

plt.ylabel("Survival Rate")

plt.title("Visualizing Survival by Fare Category", fontsize=15)

plt.show()
Bhuppi = sn.FacetGrid(titanic_df, col='Survived')

Bhuppi.map(plt.hist, 'Fare', bins=20)

plt.show()
sn.barplot(x='Sex', y='Survived', hue='Fare', data=titanic_df)

plt.ylabel("Survival Rate")

plt.title("Survival with bifurcation of Fare and Sex")

plt.show()
#Let's understand the relation between "Age" & "Survived

Vinu = sn.FacetGrid(titanic_df, col='Survived')

Vinu .map(plt.hist, 'Age', bins=20)

plt.show()
g = sn.FacetGrid(titanic_df, col="Survived")

g.map_dataframe(sn.scatterplot, x="Age", y="Fare")

g.set_axis_labels("Age", "Fare")

g.add_legend()
#Seems less age and higher class is a better combination to survive

#bins=np.arange(0, 80, 10)

PrabhU = sn.FacetGrid(titanic_df, row='Sex', col='Pclass', hue='Survived', margin_titles=True, height=3, aspect=1.1)

PrabhU.map(sn.distplot, 'Age', kde=False, bins=4, hist_kws=dict(alpha=0.6))

PrabhU.add_legend()  

plt.show()
#Categorize Age Feature

titanic_df.loc[ titanic_df['Age'] <= 23, 'Age'] = 1

titanic_df.loc[(titanic_df['Age'] > 23) & (titanic_df['Age'] <= 28), 'Age'] = 2

titanic_df.loc[(titanic_df['Age'] > 28) & (titanic_df['Age'] <= 36), 'Age'] = 3

titanic_df.loc[ titanic_df['Age'] > 36, 'Age'] = 4

titanic_df['Age'] = titanic_df['Age'].astype(int)
sn.barplot(x='Age', y='Survived', data=titanic_df)

plt.ylabel("Survival Rate")

plt.title("Survival with bifurcation of Age", fontsize=16)

plt.show()

titanic_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic_df.Age.value_counts()
sn.barplot(x='Pclass', y='Survived', hue='Age', data=titanic_df)

plt.ylabel("Survival Rate")

plt.title("Survival as bifurcation of Age and Sex")

plt.show()
sn.barplot(x='FamilySize', y='Survived', data=titanic_df)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Age", fontsize=16)

plt.show()

titanic_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanic_df.SibSp.value_counts()
test.SibSp.value_counts()
# Creating a categorical variable for Family Sizes

titanic_df['FamilyCategory'] = ''

titanic_df['FamilyCategory'].loc[(titanic_df['SibSp'] == 0)] = 'Without_Siblings_Spouses'

titanic_df['FamilyCategory'].loc[(titanic_df['SibSp'] > 0) & (titanic_df['SibSp'] <= 2 )] = 'Small'

titanic_df['FamilyCategory'].loc[(titanic_df['SibSp'] > 2) & (titanic_df['SibSp'] <= 4 )] = 'Medium'

titanic_df['FamilyCategory'].loc[(titanic_df['SibSp'] > 4)] = 'Large'
sn.barplot(x='FamilyCategory', y='Survived', data=titanic_df)

plt.ylabel("Survival Rate")

plt.title("Survival as function of FamilyCategory", fontsize=16)

plt.show()

titanic_df[['FamilyCategory', 'Survived']].groupby(['FamilyCategory'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#creating a new column "Age*Class"

titanic_df['Age*Class'] = titanic_df.Age * titanic_df.Pclass
titanic_df[["Age*Class", "Survived"]].groupby(['Age*Class'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab([titanic_df.Survived], [titanic_df.Sex,titanic_df['Age*Class']], margins=True).style.background_gradient(cmap='viridis')
pd.crosstab([titanic_df.Survived], [titanic_df.Sex,titanic_df['IsAlone']], margins=True).style.background_gradient(cmap='viridis')
titanic_df.head(4)
test.loc[ test['Fare'] <= 7.89, 'Fare'] = 0

test.loc[(test['Fare'] > 7.89) & (test['Fare'] <= 14.45), 'Fare'] = 1

test.loc[(test['Fare'] > 14.45) & (test['Fare'] <= 31.47), 'Fare'] = 2

test.loc[ test['Fare'] > 31.47, 'Fare'] = 3

test['Fare'] = test['Fare'].astype(int)
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

test['IsAlone'] = 0

test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1
test.loc[ test['Age'] <= 23, 'Age'] = 1

test.loc[(test['Age'] > 23) & (test['Age'] <= 28), 'Age'] = 2

test.loc[(test['Age'] > 28) & (test['Age'] <= 36), 'Age'] = 3

test.loc[ test['Age'] > 36, 'Age'] = 4

test['Age'] = titanic_df['Age'].astype(int)
# Creating a categorical variable for Family Sizes

test['FamilyCategory'] = ''

test['FamilyCategory'].loc[(test['SibSp'] == 0)] = 'Without_Siblings_Spouses'

test['FamilyCategory'].loc[(test['SibSp'] > 0) & (test['SibSp'] <= 2 )] = 'Small'

test['FamilyCategory'].loc[(test['SibSp'] > 2) & (test['SibSp'] <= 4 )] = 'Medium'

test['FamilyCategory'].loc[(test['SibSp'] > 4)] = 'Large'
test['Age*Class'] = test.Age * test.Pclass
test.head(2)
titanic_df.head(2)
#code categorical data

from sklearn import preprocessing

Total = [titanic_df,test]

label = preprocessing.LabelEncoder()

for dataset in Total:    

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['Age'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['Fare'])

    dataset['FamilyCategory_Code'] = label.fit_transform(dataset['FamilyCategory'])

    dataset['IsAlone_Code'] = label.fit_transform(dataset['IsAlone'])

    dataset['Age*Class_Code'] = label.fit_transform(dataset['Age*Class'])

    dataset['Pclass_Code'] = label.fit_transform(dataset['Pclass'])
Train_Ready = titanic_df.copy()

Test_Ready = test.copy()
#Dropping redundant columns and also including columns to drop based on my model accuracy findings like ["IsALone_Code","FamilyCategory_Code"]

# Trying by excluding ["Parch feature from the data set"]

drop_column = ["Sex","Name","Ticket","Embarked","Title","Age","Fare","FamilyCategory","IsAlone","Age*Class","Pclass","IsAlone_Code","FamilyCategory_Code"]

titanic_df.drop(drop_column, axis=1, inplace = True)

test.drop(drop_column, axis=1, inplace = True)
titanic_df.shape
test.shape
Train_Ready = titanic_df.copy()

Test_Ready = test.copy()
Train_Ready.info()
Test_Ready.info()
# #Dropping redundant columns

# #Will drop feature with least importance & improve my model accuracy- 1st will drop IsAlone_Code

# #Will drop feature with least importance & improve my model accuracy- 2nd will drop FamilyCategory_Code feature

# drop_column = ["FamilyCategory_Code"]

# titanic_df.drop(drop_column, axis=1, inplace = True)

# test.drop(drop_column, axis=1, inplace = True)
# titanic_df.head(4)
#Pearson Correlation of Features

colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sn.heatmap(titanic_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
# Train_Ready = titanic_df.copy()

# Test_Ready = test.copy()
F_X_train = Train_Ready.drop("Survived",axis=1)

F_y_train = titanic_df["Survived"]

F_X_test  = Test_Ready
print("Dimension of X_train:",F_X_train.shape)

print("Dimension of y_train:",F_y_train.shape)

print("Dimension of X_test:",F_X_test.shape)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(F_X_train)

X_test = sc.fit_transform(F_X_test)

y_train = F_y_train
#Stochastic Gradient Descent (SGD)

sgd = linear_model.SGDClassifier(penalty='l1',max_iter=1000, tol=1e-3)

sgd.fit(X_train, y_train)

Y_pred = sgd.predict(X_test)

sgd.score(X_train, y_train)

acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)

acc_sgd 
#Random Forest

random_forest = RandomForestClassifier(n_estimators=1000,criterion='gini', max_features='auto')

random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
#Logistic Regression

logreg = LogisticRegression(penalty='l1',solver='liblinear',max_iter=500)

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log
#K Nearest Neighbor

knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn
#K Nearest Neighbor

knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn
#Gaussian Naive Bayes

gaussian = GaussianNB() 

gaussian.fit(X_train, y_train)  

Y_pred = gaussian.predict(X_test)  

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

acc_gaussian
#Perceptron

perceptron = Perceptron(max_iter=20)

perceptron.fit(X_train, y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)

acc_perceptron
#Linear Support Vector Machine

linear_svc = LinearSVC(penalty='l2',max_iter=1000)

linear_svc.fit(X_train, y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

acc_linear_svc
#Decision Tree

decision_tree = DecisionTreeClassifier(criterion='entropy',max_features="auto") 

decision_tree.fit(X_train, y_train)  

Y_pred = decision_tree.predict(X_test)  

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree
#GradientBoostingClassifier

GB_Clf = GradientBoostingClassifier() 

GB_Clf.fit(X_train, y_train)  

Y_pred = GB_Clf.predict(X_test)  

acc_GB_Clf = round(GB_Clf.score(X_train, y_train) * 100, 2)

acc_GB_Clf
#SGDClassifier

SGD_Clf = SGDClassifier() 

SGD_Clf.fit(X_train, y_train)  

Y_pred = SGD_Clf.predict(X_test)  

acc_SGD_Clf = round(SGD_Clf.score(X_train, y_train) * 100, 2)

acc_SGD_Clf
results = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 

              'Decision Tree','GradientBoostingClassifier','SGDClassifier'],

    'Score': [acc_linear_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree,acc_GB_Clf,acc_SGD_Clf]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(10)
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(decision_tree, X_train, y_train, cv=3)

cm=confusion_matrix(y_train, predictions)

sn.heatmap(cm, annot=True, fmt='.2f',

           xticklabels= ["Not Survived", "Survived"],

           yticklabels = ["Not Survived", "Survived"] )

plt.ylabel("True label")

plt.xlabel("Predicted label")

plt.show()
# The code below will perform K-Fold Cross Validation on our random forest model, using 10 folds (K = 10). Therefore it outputs an array with 10 different scores.

from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf, X_train,y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sn

# Create a dataframe to store the features and their corresponding importances

feature_rank = pd.DataFrame( { "feature": F_X_train.columns,"importance":np.round(random_forest.feature_importances_,3)})

## Sorting the features based on their importances with most important feature at top.

feature_rank = feature_rank.sort_values("importance", ascending =False)

plt.figure(figsize=(8, 6))

# plot the values

sn.barplot( y = "feature", x = "importance", data = feature_rank )
feature_rank["cumsum"] = feature_rank.importance.cumsum() * 100 

feature_rank.head(30)
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)

random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, y_train)



acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")

#Out-of-bag (OOB) error, also called out-of-bag estimate, is a method of measuring the prediction error of

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(random_forest, X_train, y_train, cv=3)

cm=confusion_matrix(y_train, predictions)

sn.heatmap(cm, annot=True, fmt='.2f',

           xticklabels= ["Not Survived", "Survived"],

           yticklabels = ["Not Survived", "Survived"] )

plt.ylabel("True label")

plt.xlabel("Predicted label")

plt.show()
# from sklearn.model_selection import GridSearchCV

# ## Configuring parameters and values for searched

# tuned_parameters = [{ "criterion" : ["gini", "entropy"],'max_depth': [4,5,6,8,10],

#                      'n_estimators': [100, 200, 500],'max_features': ['auto', 'sqrt','log2', None]}]



# ## Initializing the RF classifier

# radm_clf = RandomForestClassifier()

# ## Configuring search with the tunable parameters

# clf = GridSearchCV(radm_clf,tuned_parameters,cv=5,scoring='accuracy')

# ## Fitting the training set

# clf.fit(X_train, y_train)
# #Mean cross-validated score of the best_estimator

# clf.best_score_
# #Parameter setting that gave the best results on the hold out data.

# clf.best_params_
## Initializing the Random Forest Model with the optimal values arrived using hyper parameter tuning

radm_clf = RandomForestClassifier(criterion= 'entropy', max_depth= 4, max_features= None,n_estimators=500)

## Fitting the model with the training set

radm_clf.fit(X_train,y_train )
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(radm_clf, X_train, y_train, cv=3)

cm=confusion_matrix(y_train, predictions)

sn.heatmap(cm, annot=True, fmt='.2f',

           xticklabels= ["Not Survived", "Survived"],

           yticklabels = ["Not Survived", "Survived"] )

plt.ylabel("True label")

plt.xlabel("Predicted label")

plt.show()
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sn

# Create a dataframe to store the features and their corresponding importances

feature_rank = pd.DataFrame( { "feature": F_X_train.columns,"importance":np.round(radm_clf.feature_importances_,3)})

## Sorting the features based on their importances with most important feature at top.

feature_rank = feature_rank.sort_values("importance", ascending =False)

plt.figure(figsize=(8, 6))

# plot the values

sn.barplot( y = "feature", x = "importance", data = feature_rank )
# #Let's Try More

# from sklearn.model_selection import GridSearchCV

# ## Configuring parameters and values for searched

# tuned_parameters = [{'criterion': ['entropy', 'gini'],

#                      'max_depth': [2,5,10,15,20,25],

#                      'max_features': ['auto', 'sqrt','log2', None],

#                      'min_samples_leaf': [4, 6, 8, 12],

#                      'min_samples_split': [5, 7, 10, 14],

#                      'n_estimators': [100, 200, 500,1000, 1500]}]



# ## Initializing the RF classifier

# radm_clf = RandomForestClassifier()

# ## Configuring search with the tunable parameters

# clf = GridSearchCV(radm_clf,tuned_parameters,cv=5,scoring='accuracy')

# ## Fitting the training set

# clf.fit(X_train, y_train)
# #Mean cross-validated score of the best_estimator

# clf.best_score_
# #Parameter setting that gave the best results on the hold out data.

# clf.best_params_
# ## Initializing the Random Forest Model with the optimal values arrived using hyper parameter tuning

# radm_clf = RandomForestClassifier(criterion= 'entropy', max_depth= 5, max_features= "sqrt", n_estimators= 500)

# ## Fitting the model with the training set

# radm_clf.fit(X_train,y_train )
# from sklearn.model_selection import cross_val_predict

# from sklearn.metrics import confusion_matrix

# predictions = cross_val_predict(radm_clf, X_train, y_train, cv=3)

# cm=confusion_matrix(y_train, predictions)

# sn.heatmap(cm, annot=True, fmt='.2f',

#            xticklabels= ["Not Survived", "Survived"],

#            yticklabels = ["Not Survived", "Survived"] )

# plt.ylabel("True label")

# plt.xlabel("Predicted label")

# plt.show()
# import seaborn as sns



# table = pd.pivot_table(pd.DataFrame(model.cv_results_),

#     values='mean_test_score', index='param_n_estimators', 

#                        columns='param_criterion')

     

# sns.heatmap(table)
# from sklearn.model_selection import cross_val_predict

# from sklearn.metrics import confusion_matrix

# predictions = cross_val_predict(model, X_train, y_train, cv=3)

# cm=confusion_matrix(y_train, predictions)

# sn.heatmap(cm, annot=True, fmt='.2f',

#            xticklabels= ["Not Survived", "Survived"],

#            yticklabels = ["Not Survived", "Survived"] )

# plt.ylabel("True label")

# plt.xlabel("Predicted label")

# plt.show()
# import numpy as np

# import matplotlib.pyplot as plt

# %matplotlib inline

# import seaborn as sn

# # Create a dataframe to store the features and their corresponding importances

# feature_rank = pd.DataFrame( { "feature": F_X_train.columns,"importance":np.round(random_forest.feature_importances_,3)})

# ## Sorting the features based on their importances with most important feature at top.

# feature_rank = feature_rank.sort_values("importance", ascending =False)

# plt.figure(figsize=(8, 6))

# # plot the values

# sn.barplot( y = "feature", x = "importance", data = feature_rank )
# feature_rank["cumsum"] = feature_rank.importance.cumsum() * 100 

# feature_rank.head(30)
# from sklearn.ensemble import GradientBoostingClassifier
# gboost_clf = GradientBoostingClassifier( n_estimators=500,max_depth=10)

# ## Fitting gradient boosting model to training set

# gboost_clf.fit(X_train,y_train)
# from sklearn.model_selection import cross_val_score

# gboost_clf = GradientBoostingClassifier( n_estimators=500,max_depth=10)

# cv_scores = cross_val_score( gboost_clf, X_train, y_train,cv = 10, scoring = 'accuracy' )
# print(cv_scores)

# print("Mean Accuracy: ", np.mean(cv_scores), "with standard deviation of:", np.std(cv_scores))
# gboost_clf.fit(X_train,y_train )

# predictions  = gboost_clf.predict(X_test )
# from sklearn.model_selection import cross_val_predict

# from sklearn.metrics import confusion_matrix

# predictions = cross_val_predict(gboost_clf, X_train, y_train, cv=10)

# cm=confusion_matrix(y_train, predictions)

# sn.heatmap(cm, annot=True, fmt='.2f',

#            xticklabels= ["Not Survived", "Survived"],

#            yticklabels = ["Not Survived", "Survived"] )

# plt.ylabel("True label")

# plt.xlabel("Predicted label")

# plt.show()
# import numpy as np

# import matplotlib.pyplot as plt

# %matplotlib inline

# import seaborn as sn

# # Create a dataframe to store the features and their corresponding importances

# feature_rank = pd.DataFrame( { "Feature": F_X_train.columns,"Importance":np.round(gboost_clf.feature_importances_,3)})

# ## Sorting the features based on their importances with most important feature at top.

# feature_rank = feature_rank.sort_values("Importance", ascending =False)

# plt.figure(figsize=(8, 6))

# # plot the values

# sn.barplot( y = "Feature", x = "Importance", data = feature_rank )
# predictions = cross_val_predict(radm_clf, X_train, y_train, cv=20)
from sklearn.metrics import precision_score, recall_score



print("Precision:", precision_score(y_train, predictions))

print("Recall:",recall_score(y_train, predictions))
#Combining precision and recall into one score is called the F-score.

from sklearn.metrics import f1_score

f1_score(y_train, predictions)
#Precision Recall Curve

#For each person the Random Forest algorithm has to classify, it computes a probability based on a function and 

#it classifies the person as survived (when the score is bigger the than threshold) or as not survived (when the score is smaller than the threshold). 

#That’s why the threshold plays an important part.

from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions

y_scores = radm_clf.predict_proba(X_train)

y_scores = y_scores[:,1]



precision, recall, threshold = precision_recall_curve(y_train, y_scores)

def plot_precision_and_recall(precision, recall, threshold):

    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)

    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)

    plt.xlabel("threshold", fontsize=19)

    plt.legend(loc="upper right", fontsize=19)

    plt.ylim([0, 1])



plt.figure(figsize=(14, 7))

plot_precision_and_recall(precision, recall, threshold)

plt.show()
def plot_precision_vs_recall(precision, recall):

    plt.plot(recall, precision, "g--", linewidth=2.5)

    plt.ylabel("recall", fontsize=19)

    plt.xlabel("precision", fontsize=19)

    plt.axis([0, 1.5, 0, 1.5])



plt.figure(figsize=(14, 7))

plot_precision_vs_recall(precision, recall)

plt.show()
from sklearn.metrics import roc_curve

# compute true positive rate and false positive rate

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)

# plotting them against each other

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'r', linewidth=4)

    plt.axis([0, 1, 0, 1])

    plt.xlabel('ROC Curve - False Positive Rate \n Precision Recall Curve - Recall', fontsize=15)

    plt.ylabel('ROC Curve - True Positive Rate \n Precision Recall Curve - Precision', fontsize=15)



plt.figure(figsize=(14, 7))

plot_roc_curve(false_positive_rate, true_positive_rate)

plt.show()
#The ROC AUC Score is the corresponding score to the ROC AUC Curve. It is simply computed by measuring the area under the curve, which is called AUC.

from sklearn.metrics import roc_auc_score

r_a_score = roc_auc_score(y_train, y_scores)

print("ROC-AUC-Score:", r_a_score)
Y_prediction = radm_clf.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": TestPassengerID,

        "Survived": Y_prediction

    })

submission.to_csv('submission.csv', index=False)