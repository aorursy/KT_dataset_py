# data processing

import pandas as pd



## linear algebra

import numpy as np



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Algorithms

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

 

from sklearn.metrics import accuracy_score  #for accuracy_score

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.metrics import confusion_matrix #for confusion matrix
titanic = pd.read_csv('../input/titanic/train.csv')

# Print the first 5 rows of the dataframe.

titanic.head(5)

titanic_test = pd.read_csv('../input/titanic/test.csv')

# Print the last 5 rows of the dataframe.

titanic_test.tail(5)
#shape command will give number of rows/samples/examples and number of columns/features/predictors in dataset

#(rows,columns)

titanic.shape
# Describe gives statistical information about numerical columns in the dataset

titanic.describe()

#you can check from count if there are missing values in columns, here we can see there are some missing values in column "Age"
# info method provides information about dataset like.

# total values in each column, null/not null, datatype, memory occupied etc.

titanic.info()
# Let's write a function to print the total percentage of the missing values.

# (this can be a good exercise for beginners to try to write simple functions like this.)



#This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage

def missing_data(df):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum().sort_values(ascending = False) * 100 /len(df),2)

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
# lets check missing values by columns

missing_data(titanic)
# Check missing values in test data set

missing_data(titanic_test)
drop_column = ['Cabin']

titanic.drop(drop_column, axis= 1, inplace = True)

titanic_test.drop(drop_column,axis = 1,inplace = True)
#COMPLETING: complete or delete missing values in train and test/validation dataset

dataset = [titanic, titanic_test]



# def missing_data(x):

for data in dataset:

    #complete missing age with median

    data['Age'].fillna(data['Age'].median(), inplace = True)



    #complete missing Embarked with Mode

    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)



    #complete missing Fare with median

    data['Fare'].fillna(data['Fare'].median(), inplace = True)

      

missing_data(titanic)
def draw(graph):

    for p in graph.patches:

        height = p.get_height()

        graph.text(p.get_x()+p.get_width()/2., height + 5,height ,ha= "center")
sns.set(style="darkgrid")

plt.figure(figsize = (8, 5))

graph= sns.countplot(x='Survived', hue="Survived", data=titanic)

draw(graph)
plt.figure(figsize = (8, 5))

graph  = sns.countplot(x ="Sex", hue ="Survived", data = titanic)

draw(graph)
 

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4)) 

x = sns.countplot(titanic['Pclass'], ax=ax[0])

y = sns.countplot(titanic['Embarked'], ax=ax[1])

draw(x)

draw(y)

fig.show()
FacetGrid = sns.FacetGrid(titanic, col='Pclass', height=4, aspect=1)

FacetGrid.map(sns.pointplot, 'Embarked', 'Survived', 'Sex', palette=None,  order=None, hue_order=None)

FacetGrid.add_legend()
drop_column = ['Embarked']

titanic.drop(drop_column, axis=1, inplace = True)

titanic_test.drop(drop_column,axis=1,inplace=True)
plt.figure(figsize = (8, 5))

pclass= sns.countplot(x='Pclass', hue='Survived', data=titanic)

draw(pclass)
plt.figure(figsize = (8, 5))

sns.barplot(x='Pclass', y='Survived', data=titanic)
# combine test and train as single to apply some function, we will use it again in Data Preprocessing

all_data=[titanic,titanic_test]



for dataset in all_data:

    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1
axes = sns.factorplot('Family','Survived', 

                      data=titanic, aspect = 2.5, )
axes = sns.factorplot('Family','Age','Survived',

                      data=titanic, aspect = 2.5, )
# create bin for age features. 

for dataset in all_data:

    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])

    

plt.figure(figsize = (8, 5))

sns.barplot(x='Age_bin', y='Survived', data=titanic)
plt.figure(figsize = (8, 5))

ag = sns.countplot(x='Age_bin', hue='Survived', data=titanic)

draw(ag)
AAS = titanic[['Sex','Age_bin','Survived']].groupby(['Sex','Age_bin'],as_index=False).mean()

sns.factorplot('Age_bin','Survived','Sex', data=AAS

                ,aspect=3,kind='bar')

plt.suptitle('Age , Sex vs Survived')
# create bin for fare features

for dataset in all_data:

    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,10,50,100,550], labels=['Low_fare','median_fare','Average_fare','high_fare'])

plt.figure(figsize = (8, 5))

ag = sns.countplot(x='Pclass', hue='Fare_bin', data=titanic)
sns.barplot(x='Fare_bin', y='Survived', data=titanic)
pd.DataFrame(abs(titanic.corr()['Survived']).sort_values(ascending = False))
# Generate a mask for the upper triangle (taken from seaborn example gallery)

corr=titanic.corr()#['Survived']



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (12,8))

sns.heatmap(corr, 

            annot=True,

            mask = mask,

            cmap = 'RdBu',

            linewidths=.9, 

            linecolor='white',

            vmax = 0.3,

            fmt='.2f',

            center = 0,

            square=True)

plt.title("Correlations Matrix", y = 1,fontsize = 20, pad = 20);
titanic.info()
# Convert ‘Sex’ feature into numeric.

genders = {"male": 0, "female": 1}



for dataset in all_data:

    dataset['Sex'] = dataset['Sex'].map(genders)

titanic['Sex'].value_counts()
for dataset in all_data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 20), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 26), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 28), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 35), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 45), 'Age'] = 5

    dataset.loc[ dataset['Age'] > 45, 'Age'] = 6

titanic['Age'].value_counts()
# As we created new fetures form existing one, so we remove that one.



# Removing SibSp & Parch because we have family now. same way Age.

# We also going to remove some other features like passenger id in list, Ticket number and Name.



for dataset in all_data:

    drop_column = ['Age_bin','Fare','Name','Ticket', 'PassengerId','SibSp','Parch','Fare_bin']

    dataset.drop(drop_column, axis=1, inplace = True)

all_features = titanic.drop("Survived",axis=1)

Targete = titanic["Survived"]

X_train,X_test,y_train,y_test = train_test_split(all_features,Targete,test_size=0.3,random_state=0)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
model = LogisticRegression()

model.fit(X_train,y_train)

prediction_lr=model.predict(X_test)

Log_acc = round(accuracy_score(prediction_lr,y_test)*100,2)

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

Log_cv_acc=cross_val_score(model,all_features,Targete,cv=10,scoring='accuracy')



print('The accuracy of the Logistic Regression is',Log_acc)

print('The cross validated score for Logistic REgression is:',round(Log_cv_acc.mean()*100,2))
knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

kfold = KFold(n_splits=10, random_state=22) 

result_knn=cross_val_score(model,all_features,Targete,cv=10,scoring='accuracy')



print('The accuracy of the K Nearst Neighbors Classifier is',round(accuracy_score(Y_pred,y_test)*100,2))

print('The cross validated score for K Nearest Neighbors Classifier is:',round(result_knn.mean()*100,2))
from sklearn.naive_bayes import GaussianNB

model= GaussianNB()

model.fit(X_train,y_train)

prediction_gnb=model.predict(X_test) 

nb_acc = round(accuracy_score(prediction_gnb,y_test)*100,2)

kfold = KFold(n_splits=12, random_state=22)

result_gnb=cross_val_score(model,all_features,Targete,cv=12,scoring='accuracy')



print('The accuracy of the Gaussian Naive Bayes Classifier is',nb_acc)

print('The cross validated score for Gaussian Naive Bayes classifier is:',round(result_gnb.mean()*100,2))
linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

kfold = KFold(n_splits=5, random_state=22)

result_svm=cross_val_score(model,all_features,Targete,cv=10,scoring='accuracy')



print('The accuracy of the Support Vector Machines Classifier is',acc_linear_svc)

print('The cross validated score for Support Vector Machines Classifier is:',round(result_svm.mean()*100,2))
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)



kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

result_rm=cross_val_score(model,all_features,Targete,cv=10,scoring='accuracy')



print('The accuracy of the Random Forest Classifier is',acc_random_forest)

print('The cross validated score for Random Forest Classifier is:',round(result_rm.mean()*100,2))
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test) 

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)



kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

result_rm=cross_val_score(model,all_features,Targete,cv=10,scoring='accuracy')



print('The accuracy of the Random Forest Classifier is',acc_decision_tree)

print('The cross validated score for Random Forest Classifier is:',round(result_rm.mean()*100,2))
results = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes',  

              'Decision Tree'],

    'Score': [acc_linear_svc, acc_knn, Log_acc, 

              acc_random_forest, nb_acc, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Model')

result_df.head(9)
predictions = cross_val_predict(random_forest, X_train, y_train, cv=3)

confusion_matrix(y_train, predictions)
from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_train, predictions))

print("Recall:",recall_score(y_train, predictions))
from sklearn.metrics import f1_score

f1_score(y_train, predictions)
from sklearn.metrics import roc_curve

# compute true positive rate and false positive rate

y_scores = random_forest.predict_proba(X_train)

y_scores = y_scores[:,1]

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)

# plotting them against each other

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'r', linewidth=4)

    plt.axis([-0.05, 1.05, -0.05, 1.05])

    plt.xlabel('False Positive Rate (FPR)', fontsize=16)

    plt.ylabel('True Positive Rate (TPR)', fontsize=16)



plt.figure(figsize=(14, 7))

plot_roc_curve(false_positive_rate, true_positive_rate)

plt.show()