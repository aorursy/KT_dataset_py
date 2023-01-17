# Data Analysis
import numpy as np 
import pandas as pd

# Data Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# allowing multiple/scrollable outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#machine Learning

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from fancyimpute import KNN

#Setting up functions for visualization

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( plt.hist , var , alpha=0.5)
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
    
# Function inspired by Helge Bjorland: An Interactive Data Science Tutorial
# Load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_full = [df_train, df_test]
# preliminary analysis
print('Shape of the Data -> Train:', df_train.shape, 'Test:', df_test.shape)
pd.crosstab(index = df_train['Survived'], columns = "count")
# preliminary analysis
df_train.describe()
df_train.isnull().sum()
df_test.isnull().sum()

#for understanding datatypes of columns
#df_train.info()
#df_test.info()
table1 = pd.pivot_table(df_train, values='Survived', index=['Pclass'])
table1
for dataset in df_full:
    family_size = dataset.SibSp + dataset.Parch +1 #including themselves
    dataset['FamilySize'] = family_size


table2 = pd.pivot_table(df_train, values = 'Survived', index= ['FamilySize'])
table2
sns.barplot(x=df_train['Embarked'],y=df_train['Survived'] )
plot_distribution( df_train , var = 'Age' , target = 'Survived', row = 'Sex')
plot_distribution( df_train , var = 'Age' , target = 'Survived')
child = df_train[(df_train['Age']<=10)]
pd.pivot_table(child, index = ['Sex'], values = 'Survived')
plot_distribution( df_train , var = 'Fare' , target = 'Survived')
#missing Embarked
port_mode = df_train.Embarked.mode()[0]
#port_mode
df_train['Embarked'] = df_train['Embarked'].fillna(port_mode)

#missing Fare
fare_median = df_test.Fare.median()
#fare_median
df_test['Fare'] = df_test['Fare'].fillna(fare_median)
#numeric values for Sex
for dataset in df_full:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#numeric values to Embarked
for dataset in df_full:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0 , 'C':1 , 'Q':2}).astype(int)
# Grouping Family Size to Ordinals
for dataset in df_full:
    dataset['FamilySize'] = dataset['FamilySize'].replace([1], 0)
    dataset['FamilySize'] = dataset['FamilySize'].replace([2,3,4], 1)
    dataset['FamilySize'] = dataset['FamilySize'].replace([5,6,7,8,9,10,11], 2)

pd.pivot_table(df_train, index = 'FamilySize' , values= 'Survived')
#quantiles for fare attribute
pd.qcut(df_train['Fare'],4, retbins=True)[1]
#creating same bins for fare bands in train and test based on quantiles in train
#giving ordinal labels 0-3

bins = [0,7.91,14.454,31.0,513.0]
labels = [0,1,2,3]

for dataset in df_full:
    dataset['Fareband'] = pd.cut(dataset['Fare'], bins=bins, labels=labels, include_lowest = True)
    dataset['Fareband'] = dataset['Fareband'].astype(int)

pd.pivot_table(df_train, index = df_train['Fareband'],values = 'Survived' )
for dataset in df_full:
 dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# to get an idea of all titles in both datasets (to make cleaning easier)
all_titles = df_test['Title'].append(df_train['Title'])
pd.crosstab(all_titles,'count')
for dataset in df_full:
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'],'Miss')
    dataset['Title'] = dataset['Title'].replace(['Mme'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Jonkheer','Major','Sir','Rev','Dr'],'Raremale')
    dataset['Title'] = dataset['Title'].replace(['Countess','Dona','Lady'],'Rarefemale')
pd.pivot_table(df_train, index = df_train['Title'], values = 'Survived')
title_map = {"Master":1, "Miss":2, "Mr":3, "Mrs":4, "Rarefemale":5, "Raremale":6}
for dataset in df_full:
    dataset['Title'] = dataset['Title'].map(title_map)
    
df_train.head()
df_test.head()
#dropping columns which we may not need / use
for dataset in df_full:
    dataset.drop(['Name','SibSp','Parch','Ticket','Cabin','Fare'], axis= 1, inplace = True)
#dataframes we are left with : age still has missing values
df_train.head()
df_test.head()
#impute age in df_train and df_test
for dataset in df_full:
    new_df = dataset[['PassengerId','Pclass','Sex','Age','Embarked','FamilySize','Fareband','Title']]
    filled = KNN(k=3).complete(new_df)
    filled = pd.DataFrame(filled, columns =['PassengerId','Pclass','Sex','Age','Embarked','FamilySize','Fareband','Title'])
#separate modifying original dataframe, add histograms for comparison
    dataset['Age'] = filled['Age']
    dataset.head()
    dataset.isnull().sum()

#hist before and after imputation
#plt.hist(filled['Age'],bins=10, alpha=0.5)
#plt.hist(new_df.Age[~np.isnan(df_train.Age)], bins =10, alpha = 0.5)
#plt.hist(filled_df['Age'],bins=10, alpha=0.5)
#plt.hist(df_train.Age[~np.isnan(df_train.Age)], bins =10, alpha = 0.5)
#plt.hist(df_test.Age[~np.isnan(df_test.Age)], bins =10, alpha = 0.5)
#Discretize age into 5 equal groups and assign ordinal agebands
pd.cut(df_train['Age'],5).unique()
bins = [0,16,32,48,64,80]
labels = [0,1,2,3,4]

for dataset in df_full:
    dataset['Ageband'] = pd.cut(dataset['Age'],bins = bins,labels = labels, include_lowest=True)
    dataset['Ageband'] = dataset['Ageband'].astype(int)

pd.pivot_table(df_train, index = ['Ageband'],values = 'Survived',columns=['Sex'])
for dataset in df_full:
    dataset.drop("Age", axis= 1, inplace = True)
    
df_train.head()
df_test.head()
# Variables needed for building prediction model
X_train = df_train.drop(["Survived","PassengerId"], axis=1)
Y_train = df_train["Survived"]
X_test  = df_test.drop(["PassengerId"], axis=1).copy()
#logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
#Y_pred
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
#KNN k=3

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
#X_train['Fareband'] = X_train['Fareband'].astype('int')
#X_train.apply(pd.to_numeric)
X_train.info()

xgb = XGBClassifier()
xgb.fit(X_train,Y_train)
y_pred = xgb.predict(X_test)

acc_xgb = round(sgd.score(X_train, Y_train) * 100, 2)
acc_xgb

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)

Y_pred

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
# Model Evaluation

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'XGBoost', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_xgb, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
# #competition submission: Random Forest Trees
# submission = pd.DataFrame({
#         "PassengerId": df_test["PassengerId"],
#         "Survived": Y_pred
#     })
# submission.to_csv('submission.csv', index=False)