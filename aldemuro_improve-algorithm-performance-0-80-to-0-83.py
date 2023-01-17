#sklearn

from sklearn.cross_validation import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve

from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn import svm,model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process



#load package

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#from math import sqrt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
## Read in file

train_original = pd.read_csv('../input/train.csv')

test_original = pd.read_csv('../input/test.csv')

train_original.sample(10)

total = [train_original,test_original]

#exploration

train_original.info()

print("----------------------------")

test_original.info()
#Retrive the salutation from 'Name' column

for dataset in total:

    dataset['Salutation'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)    
plt.subplots(figsize=(15,6))

sns.countplot(x="Salutation", hue='Survived', data=train_original,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Survive or not in each salutation')



plt.show()
#grouping the low-value data

for dataset in total:

    dataset['Salutation'] = dataset['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Salutation'] = dataset['Salutation'].replace('Mlle', 'Miss')

    dataset['Salutation'] = dataset['Salutation'].replace('Ms', 'Miss')

    dataset['Salutation'] = dataset['Salutation'].replace('Mme', 'Mrs')

    #dataset['Salutation'] = pd.factorize(dataset['Salutation'])[0]

    





#total.Salutation = pd.factorize(total.Salutation)[0]   

plt.subplots(figsize=(15,6))

sns.countplot(x="Salutation", hue='Survived', data=train_original,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Survive or not in each salutation after')



plt.show()
#Factorize the salutation

for dataset in total:    

    dataset['Salutation'] = pd.factorize(dataset['Salutation'])[0]
plt.subplots(figsize=(8,6))

sns.countplot(x="Sex", hue='Survived', data=train_original,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Survive or not in every gender')

plt.show()
#clean unused variable

train=train_original.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

test=test_original.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

total = [train,test]



train.shape, test.shape
#Detect the missing data in 'train' dataset

train.isnull().sum()
## Create function to replace missing data with the median value

def fill_missing_age(dataset):

    for i in range(1,8):

        median_age=dataset[dataset["Salutation"]==i]["Age"].median()

        dataset["Age"]=dataset["Age"].fillna(median_age)

        return dataset



train = fill_missing_age(train)
plt.subplots(figsize=(8,6))

sns.distplot(train.Age)

plt.xticks(rotation=90)

plt.title('Distribution of Passenger Age')

plt.show()
plt.subplots(figsize=(8,6))

sns.distplot(train.Fare)

plt.xticks(rotation=90)

plt.title('Distribution of fare')



plt.show()
## Embarked missing cases 

train[train['Embarked'].isnull()]
train["Embarked"] = train["Embarked"].fillna('C')
plt.subplots(figsize=(8,6))

sns.countplot(x="Embarked", hue='Survived', data=train_original,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Survive or not in ship point of embarkation')

plt.show()
plt.subplots(figsize=(8,6))

sns.countplot(x="Pclass", hue='Survived', data=train_original,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Survive or not in pasenger class')

plt.show()
test.isnull().sum()
#apply the missing age method to test dataset

test = fill_missing_age(test)
#filling the missing 'Fare' data with the  median

def fill_missing_fare(dataset):

    median_fare=dataset[(dataset["Pclass"]==3) & (dataset["Embarked"]=="S")]["Fare"].median()

    dataset["Fare"]=dataset["Fare"].fillna(median_fare)

    return dataset



test = fill_missing_fare(test)
## Re-Check for missing data

train.isnull().any()
## Re-Check for missing data

test.isnull().any()
pd.qcut(train["Age"], 6).value_counts()


for dataset in total:

    dataset.loc[dataset["Age"] <= 19, "Age"] = 0

    dataset.loc[(dataset["Age"] > 19) & (dataset["Age"] <= 25), "Age"] = 1

    dataset.loc[(dataset["Age"] > 25) & (dataset["Age"] <= 32), "Age"] = 2

    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 35), "Age"] = 3

    dataset.loc[(dataset["Age"] > 35) & (dataset["Age"] <= 40.5), "Age"] = 4

    dataset.loc[dataset["Age"] > 40.59, "Age"] = 5

pd.qcut(train["Fare"], 8).value_counts()
for dataset in total:

    dataset.loc[dataset["Fare"] <= 7.75, "Fare"] = 0

    dataset.loc[(dataset["Fare"] > 7.75) & (dataset["Fare"] <= 7.91), "Fare"] = 1

    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 9.841), "Fare"] = 2

    dataset.loc[(dataset["Fare"] > 9.841) & (dataset["Fare"] <= 14.454), "Fare"] = 3   

    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 24.479), "Fare"] = 4

    dataset.loc[(dataset["Fare"] >24.479) & (dataset["Fare"] <= 31), "Fare"] = 5   

    dataset.loc[(dataset["Fare"] > 31) & (dataset["Fare"] <= 69.487), "Fare"] = 6

    dataset.loc[dataset["Fare"] > 69.487, "Fare"] = 7
for dataset in total:

    dataset['Sex'] = pd.factorize(dataset['Sex'])[0]

    dataset['Embarked']= pd.factorize(dataset['Embarked'])[0]

train.head()
x = train.drop("Survived", axis=1)

y = train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.25,random_state=1)


MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model. RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

   #tree.ExtraTreeClassifier(),

    

    ]

MLA_columns = []

MLA_compare = pd.DataFrame(columns = MLA_columns)





row_index = 0

for alg in MLA:

    

    

    predicted = alg.fit(x_train, y_train).predict(x_test)

    fp, tp, th = roc_curve(y_test, predicted)

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(x_train, y_train), 4)

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(x_test, y_test), 4)

    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(y_test, predicted)

    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(y_test, predicted)

    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)











    row_index+=1

    

MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    

MLA_compare
plt.subplots(figsize=(15,6))

sns.barplot(x="MLA Name", y="MLA Train Accuracy",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('MLA Train Accuracy Comparison')

plt.show()
plt.subplots(figsize=(15,6))

sns.barplot(x="MLA Name", y="MLA Test Accuracy",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('MLA Test Accuracy Comparison')

plt.show()
plt.subplots(figsize=(15,6))

sns.barplot(x="MLA Name", y="MLA Precission",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('MLA Precission Comparison')

plt.show()
plt.subplots(figsize=(15,6))

sns.barplot(x="MLA Name", y="MLA Recall",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('MLA Recall Comparison')

plt.show()
plt.subplots(figsize=(15,6))

sns.barplot(x="MLA Name", y="MLA AUC",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('MLA AUC Comparison')

plt.show()
index = 1

for alg in MLA:

    

    

    predicted = alg.fit(x_train, y_train).predict(x_test)

    fp, tp, th = roc_curve(y_test, predicted)

    roc_auc_mla = auc(fp, tp)

    MLA_name = alg.__class__.__name__

    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))

   

    index+=1



plt.title('ROC Curve comparison')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0,1])

plt.ylim([0,1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')    

plt.show()
tunealg = ensemble.AdaBoostClassifier() #Select the algorithm to be tuned

tunealg.fit(x_train, y_train)



print('BEFORE tuning Parameters: ', tunealg.get_params())

print("BEFORE tuning Training w/bin set score: {:.2f}". format(tunealg.score(x_train, y_train))) 

print("BEFORE tuning Test w/bin set score: {:.2f}". format(tunealg.score(x_test, y_test)))

print('-'*10)



#tune parameters

param_grid = {'n_estimators': [10,15,25,35,45,50,55,60,65], 

              'learning_rate': [0.1,0.2,0.3,0.4,0.5,1.0],

              'algorithm': ['SAMME','SAMME.R'],                

              'random_state':  [1,2,3,4,5,50, None], 

              

             }

# So, what this GridSearchCV function do is finding the best combination of parameters value that is set above.

tune_model = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(), param_grid=param_grid, scoring = 'roc_auc')

tune_model.fit (x_train, y_train)



print('AFTER tuning Parameters: ', tune_model.best_params_)

print("AFTER tuning Training w/bin set score: {:.2f}". format(tune_model.score(x_train, y_train))) 

print("AFTER tuning Test w/bin set score: {:.2f}". format(tune_model.score(x_test, y_test)))

print('-'*10)
#re-train the model with un-split data

y_pred = tune_model.fit(x, y).predict(test)
submission = pd.DataFrame({

        "PassengerId": test_original["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('titanic.csv', index=False)
