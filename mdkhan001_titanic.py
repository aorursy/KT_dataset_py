

# Importing the libraries

import numpy as np #foundational package for scientific computing

print("NumPy version: {}". format(np.__version__))



import matplotlib #collection of functions for scientific and publication-ready visualization

print("matplotlib version: {}". format(matplotlib.__version__))

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn

seaborn.set()



import seaborn as sns



import pandas as pd

print("pandas version: {}". format(pd.__version__))



import warnings

warnings.filterwarnings('ignore')



import IPython

from IPython import display #pretty printing of dataframes in Jupyter notebook

print("IPython version: {}". format(IPython.__version__)) 



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



import math





# import data 

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



all_data = [train, test]



train.info()

#delete the cabin feature/column and others previously stated to exclude in train dataset

drop_column = ['PassengerId','Cabin', 'Ticket']

for dataset in all_data:

    dataset.drop(drop_column, axis=1, inplace = True)

    



test.info()
train.info()


print('Train columns with null values:\n', train.isnull().sum())

print("-"*10)



print('Test/Validation columns with null values:\n', test.isnull().sum())

print("-"*10)

# COMPLETING: complete or delete missing values in train and test/validation dataset

for dataset in all_data: 

    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    

# we will fill Age later on after creating a new feature named title



for dataset in all_data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])



ax = plt.subplot()

ax.set_ylabel('Average age')

train.groupby('Title').mean()['Age'].plot(kind='bar',figsize=(13,8), ax = ax)



title_mean_age=[]

title_mean_age.append(list(set(train.Title)))  #set for unique values of the title, and transform into list

title_mean_age.append(train.groupby('Title').Age.mean())

print(title_mean_age[1])

#------------------------------------------------------------------------------------------------------



#------------------ Fill the missing Ages ---------------------------



#--------------------------------------------------------------------

# training dataset

n_traning= train.shape[0]   #number of rows

n_titles= len(title_mean_age[1])

for i in range(0, n_traning):

    if np.isnan(train.Age[i])==True:

        for j in range(0, n_titles):

            if train.Title[i] == title_mean_age[0][j]:

                train.Age[i] = title_mean_age[1][j]

#-------------------------------------------------------------------- 

# test dataset

n_test= test.shape[0]   #number of rows

n_titles= len(title_mean_age[1])

for i in range(0, n_test):

    if np.isnan(test.Age[i])==True:

        for j in range(0, n_titles):

            if test.Title[i] == title_mean_age[0][j]:

                test.Age[i] = title_mean_age[1][j]

                



print('Train columns with null values:\n', train.isnull().sum())

print("-"*20)



print('Test/Validation columns with null values:\n', test.isnull().sum())

print("-"*20)



print(title_mean_age[0])

pd.crosstab(train['Title'], train['Sex'])



for dataset in all_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=True)



for dataset in all_data:

    dataset.drop('Name', axis=1, inplace = True)

    


train.info()



###CREATE: Feature Engineering for train and test/validation dataset

for dataset in all_data:    

    #Discrete variables

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1



    dataset['IsAlone'] = 1 #initialize to yes/1 is alone

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    

    


max_ = train.Age.max()

thres = 0

survival = -1

min_person = 50



for i in range(math.ceil(max_)):

    count1 = 0

    count2 = 0

    temp = train[train['Age'] < i][['Age', 'Survived']].mean()['Survived']

    

    count1 = train[train['Age'] < i][['Age', 'Survived']].count()['Survived']

    count2 = train[train['Age'] > i][['Age', 'Survived']].count()['Survived']

    

    if temp > survival and count1 > min_person and count2 > min_person:

        survival = temp

        thres = i

        

print('Threshold Age = ', thres)

print(train[train['Age'] < thres][['Age', 'Survived']].mean())

print(train[train['Age'] > thres][['Age', 'Survived']].mean())



max = train.Fare.max()

thres = 0

survival = -1

min_person = 50



for i in range(math.ceil(max)):

    count1 = 0

    count2 = 0

    

    temp = train[train['Fare'] < i][['Fare', 'Survived']].mean()['Survived']

    

    count1 = train[train['Fare'] > i][['Fare', 'Survived']].count()['Survived']

    count2 = train[train['Fare'] < i][['Fare', 'Survived']].count()['Survived']

    

    if temp > survival and count1 > min_person and count2 > min_person:

        survival = temp

        thres = i

        

print('Threshold Fare = ',thres)

print(train[train['Fare'] < thres][['Fare', 'Survived']].mean())

print(train[train['Fare'] > thres][['Fare', 'Survived']].mean())



for dataset in all_data:

    dataset['Child'] = 1

    dataset['Child'].loc[dataset['Age'] > 9] = 0

    

    

    dataset['Low_fare'] = 1

    dataset['Low_fare'].loc[dataset['Fare'] > 107] = 0

    



#Map Data

for data in all_data:



    #Mapping Sex

    sex_map = { 'female':0 , 'male':1 }

    data['Sex'] = data['Sex'].map(sex_map).astype(int)



    #Mapping Title

    title_map = {'Mr':1, 'Miss':4, 'Mrs':5, 'Master':3, 'Rare':2}

    data['Title'] = data['Title'].map(title_map)

    data['Title'] = data['Title'].fillna(0)



    #Mapping Embarked

    embark_map = {'S':0, 'C':1, 'Q':2}

    data['Embarked'] = data['Embarked'].map(embark_map).astype(int)

    

    


train.columns



def rel_to_sur(feature):

    return train[[feature, 'Survived']].groupby([feature], as_index=False).mean().sort_values(by='Survived', ascending=True)



for feature in ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'Child', 'Low_fare']:

    print(rel_to_sur(feature))

    print('-'*40)

    
train.dtypes


plt.subplot()

plt.hist(x = [train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()

plt.show()



plt.subplots(figsize=(10,8))

plt.hist(x = [train[train['Survived']==1]['Fare'], train[train['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()

plt.show()





#correlation heatmap of dataset

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(train)

train.shape
test.shape


Y = train['Survived']

X = train.drop("Survived", axis=1)

X.shape, Y.shape


# https://chrisalbon.com/machine_learning/model_selection/hyperparameter_tuning_using_grid_search/

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html



# Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV





classifier = LogisticRegression()



# Create regularization penalty space

penalty = ['l1', 'l2']



# Create regularization hyperparameter space

C = np.arange(0.1, 2, 0.1)



# Create hyperparameter options

hyperparameters = dict(C=C, penalty=penalty)



hyperparameters



# Create grid search using 5-fold cross validation

clf = GridSearchCV(classifier, hyperparameters, cv=5, verbose=0)



# Fit grid search

best_model = clf.fit(X, Y)



# View best hyperparameters

print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])

print('Best C:', best_model.best_estimator_.get_params()['C'])



best_model.best_estimator_



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = best_model.best_estimator_, X=X , y=Y , cv = 10)

print("Logistic Regression:\n Accuracy: %0.2f (+/- %0.2f)" % (accuracies.mean(), accuracies.std()),"\n")



pred = best_model.predict(test)

sample = pd.read_csv('../input/gender_submission.csv')

submission = pd.DataFrame({

        "PassengerId": sample["PassengerId"],

        "Survived": pred

    })

# submission.to_csv('./submission_logistic_regression_00.csv', index=False)



from sklearn.model_selection import GridSearchCV



# Decision Tree

# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier





model = DecisionTreeClassifier()



criterion =['gini','entropy']

max_depth = [1,3,5,None]

splitter = ['best','random']

min_samples_leaf = np.arange(1, 20, 1)

min_samples_split = np.arange(2, 20, 2)









grid = GridSearchCV(estimator=model,cv=5, param_grid=dict(criterion=criterion, max_depth=max_depth, 

                                                          splitter=splitter, min_samples_leaf=min_samples_leaf, 

                                                          min_samples_split = min_samples_split))



best_model_dt = grid.fit(X,Y)

best_model_dt.best_estimator_





accuracies = cross_val_score(estimator = best_model_dt.best_estimator_, X=X , y=Y, cv = 10)

print("Decision Tree:\n Accuracy: %0.2f (+/- %0.2f)" % (accuracies.mean(), accuracies.std()),"\n")





pred = best_model_dt.predict(test)

sample = pd.read_csv('../input/gender_submission.csv')

submission = pd.DataFrame({

        "PassengerId": sample["PassengerId"],

        "Survived": pred

    })

# submission.to_csv('./submission_decision_tree_00.csv', index=False)



# Random forest

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()



n_estimators = [10, 100, 200]

criterion =['gini','entropy']

max_depth = [1,5,None]

min_samples_leaf = [1, 10, 20]

min_samples_split = [2, 10, 20]



param_grid=dict(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,

                min_samples_leaf=min_samples_leaf, min_samples_split = min_samples_split)

param_grid



grid = GridSearchCV(estimator=classifier, cv=5, param_grid=param_grid)



best_model_rf = grid.fit(X,Y)

best_model_rf.best_estimator_





accuracies = cross_val_score(estimator = best_model_rf.best_estimator_, X=X , y=Y, cv = 10)

print("Random Forest:\n Accuracy: %0.2f (+/- %0.2f)" % (accuracies.mean(), accuracies.std()),"\n")



pred = best_model_rf.predict(test)

sample = pd.read_csv('../input/gender_submission.csv')

submission = pd.DataFrame({

        "PassengerId": sample["PassengerId"],

        "Survived": pred

    })

submission.to_csv('./submission_random_forrest_00.csv', index=False)
