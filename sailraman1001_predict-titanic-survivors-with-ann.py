#data analysis

import numpy as np

import pandas as pd

import re



#visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style="ticks", color_codes=True)



#machine learning

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, StandardScaler, Normalizer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

combine_data = [train_data, test_data]

print(train_data.info())

print(test_data.info())
#numeric data

train_data.describe()
#categorical data

train_data.describe(include=['O'])
#view first 5 rows

print(train_data.head())
#view Name feature, top 5 rows

print(train_data.Name.head())
#Survival rate per Class. "1" class passengers have highest survival rate.

train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean()
#Survival rate Gender-wise. 

#Females-  74.2% have survived. 

#Males- 18.89% have survived.

train_data[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
#compared to singles (no siblings and spouse), small families have higher survival rate

#Lagre families didnt survive

train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
#Same observation as "SibSp"

#Small families have higher survival rate, compared to passengers travelling alone and larger families

train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending = False)
#"SibSp" and "Parch" both indicate the family size. So let us have single feature "Familysize"  

#and drop these 2 features

train_data['Familysize'] = train_data['Parch'] + train_data['SibSp'] + 1

test_data['Familysize'] = test_data['Parch'] + test_data['SibSp'] + 1



train_data = train_data.drop(['Parch', 'SibSp'], axis = 1)

test_data = test_data.drop(['Parch', 'SibSp'], axis = 1)

combine_data = [train_data, test_data]
#"Cabin" has lot of missing values. 204/891 are available. It doesnt help in analysis. So, drop it.

#"Ticket" value is unique for 681/891 records. Alpha numeric ticket number, shared by 210 records. 

#No visible pattern. So drop these two columns it doesnt help in analysis.



train_data = train_data.drop(['Cabin', 'Ticket'], axis = 1)

test_data = test_data.drop(['Cabin', 'Ticket'], axis = 1)



combine_data = [train_data, test_data]
# Define function to extract titles from passenger names

def extract_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



# Create a new feature Title, containing the titles of passenger names

for dataset in combine_data:

    dataset['Title'] = dataset['Name'].apply(extract_title)



#Check if the Titles are properly aligned as per the Sex of the passenger

pd.crosstab(train_data['Title'], train_data['Sex'])



#Title feature will help in data analysis. So, retain it.

#Some rare titles can be grouped together, which have very few ( 2-3) passengers mapped.

for dataset in combine_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
#Survival rate based on Title

train_data[['Title', 'Survived']].groupby(['Title'], as_index = False).mean()



#Name feature can be dropped, as we have extracted the Title and this is not used further for analysis.

#PassengerId is just a sequential numbering for the data records. Can be droppped

train_data = train_data.drop(['Name', 'PassengerId'], axis = 1)

test_data = test_data.drop(['Name', 'PassengerId'], axis = 1)

combine_data = [train_data, test_data]
title_mapping = {'Mr' : 1, 'Miss' : 2, 'Master' : 3, 'Mrs' : 4, 'Rare' : 5}

for dataset in combine_data:

    dataset['Titlecode'] = dataset['Title'].map(title_mapping)

    dataset['Titlecode'] = dataset['Title'].fillna(0)    

train_data['Titlecode'] = train_data['Title'].map(title_mapping)

test_data['Titlecode'] = test_data['Title'].map(title_mapping)

combine_data = [train_data, test_data]
#Most of the passengers who have paid extremely higher price (Fare>80) have survived

#Most of the Passengers who have paid extremely lower price (Fare<20) have nor survived

# ore Females have paid higher Fare

grid = sns.FacetGrid(train_data, col='Survived', hue = "Sex")

grid.map(plt.hist, 'Fare', alpha=.5, bins = 6)

grid.add_legend();
# Comparitively more Female pssengers have paid higher Fare

# Age >65 senior citizens are Male passengers

# oldest Male passenger has survived.

# High Fare paying passengers have survived.

grid = sns.FacetGrid(train_data, col = 'Survived', hue='Sex')

grid.map(plt.scatter, 'Fare','Age')

grid.add_legend();
#Third class (Pclass = 3) embarked in either "S" or "Q"

#First class (Pclass = 1) embarked mostly in "C".

grid = sns.FacetGrid(train_data, hue='Embarked')

grid.map(plt.hist, 'Pclass',alpha = 0.9, bins=5)

grid.add_legend()
#First class passengers (Pclass = 1) have mostly embarked in "C". Had good survival rate.

#Third class passengers(PClass = 3) embarked in "S"have suffered the most.

grid = sns.FacetGrid(train_data, col = 'Survived', hue='Embarked')

grid.map(plt.hist, 'Pclass',alpha = 0.9, bins=5)

grid.add_legend()
#Singles and larger families (Familysize>4) have least survival rate

grid = sns.FacetGrid(train_data, col = 'Survived', hue='Sex')

grid.map(plt.hist, 'Familysize',alpha = 0.6, bins=11)

grid.add_legend()
#Singles and larger families travelled in third class (Pclass = 3)

#Smaller families and fewer singles travelled in first or second class

sns.catplot(y="Familysize", hue="Pclass", kind="count", edgecolor=".6",data=train_data)
#Survival rate Familysize-wise. 

#Smaller size families have higher survival rate, together. Large families have perished together.

train_data[['Familysize', 'Survived']].groupby(['Familysize'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
#Males- 18.89% have survived.

train_data[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)



pd.crosstab(train_data['Embarked'], train_data['Sex'])
#Larger families (Familysize > 4) have embarked in "S". Have least survival rate

pd.crosstab(train_data['Familysize'], [train_data['Survived'], train_data['Embarked']])
#Females, minor Passengers embarked in "C" and "Q" have mostly survived.

pd.crosstab(train_data['Title'], [train_data['Survived'], train_data['Embarked']])
#Almost all the children, Females travelling in first and second class have Survived. 

#However, the children travelling in third class (embarked in "S") have been penalised along with their parents.

#So families sink or survive together.

pd.crosstab(train_data['Title'], [train_data['Survived'], train_data['Pclass']])
grid = sns.FacetGrid(train_data, col = 'Survived', row = 'Pclass', height = 2.2, aspect = 2 )

grid.map(plt.hist, 'Age', bins = 10)
grid = sns.FacetGrid(train_data, row = 'Embarked')

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex',palette = 'deep')

grid.add_legend()
grid = sns.FacetGrid(train_data, row = 'Embarked', col = 'Survived')

grid.map(sns.barplot, 'Sex', 'Fare')
guessed_ages = np.zeros(5)

guessed_ages
#Impute the missing "Age" data. 

#Find the Median age per Title and use that to impute missing data

for dataset in combine_data:   

    for i in range(0,5):        

        guess_age = dataset[(dataset['Titlecode'] == i+1)]['Age'].dropna()

        guess_age = guess_age.median()

        guess_age = int( guess_age/0.5 + 0.5 ) * 0.5 

        guessed_ages[i] = guess_age

      

    for i in range(0,5):          

        dataset.loc[ (dataset.Age.isnull()) & (dataset.Titlecode == i+1), 'Age'] = guessed_ages[i]

    

    dataset['Age'] = dataset['Age'].astype(int)   
#Impute the missing "Embarked" data

#889 records have not null values. two records have missing values. 

#These can be the Mode of the dataset.

train_data['Embarked'].count() 

freq_embarked = train_data['Embarked'].mode()[0]

#print(freq_embarked)

train_data.loc[(train_data.Embarked.isnull()), 'Embarked'] = freq_embarked

test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)
#place the passengers in the bins based on their Age. 

#"pd.cut" divides the data based on the Age, into 5 bins

#train_data['AgeBins'] --> holds array-like object representing respective Age bin for each passenger





#Also, place the passengers in the bins based on the Fare.

#"pd.qcut" divides the data based on ticker fare, into ordered 5 bins.

#train_data['FareBin'] --> holds array-like object representing respective Fare bin for each passenger

for dataset in combine_data:

    dataset['AgeBin'] = pd.cut(dataset['Age'], 5)

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
# Mapping Fare

for dataset in combine_data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

train_data.head()
# Mapping Age

for dataset in combine_data:

    dataset.loc[ dataset['Age'] <= 16, 'Age']  = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 

    dataset['Age'] = dataset['Age'].astype(int)

train_data.head()
train_data = train_data.drop(['Title','AgeBin','FareBin'], axis = 1)

test_data = test_data.drop(['Title','AgeBin','FareBin'], axis = 1)

combine_data = [train_data, test_data]



train_data.head()
for dataset in combine_data:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
#there is a missing 'Sex' value in test_data

for dataset in combine_data:

    dataset['Sex'] = dataset['Sex'].map({'male' : 0, 'female' : 1}).astype(int)



train_data.head()
#Now that all the categorical features are converted into numeric, 

#Sex --> Already one hot encoded #Age and Fare -->Ordinal data #Embarked, Pclass, Titlecode --> need to one hot encode

features = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Familysize', 'Titlecode']]

target = train_data['Survived']

preprocess = make_column_transformer(    

    (OneHotEncoder(sparse=False), ['Pclass', 'Sex', 'Embarked','Titlecode']), 

    remainder= StandardScaler())



preprocess.transformers

train_x = preprocess.fit_transform(features)

train_y = target.values



train_x[1]



#Preprocess and transform test features

features_test = test_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Familysize', 'Titlecode']]

preprocess = make_column_transformer(    

    (OneHotEncoder(sparse=False), ['Pclass', 'Sex', 'Embarked','Titlecode']), 

    remainder= StandardScaler())

preprocess.transformers

test_x = preprocess.fit_transform(features_test)

#model = make_pipeline(preprocess, LogisticRegression())

#model.fit(train_x, train_y)

#print("logistic regression score: %f" % model.score(test_x, test_y))

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

import keras

#from keras.optimizers import SGD

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

import graphviz
#creating the model

def build_classifier(optimizer):  

    #initialize ANN

    classifier = Sequential()

    #Adding the input parameters and first layer to ANN

    classifier.add(Dense(units = 20, 

                         kernel_initializer = 'uniform', 

                         activation = 'relu', 

                         input_dim = 16))

    

    #Adding the dropout layer

    classifier.add(Dropout(0.5))

    

    #Adding the second layer to ANN

    classifier.add(Dense(units = 60, 

                         kernel_initializer = 'uniform', 

                         activation = 'relu'))

    

    #Adding the dropout layer

    classifier.add(Dropout(0.5))

    

    #Adding the output layer that is binary

    classifier.add(Dense(units = 1, 

                         kernel_initializer = 'uniform', 

                         activation = 'sigmoid'))

    

    classifier.summary()

    

    #with the scalar sigmoid output on a binary classification problem, use the binary cross entropy loss function

    classifier.compile(optimizer = optimizer, 

                       loss = 'binary_crossentropy', 

                       metrics = ['accuracy'] )

    

    return classifier



#Use KerasClassifier

classifier = KerasClassifier(build_fn = build_classifier)



#create a dictionary for the hyper parameters

parameters = {'batch_size' : [60, 30],

               'nb_epoch' : [30, 50],

               'optimizer' : ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']}



# 'optimizer' : ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']}

#GridSearchCV implements fit and predict

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 3)



grid_search = grid_search.fit(train_x, train_y)

best_estimator = grid_search.best_estimator_

best_parameters = grid_search.best_params_

best_score = grid_search.best_score_



print("best_estimator %s" %best_estimator)

print("best_parameters %s" %best_parameters)

print("best_score %f" %best_score)