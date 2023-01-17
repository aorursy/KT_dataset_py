#!/bin/python



# Importing data-wrangling libraries

import pandas as pd

import numpy as np

import random as rnd



# Importing visualisation libraries

import seaborn as sns

import matplotlib.pyplot as plt

# this is only used for jupyter for inline display

%matplotlib inline



# Importing machine learning data processing libraries

from sklearn import preprocessing

from sklearn.model_selection import train_test_split



# Importing machine learning modelling libraries



from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import StandardScaler



# Importing machine learning parameter tuning libraries

from sklearn.model_selection import GridSearchCV



# Importing machine learning Kfold cross validation libraries

#from sklearn.cross_validation import KFold

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



# library for my MAC for retina display

%config InlineBackend.figure_format = 'retina'



# for pretty display

from IPython.core.display import display, HTML

# Importing Titanic datasets

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



# Lets create a copy of original data (good idea for many reasons some might be unpredictable)

train = train_df.copy()

test = test_df.copy()



# Train will contain all categorical data which is further converted into categorical and numerical.

# train_std will contain standardized data.

# For Kfold cross validation we split train/train_std into X and Y dataset with their respective taining and test dataset.
# Function for describing the dataframe



# Function to print a object but in a html converted format (useful for jupyter). we will use printx instead of print command. 

def printx(str):

    display(HTML(str.to_html()))



# Generic fucntion to describe any data frame named 'df'

def describe_df(df):

    # Display the column names, data type and entries

    print("\n-- df.info --\n")

    df.info()

    #print(df.columns.values)

    

    # Lets look at top 10 and last 10 entries

    print("\n-- df.head() --")

    printx(df.head())

    print("\n-- df.tail() --")

    printx(df.tail())

    

    # Describe both numerical and categorial variables

    print("\n-- Describe the numerical values : df.describe() --")

    printx(df.describe())

    print("\n--  Describe the categorical values : df.describe(include=['O']) --")

    printx(df.describe(include=['O']))



# Function to compare a list of columns with column corresponding label.

# Require 3 arguments, data frame, list, variable

#def compare_attr(df, column_list, label_var):

#    for attrib in column_list:

#        print("\n", '*' * 40)

#        printx(train_df[[attrib, label_var]].groupby([attrib]).mean().sort_values(by=label_var, ascending=False))

        

# Describe the training dataset

print("\n Describing training data \"train\" as df:")  

describe_df(train)



# If needed we can also look at test using similar function. Although its not required.

# describe_df(test)
# Calculate the percentage of people survived of total is

(train.Survived.sum()) / (train.Survived.count()) * 100
# lets have a heatmap look at various varaibles

cols_to_transform = [ 'Sex', 'Embarked','Pclass']

train_with_dummies = pd.get_dummies( train, columns = cols_to_transform )

corr = train_with_dummies.corr()

_ , ax = plt.subplots( figsize =( 12 , 10 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = { 'fontsize' : 12 }

           )
# Functions to apply data transformations for features for both train and test datset 

# the training and test are not merged and converted seperately using same principle

# This approach can prevent data leakage.



def describe_count_null(df,column):

    print ("Total data points available are ", df[column].count(), " and missing data points are", df[column].isnull().sum())



def transform_sex(train,test):

    train.Sex = train.Sex.fillna('NA')

    test.Sex = test.Sex.fillna('NA')

    return train,test

    

def transform_Pclass(train,test):

    train.Pclass = train.Pclass.fillna(-0.5)

    test.Pclass = test.Pclass.fillna(-0.5)

    return train,test



def transform_Age(train,test):

    # Add a new transformed column Age2

    train['Age2'] = train.Age

    test['Age2'] = test.Age

    train.Age2 = train.Age2.fillna(-0.5)

    test.Age2 = test.Age2.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    #group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    #categories = pd.cut(train_df.Age2, bins,labels=group_names))

    categories = pd.cut(train.Age2, bins)

    train.Age2 = categories

    categories = pd.cut(test.Age2, bins)

    test.Age2 = categories

    return train,test



def transform_Fmembers(train,test):

    # Not checking for NA in Parch & SibSp

    train['FamilyMembers'] = train['Parch'] + train['SibSp']

    test['FamilyMembers'] = test['Parch'] + test['SibSp']

    train.FamilyMembers = train.FamilyMembers.fillna(0)

    test.FamilyMembers = test.FamilyMembers.fillna(0)

    #bins = (-1, 0, 1000)

    #categories = pd.cut(train.FamilyMembers, bins)

    #train.FamilyMembers = categories

    #categories = pd.cut(test.FamilyMembers, bins)

    #test.FamilyMembers = categories    

    return train,test    





def transform_Fare(train,test):

    train['Fare2'] = train.Fare

    test['Fare2'] = test.Fare

    train.Fare2 = train.Fare.fillna(-0.5)

    test.Fare2 = test.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    #categories = pd.cut(train_df.Fare2, bins, labels=group_names)

    categories = pd.cut(train.Fare2, bins)

    train.Fare2 = categories

    categories = pd.cut(test.Fare2, bins)

    test.Fare2 = categories

    return train,test

    

def transform_Cabin(train,test):

    train['Cabin2'] = train.Cabin

    test['Cabin2'] = test.Cabin

    train.Cabin2 = train.Cabin2.fillna('N')

    test.Cabin2 = test.Cabin2.fillna('N')

    train.Cabin2 = train.Cabin2.apply(lambda x: x[0])

    test.Cabin2 = test.Cabin2.apply(lambda x: x[0])

    return train,test



def transform_Embarked(train,test):

    train.Embarked = train.Embarked.fillna('N')

    test.Embarked = test.Embarked.fillna('N')

    return train,test    

    

def transform_Name(train,test):

    # Add a new transformed column Title

    train.Name = train.Name.fillna('NA')

    test.Name = test.Name.fillna('NA')        

    train[ 'Title' ] = train['Name'].map( lambda x: x.split( ',' )[1].split( '.' )[0].strip() )

    test[ 'Title' ] = test['Name'].map( lambda x: x.split( ',' )[1].split( '.' )[0].strip() )



    print ("\n\nCategory counts before mapping\n")

    print(train.Title.value_counts(),"\n")

    

    # a map of more aggregated titles

    Title_Dictionary = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "the Countess":"Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty",

                    "NA"   :    "NA"

                    }



    #train[ 'Title' ] = train.Title.map( Title_Dictionary )

    #test[ 'Title' ] = test.Title.map( Title_Dictionary )

    print ("\n\nCategory counts after mapping\n")

    print(train.Title.value_counts(),"\n")

    return train,test
describe_count_null(train,'Sex')

train,test = transform_sex(train,test)

print("\n",train[["Sex", "Survived"]].groupby(['Sex']).count())

sns.barplot(x="Sex", y="Survived", data=train);
describe_count_null(train,'Pclass')

train,test = transform_Pclass(train,test)

sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train);
#Extract the middle name from Name column and get its value count

describe_count_null(train,'Name')

train,test = transform_Name(train,test)



sns.barplot(x="Title", y="Survived", hue="Sex", data=train);
describe_count_null(train,'Age')



# Exploring age and plotting histogram

hist_age = sns.FacetGrid(train, col='Survived')

hist_age.map(plt.hist, 'Age', bins=15);

#sns.plt.show()
train,test = transform_Age(train,test)



print(train.Age2.value_counts(),"\n")

sns.barplot(x="Age2", y="Survived", hue="Sex", data=train);
describe_count_null(train_df,'SibSp')

sns.barplot(x="SibSp", y="Survived", hue="Sex", data=train_df);
describe_count_null(train_df,'Parch')

sns.barplot(x="Parch", y="Survived", hue="Sex", data=train_df);
train,test = transform_Fmembers(train,test)

sns.barplot(x="FamilyMembers", y="Survived", hue="Sex", data=train);
describe_count_null(train,'Fare')

hist_fare = sns.FacetGrid(train, col='Survived',size=3)

hist_fare.map(plt.hist, 'Fare', bins=100);
describe_count_null(train,'Fare')

describe_count_null(test,'Fare')

train,test = transform_Fare(train,test)

describe_count_null(train,'Fare2')

describe_count_null(test,'Fare2')



print(train.Fare2.value_counts(),"\n")

sns.barplot(x="Fare2", y="Survived", hue="Sex", data=train);
#test.to_csv('x.csv', index = False)
describe_count_null(train,'Cabin')

train,test = transform_Cabin(train,test)

sns.barplot(x="Cabin2", y="Survived", hue="Sex", data=train);

print("After categorisation\n")

print(train.Cabin2.value_counts(),"\n")

describe_count_null(train,'Embarked')

train,test = transform_Embarked(train,test)

print("\n\n")

print(train.Embarked.value_counts(),"\n")



sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train);
# Here is the list of all the columns

print(train.columns)

print(test.columns)
-1# Drop columns not deemed as useful

drop_columns = ['PassengerId', 'Name', 'Age', 'FamilyMembers', 'Ticket', 'Fare','Cabin']

#drop_columns = ['PassengerId', 'Name', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin','FamilyMembers','Cabin2','Fare2']



train_slim = train.drop(drop_columns, axis=1)

test_slim = test.drop(drop_columns, axis=1)

printx(train_slim.head())

print ("---")

printx(test_slim.head())

#test_slim.to_csv('test_slim.csv', index = False)

#train_slim.to_csv('train_slim.csv', index = False)
# Lets define some more functions to Encode label, Normalisation, 



def label_encoder(train,test,train_test):

    for feature in train.columns:

        #print(feature)

        le = preprocessing.LabelEncoder()

        le = le.fit(train_test[feature])

        train[feature] = le.transform(train[feature])

        test[feature] = le.transform(test[feature])

        

    train = pd.DataFrame(train, columns = train.columns)

    test = pd.DataFrame(test, columns = test.columns)

    return train,test



# Normalisation and standardisation

# Normalization is mainly necessary in case of algorithms which use distance measures 

# like clustering,recommender systems which use cosine similarity etc.



def data_standardisation(train,test):

    #std_scale = preprocessing.StandardScaler().fit(train)

    std_scale = preprocessing.RobustScaler().fit(train)

    train_std = std_scale.transform(train)

    test_std = std_scale.transform(test)

    return train_std,test_std


# Lets drop the label from the training data

X_all = train_slim.drop(['Survived'], axis=1)

Y_all = train_slim['Survived']



# create a merge dataset for label Encoder

train_test = X_all.append(test_slim)



# Encode variables

X_all,test_slim = label_encoder(X_all,test_slim,train_test)



# Standardize data

X_all_std,test_slim_std = data_standardisation(X_all,test_slim)
# Create a list of models and run the prediction through all models in Kfold cross validation

def evaluate_models(train,test,split_num=5,random_num=7):

    models = []

    models.append(('LR', LogisticRegression()))

    models.append(('LDA', LinearDiscriminantAnalysis()))

    models.append(('KNN', KNeighborsClassifier(n_neighbors = 3)))

    models.append(('CART', DecisionTreeClassifier()))

    models.append(('RF', RandomForestClassifier()))

    models.append(('NB', GaussianNB()))

    models.append(('SVM', SVC()))

    models.append(('GBC',GradientBoostingClassifier()))

    models.append(('ABC',AdaBoostClassifier()))

    #print(models)



    # evaluate each model in turn

    results = []

    names = []

    scoring = 'accuracy'

    for name, model in models:

        kfold = KFold(n_splits=split_num, random_state=random_num)

        cv_results = cross_val_score(model,train, test, cv=kfold, scoring=scoring)

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

        

def ensembl_voting(train,test,split_num=5,random_num=7):       

    # Lets Check the voting algorithm

    models = []

    models.append(('LR', LogisticRegression()))

    models.append(('LDA', LinearDiscriminantAnalysis()))

    models.append(('KNN', KNeighborsClassifier()))

    models.append(('CART', DecisionTreeClassifier()))

    models.append(('RF', RandomForestClassifier()))

    models.append(('NB', GaussianNB()))

    models.append(('SVM', SVC()))



    kfold = KFold(n_splits=10, random_state=7)

    ensemble = VotingClassifier(models)

    results = cross_val_score(ensemble, train, test, cv=kfold)

    print(results.mean())
# Lets do the prediction



print("\nBefore standardisation")

evaluate_models(X_all,Y_all,split_num=5,random_num=7)



print("\nAfter standardisation")

evaluate_models(X_all_std,Y_all,split_num=5,random_num=7)
# Final prediction using gradient boosting classifier

#random_forest = RandomForestClassifier()

GBC = GradientBoostingClassifier()

GBC.fit(X_all_std, Y_all)

test_pred = GBC.predict(test_slim_std)

output = pd.DataFrame({ 'PassengerId' : test_df.PassengerId, 'Survived': test_pred })

output.head()

output.to_csv('titanic-predictions_basic-1.csv', index = False)



output = pd.DataFrame({ 'Column' : X_all.columns, 'Score': GBC.feature_importances_ })

print("\nFeature importance using GBC is \n",output)
# quick look at decision tree at depth 3

# The tree graph will make moire sense when given categorical variables.

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_all, Y_all)





import pydotplus as pydot



from IPython.display import Image



from sklearn.externals.six import StringIO

from sklearn import tree



dot_data = StringIO()



tree.export_graphviz(decision_tree, out_file=dot_data,max_depth=3,feature_names=X_all.columns)



graph = pydot.graph_from_dot_data(dot_data.getvalue())



Image(graph.create_png())
# Extra looking for feature importance



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_selection import RFE

from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesClassifier



def explore_feature_imp(train,label):

    test = SelectKBest(score_func=chi2, k=4)

    fit = test.fit(train,label)

    output = pd.DataFrame({ 'Column' : train.columns, 'Score': fit.scores_ })

    print("\nScoring using Chi squre function is \n",output)

    

    model = LogisticRegression()

    rfe = RFE(model, 3)

    fit = rfe.fit(train,label)

    output = pd.DataFrame({ 'Column' : train.columns, 'Score': fit.ranking_ })

    print("\nScoring using Recursive feature elimination (regression) is \n",output)



    model = ExtraTreesClassifier()

    model.fit(train, label)

    output = pd.DataFrame({ 'Column' : train.columns, 'Score': model.feature_importances_ })

    print("\nScoring using Extra tree classifier is \n",output)



explore_feature_imp(X_all +10,Y_all)