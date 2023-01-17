# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# data manipulation

import pandas as pd

import numpy as np

import random as rnd

import re



from collections import Counter



# visualization

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

%matplotlib inline

mpl.style.use( 'ggplot' )

sns.set_style( 'white' )

pylab.rcParams[ 'figure.figsize' ] = 8 , 6



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



# Modelling Algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

import xgboost as xgb



# Modelling Helpers

from sklearn.metrics import make_scorer, accuracy_score, log_loss, confusion_matrix

from sklearn.preprocessing import Imputer, Normalizer, scale, OneHotEncoder, LabelEncoder

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, KFold, StratifiedKFold, GridSearchCV, learning_curve

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.feature_selection import RFECV



# Load Data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

# combine these datasets to run certain operations on both datasets together

full_data = [train_df, test_df]



# preview the data

train_df.head()
# Data info

train_df.info()

print('_'*40)

test_df.info()



# Distribution of numerical feature values

train_df.describe()

# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.

# Review Parch distribution using `percentiles=[.75, .8]`

# SibSp distribution `[.68, .69]`

# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`



# Distribution of categorical features

train_df.describe(include=['O'])
def check_missing_data(df):

    flag=df.isna().sum().any()

    if flag==True:

        total = df.isnull().sum()

        percent = (df.isnull().sum())/(df.isnull().count()*100)

        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

        data_type = []

        # written by MJ Bahmani

        for col in df.columns:

            dtype = str(df[col].dtype)

            data_type.append(dtype)

        output['Types'] = data_type

        return(np.transpose(output))

    else:

        return(False)



    

# Fill empty and NaNs values with NaN

train_df = train_df.fillna(np.nan)

test_df = test_df.fillna(np.nan)



check_missing_data(train_df)

#check_missing_data(test_df)



# remove rows that have NA's

#train_df = train_df.dropna()
# Outlier detection 

def detect_outliers(df, n, features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than n outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp, Parch and Fare

Outliers_to_drop = detect_outliers(train_df, 2, ["Age","SibSp","Parch","Fare"])

# Show the outliers rows

train_df.loc[Outliers_to_drop]

# Drop outliers

# train_df = train_df.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

# full_data = [train_df, test_df]
# correlation with target

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# plot helper functions

def plot_histograms( df , variables , n_rows , n_cols ):

    fig = plt.figure( figsize = ( 16 , 12 ) )

    for i, var_name in enumerate( variables ):

        ax=fig.add_subplot( n_rows , n_cols , i+1 )

        df[ var_name ].hist( bins=10 , ax=ax )

        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")

        ax.set_xticklabels( [] , visible=False )

        ax.set_yticklabels( [] , visible=False )

    fig.tight_layout()  # Improves appearance a bit.

    plt.show()



def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()



def plot_categories( df , cat , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , row = row , col = col )

    facet.map( sns.barplot , cat , target )

    facet.add_legend()



def plot_correlation_map( df ):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )



def describe_more( df ):

    var = [] ; l = [] ; t = []

    for x in df:

        var.append( x )

        l.append( len( pd.value_counts( df[ x ] ) ) )

        t.append( df[ x ].dtypes )

    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )

    levels.sort_values( by = 'Levels' , inplace = True )

    return levels



def plot_variable_importance( X , y ):

    tree = DecisionTreeClassifier( random_state = 99 )

    tree.fit( X , y )

    plot_model_var_imp( tree , X , y )

    

def plot_model_var_imp( model , X , y ):

    imp = pd.DataFrame( 

        model.feature_importances_  , 

        columns = [ 'Importance' ] , 

        index = X.columns 

    )

    imp = imp.sort_values( [ 'Importance' ] , ascending = True )

    imp[ : 10 ].plot( kind = 'barh' )

    print (model.score( X , y ))

# visualization

# plot_distribution(train_df, var = 'Age', target = 'Survived', row = 'Sex')

# plot_categories(train_df, cat = 'Embarked', target = 'Survived')

plot_correlation_map(train_df)



# scatterplot: to identify the type of relationship (if any) between two quantitative variables

# g = sns.FacetGrid(train_df, hue="Survived", col="Pclass", margin_titles=True, palette={1:"seagreen", 0:"gray"})

# g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();



# boxplot: depicting groups of numerical data through their quartiles

# ax= sns.boxplot(x="Pclass", y="Age", data=train_df)

# ax= sns.stripplot(x="Pclass", y="Age", data=train_df, jitter=True, edgecolor="gray")

# plt.show()



# pairplot

# g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',

#        u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

# g.set(xticklabels=[])



# histogram

# g = sns.FacetGrid(train_df, col='Survived')

# g.map(plt.hist, 'Age', bins=20)



# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

# grid.map(plt.hist, 'Age', alpha=.5, bins=20)

# grid.add_legend();



# pointplot

# grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

# grid.add_legend()



# barplot

# grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

# grid.add_legend()



# other plots

# train_df['Age'].hist(bins=70)

# sns.factorplot('Embarked','Survived', data=train_df,size=4,aspect=3)



# peaks for survived/not survived passengers by their age

# facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)

# facet.map(sns.kdeplot,'Age',shade= True)

# facet.set(xlim=(0, train_df['Age'].max()))

# facet.add_legend()



# fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

# sns.factorplot('Embarked',data=train_df,kind='count',order=['S','C','Q'],ax=axis1)

# sns.factorplot('Survived',hue="Embarked",data=train_df,kind='count',order=[1,0],ax=axis2)

# sns.countplot(x='Embarked', data=train_df, ax=axis1)

# sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0], ax=axis2)



# group by embarked, and get the mean for survived passengers for each value in Embarked

# embark_perc = train_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

# sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)



# Multivariate plots

# scatter plot matrix

# pd.plotting.scatter_matrix(train_df,figsize=(10,10))

# plt.figure();



# violinplots on petal-length for each species

# f,ax=plt.subplots(1,2,figsize=(18,8))

# sns.violinplot("Pclass","Age", hue="Survived", data=train_df,split=True,ax=ax[0])

# ax[0].set_title('Pclass and Age vs Survived')

# ax[0].set_yticks(range(0,110,10))

# sns.violinplot("Sex","Age", hue="Survived", data=train_df,split=True,ax=ax[1])

# ax[1].set_title('Sex and Age vs Survived')

# ax[1].set_yticks(range(0,110,10))

# plt.show()



# kdeplot

# sns.FacetGrid(train_df, hue="Survived", size=5).map(sns.kdeplot, "Fare").add_legend()

# plt.show();



# jointplot

# sns.jointplot(x='Fare',y='Age' ,data=train_df, kind='reg');



# Swarm plot

# sns.swarmplot(x='Pclass',y='Age',data=train_df);



# Heatmap

# plt.figure(figsize=(7,4)) 

# sns.heatmap(train_df.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())

# plt.show();



# distplot

# f,ax=plt.subplots(1,3,figsize=(20,8))

# sns.distplot(train_df[train_df['Pclass']==1].Fare,ax=ax[0])

# ax[0].set_title('Fares in Pclass 1')

# sns.distplot(train_df[train_df['Pclass']==2].Fare,ax=ax[1])

# ax[1].set_title('Fares in Pclass 2')

# sns.distplot(train_df[train_df['Pclass']==3].Fare,ax=ax[2])

# ax[2].set_title('Fares in Pclass 3')

# plt.show()
# drop the ticket and cabin features

print("Before", train_df.shape, test_df.shape, full_data[0].shape, full_data[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

full_data = [train_df, test_df]

print("After", train_df.shape, test_df.shape, full_data[0].shape, full_data[1].shape)



# extract Title as a new feature

for dataset in full_data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#title[ 'Title' ] = full_data[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )



# replace many titles with a more common name or classify them as Rare

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()



# convert the categorical titles to ordinal.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in full_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

# drop name feature

train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

full_data = [train_df, test_df]

train_df.shape, test_df.shape



# sex feature to numerical values

for dataset in full_data:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# sex = pd.Series( np.where( full_data.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.

# So, we can classify passengers as males, females, and child

def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

train_df['Person'] = train_df[['Age','Sex']].apply(get_person,axis=1)

test_df['Person'] = test_df[['Age','Sex']].apply(get_person,axis=1)



# No need to use Sex column since we created Person column

# train_df.drop(['Sex'],axis=1,inplace=True)

# test_df.drop(['Sex'],axis=1,inplace=True)



# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers

person_dummies_titanic  = pd.get_dummies(train_df['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(test_df['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



train_df = train_df.join(person_dummies_titanic)

test_df = test_df.join(person_dummies_test)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



# sns.factorplot('Person',data=train_df,kind='count',ax=axis1)

sns.countplot(x='Person', data=train_df, ax=axis1)



# average of survived for each Person(male, female, or child)

person_perc = train_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])



train_df.drop(['Person'],axis=1,inplace=True)

test_df.drop(['Person'],axis=1,inplace=True)

full_data = [train_df, test_df]
# Age feature

guess_ages = np.zeros((2,3))

guess_ages



# Fill missing values of Age with the average of Age (mean)

#imputed = pd.DataFrame()

#imputed[ 'Age' ] = full_data.Age.fillna( full.Age.mean() )



for dataset in full_data:

# fill with random value

#     age_avg = dataset['Age'].mean()

#     age_std = dataset['Age'].std()

#     age_null_count = dataset['Age'].isnull().sum()

#     age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

#     dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)



for dataset in full_data:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']



train_df = train_df.drop(['AgeBand', 'Sex'], axis=1)

test_df.drop(['Sex'],axis=1,inplace=True)



full_data = [train_df, test_df]



for dataset in full_data:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

train_df.head()
# New feature based on Parch and SibSp 

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)



# introducing other features based on the family size

for dataset in full_data:

    dataset[ 'Family_Single' ] = dataset[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )

    dataset[ 'Family_Small' ]  = dataset[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )

    dataset[ 'Family_Large' ]  = dataset[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )



train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)



# for dataset in full_data:

#     dataset['IsAlone'] = 0

#     dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

# train_df = train_df.drop(['FamilySize'], axis=1)

# test_df = test_df.drop(['FamilySize'], axis=1)



full_data = [train_df, test_df]



train_df.head()
# Embarked feature

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)



for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
# Fare feature

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)



for dataset in full_data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

full_data = [train_df, test_df]

    

train_df.head()
# The preprocessing phase is to normalize labels. The LabelEncoder in Scikit-learn will convert 

# each unique string value into a number, making out data more flexible for various algorithms.



# def encode_features(df_train, df_test):

#     features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']

#     df_combined = pd.concat([df_train[features], df_test[features]])

    

#     for feature in features:

#         le = LabelEncoder()

#         le = le.fit(df_combined[feature])

#         df_train[feature] = le.transform(df_train[feature])

#         df_test[feature] = le.transform(df_test[feature])

#     return df_train, df_test

    

# data_train, data_test = encode_features(train_df, test_df)

# data_train.head()
train, test = train_test_split(train_df, test_size = 0.3, random_state = 0)

split_train_X = train.drop("Survived", axis=1)

split_train_Y = train["Survived"]

split_test_X = test.drop("Survived", axis=1)

split_test_Y = test["Survived"]
logreg = LogisticRegression()

logreg.fit(split_train_X, split_train_Y)

logreg.score(split_test_X, split_test_Y)
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
svc = SVC(kernel='linear')

# svc.fit(split_train_X, split_train_Y)

# svc.score(split_test_X, split_test_Y)



# Confusion Matrix: gives the number of correct and incorrect classifications made by the classifier.

X = train_df[train_df.columns[1:]]

Y = train_df['Survived']

_ , ax = plt.subplots(figsize =(12 , 10))

y_pred = cross_val_predict(svc, X, Y, cv = 10)

sns.heatmap(confusion_matrix(Y, y_pred), ax = ax, annot = True, fmt = '2.0f')
a_index = list(range(1,11))

a = pd.Series()

x = [0,1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model = KNeighborsClassifier(n_neighbors = i) 

    model.fit(split_train_X, split_train_Y)

    a = a.append(pd.Series(model.score(split_test_X, split_test_Y)))

plt.plot(a_index, a)

plt.xticks(x)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()

print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())
gaussian = GaussianNB()

gaussian.fit(split_train_X, split_train_Y)

gaussian.score(split_test_X, split_test_Y)
perceptron = Perceptron()

perceptron.fit(split_train_X, split_train_Y)

perceptron.score(split_test_X, split_test_Y)
decision_tree = DecisionTreeClassifier()

decision_tree.fit(split_train_X, split_train_Y)

decision_tree.score(split_test_X, split_test_Y)
# Hyper-Parameters Tuning

random_forest = RandomForestClassifier(n_estimators=100)

# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }

# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

kfold = StratifiedKFold(n_splits=10)

grid_obj = GridSearchCV(random_forest, parameters, cv=kfold, scoring=acc_scorer)

grid_obj.fit(X, Y)



# Set the clf to the best combination of parameters

random_forest = grid_obj.best_estimator_

grid_obj.best_score_
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



g = plot_learning_curve(random_forest,"random_forest learning curves", X, Y)
# Feature importance

fig, axes = plt.subplots(1, 1, sharex="all", figsize=(15,15))



name = "RandomForest"

indices = np.argsort(random_forest.feature_importances_)[::-1][:40]

g = sns.barplot(y = train_df.columns[indices][:40],x = random_forest.feature_importances_[indices][:40] , orient='h')

g.set_xlabel("Relative importance",fontsize=12)

g.set_ylabel("Features",fontsize=12)

g.tick_params(labelsize=9)

g.set_title("RandomForest feature importance")
train_df_X = train_df.drop("Survived", axis=1)

train_df_Y = train_df["Survived"]

test_df_X  = test_df.drop("PassengerId", axis=1).copy()

train_df_X.shape, train_df_Y.shape, test_df_X.shape

# plot_variable_importance(train_X, train_Y)

train_X = train_df_X.values

train_Y = train_df_Y.values

test_X = test_df_X.values



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]



kfold = StratifiedKFold(n_splits=10, random_state=0)

cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, train_X, y = train_Y, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")



# acc_dict = {}

# for train_index, test_index in kfold.split(train_X, train_Y):

#     split_train_X, split_test_X = train_X[train_index], train_X[test_index]

#     split_train_Y, split_test_Y = train_Y[train_index], train_Y[test_index]



#     for model in classifiers:

#         name = model.__class__.__name__

#         model.fit(split_train_X, split_train_Y)

#         #plot_model_var_imp(model, split_train_X, split_train_Y)

#         train_predictions = model.predict(split_test_X)

#         acc = accuracy_score(split_test_Y, train_predictions)

#         if name in acc_dict:

#             acc_dict[name] += acc

#         else:

#             acc_dict[name] = acc



# log_cols = ["Classifier", "Accuracy"]

# log = pd.DataFrame(columns=log_cols)

# for model in acc_dict:

#     acc_dict[model] = acc_dict[model] / 11.0

#     log_entry = pd.DataFrame([[model, acc_dict[model]]], columns=log_cols)

#     log = log.append(log_entry)



# plt.xlabel('Accuracy')

# plt.title('Classifier Accuracy')

# sns.set_color_codes("muted")

# sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier



# Voting Classifier

ensemble_lin_rbf = VotingClassifier(estimators=[('KNN', KNeighborsClassifier(n_neighbors=10)),

                                              ('RBF', SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),

                                              ('RFor', RandomForestClassifier(n_estimators=500,random_state=0)),

                                              ('LR', LogisticRegression(C=0.05)),

                                              ('DT', DecisionTreeClassifier(random_state=0)),

                                              ('NB', GaussianNB()),

                                              ('svm', SVC(kernel='linear',probability=True))

                                             ], 

                       voting='soft').fit(split_train_X, split_train_Y)

print('The accuracy for ensembled model is:', ensemble_lin_rbf.score(split_test_X, split_test_Y))

cross = cross_val_score(ensemble_lin_rbf, X, Y, cv = 10, scoring = "accuracy")

print('The cross validated score is', cross.mean())



# Bagged DecisionTree

model = BaggingClassifier(base_estimator = DecisionTreeClassifier(), random_state = 0, n_estimators = 100)

model.fit(split_train_X, split_train_Y)

prediction = model.predict(split_test_X)

print('The accuracy for bagged Decision Tree is:', accuracy_score(prediction, split_test_Y))

result = cross_val_score(model, X, Y, cv=10, scoring='accuracy')

print('The cross validated score for bagged Decision Tree is:', result.mean())



# AdaBoost (Adaptive Boosting)

ada = AdaBoostClassifier(n_estimators = 200, random_state = 0, learning_rate = 0.1)

result = cross_val_score(ada,X, Y, cv = 10, scoring='accuracy')

print('The cross validated score for AdaBoost is:', result.mean())



# Hyper-Parameter Tuning for AdaBoost

# n_estimators = list(range(100,1100,100))

# learn_rate = [0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]

# hyper = {'n_estimators':n_estimators, 'learning_rate':learn_rate}

# gd = GridSearchCV(estimator = AdaBoostClassifier(), param_grid = hyper,verbose = True)

# gd.fit(X,Y)

# print(gd.best_score_)

# print(gd.best_estimator_)



# Confusion Matrix for the Best Model

ada = AdaBoostClassifier(n_estimators = 200, random_state = 0, learning_rate = 0.05)

result = cross_val_predict(ada, X, Y, cv = 10)

sns.heatmap(confusion_matrix(Y,result), cmap='winter', annot=True, fmt='2.0f')

plt.show()
# Helper function for Stacking

ntrain = train_X.shape[0]

ntest = test_X.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state = SEED)



# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)

        

# Out-of-Fold prediction function        

def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    fold = 0

    for (train_index, test_index) in kf.split(x_train, y_train):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[fold, :] = clf.predict(x_test)

        fold += 1



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Parameters of the listed classifiers

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    #'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

}



# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}



# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'linear',

    'C' : 0.025

    }



# Create 5 objects that represent our 4 models

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)



# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, train_X, train_Y, test_X) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf, train_X, train_Y, test_X) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, train_X, train_Y, test_X) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb, train_X, train_Y, test_X) # Gradient Boost

svc_oof_train, svc_oof_test = get_oof(svc, train_X, train_Y, test_X) # Support Vector Classifier



print("Training is complete")

rf_feature = rf.feature_importances(train_X, train_Y)

et_feature = et.feature_importances(train_X, train_Y)

ada_feature = ada.feature_importances(train_X, train_Y)

gb_feature = gb.feature_importances(train_X, train_Y)
# Second-Level Predictions from the First-level Output

base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

    })

base_predictions_train.head()



# Correlation Heatmap of the Second Level Training set

data = [

    go.Heatmap(

        z= base_predictions_train.astype(float).corr().values ,

        x=base_predictions_train.columns.values,

        y= base_predictions_train.columns.values,

          colorscale='Viridis',

            showscale=True,

            reversescale = True

    )

]

py.iplot(data, filename='labelled-heatmap')
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
# Having now concatenated and joined both the first-level train and test predictions as x_train and x_test, 

# we can now fit a second-level learning model.

gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(train_X, train_Y)



predictions = gbm.predict(test_X)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": predictions

    })

submission.to_csv('submission.csv', index=False)