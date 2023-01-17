# Settings

from IPython.core.interactiveshell import InteractiveShell  

InteractiveShell.ast_node_interactivity = "all"
#imports

# os 

import os



# data analysis and wrangling

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random 



# visualization and reporting

import pandas_profiling # dataframe profiling @

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

#import calmap





# machine learning

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# combination'

from itertools import combinations
# walk file structure and find input data

input_files=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path=os.path.join(dirname, filename)

        if path.endswith('submission.csv'):

            path_gs=path

        elif path.endswith('test.csv'):

            path_test=path

        elif path.endswith('train.csv'):

            path_train=path

print(path_gs,path_test,path_train)            
# create a dataframe for each csv file

test_df=pd.read_csv(path_test)

train_df=pd.read_csv(path_train)



combo=[test_df,train_df]
# display

if False: # test report

    test_df.head()

    test_df.info()

    test_df.describe(include='all')

if True: # Train report

    train_df.head()

    train_df.info()

    #train_df.describe()

    train_df.describe(include='all')
#heat map

sns.heatmap(train_df.isnull(), cbar=False)
#profile report inline



if True: # Train Data report generation

    profile_train = train_df.profile_report(title='Titanic train data') 

    #profile_train.to_file(output_file="kaggle/profile_report/Titanic_train_data.html") # save the report 

    profile_train # display inline

if True: # Test Data report generation

    profile_test = test_df.profile_report(title='Titanic test data')

    #profile_test.to_file(output_file="kaggle/profile_report/Titanic_test_data.html") # save the report 

    profile_test # in order to display inline
train_df['Surname'] = train_df.Name.str.split(',').str[0]

train_df.head()
# extract Title

for df in combo:

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Surname'] = df.Name.str.split(',').str[0]

    

#display    

train_df[['Title','Survived']].groupby(['Title'],as_index=True).mean().sort_values(by='Survived',ascending=False).style.background_gradient(cmap='Reds')   
for df in combo:

    #Rare Titles

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    

    #Miss, Mlle, Ms,Mme and so on

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    

#display    

train_df[['Title','Survived']].groupby(['Title'],as_index=True).mean().sort_values(by='Survived',ascending=False).style.background_gradient(cmap='Reds')  
for df in combo:

    df['FamilySize'] = df.SibSp + df.Parch + 1   

    #df.drop(['SibSp','Parch'],axis=True,inplace=True)
train_df[['FamilySize','Survived']].groupby('FamilySize',as_index=True).mean().sort_values('FamilySize',ascending=True).style.background_gradient(cmap='Reds')
title_mapping = {1: "alone",2: "small" ,3:"small",4: "medium",5: "large",6:"large",7:"large",8:"large",9:"large",10:"large",11: "large"}

#title_mapping = {1: "alone",2: "family" ,3:"family",4: "family",5: "family",6:"family",7:"family",8:"family",9:"family",10:"family",11: "family"}



for df in combo:

    df['FamilySize'] = df['FamilySize'].map(title_mapping)   

train_df[['FamilySize','Survived']].groupby('FamilySize',as_index=True).mean().sort_values('Survived',ascending=True).style.background_gradient(cmap='Reds')
test_df.loc[test_df.Fare.isna()]
median_train=train_df.loc[(train_df.Pclass==3)&

             (train_df.Sex=='male')&

             (train_df.Embarked=='S')]['Fare'].median()

median_train
# replace missing value by median considering known information

test_df.loc[test_df.Fare.isna(),'Fare']=median_train
#check for more missing value

test_df.loc[test_df.Fare.isna()]['Fare'].sum()
#convert float to int

for df in combo:

    df['Fare'] = df['Fare'].astype(int)
for df in combo:

    df['Pclass'] = df['Pclass'].astype(str)

    

train_df[['Pclass','Survived']].groupby('Pclass',as_index=True).mean().sort_values('Survived',ascending=True).style.background_gradient(cmap='Reds')
train_df.loc[train_df.Embarked.isna()]

test_df.loc[test_df.Embarked.isna()]
train_df[['Pclass','Fare','Embarked']].dropna().groupby(['Pclass','Embarked'],as_index=True).describe()
Fare_price=train_df.loc[train_df.PassengerId==62,'Fare'].values[0]

ax=train_df[['Pclass','Fare','Embarked']].dropna().boxplot(column='Fare',by=['Embarked','Pclass'],figsize=(12,6))

ax.axhline(y=Fare_price, color="red",linewidth=2,linestyle='dotted')
#fill missing 

train_df.loc[train_df.Embarked.isna(),'Embarked']='C'
#check

train_df.loc[train_df.Embarked.isna()]
for df in combo:

    # get average, std, and number of NaN values in train_df

    average_age_df   = df["Age"].mean()

    std_age_df       = df["Age"].std()

    count_nan_age_df = df["Age"].isnull().sum()



    # generate random numbers between (mean - std) & (mean + std)

    rand = np.random.randint(average_age_df - std_age_df, average_age_df + std_age_df, size = count_nan_age_df)



    # fill NaN values in Age column with random values generated

    df.loc[df.Age.isnull(),'Age'] = rand



    # convert from float to int

    df['Age'] = df['Age'].astype(int)

#plot old vs new overlap grid 2,1

if True:

    fig, axs = plt.subplots(2, 1,figsize=(12,10))

    axs[0].set_title('\n Age - Train \n')

    axs[1].set_title('\n Age - Test \n')





    # plot Old Age Values

    train_df['Age'].dropna().astype(int).hist(bins=70,width=1, ax=axs[0],label="old")

    test_df['Age'].dropna().astype(int).hist(bins=70,width=1, ax=axs[1],label="old")



    # plot new Age Values

    train_df['Age'].hist(bins=70,width=0.5, ax=axs[0],label="new")

    test_df['Age'].hist(bins=70,width=0.5, ax=axs[1],label="new")

    axs[1].legend()

    axs[0].legend()
#plot old vs new grid 2x2

if False:

    fig, axs = plt.subplots(2, 2,figsize=(15,8))

    axs[0, 0].set_title(' Age - Train')

    axs[0, 1].set_title('New Age values - Train')

    axs[1, 0].set_title(' Age - Test')

    axs[1, 1].set_title('New Age values - Test')



    # plot Old Age Values

    train_df['Age'].dropna().astype(int).hist(bins=50, ax=axs[0, 0])

    test_df['Age'].dropna().astype(int).hist(bins=50, ax=axs[1, 0])



    # plot new Age Values

    train_df['Age'].hist(bins=50, ax=axs[0, 1])

    test_df['Age'].hist(bins=50, ax=axs[1, 1])

    #train_df['Age'].hist(bins=50, ax=axs[0, 0])

    #test_df['Age'].hist(bins=50, ax=axs[1, 0])
#check if anymore missing values

sns.heatmap(test_df.isnull(), cbar=False)
Age_adulthood=16

for df in combo:

    df['isChild'] = 'yes'

    df.loc[df['Age']>=Age_adulthood,'isChild']='no'
train_df[['isChild', 'Survived']].groupby(['isChild'], as_index=True).mean().sort_values(by='Survived',ascending=False).style.background_gradient(cmap='Reds')
#Motherhood

train_df.loc[(train_df.Sex=='male')&

             (train_df.Parch!=0)&

             (train_df.Title=='Mr')].head()
for df in combo:

    df['isParent']='no'

    df.loc[(df.Sex=='female') & (df.Parch!=0) & (df.Title=='Mrs'),'isParent']='mother'

    df.loc[(df.Sex=='male') & (df.Parch!=0) & (df.Title=='Mr'),'isParent']='father'
# for categorical data only

#pd.crosstab(train_df.isMother, train_df.Survived)
train_df[['isParent', 'Survived']].groupby(['isParent'], as_index=True).mean().sort_values(by='Survived',ascending=False).style.background_gradient(cmap='Reds')
for df in combo:

    df.loc[:,'Person']=df.loc[:,'Sex']

    df.loc[df.isChild=='yes','Person']='Child'

    #df.loc[df.isParent=='mother','Person']='Mother'

    #df.loc[df.isParent=='father','Person']='father'
train_df[['Person', 'Survived']].groupby(['Person'], as_index=True).mean().sort_values(by='Survived',ascending=False).style.background_gradient(cmap='Reds')
for df in combo:

    df.drop(['Ticket','Cabin'],axis=1,inplace=True)
for df in combo:



    df.drop(['Name'],axis=1,inplace=True)
#check

sns.heatmap(train_df.isnull(), cbar=False)



#sns.heatmap(test_df.isnull(), cbar=False)
# all possible feature

Possible_features=train_df.columns.to_list()

Possible_features.remove('PassengerId')

Possible_features.remove('Survived')

print(Possible_features)
# remove feature

# orginal

#Possible_features.remove('Age')

#Possible_features.remove('Fare')

#Possible_features.remove('Embarked')

#Possible_features.remove('Sex')

#Possible_features.remove('SibSp')

#Possible_features.remove('Parch')



# new

#Possible_features.remove('FamilySize')

#Possible_features.remove('Title')

Possible_features.remove('Surname')

Possible_features.remove('isParent')

Possible_features.remove('isChild')

#Possible_features.remove('Person')



features=Possible_features

print(features)
feature_combi = sum([list(map(list, combinations(Possible_features, i))) for i in range(len(Possible_features) + 1)], [])
len(feature_combi)
ml_results=pd.DataFrame(columns=['feature_set','Ml_score'])
i=0

for feature_set in feature_combi:

    if len(feature_set)>=2:

        

        #prepare data

        Y_train  = train_df["Survived"]

        X_train  = pd.get_dummies(train_df[feature_set])

        X_test = pd.get_dummies(test_df[feature_set])

        

        #compute model

        model = RandomForestClassifier(n_estimators=100)

        model.fit(X_train, Y_train)

        predictions = model.predict(X_test)

        

        #Determine Score 

        score=round(model.score(X_train, Y_train)*100,2)

        

        #update ml result

        ml_results.loc[i]=feature_set,score

        i+=1
best_score=ml_results.loc[ml_results.Ml_score==ml_results.Ml_score.max()]

best_score
feature_Set_small = best_score.loc[ best_score.index.min(),'feature_set']

feature_Set_small
Y_train  = train_df["Survived"]

X_train  = pd.get_dummies(train_df[feature_Set_small])

X_test = pd.get_dummies(test_df[feature_Set_small])



model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, Y_train)

predictions_small_set = model.predict(X_test)



# importance

variable_importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_*100})

variable_importance['feature']=variable_importance['feature'].str.split("_", n = 1, expand = True) 

variable_importance=variable_importance.groupby('feature',as_index=True).sum().sort_values(by='importance', ascending=True)

variable_importance.plot.barh()

variable_importance.sort_values(by='importance', ascending=False).style.background_gradient(cmap='Greens')
feature_Set_big = best_score.loc[ best_score.index.max(),'feature_set']

feature_Set_big
Y_train  = train_df["Survived"]

X_train  = pd.get_dummies(train_df[feature_Set_big])

X_test = pd.get_dummies(test_df[feature_Set_big])



model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, Y_train)

predictions_big_set = model.predict(X_test)



# importance

variable_importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_*100})

variable_importance['feature']=variable_importance['feature'].str.split("_", n = 1, expand = True) 

variable_importance=variable_importance.groupby('feature',as_index=True).sum().sort_values(by='importance', ascending=True)

variable_importance.plot.barh()

variable_importance.sort_values(by='importance', ascending=False).style.background_gradient(cmap='Greens')
ml_model_results=pd.DataFrame(index=range(9),columns=['N','Name'])

Model_name_short=['LR','KNN','SVM','GNB','DT','RF','P','SVC','SGD']

Model_name_long=['Logistic Regression','k-Nearest Neighbors','Support Vector Machines',

                 'Gaussian Naive Bayes classifier','Decision Tree','Random Forrest','Perceptron',

                 'Linear SVC','Stochastic Gradient Descent']



ml_model_results.loc[:,'N']=Model_name_short

ml_model_results.loc[:,'Name']=Model_name_long

model_list = [LogisticRegression(solver ='liblinear'),

              KNeighborsClassifier(n_neighbors = 3),

              SVC(gamma='auto'),

              GaussianNB(),

              DecisionTreeClassifier(),

              RandomForestClassifier(n_estimators=100),

              Perceptron(),

              LinearSVC(max_iter=10000),

              SGDClassifier()         

             ]



feature_sets=[feature_Set_small,feature_Set_big]
j=0

for feature_set in feature_sets:

    j+=1

    colname='Model_score_'+str(j)

    ml_model_results[colname]=np.nan

    #small featureset

    X_train  = pd.get_dummies(train_df[feature_set])

    X_test = pd.get_dummies(test_df[feature_set])

    i=0

    for model in model_list:

        mod=model

        mod.fit(X_train, Y_train)

        prediction = mod.predict(X_test)

        score = round(mod.score(X_train, Y_train), 4)

        ml_model_results.loc[i,colname]=float(score)

        i+=1

    ml_model_results[colname]=ml_model_results[colname].astype(float)

  
#display best result

ml_model_results.style.background_gradient(cmap='Greens')
output_small_set = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions_small_set})

output_big_set = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions_big_set})

#output.set_index('PassengerId',inplace=True)

#output.head()
output_small_set.to_csv('small_set_submission.csv', index=False)

output_big_set.to_csv('big_set_submission.csv', index=False)

#output.to_csv('my_ml_submission.csv', index=False)
