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
import matplotlib.pyplot as plt

import seaborn as sns

## Display all the columns of the dataframe

pd.pandas.set_option('display.max_columns',None)
"""Machine learning models."""



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.svm import SVC

from xgboost import XGBClassifier



"""Classification (evaluation) metrices."""

from sklearn.metrics import accuracy_score, r2_score, f1_score,  classification_report 

from sklearn.model_selection import cross_val_score, GridSearchCV,RandomizedSearchCV, cross_val_predict
data_train=pd.read_csv("../input/titanic/train.csv")

print(data_train.shape)

data_train.head(2)
display(data_train.isnull().sum())
sns.countplot("Survived", data=data_train)

plt.ylabel("Total no of people",size=10)

plt.title("Survived vs Dead")

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(121)

sns.countplot("Sex", data=data_train)

plt.title("male vs female ")

plt.ylabel("Total no of people",size=10)



plt.subplot(122)

sns.countplot("Sex", data=data_train, hue="Survived")

plt.title("male vs female survived and death")

plt.ylabel("no of people")

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(121)

sns.countplot("Pclass", data=data_train)

plt.ylabel("Total no of people",size=10)

plt.title("Pclass Data")

plt.xticks(rotation=90)



plt.subplot(122)

sns.countplot("Pclass", data=data_train, hue="Survived")

plt.title("Pclass survived vs Dead")

plt.ylabel("no of people")

plt.show()
fig, ax = plt.subplots(1,2, figsize=(15,5) )



sns.countplot("Embarked", data=data_train, ax=ax[0])

ax[0].set_title('Embarked ', fontsize=16)

ax[0].set_ylabel('No. of People count ', fontsize=10)



sns.countplot("Embarked", data=data_train, hue="Survived")

plt.title("Embarked survived vs Dead")



plt.show()
data_train['Cabin']=data_train['Cabin'].fillna('Missing').apply(lambda x: x[0] if x!='Missing' else 'Missing')#.value_counts()
sns.countplot("Cabin", data=data_train, hue="Survived")

plt.show()
data_train["familySize"] = data_train["SibSp"]+data_train["Parch"]+1# Adding 1 for single person

#then delete [ SibSp, Parch ] 

data_train.drop(['SibSp', 'Parch'], inplace=True, axis=1)



#create 4 buckets namely single, small, medium, and large for rest of them.

data_train["familySize"] = data_train["familySize"].apply(lambda x: 'single' if x==1 else 

                                                    ('small' if (x==2 or x==3) else

                                                    ('medium' if (x==4 or x==5) else 'large' )))

sns.countplot("familySize", data=data_train, hue="Survived")

plt.show()
#lets take only calling name 

#Mr ,Miss , Mrs ,Master ----------etc

data_train['Name']=data_train['Name'].apply(lambda x: (x.split('.')[0]).split(',')[1] )



# Professionals like Dr, Rev, Col, Major, Capt will be put into 'Officer'

# Dona, Jonkheer, Countess, Sir, Lady, Don will be put into aristocrats

# Mlle and Ms with Miss and Mme by Mrs as these are French titles.

data_train["Name"] = data_train["Name"].replace(to_replace=["Dr", "Rev", "Col", "Major", "Capt"], value = "Officer", regex=True )

data_train["Name"].replace(to_replace=["Dona", "Jonkheer", "the Countess", "Sir", "Lady", "Don"], value = "Aristocrat", inplace = True,regex=True)

data_train["Name"].replace({"Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs"}, inplace = True,regex=True)

data_train['Name'].value_counts()
f, ax = plt.subplots(1,2, figsize=(15,5))



data_train['Name'].value_counts().plot.pie( autopct='%1.2f%%', ax=ax[0], shadow=True, startangle=180)

ax[0].set_title('Pie Chart Plot of Name', fontsize=16)

ax[0].legend(loc='lower left')





sns.countplot('Name', data=data_train, hue="Survived" )



plt.show()
#Assign 'N' if there is only digits in Ticket. Otherwise just get the 1st character from Ticket.

data_train['Ticket']=np.where(data_train['Ticket'].str.isdigit(), "N", data_train['Ticket'].str[0] )
sns.countplot("Ticket", data=data_train, hue="Survived")

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(121)

bar=sns.distplot(data_train.Fare)

bar.legend(['Fare skewness :: {:.2f}'.format(data_train.Fare.skew())])



plt.subplot(122)

bar=sns.distplot(data_train.Age)

bar.legend(['Age skewness :: {:.2f}'.format(data_train.Age.skew())])

plt.xticks(range(0,int(np.max(data_train.Age))+10,10))



plt.show()
plt.figure(figsize=(15,5))



plt.subplot(121)

sns.boxplot('Fare',data=data_train)

plt.title('outlier')



plt.subplot(122)

sns.boxplot(np.log(data_train['Fare']))

plt.title('with log distribution')



plt.show()
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')

submis=pd.read_csv("../input/titanic/gender_submission.csv")



#display shape of datasets

print(train.shape)

print(test.shape)

print(submis.shape)
# store Survived values from train to survived

# then drop it and display first 2 rows

survived=train['Survived']

train.drop('Survived', axis=1, inplace=True)

display(train.head(2))
##concatinate test and train for data pre-processing

# display 5 rows 

dataset=pd.concat([train,test],axis=0)

dataset=dataset.reset_index(drop=True)

#shape

display(dataset.shape)

#check null

display(dataset.isnull().sum())

dataset.head()
## Here we will check the percentage of nan values present in each feature

## 1 -step make the list of features which has missing values

features_with_na=[features for features in dataset.columns if dataset[features].isna().sum()>=1]



## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:

    print(feature, np.round(dataset[feature].isnull().mean(), 4),  ' % missing values')

    

print(f"The total feature which have missing values are :: {len(features_with_na)}")
Age_median_value=dataset['Age'].median() 

dataset['Age'].fillna(Age_median_value,inplace=True)
# As fare also contain zero (0) values,  which is not suitable values for fare , that must be NaN values

dataset['Fare'].replace(to_replace=0, value=np.nan, inplace = True,regex=True)
# ## replace Fare by using median as it have outlier

Fare_median_value=int(dataset['Fare'].median())    

dataset['Fare'].fillna(Fare_median_value,inplace=True)
#lets fill nan value with mode of Embarked column

mode_value=dataset['Embarked'].mode()[0]    

dataset['Embarked'].fillna(mode_value,inplace=True)
# #let take first char for cabin feature and escape for NaN

dataset['Cabin']=dataset['Cabin'].apply(lambda x: x[0] if type(x)==str else x )#.value_counts()
# lets fill nan with ffil and then bfill

dataset['Cabin'].fillna(method='ffill', inplace=True)

dataset['Cabin'].fillna(method='bfill', inplace=True)
# Again check for NaN

dataset[features_with_na].isnull().sum()
dataset["familySize_num"] = dataset["SibSp"]+dataset["Parch"]+1# Adding 1 for single person

#then delete [ SibSp, Parch ] 

dataset.drop(['SibSp', 'Parch'], inplace=True, axis=1)



#create 4 buckets namely single, small, medium, and large for rest of them.

dataset["familySize"] = dataset["familySize_num"].apply(lambda x: 'single' if x==1 else 

                                                    ('small' if (x==2 or x==3) else

                                                    ('medium' if (x==4 or x==5) else 'large' )))

#lets take only calling name 

#Mr ,Miss , Mrs ,Master ----------etc

dataset['Name']=dataset['Name'].apply(lambda x: (x.split('.')[0]).split(',')[1] )



# Professionals like Dr, Rev, Col, Major, Capt will be put into 'Officer'

# Dona, Jonkheer, Countess, Sir, Lady, Don will be put into aristocrats

# Mlle and Ms with Miss and Mme by Mrs as these are French titles.

dataset["Name"] = dataset["Name"].replace(to_replace=["Dr", "Rev", "Col", "Major", "Capt"], value = "Officer", regex=True )

dataset["Name"].replace(to_replace=["Dona", "Jonkheer", "the Countess", "Sir", "Lady", "Don"], value = "Aristocrat", inplace = True,regex=True)

dataset["Name"].replace({"Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs"}, inplace = True,regex=True)
plt.figure(figsize=(15,5))



plt.subplot(121)

bar=sns.distplot(dataset.Fare)

bar.legend(['Fare skewness :: {:.2f}'.format(dataset.Fare.skew())])



plt.subplot(122)

bar=sns.distplot(dataset.Age)

bar.legend(['Age skewness :: {:.2f}'.format(dataset.Age.skew())])

plt.xticks(range(0,int(np.max(dataset.Age))+10,10))



plt.show()
dataset.Fare=np.log(dataset.Fare)



plt.figure(figsize=(10,5))

bar=sns.distplot(dataset.Fare)

bar.legend(['Fare skewness :: {:.2f}'.format(dataset.Fare.skew())])



plt.show()
#lets drop PassengerId and Ticket

dataset.drop(['PassengerId','Ticket'],axis=1,inplace=True)
dataset.dtypes
dataset['Pclass']=dataset['Pclass'].astype('category') 
dataset.head()
# use one hot encoding instead of label encoding because algorithm might give weights to higher values if label encoding is 

#used to encode numeric variables.

dataset1=pd.get_dummies(dataset)

display(dataset1.shape)

dataset1.head()
scal_fet=['Age','Fare','familySize_num']
from sklearn.preprocessing import MinMaxScaler



scale=MinMaxScaler(feature_range=(0,1))



dataset1[scal_fet]=scale.fit_transform(dataset1[scal_fet])
dataset1.head()
#Let's split the train and test set 

df_train=dataset1.iloc[:891,:]

df_test=dataset1.iloc[891:,:]



X_train=df_train

X_test=df_test



y_train=survived



print(f"Train Data Dimension: {X_train.shape}")

print(f"Train output: {y_train.shape}")

print(f"Test Data Dimension: {X_test.shape}")

"""The  different  model we will try are::"""

# 1. Logistic Regression

lr = LogisticRegression()

# 2. DecisionTreeClassifier

dtree=DecisionTreeClassifier()

# 3. Random Forest Classifier

rand_forest = RandomForestClassifier()

# 4. KNN

knn=KNeighborsClassifier()

# 5. Support Vector Machines [ SV Classifier ] 

svc=SVC()

# 6. Gaussian Naive Bayes

gau_nb = GaussianNB()

# 7. Multinomial Naive Bayes

mult_nb=MultinomialNB()

# 8. Gradient Boosting Classifier

gbc = GradientBoostingClassifier()

# 9. Adaboost Classifier

abc = AdaBoostClassifier()

# 10. ExtraTrees Classifier

etc = ExtraTreesClassifier()

# 11. Extreme Gradient Boosting

xgbc = XGBClassifier()



#list of model defined above 

models = [lr, dtree, rand_forest, knn, svc, gau_nb, mult_nb, gbc,abc, etc, xgbc]

modelNames=['LogisticRegression','DecisionTreeClassifier',"RandomForestClassifier",'KNeighborsClassifier',

          'SupportVectorClassifier','GaussianNB','MultinomialNB','GradientBoostingClassifier','AdaBoostClassifier',

          'ExtraTreesClassifier','XGBClassifier']
"""Create a function that returns train accuracy of different models."""

def calculateTrainAccuracy(model):    

    model.fit(X_train,y_train)

    trainAccuracy = model.score(X_train,y_train)

    trainAccuracy = round(trainAccuracy*100, 2)

    return trainAccuracy



# Calculate train accuracy of all the models and store them in a dataframe

trainModelScores = list(map(calculateTrainAccuracy, models))

trainAccuracy_df = pd.DataFrame(trainModelScores, columns = ["trainAccuracy"], index=modelNames)

trainAccuracy_df=trainAccuracy_df.sort_values(by="trainAccuracy", ascending=False)

trainAccuracy_df
"""Create a function that returns mean cross validation score for different models."""

def calculateXValScore(model):    

    xValScore = cross_val_score(model,

                                X_train, y_train,

                                cv = 10, n_jobs =4,

                                scoring="accuracy"

                               ).mean()

    xValScore = round(xValScore*100, 2)

    return xValScore



# Calculate cross validation scores of all the models and store them in a dataframe

CVScores = list(map(calculateXValScore, models))

xValScores_df = pd.DataFrame(CVScores, columns = ["Xcross_val_score"], index=modelNames)

xValScores_df_sort = xValScores_df.sort_values(by="Xcross_val_score", ascending=False)



print("---------------Model Evaluation cross_val_score ---------------")

xValScores_df_sort
"""Define all the models hyperparameters one by one first::"""



# 1. Logistic Regression

lrParams = {"penalty":["l1", "l2"],

            "C": np.logspace(0, 4, 10),

            "max_iter":[100,500,1000,5000]}



# 2. DecisionTreeClassifier

dtreeParams = {'criterion':['gini','entropy'],

               "max_features": ["auto", "sqrt", "log2"],

               "min_samples_split": [ 2,  4,  6,  8, 10, 12, 14], 

               "min_samples_leaf":[ 1,  3,  5,  7,  9, 11],#np.arange(1,12,2)

               "random_state":[40,43,None]}



# 3. Random Forest Classifier

rand_forestParams = {"criterion":["gini","entropy"],

                     "n_estimators":[80,100, 300, 500, 800,1100,1500],

                     "min_samples_leaf":[ 2,3,4,6,8],

                     "min_samples_split":[2,3,5,7,9], #np.arange(1,12,2)

                     "max_features":["sqrt", "auto", "log2"],

                     "random_state":[40,43,44,None]}



# 4. KNN

knnParams = {"n_neighbors":[3,5,7,11],#np.arange(3,9)

             "leaf_size":[2,25,30,35,40],

             "weights":["uniform", "distance"],

             "algorithm":["auto", "ball_tree","kd_tree","brute"]}



# 5. Support Vector Machines [ SV Classifier ] 

svcParams = {"C": [0.1,1,5,10,100],

             "kernel": ["linear","rbf"],

             "gamma": [5 ,1 ,0.1,0.5, 0.001,'scale','auto'],

             'random_state':[40,43,45,None] 

            }



# 6. Gaussian Naive Bayes

gau_nbParams = {"priors":[0.5,0.8,1,None],

               "var_smoothing":[1e-09,1e-06,0.01]

                }



# 7. Multinomial Naive Bayes

mult_nbParams={"alpha":[0.1,1,10],

               "fit_prior":[True,False],

               'class_prior': [0.5,0.8,1,None]

                }



# 8. Gradient Boosting Classifier

gbcParams = {'n_estimators':[100,200,300,500,800],

             "learning_rate": [5,1, 0.1, 0.01,0.02,0.05,0.001],

             "max_depth": [4, 6, 8],

             "max_features": [1,0.8,0.5, 0.3, 0.1], 

             "min_samples_split": [ 2, 3, 4,5],

             "random_state":[40,43,45,None]}



# 9. Adaboost Classifier

abcParams = {"n_estimators":[50,70,100,200,300,500],

             "learning_rate":[ 0.02,0.01,0.2,0.1,1,5,10],

             "random_state":[40,43,45,None]}



# 10. ExtraTrees Classifier

etcParams = {"n_estimators":[10,20,30,50,100],

             "criterion":["gini","entropy"],

             "min_samples_split":[2, 3, 4],

             "min_samples_leaf":[1, 2,3 ],

             "max_depth":[2,3,5,10,None],

             "max_features":[3, 10,"auto"],

             "random_state":[40,43,45,None]}



# 11. Extreme Gradient Boosting

xgbcParams = {"n_estimators": [100, 250, 400, 550,700, 850, 1000,1200],

              "learning_rate": [0.01,0.1,0.5,0.300000012],

              'booster':[ 'gbtree', 'gblinear'],

              "max_depth": [ 3,4, 6, 10, 15],## np.arange(3,15,2)

              "min_child_weight": [1, 2, 3],

              "subsample": [0.6,0.7,0.8,0.9],

              "colsample_bytree": (0.5,0.7, 0.9,1),

              'base_score':[0.25,0.5,0.75,1],

              "random_state":[40,43,45,None]}



# list of Paramaters defined

parametersLists = [lrParams,dtreeParams,rand_forestParams,knnParams, svcParams,gau_nbParams,mult_nbParams,

                   gbcParams, abcParams, etcParams,xgbcParams]
"""Create a function to tune hyperparameters of the selected models."""

def GSCVtuneHyperParam(model, params):

    # Construct grid search object with 10 fold cross validation.

    gridSearch = GridSearchCV(estimator=model, param_grid=params, 

                              verbose=0, cv=10, n_jobs =4, 

                              return_train_score=False,)

    # Fit using grid search.

    gridSearch.fit(X_train,y_train)

    bestParams, bestScore = gridSearch.best_params_, round(gridSearch.best_score_*100, 2)

    return bestScore, bestParams



def RSCVtuneHyperparam(model, params):

    randSearch = RandomizedSearchCV(estimator=model,

                                    param_distributions=params, 

                                    verbose=0, cv=10,

                                    n_jobs =4, n_iter=30,

                                    return_train_score=False,)

    randSearch.fit(X_train,y_train)

    bestParams, bestScore = randSearch.best_params_, round(randSearch.best_score_*100, 2)

    model=randSearch.best_estimator_#redefine the model with best parameters

    return bestScore, bestParams
# ## #Perform RandomizedSearchCV 

bestScore=[]

bestParam=[]

for model,param in zip(models,parametersLists):

    Score,Param=RSCVtuneHyperparam(model,param)

    bestScore.append(Score)

    bestParam.append(Param)

    

""" create a dataframe to store best score and best params."""

rsCV_df=pd.DataFrame()

# rsCV_df['modelNames']=modelNames

rsCV_df['bestScore']=bestScore

rsCV_df['bestParam']=bestParam

rsCV_df['Xcross_val_score']=xValScores_df.iloc[:,0].values

rsCV_df.index=modelNames

rsCV_df = rsCV_df.sort_values(by="bestScore", ascending=False)

# rsCV_df=rsCV_df.reset_index(drop=True)

print(" --------------------- RSCVtuneHyperparam --------------------- ")

rsCV_df
rsCV_df.to_csv('rsCV_df.csv',index=False)
"""Define the models with optimized hyperparameters."""



optParam=rsCV_df.sort_index().loc[:,'bestParam']



# 1. Logistic Regression

lr = LogisticRegression( **optParam["LogisticRegression"] )

# 2. DecisionTreeClassifier

dtree=DecisionTreeClassifier(**optParam["DecisionTreeClassifier"])

# 3. Random Forest Classifier

rand_forest = RandomForestClassifier( **optParam["RandomForestClassifier"] )

# 4. KNN

knn=KNeighborsClassifier(**optParam["KNeighborsClassifier"] )

# 5. Support Vector Machines [ SV Classifier ] 

svc=SVC(**optParam["SupportVectorClassifier"])

# 6. Gaussian Naive Bayes

gau_nb = GaussianNB(**optParam["GaussianNB"] )

# 7. Multinomial Naive Bayes

mult_nb=MultinomialNB( **optParam["MultinomialNB"] )

# 8. Gradient Boosting Classifier

gbc = GradientBoostingClassifier( **optParam["GradientBoostingClassifier"] )

# 9. Adaboost Classifier

abc = AdaBoostClassifier( **optParam["AdaBoostClassifier"] )

# 10. ExtraTrees Classifier

etc = ExtraTreesClassifier(**optParam["ExtraTreesClassifier"] )

# 11. Extreme Gradient Boosting

xgbc = XGBClassifier( **optParam["XGBClassifier"] )



#list of model defined above  again

models = [lr, dtree, rand_forest, knn, svc, gau_nb, mult_nb, gbc,abc, etc, xgbc]

modelNames=['LogisticRegression','DecisionTreeClassifier',"RandomForestClassifier",'KNeighborsClassifier',

          'SupportVectorClassifier','GaussianNB','MultinomialNB','GradientBoostingClassifier','AdaBoostClassifier',

          'ExtraTreesClassifier','XGBClassifier']
"""Instantiate the models with optimized hyperparameters."""

Score=[]

for model in models:

    model.fit(X_train,y_train)

    cv_score=cross_val_score(model, X_train, y_train, cv = 10, scoring="accuracy").mean()

    cv_score = round(cv_score*100, 2)

    Score.append(cv_score)



    

""" create a dataframe to store score after optimized parameter is fit."""

optParamTrainScore_df=pd.DataFrame()

optParamTrainScore_df['CV_Score_now']=Score

optParamTrainScore_df['bestScore']=bestScore

optParamTrainScore_df['Xcross_val_score_before_hypParmet']=xValScores_df.iloc[:,0].values

optParamTrainScore_df.index=modelNames

optParamTrainScore_df = optParamTrainScore_df.sort_values(by="CV_Score_now", ascending=False)

print(" --------------------- RSCVtuneHyperparam --------------------- ")

optParamTrainScore_df
"""Make prediction using all the trained models with best parameter."""



modelPrediction = pd.DataFrame({"LogisticRegression":lr.predict(X_test),

                                "DecisionTreeClassifier":dtree.predict(X_test),

                                "RandomForestClassifier":rand_forest.predict(X_test),

                                "KNeighborsClassifier":knn.predict(X_test), 

                                "SVC":svc.predict(X_test),

                                "GaussianNB":gau_nb.predict(X_test),

                                "MultinomialNB":mult_nb.predict(X_test), 

                                "GradientBoostingClassifier":gbc.predict(X_test), 

                                "AdaBoostClassifier":abc.predict(X_test),

                                "ExtraTreesClassifier":etc.predict(X_test), 

                                "XGBClassifier":xgbc.predict(X_test)

                                })

modelPrediction.head()
for modelName in modelPrediction.columns:

    submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": modelPrediction[modelName] })

    submission.to_csv(modelName+'.csv',index=False)