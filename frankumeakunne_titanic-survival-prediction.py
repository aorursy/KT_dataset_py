# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Project Planning Overview



""" Workflow Stages

1. Problem Understanding | Question or problem definition.

2. Data Collection | Acquire training and testing data.

3. Data Preprocessing | Wrangle, prepare, cleanse the data. 

4. Data Exploration | Analyze, identify patterns, and explore the data.  

5. Data Model Building | Model, predict and solve the problem.

6. Data Model Validation and Reporting | Visualize, report, and present the problem solving steps and final solution.

7. Solution Submission | Supply or submit the results.

"""



""" Workflow Goals: Follow 7 C's 

- Classifying 

- Correlating

- Converting 

- Completing

- Correcting 

- Creating 

- Charting

"""
#References: this notebook was created based on amazing work by the following

 #   https://www.kaggle.com/startupsci/titanic-data-science-solutions

 #   https://www.kaggle.com/miguelquiceno/titanic-eda/execution

 #   https://www.youtube.com/watch?v=I3FBJdiExcg&t=371s
#1. Problem Understanding | Question or problem definition.
"""Problem Statement   

In this challenge, we ask you to build a predictive model that answers the question: 

“what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, 

socio-economic class, etc).



Problem Interpretation: Can we develop an 'explainable' predictive model that predicts which passenger survives

given their personal log records data? Explainable, in this context, refers to the use of a type of model that 

can reliably associate the predictive power/assocaition with the original or engineered features from the 

passenger data set.



Domain Knowledge

The sinking of the Titanic is one of the most infamous shipwrecks in history.



On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding 

with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502

out of 2224 passengers and crew.



While there was some element of luck involved in surviving, it seems some groups of people were more likely to

survive than others.

"""
#data wrangling

import pandas as pd

import numpy as np

import re

import time, datetime  



#visualizations

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

%matplotlib inline



#data scaler

from sklearn.preprocessing import StandardScaler as sclr



#machine learning

from sklearn.linear_model import LogisticRegression as logr

from sklearn import tree

from sklearn.naive_bayes import GaussianNB as gnb

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



#ml ensembles

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier



#model tuning

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



#model scoring

from sklearn.model_selection import cross_val_score as cv_score
#2. Data Collection | Acquire training and testing data.
#read in training and test sets

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



#1st look at data

train.head()
#identify all training set columns

train.columns
#identify all test set columns

test.columns
#visualize missing values as a matrix 

msno.matrix(train)



#'Age', 'Cabin', and 'Embarked' have missing values

#missing values appear randomly dispersed throughout the dataset
#check training set for null values and mismatched data types

train.info()



#~20% of 'Age' and ~80% of 'Cabin' are missing values | 'Cabin' seems to have too few to use, 'Age' appears useable

#only 2 records have missing 'Embarked' entry which is negligible



#we expected 891 entries and recieved 891 entries, so we can conclude the import did not lead to present nulls
#check test set for null values and mismatched data types

test.info()



#~20% of 'Age' and ~80% of 'Cabin' are missing values. Consistent with traing set.

#only 1 record for Fare is missing



#we expected 891 entries and recieved 891 entries, so we can conclude the import did not lead to present nulls
#3. Data Preprocessing | Wrangle, prepare, feature engineer, cleanse the data.
# Data Categorization - we can look at the df index, head() and data dictionary to classify data



"""

Variables: Categorical 

- Factor | str

Pclass, Name, Ticket, Cabin, Embarked



- Factor | discrete numeric

PassengerID, Pclass



- Binary | discrete numeric and str

Survived, set_type, and Sex 



Variables: Continuous & Discrete | numeric 

- Age, Fare, Parch, SibSp

"""
#3.1 Feature Engineering
#New feature: add a title feature to track correlation of surival with passenger title

train['title'] = train['Name'].apply(lambda x: x.split(', ')[1].split('.')[0])

test['title'] = test['Name'].apply(lambda x: x.split(', ')[1].split('.')[0])

test['title'].value_counts()
#Personal Curiosity: Let's peak at the survival of the ship's Captain and a Countess!

train[train['Name'].str.contains(r'C[ao][pu][tn]',regex=True)][2:] #The Royal surives while the Captain goes down with his ship.
#New feature: Cabin letter - see if a passengers cabin section correlates with survival

train['Cabin_letter'] = train.Cabin.apply(lambda x: 0 if pd.isna(x) else x[0])

test['Cabin_letter'] = test.Cabin.apply(lambda x: 0 if pd.isna(x) else x[0]) 

test.Cabin_letter.value_counts()
#3.2 Data Cleaning and Harmonization
#Data Correction

    #drop embarked nulls since there are only 2

train.dropna(subset=['Embarked'], inplace=True)

test.dropna(subset=['Embarked'], inplace=True)



#Data Completion

    #impute Age value for models can't handle Nulls | used the mode in this case 4 minus the median = 24.

train['Age'] = train['Age'].fillna(value=(train['Age'].median()-4))

test['Age'] = test['Age'].fillna(value=(train['Age'].median()-4))



#Data Convertion

    #normalize Fare feature using a log transformation since it is exponential

train['Fare_norm'] = np.log(train.Fare-1) #include -1 to address -inf for zero dollar Fares

train['Fare_norm'] = train['Fare_norm'].fillna(value=1) #fix zero Fares not correctly scaled to 1

test['Fare_norm'] = np.log(test.Fare-1)

test['Fare_norm'] = test['Fare_norm'].fillna(value=1)



    #convert factors to numeric type | aka create dummy variables 

train_dum = pd.get_dummies(train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Fare_norm', 'title', 'Cabin_letter']])

test_dum = pd.get_dummies(test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Fare_norm', 'title', 'Cabin_letter']])



    #drop Cabin column 

train.drop(columns='Cabin',inplace=True)
#check 'Fare' feature normalized with histogram

train.Fare_norm.hist()
#partition x, y training and test sets

x_train = train_dum

y_train = train['Survived']

x_test = test_dum
#add missing columns from test set that is in training set and vice-versa

x_test['title_Capt'] = 0 

x_test['title_Lady'] = 0 

x_test['title_Major'] = 0 

x_test['title_Jonkheer'] = 0 

x_test['title_Mlle'] = 0 

x_test['title_Mme'] = 0 

x_test['title_Sir'] = 0 

x_test['title_the Countess'] = 0 

x_test['Cabin_letter_T'] = 0 

x_test['title_Don']=0

x_train['title_Dona']=0



#rearrange dataframe columns to match

x_test = x_test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare_norm', 'Sex_female',

       'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'title_Capt',

       'title_Col', 'title_Don', 'title_Dr', 'title_Jonkheer', 'title_Lady',

       'title_Major', 'title_Master', 'title_Miss', 'title_Mlle', 'title_Mme',

       'title_Mr', 'title_Mrs', 'title_Ms', 'title_Rev', 'title_Sir',

       'title_the Countess', 'Cabin_letter_0', 'Cabin_letter_A',

       'Cabin_letter_B', 'Cabin_letter_C', 'Cabin_letter_D', 'Cabin_letter_E',

       'Cabin_letter_F', 'Cabin_letter_G', 'Cabin_letter_T', 'title_Dona']]



#check order matches

x_train.columns
x_test.columns
#Data Conversion | Harmonize the scale of all data

scale = sclr()

x_train_scl = x_train.copy() # copy dummy df to leave it intact after scaling for performance comparison

x_test_scl = x_test.copy()



#scale appropriate variables within the train and test sets

x_train_scl[['Age','SibSp','Parch','Fare_norm']]=scale.fit_transform(x_train_scl[['Age','SibSp','Parch','Fare_norm']])

x_test_scl[['Age','SibSp','Parch','Fare_norm']]=scale.fit_transform(x_test_scl[['Age','SibSp','Parch','Fare_norm']])
#4. Data Exploration | Analyze, identify patterns, and explore the data.
#check the central tendency and dispersion of our numeric features

train.describe()



#Some columns are not relavant; PassengerId, set_type



#Survived: over 60% of passengers did not survive

#Pclass: Over 75% of the passengers were in the lower classes (2 and 3)

#Age: the average passenger was 29, but with a 14 year std this could mean more older or more younger people were present 

# -looking at percentiles, we better see 50% of passengers were under 28, 25% between 28 and 38, and 25% older than 38

#SibSp: 50% of passengers had no sibling or spouse, 25% had 1, and 25% had 1 or more

#parch: 75% of passengers did not travel with a parent or child. Does not include nannies traveling with children

#Fare: average fare was $32, while 75% of passengers paid less than $32. This is possibly high skew from large fares like $512. 
#Exploring Continuous Variables
#1st by Histograms



#here we check out the frequency of continuous and discrete numeric variables
hist_col = ['Age','SibSp','Parch','Fare'] 



for e in hist_col:

    train[e].hist()

    plt.title('Graph: %s' % e) 

    plt.show()
#Bar Plot Review



#'Age' - more normally distributed with most between 16 and 38 | we need to select appropriate bin (current = 5)

#'SibSp' - vast majority have 0 or 1 | small range, likely not need to normalize

#'Parch' - vast majority have 0 or 1 | small range, likely not need to normalize

#'Fare' - vast majority paid less than $100 | likely need to normalize
#2nd by Box Plot



#Here we check out the numeric fields that have interesting percentile values
box_col =['Pclass', 'Age','SibSp', 'Fare','Fare_norm'] #normalized 'Fare' seems to re-include most of the outliers

for e in box_col:

    train.boxplot(column = e)

    plt.title('Graph: %s' % e)

    plt.show()
#Boxplot Review



#'Pclass' - Most dispersion in class 2 and 3

#'Age' - dispersion is so low in age that passengers over 65 are outliers

#'SibSp' - 5 to 8 Siblings/Spouses are considered outliers

#'Fare' - Vast majority of fares are less than $50



train[train['SibSp'] >= 5] #Only two familes appear to have 5 or more siblings/spouses. And none of them survived. Tragic.
train[train['Fare'] >= 270] #Only 3 passengers paid tickets over $270. Outliers, but notice they all survived.
#3rd by Correlation Matrix
corr = train.corr()

corr
#generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



#set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



#generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



#draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#Correlation Matrix Review



#Note: the target variable is 'Survived'

#'Pclass' and 'Fare_norm' have the highest correlation with 'Survived' at -33.8% and 33.3% respectively.

#multicollinearity watch: 

# - 'Pclass' with 'Age' and 'Pclass' with 'Fare_norm' (makes sense if higher class tickets have higher fare cost)

# - Our engineered 'Fare_norm' more correlated with 'Survived' and highly correlated with 'Pclass'

# - 'SibSp' with 'Age' and SibSp with 'Parch' 



#now let's drop the 'Fare' as we will use the 'Fare_norm'

train.drop(columns='Fare',inplace=True)

test.drop(columns='Fare',inplace=True)
#Exploring Categorical Variables
#1st by Barplot
#build bar plots for each cat feature

bar_col1 = train[['Survived', 'Pclass', 'Sex', 'Embarked','title']] 



for e in bar_col1:

    col_val = bar_col1[e].value_counts()

    plt.title('Graph: %s | Total Groups: %d' % (e,len(col_val)))

    plot = sns.barplot(x=col_val.index, y=col_val)

    plot.set_xticklabels(plot.get_xticklabels())

    plt.show()
#Plots with vertical x labels

bar_col2 = train[['Name', 'title']] #'Name' feature values are too distinct and will be excluded



for e in bar_col2:

    col_val = bar_col2[e].value_counts()

    plt.title('Graph: %s | Total Groups: %d' % (e,len(col_val)))

    plot = sns.barplot(x=col_val.index, y=col_val)

    plot.set_xticklabels(plot.get_xticklabels(), rotation=90)

    plt.show()
#Barplot Review



#Survived - majority did not survive | binary 

#Pclass - majority of passengers 3rd, then sharp drop to 1st, followed closely by 2nd



#now let's drop the 'Name' column 

train.drop(columns='Name',inplace=True)

test.drop(columns='Name',inplace=True)
#Explore variable relationships with passenger survival by Pivot Table

pivot_col = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Embarked', 'Fare_norm', 'title']

for e in pivot_col:

    ptab=pd.pivot_table(train, index = e, values='Survived')

    print(ptab)
#Pivot Review



#Survival by

#'Class: 60% of 1st class survived, just of 50% of 2nd class, and only just over 20% of 3rd class. Note: The majority of passengers were in 3rd class. Tragic.

#'Sex': Over74% of females survived while only 18.9% of males survived.

#'SibSp': Passengers with 1 Sibling/Spouse had the highest proportion og survival. Passengers who had more 5 or more Sib/Sp did not survive

#'Parch': Passengers with 3 Parents or Children had the highest proportion of survival

#'Fare_norm': Passengers who paid higher Fare's had a higher proporiton of surival

#'Embarked': Over 55% of passgeners who embarked from Cherbourg survived.

#'title': very telling on survival. All young, unmarried women (with high status) survived ('Ms'/1.0) while very few 'average' men proportionately survived ('Mr'/0.16). and other insights. 
#Let's compare the average values for our continuous variables along Survivorship

pd.pivot_table(train, index='Survived', values=['Pclass', 'SibSp', 'Parch', 'Fare_norm'])
#Let's use pivots to drill-down and count the number of survivors by interesting subcategories

piv_col1 = ['Pclass', 'SibSp', 'Sex', 'Parch', 'Embarked','title']

for e in piv_col1:

    dd_pivot = pd.pivot_table(train, index='Survived', columns=e, values='PassengerId', aggfunc='count')

    print(dd_pivot)

    print('\n')

#5. Data Model Building | Model, predict and solve the problem.
#1st Iteration: Build of Plausible Models on Default Setting
#Results: Format is (Scaled Score | Unscaled Score)



"""

Should Scale

- Logistic Regression (82.6% | 82.5%)

- K-Nearest Neighbors (82.5% | 81.1%)

- Support Vector Classifier (83.1% | 82.2%)

- Soft Vote Classifier (83.7% | 84.03%)

- Hard Vote Classifier (84.1% | 83.6%)  #Best Ensembler



Should not Scale

- Naive Bayes (72.1% | 73.8%)

- Classification Tree (78.6% | 78.6%)

- Random Forest (81.2% | 81.3%)

- AdaBoost Classifier (82.0% | 82.0%)

- Gradient Descent Classifier (82.7% | 82.7%)     

- XGBoost Classifier (82.9% | 82.9%)              



"""
#Run baseline models | Model 1



#Gaussian Naive Bayes 

start_time = time.time()



#Gaussian Naive Bayes is generally used as a good baseline model for classification problems with continuous variables

nb = gnb()

score_nb = cv_score(nb,x_train_scl,y_train,cv=10)



run_time = (time.time() - start_time)



print(score_nb.round(2)*100) #print 5 cross validation scores

print("Naive Bayes Accuracy: %s" % round(score_nb.mean()*100,2)+'%') #print the average score

print("Run time: %s" % datetime.timedelta(seconds=run_time))

#Re-Run baseline models | Model 1



#we re-run our model on the unscaled training set to compare performance

start_time = time.time()



nb = gnb()

score_nb_unscl = cv_score(nb,x_train,y_train,cv=10)



run_time = (time.time() - start_time)



print(score_nb_unscl.round(2)*100) #print 5 cross validation scores

print("Unscaled Naive Bayes Accuracy: %s" % round(score_nb_unscl.mean()*100,2)+'%') #print the average score

print("Run time: %s" % datetime.timedelta(seconds=run_time))



#the unscaled training set accuracy outperforms the scaled set 
#Run baseline models | Model 2



#Decision ('Classification') Tree



start_time = time.time()



#as we will incorporate more robust tree based models, this standard tree serves as a baseline model

dt = tree.DecisionTreeClassifier(random_state=0)

score_dt = cv_score(dt, x_train_scl, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_dt.round(2)*100) #print 10 cross validation scores

print("Classification Tree Accuracy: %s" % round(score_dt.mean()*100,2)+'%') #print the average score

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#Re-run baseline models | Model 2



#we run the Decision ('Classification') Tree on the unscaled training set



start_time = time.time()



dt = tree.DecisionTreeClassifier(random_state=0)

score_dt_unscl = cv_score(dt, x_train, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_dt_unscl.round(2)*100) 

print("Classification Tree Accuracy: %s" % round(score_dt_unscl.mean()*100,2)+'%') 

print("Run time: %s" % datetime.timedelta(seconds=run_time))



#the unscaled training set accuracy approximately matches the scaled set, with a faster run time
#Repeat for remaining models with parameters set to default
#Logistic Regression



start_time = time.time()



lr = logr(solver='liblinear')

score_lr = cv_score(lr, x_train_scl, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_lr.round(2)*100) #again, print 10 cross validation scores

print("Logistic Regression Accuracy: %s" % round(score_lr.mean()*100,2)) #again, print the average score

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#Logistic Regression (Unscaled)



start_time = time.time()



lr = logr(solver='liblinear')

score_lr_unscl = cv_score(lr, x_train, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_lr_unscl.round(2)*100) 

print("Logistic Regression Accuracy: %s" % round(score_lr_unscl.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))



#the scaled training set accuracy outperforms the unscaled set 
#Random Forests



start_time = time.time()



rf = RandomForestClassifier(random_state=0, n_estimators=100)

score_rf = cv_score(rf, x_train_scl, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_rf.round(2)*100) 

print("Random Forest Accuracy: %s" % round(score_rf.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#Random Forests (Unscaled)



start_time = time.time()



rf = RandomForestClassifier(random_state=0, n_estimators=100)

score_rf_unscl = cv_score(rf, x_train, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_rf_unscl.round(2)*100) 

print("Unscaled Random Forest Accuracy: %s" % round(score_rf_unscl.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))



#The unscaled training set accuracy outperforms the scaled set 
#K-Nearest Neighbors 



start_time = time.time()



kn = KNeighborsClassifier(n_neighbors=5)

score_knn = cv_score(kn, x_train_scl, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_knn.round(2)*100) 

print("K-Nearest Neighbor Accuracy: %s" % round(score_knn.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#K-Nearest Neighbors (Unscaled)



start_time = time.time()



kn = KNeighborsClassifier(n_neighbors=5)

score_knn_unscl = cv_score(kn, x_train, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_knn_unscl.round(2)*100) 

print("Unscaled K-Nearest Neighbor Accuracy: %s" % round(score_knn_unscl.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))



#The scaled training set accuracy outperforms the unscaled set | 82.45 > 81.11
#Support Vector Classification



start_time = time.time()



svc = SVC(gamma='auto',probability=True)

score_svc = cv_score(svc, x_train_scl, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_svc.round(2)*100) 

print("Support Vector Accuracy: %s" % round(score_svc.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#Support Vector Classification (Unscaled)



start_time = time.time()



svc = SVC(gamma='auto',probability=True)

score_svc_unscl = cv_score(svc, x_train, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_svc_unscl.round(2)*100) 

print("Unscaled Support Vector Accuracy: %s" % round(score_svc_unscl.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))



#The scaled training set accuracy outperforms the unscaled set | 83.13 > 82.23
#AdaBoost Classifier



start_time = time.time()



ada = AdaBoostClassifier()

score_ada = cv_score(ada, x_train_scl, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_ada.round(2)*100) 

print("AdaBoost Accuracy: %s" % round(score_ada.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#AdaBoost Classifier (Unscaled)



start_time = time.time()



ada = AdaBoostClassifier()

score_ada_unscl = cv_score(ada, x_train, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_ada_unscl.round(2)*100) 

print("Unscaled AdaBoost Accuracy: %s" % round(score_ada_unscl.mean()*100,2))

print("Run time: %s" % datetime.timedelta(seconds=run_time))



#The scaled training set accuracy matches the unscaled set | 82.0 = 82.0, while the unscaled set runs faster
#Gradient Boost Classifier



start_time = time.time()



gb = GradientBoostingClassifier(random_state=0)

score_gb = cv_score(gb, x_train_scl, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_gb.round(2)*100) 

print("Gradient Boost Accuracy: %s" % round(score_gb.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#Gradient Boost Classifier (Unscaled)



start_time = time.time()



gb = GradientBoostingClassifier(random_state=0)

score_gb_unscl = cv_score(gb, x_train, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_gb_unscl.round(2)*100) 

print("Unscaled Gradient Boost Accuracy: %s" % round(score_gb_unscl.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))



#The scaled training set accuracy matches unscaled set | 82.68 = 82.68, and is slightly faster
#XGBoost Classifier



start_time = time.time()



xgb = XGBClassifier(random_state=0)

score_xgb = cv_score(xgb, x_train_scl, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_xgb.round(2)*100) 

print("XGBoost Accuracy: %s" % round(score_xgb.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#XGBoost Classifier (Unscaled)



start_time = time.time()



xgb = XGBClassifier(random_state=0)

score_xgb_unscl = cv_score(xgb, x_train, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_xgb_unscl.round(2)*100) 

print("Unscaled XGBoost Accuracy: %s" % round(score_xgb_unscl.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))



#The unscaled training set accuracy matches the scaled set | 82.9 = 82.9, while running a little faster
#Soft Voting Classifier



start_time = time.time()



soft_vote = VotingClassifier(estimators=[('xgb',xgb), ('gb',gb), ('ada',ada), ('rf',rf),('svc',svc), ('kn',kn), ('lr',lr)], voting='soft')

score_vote = cv_score(soft_vote, x_train_scl, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_vote.round(2)*100) 

print("Voting Classifier Accuracy: %s" % round(score_vote.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#Soft Voting Classifier (Unscaled)



start_time = time.time()



soft_vote = VotingClassifier(estimators=[('xgb',xgb), ('gb',gb), ('ada',ada), ('rf',rf),('svc',svc), ('kn',kn), ('lr',lr)], voting='soft')

score_vote_unscl = cv_score(soft_vote, x_train, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_vote_unscl.round(2)*100) 

print("Unscaled Voting Classifier Accuracy: %s" % round(score_vote_unscl.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))



#The unscaled training set accuracy outperforms the scaled set | 84.03 > 83.69
#Hard Voting Classifier



start_time = time.time()



hard_vote = VotingClassifier(estimators=[('xgb',xgb), ('gb',gb), ('ada',ada), ('rf',rf),('svc',svc), ('kn',kn), ('lr',lr)], voting='hard')

score_vote = cv_score(hard_vote, x_train_scl, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_vote.round(2)*100) 

print("Voting Classifier Accuracy: %s" % round(score_vote.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#Hard Voting Classifier (Unscaled)



start_time = time.time()



hard_vote = VotingClassifier(estimators=[('xgb',xgb), ('gb',gb), ('ada',ada), ('rf',rf),('svc',svc), ('kn',kn), ('lr',lr)], voting='hard')

score_vote_unscl = cv_score(hard_vote, x_train, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_vote_unscl.round(2)*100) 

print("Voting Classifier Accuracy: %s" % round(score_vote_unscl.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#Create baseline submission using best ensemble | Hard Voting Classifier

hard_vote.fit( x_train_scl, y_train)

y_pred_hardvote = hard_vote.predict(x_test_scl)

submission_hardvote = {'PassengerID': test.PassengerId, 'Survived': y_pred_hardvote}

submission_hv = pd.DataFrame(data=submission_hardvote)

submission_hv.to_csv('baseline_submission.csv', index=False)
#2nd Iteration: Tune Parameters and Reubild Models
#let's define a function to report the best parameters to tune for each model and resulting score

def tune_report(mod, name):

    print(name)

    print('Best Score: %s' % str(round(mod.best_score_,3)*100)+'%')

    print('Best Parameters: %s' % str(mod.best_params_))
#We'll skip the baseline models and jump right into our superior scoring and more robust models

#Logistic Regression | Exhaustive Search



lr = logr(solver='liblinear')

param_grid = {'max_iter': [100,500,200],

             'penalty': ['l1','l2'],

             'C': np.logspace(-5,5,num=25)} #try 25 values between x^-5 and x^5

search_lr = GridSearchCV(lr, param_grid=param_grid, cv=5, verbose=True, n_jobs = 1)

tuned_lr = search_lr.fit(x_train_scl,y_train)

tune_report(tuned_lr,'Logistic Regression')
#Random Forest | Random Search  - run a few times to get an effiecient taste of local optimal parameters



param_grid = {'n_estimators': [100,250,500],

             'max_depth': [3,5,9,15,20,100,None],

             'min_samples_split': [2, 5,10], 

             'min_samples_leaf': [1,2,5,7,10],

             'max_features': ['auto', 'sqrt', 'log2'],

             'bootstrap': [True],

             'oob_score': [True, False]}

rand_search_rf = RandomizedSearchCV(rf, param_distributions=param_grid, cv=5, verbose=True, n_jobs = 1)

tuned_rf_rand = rand_search_rf.fit(x_train_scl,y_train)

tune_report(tuned_rf_rand,'Random Forest')
#Random Forest | Focused Search - exhaustive search with parameter set adjusted from randomized results

rf = RandomForestClassifier()

param_grid = {'n_estimators': [500],

             'max_depth': [9,20,50,None],

             'min_samples_split': [2,3], 

             'min_samples_leaf': [1,2],

             'max_features': ['auto'],

             'bootstrap': [True],

             'oob_score': [True]}

search_rf = GridSearchCV(rf, param_grid=param_grid, cv=5, verbose=True, n_jobs = 1)

tuned_rf = search_rf.fit(x_train_scl,y_train)

tune_report(tuned_rf,'Random Forest')
#K Nearest Neighbors | Exhaustive Search



kn = KNeighborsClassifier()

param_grid = {'n_neighbors': [3,5,7,9],

             'algorithm': ['ball_tree', 'kd_tree', 'brute'],

             'weights': ['uniform', 'distance'],

             'p': [1,2]}

search_kn = GridSearchCV(kn, param_grid=param_grid, cv=5, verbose=True, n_jobs = 1)

tuned_kn = search_kn.fit(x_train_scl,y_train)

tune_report(tuned_kn,'K Nearest Neighbors')
#Support Vector Classifier | Exhaustive Search



param_grid = {'kernel': ['rbf'],

              'probability': [True],

             'gamma': ['scale','auto']}

search_svc = GridSearchCV(svc, param_grid=param_grid, cv=5, verbose=True, n_jobs = 1)

tuned_svc = search_svc.fit(x_train,y_train)

tune_report(tuned_svc,'Support Vector Classifier')
#AdaBoost Classifier | Exhaustive Search



ada =AdaBoostClassifier()

param_grid = {'n_estimators': [50],

             'learning_rate': [0.2,0.5,1],

             'algorithm': ['SAMME.R'],

             'random_state': [0]}

search_ada = GridSearchCV(ada, param_grid=param_grid, cv=5, verbose=True, n_jobs = 1)

tuned_ada = search_ada.fit(x_train_scl,y_train)

tune_report(tuned_ada,'AdaBoost Classifier')
#XGBoost Classifier | Exhaustive Search

xgb = XGBClassifier(silent = True,verbosity=0)

param_grid = {'booster': ['gbtree'],

             'eta': [.3,.7],

             'gamma': [3,5],

             'max_depth': [6,9], 

             'random_state': [0]}

search_xgb = GridSearchCV(xgb, param_grid=param_grid, cv=5, verbose=0, n_jobs = 1)

tuned_xgb = search_xgb.fit(x_train_scl,y_train)

tune_report(tuned_xgb,'XGBoost Classifier')
#Create submission using XGBoost

y_pred_xgb = tuned_xgb.best_estimator_.predict(x_test_scl)

data_xgb = {'PassengerID': test.PassengerId, 'Survived': y_pred_xgb}

submission_xgb = pd.DataFrame(data=data_xgb)

submission_xgb.to_csv('xgb_submission.csv', index=False)
#6. Data Model Validation and Reporting | Visualize, report, and present the problem solving steps and final solution.
#Report: Model Post-Tuned Performance  (Pre-tuned | Post-tuned)



#Note: only tuned non-baseline models with likely superior performance

"""

- Logistic Regression (82.6% | 82.1%)

- Random Forest (81.2% | 83.0%)

- K-Nearest Neighbors (82.5% | 81.2%)

- Support Vector Classifier (83.1% | 81.2%)

- AdaBoost Classifier (82.0% | 81.9%)

- XGBoost Classifier (82.9% | 83.8%) 



- Soft Vote Classifier (83.7% | 83.69%)

- Hard Vote Classifier (84.1% | 83.91%)  #Best Ensemble     

"""
#let's visualize the importance of each factor in random forest model

best_rf = tuned_rf.best_estimator_.fit(x_train_scl,y_train)

feat_rank = pd.Series(best_rf.feature_importances_, index=x_train_scl.columns)

feat_rank.nlargest(10).plot(kind='pie', title='Feature Importance Pie by Random Forest')
#let's view the value of importance of each factor in random forest model

best_rf = tuned_rf.best_estimator_.fit(x_train_scl,y_train)

df = tuple(zip(x_train_scl.columns,best_rf.feature_importances_))

feat_rank = pd.DataFrame(df, columns=['feat_name','feat_rank'])

feat_rank.nlargest(10,'feat_rank').plot(kind='barh',x='feat_name',y='feat_rank', title='Feature Importance by Random Forest')
#Create submission using random forest 

y_pred_rf = tuned_rf.best_estimator_.predict(x_test_scl)

data_rf = {'PassengerID': test.PassengerId, 'Survived': y_pred_rf}

submission_rf = pd.DataFrame(data=data_rf)

submission_rf.to_csv('rf_submission.csv', index=False)
#let's visualize the importance of each factor in adaboost model in copmarison

best_ada = tuned_ada.best_estimator_.fit(x_train_scl,y_train)

df = tuple(zip(x_train_scl.columns,best_ada.feature_importances_))

feat_rank = pd.DataFrame(df, columns=['feat_name','feat_rank'])

feat_rank.nlargest(10,'feat_rank').plot(kind='barh',x='feat_name',y='feat_rank', title='Feature Importance by ADA')
#Create submission using Adaboost 

y_pred_ada = tuned_ada.best_estimator_.predict(x_test_scl)

data_ada = {'PassengerID': test.PassengerId, 'Survived': y_pred_ada}

submission_ada = pd.DataFrame(data=data_ada)

submission_ada.to_csv('ada_submission.csv', index=False)
#Lastly, let's see if we can improve performance with a Hard and Soft Vote Ensemble
#create best models

best_lr = tuned_lr.best_estimator_

best_rf = tuned_rf.best_estimator_

best_kn = tuned_kn.best_estimator_

best_svc = tuned_svc.best_estimator_

best_ada = tuned_ada.best_estimator_

best_xgb = tuned_xgb.best_estimator_
#Hard Voting Ensemble



start_time = time.time()



hard_vote = VotingClassifier(estimators=[('xgb',best_xgb), ('ada',best_ada), ('rf',best_rf),('svc',best_svc), ('kn',best_kn), ('lr',best_lr)], voting='hard')

score_vote = cv_score(hard_vote, x_train_scl, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_vote.round(2)*100) 

print("Voting Classifier Accuracy: %s" % round(score_vote.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#Soft Voting Ensemble



start_time = time.time()



soft_vote = VotingClassifier(estimators=[('xgb',best_xgb), ('ada',best_ada), ('rf',best_rf),('svc',best_svc), ('kn',best_kn), ('lr',best_lr)], voting='soft')

score_vote = cv_score(soft_vote, x_train_scl, y_train,cv=10)



run_time = (time.time() - start_time)



print(score_vote.round(2)*100) 

print("Voting Classifier Accuracy: %s" % round(score_vote.mean()*100,2)) 

print("Run time: %s" % datetime.timedelta(seconds=run_time))
#7. Solution Submission | Supply or submit the prediction output.
#create final submissions using hard and soft ensembles | Hard Voting Ensemble

hard_vote.fit(x_train_scl, y_train)

y_pred_hardvote = hard_vote.predict(x_test_scl)

submission_hardvote = {'PassengerID': test.PassengerId, 'Survived': y_pred_hardvote}

submission_hv = pd.DataFrame(data=submission_hardvote)

submission_hv.to_csv('hardvote_submission.csv', index=False)
#create final submissions using hard and soft ensembles | Soft Voting Ensemble

soft_vote.fit(x_train_scl, y_train)

y_pred_softvote = soft_vote.predict(x_test_scl)

submission_softvote = {'PassengerID': test.PassengerId, 'Survived': y_pred_softvote}

submission_sv = pd.DataFrame(data=submission_softvote)

submission_sv.to_csv('softvote_submission.csv', index=False)
#let's compare the outputs of all our submissions

all_subs = {'PassengerID': test.PassengerId, 'Survived_hv': y_pred_hardvote,'Survived_sv': y_pred_softvote,

           'Survived_ada': y_pred_ada, 'Survived_rf': y_pred_rf, 'Survived_xgb': y_pred_xgb, 

            'Survived_baseline': y_pred_hardvote}



df_allsubs = pd.DataFrame(data=all_subs)
df_allsubs['diff_baseline_other'] = df_allsubs.apply(lambda x: 1 if x.Survived_baseline != x.Survived_rf else 0, axis=1)

df_allsubs['diff_baseline_other'].value_counts()