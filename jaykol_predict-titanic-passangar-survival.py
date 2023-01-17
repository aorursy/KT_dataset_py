# This script shows you how to make a submission using a few

# useful Python libraries.

# It gets a public leaderboard score of 0.76077.

# Maybe you can tweak it and do better...?



import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

import numpy as np



# Load the data

train_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)





# We'll impute missing values using the median for numeric columns and the most

# common value for string columns.

# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],

            index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.fill)



feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']

nonnumeric_columns = ['Sex']



# Join the features from train and test together before imputing missing values,

# in case their distribution is slightly different

big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])

big_X_imputed = DataFrameImputer().fit_transform(big_X)



# XGBoost doesn't (yet) handle categorical features automatically, so we need to change

# them to columns of integer values.

# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more

# details and options

le = LabelEncoder()

for feature in nonnumeric_columns:

    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])



# Prepare the inputs for the model

train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()

test_X = big_X_imputed[train_df.shape[0]::].as_matrix()

train_y = train_df['Survived']



# You can experiment with many other options here, using the same .fit() and .predict()

# methods; see http://scikit-learn.org

# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)

predictions = gbm.predict(test_X)



# Kaggle needs the submission to have a certain format;

# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv

# for an example of what it's supposed to look like.

submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],

                            'Survived': predictions })

submission.to_csv("submission.csv", index=False)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Add packages as required by code below and execute this cell.





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Kaggle you need to specify "../input" path for all your input files.



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.info()
test.info()
# Count of People survived and dead

print("People survived count : ")

survive_cnt = train.Survived.value_counts()

print(survive_cnt)

# We can see out of 891 people in train DataSet only 342 survived and rest are dead



# Let's find surviving percentage 

print("Survived percentage : ")

survive_per = (survive_cnt/survive_cnt.sum()) * 100

print(survive_per)

# only 38.3 percent chance of survival



# Divide figure into two plots having 1 row and 2 columns

ax1 = plt.subplot2grid((1,2),(0,0))



# Define figure size on (x,y) axis

plt.figsize=(10,5)



# plot survived count to grid 1 (plot 1)

_ = survive_cnt.plot(kind='bar', ax=ax1)

_ = plt.title('Survived Count')

_ = plt.xlabel('Survived')

_ = plt.ylabel('Count')

plt.margins(0.04)



# Divide figure into two plots having 1 row and 2 columns

ax2 = plt.subplot2grid((1,2),(0,1))



# plot survived percent to grid 2 (plot 2)

_ = survive_per.plot(kind='bar', ax=ax2)

_ = plt.title('Survived Percentage')

_ = plt.xlabel('Survived')

_ = plt.ylabel('percentage')

plt.margins(0.04)
# Cound number of categorical data

plt.figure(figsize=(8,3))

pclass_cnt = train.Pclass.value_counts()

print(pclass_cnt)

_ = pclass_cnt.plot(kind='barh')

_ = plt.title('Class Count')

_ = plt.xlabel('Count')

_ = plt.ylabel('Pclass')

# Do you see there are three class available 1,2 and 3.
# count the number of male and female

print("Male Female count : ")

sex_cnt = train.Sex.value_counts()

print(sex_cnt)



# percentage the number of male and female 

print("Male Female percentage : ")

sex_per = (sex_cnt/sex_cnt.sum()) * 100

print(sex_per)



# Divide figure into two plots having 1 row and 2 columns

ax1 = plt.subplot2grid((1,2),(0,0))



# Define figure size on (x,y) axis

plt.figsize=(10,5)



# plot survived count to grid 1 (plot 1)

_ = sex_cnt.plot(kind='bar', ax=ax1)

_ = plt.title('Sex Count')

_ = plt.xlabel('Sex')

_ = plt.ylabel('Count')

plt.margins(0.04)



# Divide figure into two plots having 1 row and 2 columns

ax2 = plt.subplot2grid((1,2),(0,1))



# plot survived percent to grid 2 (plot 2)

_ = sex_per.plot(kind='bar', ax=ax2)

_ = plt.title('Sex Percentage')

_ = plt.xlabel('Sex')

_ = plt.ylabel('percentage')

plt.margins(0.04)
# Plot for number of sibling and spouse count.

plt.figsize=(15,3)

ax1 = plt.subplot2grid((1,2),(0,0))

_ = train.SibSp.value_counts().plot(kind='bar', ax=ax1)

_ = plt.title('Frequency of Siblings or Spouse on-board',  fontsize=8)

_ = plt.xlabel('SibSp')

_ = plt.ylabel('Frequency')

plt.margins(.04)



plt.figsize=(10,5)

ax2 = plt.subplot2grid((1,2),(0,1))

_ = train.Parch.value_counts().plot(kind='bar', ax=ax2)

_ = plt.title('Frequency of Parant or Children on-board', fontsize=8)

_ = plt.xlabel('Parch')

_ = plt.ylabel('Frequency')

plt.margins(.04)



# As there can be any size of family on-board we can call this variables as continuous. 

# (My understanding can be wrong here.)
# Lets plot Age by dropping null values

_ = train.Age.dropna().plot(kind='hist', bins=50) 

_ = plt.xlabel('Age')

# As age is floating point value we condider it in continuous variable catategory.

# Age 18 to 35 is commomn.



# Get central tendency for Age

print('Age average : ',train.Age.dropna().mean())

print('Age mode : ',train.Age.dropna().mode())

print('Age median : ',train.Age.dropna().median())
# As Fare available in floating point value for time being we will change it to integer and plot

fare = train.Fare.astype(int)

_ = sns.boxplot(data=fare)

_ = plt.title('Fare distribution plot')

_ = plt.ylabel('Fare')

# from box plot it is easy to infer most of the people carrying ticket price less than 10 and 

# there are some ouliers (dots in plot) with high ticket value.



# Get central tendency for Fare

print('Fare average : ',train.Fare.dropna().mean())

print('Fare mode : ',train.Fare.dropna().mode())

print('Fare median : ',train.Fare.dropna().median())
# Plot Embarked frequency

_ = train.Embarked.value_counts().plot(kind='barh')

_ = plt.title('Embarked frequency')

_ = plt.xlabel('Embarked')

_ = plt.ylabel('Frequency')

# Embarked = S is most common
plt.figure(figsize=(10,20))



# distribution of class

ax1 = plt.subplot2grid((4,2),(0,0))

_ = train['Pclass'].value_counts().plot(kind='barh', color='red', label='train', alpha=.6)

_ = test['Pclass'].value_counts().plot(kind='barh', color='blue', label='test', alpha=.6)

_ = ax1.set_title('class distribustion')

_ = ax1.set_xlabel('Count')

_ = ax1.set_ylabel('Class')

_ = ax1.legend()

plt.margins(0.04, ax=ax1)



# distribution of Sex

ax2 = plt.subplot2grid((4,2),(0,1))

_ = train['Sex'].value_counts().plot(kind='barh', color='red', label='train', alpha=.6)

_ = test['Sex'].value_counts().plot(kind='barh', color='blue', label='test', alpha=.6)

_ = ax2.set_title('Sex distribustion')

_ = ax2.set_xlabel('Count')

_ = ax2.legend()

plt.margins(0.04, ax=ax2)



# distribution of Age

ax3 = plt.subplot2grid((4,2),(1,0))

_ = train['Age'].value_counts().plot(kind='kde', color='red', label='train', alpha=.6)

_ = test['Age'].value_counts().plot(kind='kde', color='blue', label='test', alpha=.6)

_ = ax3.set_title('Age distribustion')

_ = ax3.set_xlabel('Age')

_ = ax3.legend()

plt.margins(0.04, ax=ax3)



# distribution of SibSp

ax4 = plt.subplot2grid((4,2),(1,1))

_ = train['SibSp'].value_counts().plot(kind='kde', color='red', label='train', alpha=.6)

_ = test['SibSp'].value_counts().plot(kind='kde', color='blue', label='test', alpha=.6)

_ = ax4.set_title('SibSp distribustion')

_ = ax4.set_xlabel('SibSp')

_ = ax4.legend()

plt.margins(0.04, ax=ax4)



# distribution of Parch

ax5 = plt.subplot2grid((4,2),(2,0))

_ = train['Parch'].value_counts().plot(kind='kde', color='red', label='train', alpha=.6)

_ = test['Parch'].value_counts().plot(kind='kde', color='blue', label='test', alpha=.6)

_ = ax5.set_title('Parch distribustion')

_ = ax5.set_xlabel('Parch')

_ = ax5.legend

plt.margins(0.04, ax=ax5)



# distribution of Fare

ax6 = plt.subplot2grid((4,2),(2,1))

_ = train['Fare'].value_counts().plot(kind='kde', color='red', label='train', alpha=.6)

_ = test['Fare'].value_counts().plot(kind='kde', color='blue', label='test', alpha=.6)

_ = ax6.set_title('Fare distribustion')

_ = ax6.set_xlabel('Fare')

_ = ax6.legend()

plt.margins(0.04, ax=ax6)



# distribution of Embarked

ax7 = plt.subplot2grid((4,2),(3,0), colspan=2)

_ = train['Embarked'].value_counts().plot(kind='barh', color='red', label='train', alpha=.6)

_ = test['Embarked'].value_counts().plot(kind='barh', color='blue', label='test', alpha=.6)

_ = ax7.set_title('Embarked distribustion')

_ = ax7.set_xlabel('Count')

_ = ax7.set_ylabel('Embarked')

_ = ax7.legend()

plt.margins(0.04, ax=ax7)
# Class with survival

# It is always better to use stacked chart for bi-variate analysis of categorical vs categorical data.

fig = plt.figure(figsize=(10,5))



survived_dead = pd.DataFrame({'Survived' : train[train.Survived==1].Pclass.value_counts(),

                         'Dead' : train[train.Survived==0].Pclass.value_counts()

                        })



_ = survived_dead.plot(kind='bar')

_ = plt.xlabel('Class')

_ = plt.ylabel('Frequency')

_ = plt.title('Class Vs Survived')



# Class 2 and 3 passangar are less likely to survive
# Sex with survival



mf_serv = pd.DataFrame(train.Sex)

mf_serv['Survived'] = train[train.Survived==1].Survived

mf_serv['Dead'] = train[train.Survived==0].Survived

mf_serv = mf_serv.sort_values('Sex').groupby('Sex').agg('count')

_ = mf_serv.plot(kind='bar')

_ = plt.title('Sex Vs Survived')



#Females are more likely to survive
# Well the passenger with its siblings/ spouse and Prarent/ children form a family and 

# number of family member can decide their survival



family = train['SibSp'] + train['Parch'] + 1

fam_serv = pd.DataFrame({'family':family})

fam_serv['Survived'] = np.nan

fam_serv['Dead'] = np.nan

fam_serv['Survived'] = train[train.Survived==1].Survived

fam_serv['Dead'] = train[train.Survived==0].Survived

fam_serv = fam_serv.sort_values('family').groupby('family').agg('count')

_ = fam_serv.plot(kind='bar')

_ = plt.title('Family Vs Survived')



# From graph it is quite clear an individual person has minimal chance of survival 

# compared with when he is in accompany with 2 to 4.
# Age with survival

# we will get three columns 1st one will hold Ages 2nd are they survived and 3rd if they dead



age_serv = train[['Survived','Age']]

fig = sns.FacetGrid(data=age_serv, hue='Survived', aspect=2).map(sns.kdeplot,'Age', shade=True)

_ = fig.add_legend()



# from fig. we can infer the survival changes of childern are high.
# In uni-variate analysis we found that Fare has outliers we will try to normalised this



plt.figure(figsize=(8,3))



fare = train.Fare

fare_norm = np.log(train.Fare.dropna()+1) # added 1 to avoid log(0) as undefine, log(1)=0

fare_plt = pd.DataFrame({'Fare':fare, 'Fare_norm': fare_norm })

fare_plt['points'] = np.arange(len(fare))



ax1 = plt.subplot2grid((1,2),(0,0))

_ = fare_plt.plot(kind='scatter',x='Fare', y='points', title='Fare', ax=ax1)

ax2 = plt.subplot2grid((1,2),(0,1))

_ = fare_plt.plot(kind='scatter',x='Fare_norm', y='points', title='Normalized Fare', ax=ax2)



# Now we will check distribution of Normalized Fare with that of survived



#fare_serv = pd.DataFrame({'Fare' : fare_norm, 

#                          'Survived':train[train.Survived==1].Survived,

#                          'Dead':train[train.Survived==0].Survived,

#                         })



fare_serv = pd.DataFrame({'Fare' : fare_norm, 'Survived':train.Survived })

#ax3 = plt.subplot2grid((2,2),(1,0))

_ = sns.FacetGrid(data=fare_serv,aspect=2).map(sns.boxplot,'Survived','Fare')



#ax4 = plt.subplot2grid((2,2),(1,1))

fig = sns.FacetGrid(data=fare_serv,hue='Survived', aspect=2).map(sns.kdeplot,'Fare')

fig.add_legend()

# it is clear there is higher chances of survival with normalized fare of 3 to 7. 
# Embarked vs survival



emb_serv = pd.DataFrame(train['Embarked'])

emb_serv['Survived'] = train[train.Survived==1].Survived

emb_serv['Dead'] = train[train.Survived==0].Survived

emb_serv = emb_serv.sort_values('Embarked').groupby('Embarked').agg('count')

emb_serv.plot(kind='bar')



# Embark doesn't give clear picture of survival, we can consider survival rate of embarked=C is higher
full = pd.concat([train,test], ignore_index=True)

full.head()
full.tail()
# get missing count for each field

full.isnull().sum()

# total entries are 1309 and we can see data is missing for Age, Embarked and Fare predictors.
# Filing missing Embarked



# We have two option to fill missing value either by using most common value for categorical data 

# or by finding the relation of Embarked with other predictors 

# It will be advisable to use 2nd approach as we can fill more accurate data and can increase the 

# accuracy of prediction.



# let consider Embarked dependency on other variables.



print(full[full.Embarked.isnull()])



Embrk_fill = full[['Embarked','Pclass','Fare']]

Embrk_fill['Fare'] = Embrk_fill['Fare'].dropna().astype(int)

_ = Embrk_fill.boxplot(by=['Embarked','Pclass'])

# draw horizontal line through y=80 

_ = plt.axhline(y=80)

# Set missing value to C as we can clearly see y=80 has better intersection through Embarked=C and Pclass=1

full.Embarked.fillna('C', inplace=True)
full.isnull().sum()
1
# Fill Embarked

train[train.Embarked.isnull()]



# Check Fare,Pclass and Embarked trend

Embrk_fare = train[['Embarked','Pclass','Fare']]

Embrk_fare['Fare'] = Embrk_fare.Fare.astype(int)

#Embrk_fare = Embrk_fare.sort_values('Fare').groupby(['Embarked','Fare']).agg('count').reset_index()

Embrk_fare.boxplot(by=['Embarked','Pclass'])

plt.axhline(y=80, color='green')



_ = train.set_value(train.Embarked.isnull(),'Embarked','C')
# fill missing fare

from collections import Counter

test[test.Fare.isnull()]

common_fare = Counter(test[(test.Pclass==3) & (test.Embarked=='S')].Fare.sort_values().dropna().astype(int)).most_common(5)

obsr_mul = [fare[0]*fare[1] for fare in common_fare]

obsr_num = [fare[1] for fare in common_fare]

new_fare = sum(obsr_mul)/sum(obsr_num)

new_fare

_ = test.set_value(test.Fare.isnull(),'Fare',new_fare)
# fill age

plt.figure(figsize=(8,5))



train_mean = train.Age.mean()

train_std  = train.Age.std()

test_mean = test.Age.mean()

test_std  = test.Age.std()

print(train_mean,train_std)

print(test_mean,test_std)



ax1 = plt.subplot2grid((2,1),(0,0))

train_copy = train.copy()

train_copy.Age.hist(bins=70)

del(train_copy)



test_age = np.random.randint(low=train_mean-train_std,

                             high=train_mean+train_std,

                             size=train.Age.isnull().sum()

                            )

_ = train.set_value(train.Age.isnull(), 'Age', test_age)



train_age = np.random.randint(low=test_mean-test_std,

                              high=test_mean+test_std,

                              size=test.Age.isnull().sum()

                             )



_ = test.set_value(test.Age.isnull(), 'Age', train_age)

ax2 = plt.subplot2grid((2,1),(1,0))

train.Age.hist(bins=70)
# Set numeric value for Sex

# 1 for male and 0 for female



train['Sex_n'] = np.nan

_ = train.set_value(train.Sex=='male','Sex_n',1)

_ = train.set_value(train.Sex!='male','Sex_n',0)



test['Sex_n'] = np.nan

_ = test.set_value(test.Sex=='male','Sex_n',1)

_ = test.set_value(test.Sex!='male','Sex_n',0)



train['Sex_n'] = train['Sex_n'].astype(int)

test['Sex_n'] = test['Sex_n'].astype(int)
# Set normalized value for Fare

train['Fare_n'] = np.nan

test['Fare_n'] = np.nan

_ = train.set_value(train.index,'Fare_n',train.Fare.astype(int))

_ = test.set_value(test.index,'Fare_n',test.Fare.astype(int))
# Set numeric values for Embarked

# S=0, C=1 and Q=2

train['Embarked_n'] = np.nan

_ = train.set_value(train.Embarked=='S','Embarked_n',0)

_ = train.set_value(train.Embarked=='C','Embarked_n',1)

_ = train.set_value(train.Embarked=='Q','Embarked_n',0)



test['Embarked_n'] = np.nan

_ = test.set_value(test.Embarked=='S','Embarked_n',0)

_ = test.set_value(test.Embarked=='C','Embarked_n',1)

_ = test.set_value(test.Embarked=='Q','Embarked_n',0)



train['Embarked_n'] = train['Embarked_n'].astype(int)

test['Embarked_n'] = test['Embarked_n'].astype(int)
# create copy of pclass

train['class'] = train['Pclass']

test['class'] = test['Pclass']
# create variable taht list male fimale and child

# male=0, female=1, child=2

train['MFC'] = np.nan

_ = train.set_value((train.Sex=='male') & (train.Age>18), 'MFC', 0)

_ = train.set_value((train.Sex!='male') & (train.Age>18), 'MFC', 1)

_ = train.set_value(train.Age<=18, 'MFC', 2)



test['MFC'] = np.nan

_ = test.set_value((test.Sex=='male') & (test.Age>18), 'MFC', 0)

_ = test.set_value((test.Sex!='male') & (test.Age>18), 'MFC', 1)

_ = test.set_value(test.Age<=18, 'MFC', 2)



train['MFC'] = train['MFC'].astype(int)

test['MFC'] = test['MFC'].astype(int)
# setup family count

# 'low',0 for count =1, 'med',1 for count =(2,3,4), 'high',2 for count>4 

train['family'] = np.nan

test['family'] = np.nan

train['family'] = (train['SibSp'] + train['Parch'] + 1).astype(int)

test['family'] = (test['SibSp'] + test['Parch'] + 1).astype(int)





train['family_size'] = 1

_ = train.set_value(train['family']==1,'family_size',0)

_ = train.set_value(train['family']>4,'family_size',2)

test['family_size'] = 1

_ = test.set_value(test['family']==1,'family_size',0)

_ = test.set_value(test['family']>4,'family_size',2)
train.info()
from sklearn.ensemble import RandomForestClassifier



train_features = train[['Sex_n','class','family_size']]

test_features = test[['Sex_n','class','family_size']]

train_target = train.Survived

rand_forest = RandomForestClassifier(n_estimators=100,#random_state=42, #criterion='entropy', 

                                     min_samples_split=2, #oob_score=True, min_samples_leaf=12

                                     max_depth= 10

                                    )

rand_forest = rand_forest.fit(train_features,train_target)

predicted = rand_forest.predict(test_features)

print(rand_forest.score(train_features,train_target))
# write predicted values for submission

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predicted

    })

submission.to_csv('Predicted_Survival.csv', index=False)
rand_forest.feature_importances_
array([ 0.59298075,  0.05196973,  0.24705396,  0.10799557])