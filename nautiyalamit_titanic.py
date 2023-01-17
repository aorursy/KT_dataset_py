%matplotlib inline

import  numpy as np 



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont

import xgboost as xgb

from multiprocessing import Pool

## access the data 

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

## analyse the data 

gender_submission.info()

print("--------------------------------------------------")

test.info()

print("--------------------------------------------------")

train.info()

original_train = train.copy()
## initial anysis says that this is regression issue as  features can increaSE in y. And there lot of dependency in the data  



#lets dig deeper in the data 

train.describe(include='all')
## we will perform data visualization to see the  relation between various features:
##grouping the data 

sns.set()

analysis1=train.groupby(['Sex'])[['Survived']].mean()

print(analysis1)

analysis1.plot(kind='bar',stacked=True)
analysis2=train.groupby(['Sex', 'Pclass'])['Survived'].aggregate('mean').unstack()

print(analysis2)

analysis2.plot(kind='bar')
age = pd.cut(train['Age'], [0, 18, 80])

#titanic.pivot_table('survived', index='sex', columns='class')

#  call signature as of Pandas 0.18

# DataFrame.pivot_table(data, values=None, index=None, columns=None,

#                       aggfunc='mean', fill_value=None, margins=False,

#                       dropna=True, margins_name='All')



analysis3=train.pivot_table('Survived', ['Sex', age], 'Pclass')

print(analysis3)

analysis3.plot(kind='bar')
fare = pd.qcut(train['Fare'], 2)

analysis4=train.pivot_table('Survived', ['Sex', age], [fare, 'Pclass'])

print(analysis4)

analysis4.plot(kind='bar',stacked=True)


analysis5=train.pivot_table(index='Sex', columns='Pclass',

                    aggfunc={'Survived':sum, 'Fare':'mean'})

print(analysis5)

analysis5.plot(kind='bar')
analysis6=train.pivot_table('Survived', index='Sex', columns='Pclass', margins=True)

print(analysis6)

analysis6.plot(kind='bar')
#Feature engineering


# if we see the data and plotting above we can see that cabin has a relationship between the survival rates. 

#but it is also to be noticed that the data  in cabin is not avialble and has lot of missing data 

## let me create sample column for addressing the issue  to convert into binary data



#pandas.apply(): Apply a function to each row/column in Dataframe

##having cabin = Ycabin

train['YCabin']=train['Cabin'].apply(lambda a :0 if type(a)==float else 1)



test['YCabin']=test['Cabin'].apply(lambda a :0 if type(a)==float else 1)



##  let me not deal with the ----> fare data<------ as there is not very specific corelation between the survival rate  ....



## but age should contain tnull for better accuracy

combined_DataSet=[train,test]



##create a new feature= faminly size 

for data in combined_DataSet:

    data['family_size']=data['SibSp'] + data['Parch'] +1



    # Remove all NULLS in the Embarked column

for data in combined_DataSet:

    data['Embarked'] = data['Embarked'].fillna('S')

# Remove all NULLS in the Fare column

for data in combined_DataSet:

    data['Fare'] = data['Fare'].fillna(train['Fare'].median())

    





# Create new feature Alone from FamilySize

for data in combined_DataSet:

    data['Alone'] = 0

    data.loc[data['family_size'] == 1, 'Alone'] = 1



for data in combined_DataSet:

    age_avg = data['Age'].mean()

    age_std = data['Age'].std()

    age_null_count = data['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    

    data.loc[np.isnan(data['Age']), 'Age'] = age_null_random_list

    data['Age'] = data['Age'].astype(int)



## Define function to extract titles from passenger names

def getTitle(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



for data in combined_DataSet:

    data['Title'] = data['Name'].apply(getTitle)

    

# Group all non-common titles into one single grouping "Rare"

for data in combined_DataSet:

    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Ms', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')



for data in combined_DataSet:

    # Map Sex

    data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Map titles

    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}

    data['Title'] = data['Title'].map(title_mapping)

    data['Title'] = data['Title'].fillna(0)



    # Map Embarked

    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Map Fare

    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2

    data.loc[ data['Fare'] > 31, 'Fare']        = 3

    data['Fare'] = data['Fare'].astype(int)

    

    # MAp Age

    data.loc[ data['Age'] <= 16, 'Age']     = 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[ data['Age'] > 64, 'Age'] ;
# Feature selection: remove variables not reqired now



train = train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp'], axis = 1)



test = test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp'], axis = 1)





train.head()

test.head()
train[['Title', 'Survived']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum'])



# Since "Survived" is a binary class (0 or 1), these metrics grouped by the Title feature represent:

    # MEAN: survival rate

    # COUNT: total observations

    # SUM: people survived



# title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title(' Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean', 'count', 'sum'])

# Since Survived is a binary feature, this metrics grouped by the Sex feature represent:

    # MEAN: survival rate

    # COUNT: total observations

    # SUM: people survived

    

# sex_mapping = {{'female': 0, 'male': 1}}
# Let's use our 'original_train' dataframe to check the sex distribution for each title.

# We use copy() again to prevent modifications in out original_train dataset

title_and_sex = original_train.copy()[['Name', 'Sex']]



# Create 'Title' feature

title_and_sex['Title'] = title_and_sex['Name'].apply(getTitle)



# Map 'Sex' as binary feature

title_and_sex['Sex'] = title_and_sex['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



# Table with 'Sex' distribution grouped by 'Title'

title_and_sex[['Title', 'Sex']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum'])



# Since Sex is a binary feature, this metrics grouped by the Title feature represent:

    # MEAN: percentage of men

    # COUNT: total observations

    # SUM: number of men
# Define function to calculate Gini Impurity

def get_gini_impurity(survived_count, total_count):

    survival_prob = survived_count/total_count

    not_survival_prob = (1 - survival_prob)

    random_observation_survived_prob = survival_prob

    random_observation_not_survived_prob = (1 - random_observation_survived_prob)

    mislabelling_survided_prob = not_survival_prob * random_observation_survived_prob

    mislabelling_not_survided_prob = survival_prob * random_observation_not_survived_prob

    gini_impurity = mislabelling_survided_prob + mislabelling_not_survided_prob

    return gini_impurity

# Gini Impurity of starting node

gini_impurity_starting_node = get_gini_impurity(342, 891)

gini_impurity_starting_node


# Gini Impurity decrease of node for 'male' observations

gini_impurity_men = get_gini_impurity(109, 577)

gini_impurity_men

# Gini Impurity decrease if node splited for 'female' observations

gini_impurity_women = get_gini_impurity(233, 314)

gini_impurity_women
# Gini Impurity decrease if node splited by Sex

men_weight = 577/891

women_weight = 314/891

weighted_gini_impurity_sex_split = (gini_impurity_men * men_weight) + (gini_impurity_women * women_weight)



sex_gini_decrease = weighted_gini_impurity_sex_split - gini_impurity_starting_node

sex_gini_decrease
# Gini Impurity decrease of node for observations with Title == 1 == Mr

gini_impurity_title_1 = get_gini_impurity(81, 517)

gini_impurity_title_1
# Gini Impurity decrease if node splited for observations with Title != 1 != Mr

gini_impurity_title_others = get_gini_impurity(261, 374)

gini_impurity_title_others

# Gini Impurity decrease if node splited for observations with Title == 1 == Mr

title_1_weight = 517/891

title_others_weight = 374/891

weighted_gini_impurity_title_split = (gini_impurity_title_1 * title_1_weight) + (gini_impurity_title_others * title_others_weight)



title_gini_decrease = weighted_gini_impurity_title_split - gini_impurity_starting_node

title_gini_decrease

cv = KFold(n_splits=10)            # Desired number of Cross Validation folds

accuracies = list()

max_attributes = len(list(test))

depth_range = range(1, max_attributes + 1)



# Testing max_depths from 1 to max attributes

# Uncomment prints for details about each Cross Validation pass

for depth in depth_range:

    fold_accuracy = []

    tree_model = tree.DecisionTreeClassifier(max_depth = depth)

    print("Current max depth: ", depth, "\n")

    for train_fold, valid_fold in cv.split(train):

        f_train = train.loc[train_fold] # Extract train data with cv indices

        f_valid = train.loc[valid_fold] # Extract valid data with cv indices



        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), 

                               y = f_train["Survived"]) # We fit the model with the fold train data

        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), 

                                y = f_valid["Survived"])# We calculate accuracy with the fold validation data

        fold_accuracy.append(valid_acc)



    avg = sum(fold_accuracy)/len(fold_accuracy)

    accuracies.append(avg)

    print("Accuracy per fold: ", fold_accuracy, "\n")

    print("Average accuracy: ", avg)

    print("\n")

    

# show results conveniently

df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})

df = df[["Max Depth", "Average Accuracy"]]

print(df.to_string(index=False))

