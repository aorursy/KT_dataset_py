import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sbn # plotting tool



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



full = [train, test]



train.head(10)
# Normalize Sex Feature, as we know women lived more than man



for df in full:

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    
# Add the median age of each sex for NAs and makes ages as integers

male_age_medians = pd.concat([train, test])[lambda x: x.Sex == 0].median()['Age']

female_age_medians = pd.concat([train, test])[lambda x: x.Sex == 1].median()['Age']



for df in full:

    df.loc[ df.Sex == 0, 'Age' ] = df.loc[ df.Sex == 0, 'Age' ].fillna(male_age_medians)

    df.loc[ df.Sex == 1, 'Age' ] = df.loc[ df.Sex == 1, 'Age' ].fillna(female_age_medians)

    

    df.Age = df.Age.astype(int)

    

train.head(10)
# To avoid overfitting, we set ages in some groups



# [0] -> 0 - 12 child

# [1] -> 12 - 18 teenager

# [2] -> 18 - 60 adult

# [3] -> 60+ aged   

for df in full:

    df['AgeGroup'] = None

    df.loc[ df.Age < 10, 'AgeGroup' ] = 0

    df.loc[ df.Age >= 10, 'AgeGroup'] = 1

    #df.loc[ (df.Age >= 12) & (df.Age < 18), 'AgeGroup' ] = 1

    #df.loc[ (df.Age >= 18) & (df.Age < 60), 'AgeGroup'] = 2

    #df.loc[ df.Age >= 60, 'AgeGroup'] = 3



train[['AgeGroup', 'Survived']].groupby('AgeGroup').mean()
#Done with ages, Let's check Fare



sbn.set(style = 'whitegrid')

sbn.barplot(x = 'Fare', y = 'Survived', data = train[['Fare', 'Survived']])
# Too bad, let's try grouping the fares into bands



fare_df = train[['Fare', 'Survived']].copy()

fare_df['FareBand'] = None



groups = np.linspace(fare_df.Fare.min(), fare_df.Fare.max(), 5)



fare_df.groupby(pd.cut(fare_df.Fare, groups)).mean()
def fare_band_map(fare):

    if fare < 128.082:

        return 0

    if fare < 256.165:

        return 1

    if fare < 384.247:

        return 2

    

    return 3



# Set FareBand values

fare_df['FareBand'] = fare_df.Fare.apply(fare_band_map)



#fa[['FareBand', 'Survived']].groupby('FareBand').mean()

sbn.barplot(x = 'FareBand', y = 'Survived', data = fare_df[['FareBand', 'Survived']])
#Apply fare changes to our modeling datasets

for df in full:

    df['FareBand'] = None

    df['FareBand'] = df.Fare.apply(fare_band_map)

    

train.head(10)
#Adding family size into account

for df in full:

    df['FamilyMembersCount'] = df.SibSp + df.Parch + 1

    

train[['FamilyMembersCount', 'Survived']].groupby('FamilyMembersCount').mean()
# From 2 to 4 survival rate is high, let's group the numbers taking this in account



def family_size_map(members_count):

    if members_count == 1:

        return 'alone'

    #if members_count == 4:

    #    return 'medium'

    if members_count < 5:

        return 'small'

    

    return 'large'

        

for df in full:

    df['FamilySize'] = df.FamilyMembersCount.apply(family_size_map)

    

train[['FamilySize', 'Survived']].groupby('FamilySize').mean()
# Mapping family size to numerical values



for df in full:

    df.FamilySize = df.FamilySize.map({

        'alone': 0,

        'small': 1,

        'medium': 2,

        'large': 3

    })

    

train[['FamilySize', 'Survived']].groupby('FamilySize').mean()
#Now let's check the titles for those who were traveling

import re



#Based on by https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic

title_df = train[['Name', 'Survived']].copy()



# Grab title from passenger names

title_df['Title'] = title_df.Name.apply(lambda x: re.sub('(.*, )|(\\..*)', '', x) )



# Titles with very low cell counts to be combined to "rare" level

rare_titles = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']



title_df.Title = title_df.Title.apply(lambda t: 'Rare Title' if t in rare_titles else t)

title_df.Title = title_df.Title.apply(lambda t: 'Miss' if t in ['Mlle', 'Ms'] else t)

title_df.Title = title_df.Title.apply(lambda t: 'Mrs' if t == 'Mme' else t)



title_df[['Title', 'Survived']].groupby('Title').mean()
# Send titles to train and test datasets



for df in full:

    df['Title'] = None

    

    df.Title = df.Title.apply(lambda t: 'Rare Title' if t in rare_titles else t)

    df.Title = df.Title.apply(lambda t: 'Miss' if t in ['Mlle', 'Ms'] else t)

    df.Title = title_df.Title.apply(lambda t: 'Mrs' if t == 'Mme' else t)

    

train[['Title', 'Survived']].groupby('Title').mean()
# Mapping to numeric values

for df in full:

    df.Title = df.Title.map({

        'Master': 0,

        'Miss': 1,

        'Mr': 2,

        'Mrs': 3,

        'Rare Title': 4

    })

    

train[['Title', 'Survived']].groupby('Title').mean()
# Checking what we have so far

train.describe(include='all')
#Dropping common columns we won't use as a feature



test_output_df = test.copy() # Saves a copy as we'll need PassengerID in the future



for df in full:

    for column in ['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 

                   'Fare', 'Cabin', 'Embarked', 'FamilyMembersCount']:

        if column in df:

            df.drop(column, axis=1, inplace=True)

        

train
# First let's prepare our data to train



from sklearn.utils import shuffle



# Shuffling train dataset

train_sf = shuffle(train.copy())



# Splitting the train dataset into train and test blocks

split_idx = int(len(train_sf) * 0.8)

train_features = train_sf[:split_idx]

test_features = train_sf[split_idx:]



# Isolating the labels

train_labels = train_features['Survived']

train_features = train_features.drop('Survived', axis=1)



test_labels = test_features['Survived']

test_features = test_features.drop('Survived', axis=1)



# Let's try a few classifiers

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



classifiers = [

    SVC(C=100000), 

    SVC(C=4), 

    SVC(),

    DecisionTreeClassifier(min_samples_split=50, presort=True),

    DecisionTreeClassifier(),

    GaussianNB(),

    RandomForestClassifier(),

    RandomForestClassifier(n_estimators = 50),

    RandomForestClassifier(min_samples_split=50),

    RandomForestClassifier(n_estimators = 50, min_samples_split=50)

]



for clf in classifiers:

    print ("======================================================")

    print(clf)

    clf.fit(train_features, train_labels)

    prediction = clf.predict(test_features)

    accuracy = accuracy_score(test_labels, prediction)

    print ("\nSCORE: {0}".format(accuracy))

    

classifier =     RandomForestClassifier()

classifier.fit(train_features, train_labels)

prediction = classifier.predict(test)



test_output_df['Survived'] = prediction



output_df = test_output_df[['PassengerId', 'Survived']].copy()
from datetime import datetime



output_file = "submission.%s.csv" % (datetime.now().strftime('%Y%m%d.%H%M'))



output_df.to_csv(output_file, index = False)



print ("Output writen to %s" % (output_file))