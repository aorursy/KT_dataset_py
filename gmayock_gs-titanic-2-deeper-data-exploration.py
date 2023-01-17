# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
training_data = pd.read_csv("../input/train.csv")
# Let's import the tools
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() # sets as default for plots
training_data['female'] = training_data.Sex == 'female'
training_data.head()
# This time, before filling NaN (such as age), I am going to look to see if there are other data which provide signal
training_data.isnull().sum()
# Let's see how many unique values there are
n = 0
for col in training_data:
    print (training_data.dtypes.index[n], "         ", len(training_data[col].unique()))
    n += 1
# one quick way to view the data is to show a bar chart
survived = training_data[training_data['Survived']==1]['Sex'].value_counts()
perished = training_data[training_data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived, perished])
df.index = ['Survived', 'Perished']
df.plot(kind='bar', stacked = False, figsize = (6,4.5));
# however it gets repetitive to continually type so much, so let's put it in a function
def show_bar_chart(feature):
    survived = training_data[training_data['Survived']==1][feature].value_counts()
    perished = training_data[training_data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, perished])
    df.index = ['Survived', 'Perished']
    df.plot(kind='bar', stacked = False, figsize = (6,4.5));
show_bar_chart('Pclass')
show_bar_chart('SibSp')
survived = training_data[training_data['Survived']==1]['SibSp'].value_counts()
perished = training_data[training_data['Survived']==0]['SibSp'].value_counts()
df = pd.DataFrame([survived, perished])
df.index = ['Survived', 'Perished']
g = df.plot(kind='bar', stacked = False, figsize = (6,4.5))
g.set(ylim=(0,20));
show_bar_chart('Parch')
survived = training_data[training_data['Survived']==1]['Parch'].value_counts()
perished = training_data[training_data['Survived']==0]['Parch'].value_counts()
df = pd.DataFrame([survived, perished])
df.index = ['Survived', 'Perished']
g = df.plot(kind='bar', stacked = False, figsize = (6,4.5))
g.set(ylim=(0,70));
show_bar_chart('Embarked')
# Let's use a for loop to splice the title out of the strings. I've seen people use regex and str.extract, but I am not yet experienced with those so I'll just build my own way to do it
title = []
for i in training_data['Name']:
    period = i.find(".")
    comma = i.find(",")
    title_value = i[comma+2:period]
    title.append(title_value)
training_data['title'] = title
# Sweet, that works. Let's use a dictionary to make sure that synonyms typed differently (like Ms vs Miss) are understood the same.
# I want to make sure I know how much of each title there is
from collections import Counter
title_count = pd.DataFrame([Counter(title).keys(), Counter(title).values()])
title_count.head()
rev_list = []
for row in training_data.title:
    if row == "Rev":
        rev_list.append(True)
    else:
        rev_list.append(False)

is_rev = pd.Series(rev_list)
is_rev.head()
training_data[is_rev]
masters_list = []
for row in training_data.title:
    if row == "Master":
        masters_list.append(True)
    else:
        masters_list.append(False)

is_master = pd.Series(masters_list)
training_data[is_master]
survived = training_data[training_data['Survived']==1]['title'].value_counts()
perished = training_data[training_data['Survived']==0]['title'].value_counts()
df = pd.DataFrame([survived, perished])
df.index = ['Survived', 'Perished']

# fill NaN with 0
df = df.fillna(0)

#calc survival rates
survival_rates = []
for i in df.dtypes.index:
    survival_rate = round((df[i][0]/(df[i][0] + df[i][1])),4)
    survival_rates.append([i,survival_rate])
# making a df for survival rates
dfsr = pd.DataFrame(survival_rates)
dfsr.columns = ['title', 'survival_rate']
dfsr
# When ready, we'll clean the titles into a new cleaned_title list
title_arr = pd.Series(title)
title_dict = {
    'Mr' : 'Mr',
    'Mrs' : 'Mrs',
    'Miss' : 'Miss',
    'Master' : 'Master',
    'Don' : 'Formal',
    'Rev' : 'Religious',
    'Dr' : 'Academic',
    'Mme' : 'Mrs',
    'Ms' : 'Miss',
    'Major' : 'Formal',
    'Lady' : 'Formal',
    'Sir' : 'Formal',
    'Mlle' : 'Miss',
    'Col' : 'Formal',
    'Capt' : 'Formal',
    'the Countess' : 'Formal',
    'Jonkheer' : 'Formal',
}

cleaned_title = title_arr.map(title_dict)
training_data['cleaned_title'] = cleaned_title

cleaned_title_count = pd.DataFrame([Counter(cleaned_title).keys(), Counter(cleaned_title).values()])
cleaned_title_count.head()

# I'm not sure if this replacement/dict is the best.
survived = training_data[training_data['Survived']==1]['cleaned_title'].value_counts()
perished = training_data[training_data['Survived']==0]['cleaned_title'].value_counts()
df = pd.DataFrame([survived, perished])
df.index = ['Survived', 'Perished']

# fill NaN with 0
df = df.fillna(0)

#calc survival rates
survival_rates_cleaned = []
for i in df.dtypes.index:
    survival_rate = round((df[i][0]/(df[i][0] + df[i][1])),4)
    survival_rates_cleaned.append([i,survival_rate])

# making a df for survival rates
dfsrf = pd.DataFrame(survival_rates_cleaned)
dfsrf.columns = ['title', 'survival_rate']
dfsrf
training_data.isnull().sum()
embarked_nulls_list = []
for row in training_data.Embarked:
    if pd.isnull(row):
        embarked_nulls_list.append(True)
    else:
        embarked_nulls_list.append(False)

embarked_null = pd.Series(embarked_nulls_list)
training_data[embarked_null]
survived = training_data[training_data['Survived']==1]['Embarked'].value_counts()
perished = training_data[training_data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived, perished])
df.index = ['Survived', 'Perished']

# fill NaN with 0
df = df.fillna(0)

#calc survival rates
survival_rates_cleaned = []
for i in df.dtypes.index:
    survival_rate = round((df[i][0]/(df[i][0] + df[i][1])),4)
    survival_rates_cleaned.append([i,survival_rate])

# making a df for survival rates
dfsre = pd.DataFrame(survival_rates_cleaned)
dfsre.columns = ['embarked', 'survival_rate']
dfsre
test_data = pd.read_csv("../input/test.csv")
test_data.isnull().sum()
training_data['Embarked'] = training_data['Embarked'].fillna("S") 
# OHE 
training_data = pd.concat([training_data, pd.get_dummies(training_data['Embarked'])], axis = 1)

# Port mapping in order of passenger pickups
port = {
    'S' : 1,
    'C' : 2,
    'Q' : 3
}
training_data['pickup_order'] = training_data['Embarked'].map(port)
training_data.head()
training_data.isnull().sum()
age_nulls = []
for row in training_data.Age:
    if pd.isnull(row):
        age_nulls.append(True)
    else:
        age_nulls.append(False)

age_null = pd.Series(age_nulls)
training_data[age_null]
# First I group by some relevant features. I don't want to overfit, but using gender, title, and ticket class, we can get a fair number of groups.
grouped = training_data.groupby(['female','Pclass', 'cleaned_title'])  
# I can also view the median age by group
grouped.Age.median()
# We'll check the counts to make sure it's not toooo overfit
grouped.Age.count()
# It looks like there may be some overfit with some of the more rare titles, but there's not much we can do about that. 
# Now I want to fill the NaN will the medians from those groups
training_data['Age'] = grouped.Age.apply(lambda x: x.fillna(x.median()))
training_data.isnull().sum()
# Voila!
# Encode childhood
training_data['child'] = training_data.Age < 16
# Using OHE to assign each title its own column as they are categorical but not necessarily ordinal
training_data = pd.concat([training_data, pd.get_dummies(training_data['cleaned_title'])], axis = 1)
training_data.head()
training_data['Cabin'] = training_data['Cabin'].fillna("U") 
# I am sure there are ways to do this with less code but I don't know those yet.
cabin_group = []
for i in training_data['Cabin']:
    cabin_group.append(i[0])

training_data['cabin_group'] = cabin_group
# Dummies for cabin group
training_data = pd.concat([training_data, pd.get_dummies(training_data['cabin_group'])], axis = 1)
training_data['family_size'] = training_data.Parch + training_data.SibSp + 1
training_data.head()
# Dummies for Pclass
training_data = pd.concat([training_data, pd.get_dummies(training_data['Pclass'])], axis = 1)
training_data.head()
training_data.dtypes.index
# wait, what's this? Are $0 fare people crew members? Does that provide signal?
min(training_data['Fare'].unique())
# How many were there? ... 15
free_board = []
for row in training_data.Fare:
    if row == 0:
        free_board.append(True)
    else:
        free_board.append(False)

was_free = pd.Series(free_board)
training_data[was_free]
# Well, Johnkheer. John Reuchlin was most assuredly not a crew member, which is what I was hoping free tickets would indicate. But hm, what's that LINE ticket?
LINE_ticket = []
for row in training_data.Ticket:
    if row == "LINE":
        LINE_ticket.append(True)
    else:
        LINE_ticket.append(False)

was_LINE = pd.Series(LINE_ticket)
training_data[was_LINE]
# here are the models I'll use for a first-try
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# here are the metrics I'll check them with
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
# and the code to split the test/train data
from sklearn.model_selection import train_test_split

# I also want to split the training_data dataframe into a training and testing portion
train_baseline, test_baseline = train_test_split(training_data, random_state = 0)

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'female', 'child', 'S', 'C', 'Q', 'pickup_order', 'Academic', 'Formal', 'Master', 'Miss', 'Mr', 'Mrs', 'Religious', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U', 'family_size', 1, 2, 3]
target = 'Survived'
# Define the models
################################################################################################### NOTE - I cannot justify setting random_state=0 for any reason other than reproducability of results
model01 = RandomForestClassifier(random_state=0);
model02 = DecisionTreeClassifier(random_state=0);
model03 = LogisticRegression();

# Fit the models
model01.fit(train_baseline[features], train_baseline[target]);
model02.fit(train_baseline[features], train_baseline[target]);
model03.fit(train_baseline[features], train_baseline[target]);

# Define a function to make reading recall score easier
def printTestRecall(model_number):
    print("Test Recall: ", round(recall_score(test_baseline[target], model_number.predict(test_baseline[features]))*100,2), "%")

# Print results
print("Random Forest Classifier")
printTestRecall(model01);
print("\n\nDecision Tree Classifier")
printTestRecall(model02);
print("\n\nLogistic Regression")
printTestRecall(model03);
