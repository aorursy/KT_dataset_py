# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Import CSV FILE

df = pd.read_csv('../input/NationalNames.csv')

df = df.drop_duplicates(subset="Name")

df.head()
# Apply some methods to it

# Check the gender and return numeric value

def checkGender(gender):

    if gender == 'F':

        return 0

    return 1

# Check if the name ends in vowel

def checkVowelEnd(name):

    if name[-1] in "aeiou":

        return 0

    return 1
# Check if name starts in vowel and ends with consonant

def checkVowelStartConsonantEnd(name):

    if name[0] in "AEIOU" and not name[-1] in "aeiou":

        return 0

    return 1
# Check if name starts with a vowel 

def checkVowelStart(name):

    if name[0] in "AEIOU":

        return 0

    return 1
# Check length of name

def checkNameLength(name):

    return len(name)
# check if name is short or long

def checkShortLongName(name):

    if len(name) < 5:

        return 0

    return 1
df["GenderValue"] = df["Gender"].apply(checkGender)

df["VowelEnd"] = df["Name"].apply(checkVowelEnd)

df["VowelStart"] = df["Name"].apply(checkVowelStart)

df["NameLength"] = df["Name"].apply(checkNameLength)

df["Short/Long"] = df["Name"].apply(checkShortLongName)

df["ConsonantEnd/VowelStart"] = df["Name"].apply(checkVowelStartConsonantEnd)

df.head()
training_data = df[["GenderValue", "VowelStart", "VowelEnd", "NameLength", "Short/Long", "ConsonantEnd/VowelStart"]]

training_data.head()
train, test = train_test_split(training_data, test_size = 0.15)
clf = DecisionTreeClassifier(max_leaf_nodes=21)

clf = clf.fit(train[["VowelStart", "VowelEnd", "NameLength", "Short/Long", "ConsonantEnd/VowelStart"]], train["GenderValue"])
predictions = clf.predict(test[["VowelStart", "VowelEnd", "NameLength", "Short/Long", "ConsonantEnd/VowelStart"]])

accuracy_score(test["GenderValue"], predictions)