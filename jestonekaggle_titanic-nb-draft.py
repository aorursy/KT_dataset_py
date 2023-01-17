# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Imports







# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# preview the data

titanic_df.head()
print("Total passengers represented in train+test data is")

print(titanic_df["PassengerId"].count()+test_df["PassengerId"].count())
wiki_list=["The Countess of Rothes", "Thomas Dyer-Edwards", "Clementina Dyer-Edwards", "Gladys Cherry",

          "Sir Cosmo Duff-Gordon", "Lady Duff Gordon","Colonel Archibald Gracie IV", "Lord Pirrie", "Thomas Andrews",

          "Colonel John Jacob Astor", "Benjamin Guggenheim", "Isidor Straus", "Ida Straus", "George Dennick Wick",

          "George Dunton Widener","John Thayer", "Marian Thayer", "Charles Hayes","William Ernest Carter", "Lucile Carter",

          "Margaret Brown", "Karl Behr", "Dorothy Gibson", "Edward Austin Kent","Major Archibald Butt", 

          "Mauritz Hakan Bjornstrom-Steffansson","Rev John Harper", "Lawrence Beesley", "Joseph Laroche",

          "Simonne Laroche","Louise Laroche", "Juliette Laroche", "Margaret Hays", "John Sage", "Annie Sage",

          "Milvina Dean"]



#wiki notes several people who didn't sail - just curious, could any of these still been included in the dataset?

did_not_sail = ["Theodore Dreiser","Henry Clay Frick","Milton S. Hershey","Guglielmo Marconi","John Pierpont Morgan",

               "Edgar Selwyn","Hugh Sullivan","Alfred Gwynne Vanderbilt"]
#create a merged train+test set to do some of this analysis



frames = [titanic_df,test_df]

titanic_all = pd.concat(frames, keys = ['tr','te'])



#to avoid a more complicated 'comparison' of the names from my list with the passenger list, we'll just

#  start with last names



wiki_list_last = [ x.split()[-1] for x in wiki_list ]

did_not_sail_last = [ x.split()[-1] for x in did_not_sail ]



#pull the last names and strip out the comma (since the dataset names are of the form "Surname, yada..." )

data_list_last = [ x.split()[0].rstrip(',') for x in titanic_all["Name"] ]



pass_found = list(set(wiki_list_last) & set(data_list_last))

pass_not_found = list(set(wiki_list_last)-set(data_list_last))





print("These 'infamous' people were found:")

print(pass_found)

print()

print("These 'infamous' people were not found:")

print(pass_not_found)





# this checks to see if anyone who DIDN'T sail, but had a ticket, was in the dataset - alas, they were not

dns_found = list(set(did_not_sail_last) & set(data_list_last))

print()

print("Yep, the folks with tickets who never boarded weren't included in the data...")

print(dns_found)
print(titanic_all.sort_values(by='Fare', ascending=False).head())



#training only...



# counting male/female by knowing categorical values ahead of time

print("male", titanic_df[titanic_df["Sex"] == "male"]["Sex"].count())

print("female",titanic_df[titanic_df["Sex"] == "female"]["Sex"].count())



# a more natural "pandas" way to do the same thing without having to manually determine the categories ahead of time

print(titanic_df["Sex"].value_counts())

# Now we want to get % males survived, % females survived

print("male survivors (%)",titanic_df[(titanic_df["Sex"] == "male") & (titanic_df["Survived"] == 1)]["Sex"].count() /titanic_df["Sex"].value_counts()['male'])

print("female survivors (%)",titanic_df[(titanic_df["Sex"] == "female") & (titanic_df["Survived"] == 1)]["Sex"].count() /titanic_df["Sex"].value_counts()['female'])