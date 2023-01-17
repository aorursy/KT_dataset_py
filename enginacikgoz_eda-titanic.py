# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


train = pd.read_csv("../input/train.csv")

train.drop(["PassengerId"], axis=1, inplace=True)



test = pd.read_csv("../input/test.csv")

test.drop(["PassengerId"], axis=1, inplace=True)



test_result = pd.read_csv("../input/gender_submission.csv")

test_result.drop(["PassengerId"], axis=1, inplace=True)



df_tmp = pd.concat([test, test_result], axis=1)



df = pd.concat([train,df_tmp], axis=0, sort=False)



df.reset_index(inplace=True)

df.drop(["index"], inplace=True, axis=1)



del df_tmp


# Adding Average Fare. 1 = Over payment, 0 = Normal Price.

favg = df.Fare.mean()

df["Average_fare"] = [ 1 if i > favg else 0 for i in df.Fare ]





def set_age(x):

    if x in range(0,4):

        return "0-3"



    if x in range(4,8):

        return "4-7"



    if x in range(8,13):

        return "8-12"



    if x in range(13,18):

        return "13-17"



    if x in range(18,26):

        return "18-25"



    if x in range(25,31):

        return "25-30"



    if x in range(30,41):

        return "30-40"



    if x in range(40,51):

        return "40-50"



    if x > 50:

        return ">50"





df["Age_range"] = df.Age.apply(set_age)

# %% Passenger info



age_x = df.Age_range.value_counts().index

age_y = df.Age_range.value_counts().values



plt.figure(figsize=(18,7))



f = sns.barplot(age_x, age_y, alpha=.7, palette="cool")

plt.title("Age range of all passangers", size=15)

plt.xlabel("Age range", size=16)

plt.ylabel("Number of people", size=14)

plt.xticks(size=15)

plt.yticks(size=14)

plt.show()
overpay = df[ (df.Average_fare == 1)]

normal = df[ (df.Average_fare == 0)]



# Survive rate by genders



plt.figure(figsize=(18,7))

plt.style.use('dark_background')



f = sns.swarmplot(df.Sex, df.Age, hue=df.Survived, palette="Set1", dodge=True, size=8)

f.legend(title="Is Survived?", labels=["No","Yes"])

plt.title("Survive by genders", size=16)

plt.xlabel("Sex", size=18)

plt.ylabel("Age", size=18)

plt.xticks(size=18)

plt.yticks(size=15)

plt.show()

# Survive rate by price of tickets

plt.figure(figsize=(14,7))

plt.style.use('dark_background')



plt.title("Survive rate by price of tickets", size=16)

sns.countplot(normal.Survived, label="Normal", color="purple", alpha=.7)

sns.countplot(overpay.Survived, label="Overpay", color="red", alpha=.7)

plt.xlabel("0 = Not Survived"+" "*64+"1 = Survived", size=14)

plt.ylabel("Survive rate",size=16)

plt.xticks(size=15)

plt.yticks(size=15)

plt.legend()

plt.show()

# # Payment rate by genders

plt.figure(figsize=(14,8))

plt.style.use('dark_background')



f = sns.swarmplot(df.Sex, df.Fare, hue=df.Pclass, dodge=True, palette="Set1", size=6)

plt.title("Price paid from by genders", size=18)

plt.xlabel("Sex", size=18)

plt.ylabel("Fare", size=18)

plt.xticks(size=17)

plt.yticks(size=15)

f.legend(title="Ticket Class")

plt.show()

# Ticket class by embark

plt.figure(figsize=(18,8))

plt.style.use('dark_background')



f = sns.swarmplot(df.Pclass, df.Age, hue=df.Embarked, palette="Set1", dodge=True)

plt.title("Ticket class by embark", size=20)

plt.xlabel("Ticket class", size=17)

plt.ylabel("Age", size=18)

plt.xticks(size=22)

plt.yticks(size=13)

f.legend(title="Embarked", labels=["Southampton","Cherbourg","Queenstown"])

plt.show()
words = [i for i in df.Name.dropna()]

    

words = " ".join(words)

words = words.split()

words = np.array(words)

words = words[ (words != "Mr") & (words != "Mr.") & (words != "Mrs") & (words != "Mrs.") & ( len(words) > 2 )]



words = str(list(words))



plt.subplots(figsize=(28,12))

wordcloud = WordCloud(

                          background_color='black',

                          width=2048,

                          height=1024

                          ).generate(words)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
