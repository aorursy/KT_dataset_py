# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

cocktails = pd.read_csv("/kaggle/input/cocktails-hotaling-co/hotaling_cocktails - Cocktails.csv")
cocktails.head()
cocktails.describe()
cocktails.info()
cocktails.isnull().sum()

# quite lots of missing values

# I will check percentage of missing values
cocktails.isnull().mean()

# most of features have more than 50% of missing values
# I decided to impute those missing values with "Others" 

cocktails = cocktails.fillna("Others")
cocktails.isnull().sum()

# all missing values are imputed
########################### Exploratory Data Analysis ################################

cocktails["Cocktail Name"].nunique()

# there are 687 rows but there are only 684 different cocktail names in the data.

# there is cocktail name which appears more than once.

# perhaps there is different preparations by different bartender for that particular cocktail.
cocktails["Cocktail Name"].value_counts()

# Martinez, Negroni and Rita's Song have appeared more than once
# Let's explore on those mentioned cocktails

cocktails[cocktails["Cocktail Name"] == "Martinez"]

# there is indeed slight variation in terms of ingredients, garnish, glassware used and preparation.

# however, it is impossible to tell whether these two variations are being praticed by the same bartender or not
cocktails[cocktails["Cocktail Name"] == "Negroni"]

# there is a slight variation in terms of ingredients, garnish, glassware used and preparation.

# not much useful information can be extracted from here 
cocktails[cocktails["Cocktail Name"] == "Rita's Song"]

# they are the same in terms of ingredients and preparation used

# however, it is impossible to tell whether they are practiced by the same bertender or different bars at the same location.
df_bartender = pd.DataFrame(cocktails["Bartender"].value_counts().reset_index())

df_bartender.columns = ["Bartender","Counts"]

df_bartender
df_bartender.drop(0, inplace=True)

# I am not interested in "Others" as being the highest counts does not bring any value.

df_bartender
plt.figure(figsize=(16,8))

sns.barplot(x="Bartender",y="Counts", data=df_bartender.head(10))

plt.show()

# from this graph, I will suggest Francesco Lafranconi is the most experienced bartender.

# he has practiced the highest number of different coctails among other bertenders.
df_bar = pd.DataFrame(cocktails["Bar/Company"].value_counts().reset_index())

df_bar.columns = ["Bar/Company","Counts"]

df_bar
df_bar.drop(0, inplace=True)

# I am not interested in "Others" as being the highest counts does not bring any value.

df_bar
plt.figure(figsize=(16,8))

sns.barplot(x="Bar/Company",y="Counts", data=df_bar.head(10))

plt.show()

# Dirty Habit is the bar that serves the highest different variety of cocktails 

# perhaps this is because that company has hired lots of bartenders who practiced different cocktails
cocktails[cocktails["Bartender"] == "Francesco Lafranconi"]["Bar/Company"].unique()

# it is impossible to tell whether Francesco Lafranconi is working at Dirty Habit or not due to those missing values
df_location = pd.DataFrame(cocktails["Location"].value_counts().reset_index())

df_location.columns = ["Location","Counts"]

df_location
df_location.drop(0, inplace=True)

# I am not interested in "Others" as being the highest counts does not bring any value.

df_location
plt.figure(figsize=(16,8))

sns.barplot(x="Location",y="Counts", data=df_location.head(10))

plt.show()

# Most of bartenders are prefer to work at San Francisco.

# Maybe San Francisco has the highest pay.
cocktails[cocktails["Bar/Company"] == "Dirty Habit"]["Location"].value_counts()

# almost all of Dirty Habit outlets are opened at San Francisco

# at this point, I would say lots of bartenders are working for Dirty Habit

# maybe Dirty Habit pays the highest salary among other companies
# I will try to do text mining on "Ingredients"

import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



stop_words = stopwords.words("english")

lem = WordNetLemmatizer()
# I will create a function to clean texts on "Ingredients"

def cleaning_text(i):

    i = re.sub("[^A-Za-z]+"," ",i).lower()

    i = re.sub("[0-9]+"," ",i)

    lemmed_words = []

    for word in i.split(" "):

        word = lem.lemmatize(word)

        lemmed_words.append(word)

    cleaned_words = []

    for word in lemmed_words:

        if word not in stop_words:

            cleaned_words.append(word)

    completed_words = []

    for word in cleaned_words:

        if len(word) > 3:

            completed_words.append(word)

    return(" ".join(completed_words))
cocktails.bow = cocktails["Ingredients"].apply(cleaning_text)

cocktails.bow = " ".join(cocktails.bow)
# I will create wordcloud on it

from wordcloud import WordCloud



wordcloud_cocktails = WordCloud(background_color="black",height=1400, width=1800).generate(cocktails.bow)

plt.figure(figsize=(16,8))

plt.imshow(wordcloud_cocktails)

# among all cocktails mentioned in this data, it seems that lemon juice is the most common ingredient.
# There are limited information I can extract from this data as it contains too many missing values.

# As a conclusion, at this point, I would say San Francesco is the primary choice for those companies that want to open bar.

# Dirty Habit chose to open almost all of its outlets at San Francesco and served lots of different cocktails.