# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")

df.tail()
df.describe()
df.columns
df.isnull().sum()
df.info()
df.drop(["PassengerId"],axis=1,inplace = True)
def bar_plot(feature,figsize = (22,5)):

    

    value = df[feature].value_counts()

    

    plt.figure(figsize = figsize)

    plt.bar(value.index,value.values)

    plt.ylabel("Frequency")

    plt.xlabel(feature)

    plt.title("Distribution of " +str(feature))

    plt.show()

    

    print(value)
categorical_features = ["Country", "Sex", "Category", "Survived"]



for c in categorical_features:

    bar_plot(c)
plt.figure(figsize = (9,5))

plt.hist(df.Age,bins = 50)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.title("Age Distribution")

plt.show()

# Country -- Survived

df_country_survived = df[["Country","Survived"]].groupby("Country").sum()

countries = df.Country.value_counts()



[(countries[countries.index == c].values[0] - df_country_survived[df_country_survived.index == c].values[0])[0] for c in df_country_survived.index]

df_country_survived["Dead"] = [(countries[countries.index == c].values[0] - df_country_survived[df_country_survived.index == c].values[0])[0] for c in df_country_survived.index]

df_country_survived["Mean_of_Survived"] = df[["Country","Survived"]].groupby("Country").mean()

df_country_survived
# Category -- Survived

df_category_survived = df[["Category","Survived"]].groupby("Category").sum()

number_of_c = df.Category[df.Category == "C"].value_counts().values[0]

number_of_p = df.Category[df.Category == "P"].value_counts().values[0]

df_category_survived["Dead"] = [(number_of_c - df_category_survived.Survived.values[0]),(number_of_p - df_category_survived.Survived.values[1])]

df_category_survived["Mean_of_Survived"] = df[["Category","Survived"]].groupby("Category").mean()

df_category_survived
# Sex -- Survived

df_sex_survived = df[["Sex","Survived"]].groupby("Sex").sum()

number_of_c = df.Category[df.Category == "C"].value_counts().values[0]

number_of_p = df.Category[df.Category == "P"].value_counts().values[0]

df_sex_survived["Dead"] = [(number_of_c - df_sex_survived.Survived.values[0]),(number_of_p - df_sex_survived.Survived.values[1])]

df_sex_survived["Mean_of_Survived"] = df[["Sex","Survived"]].groupby("Sex").mean()

df_sex_survived
#Country -- Category -- Survived

df.groupby(["Country","Category","Survived"]).size().reset_index(name = "Count")
#Sex -- Category -- Survived

df.groupby(["Sex","Category","Survived"]).size().reset_index(name = "Count")
#Sex -- Country -- Survived

df.groupby(["Sex","Country","Survived"]).size().reset_index(name = "Count")
g = sns.catplot(x = "Category", y="Survived", kind = 'bar',data = df, size = 5)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.catplot(x = "Sex", y="Survived", kind = 'bar',data = df, size = 5)

g.set_ylabels("Survived Probability")

plt.show()
#df_age_categorical = df.copy()

df.Age = [0 if a < 10 

                          else 1 if a >= 10 | a < 20 

                          else 2 if a >= 20 | a < 30 

                          else 3 if a >= 30 | a < 40 

                          else 4 if a >= 40 | a < 50 

                          else 5 if a >= 50 | a < 60 

                          else 6 if a >= 60 | a < 70 

                          else 7 if a >= 70 | a < 80 

                          else 8 

                          for a in df.Age.values]



g = sns.catplot(x = "Age", y="Survived", kind = 'bar',data = df, size = 5,aspect = 3)

g.set_xticklabels(["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80-87"])

g.set_ylabels("Survived Probability")

plt.show()
g = sns.catplot(x="Country", y="Survived", hue="Category", kind="bar", data=df,height = 5,aspect = 3);

g.set_ylabels("Survived Probability")

plt.show()
g = sns.catplot(x="Category", y="Survived", hue="Sex", kind="bar", data=df,height = 5,aspect = 3);

g.set_ylabels("Survived Probability")

plt.show()
g = sns.catplot(x="Country", y="Survived", hue="Sex", kind="bar", data=df,height = 5,aspect = 3);

g.set_ylabels("Survived Probability")

plt.show()