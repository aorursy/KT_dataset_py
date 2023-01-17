#import python libraries for data analysis and visualization
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns



#import the input dataset as python dataframe object
titanic_df = pd.read_csv("../input/train.csv")
#get the first few rows in the dataframe
titanic_df.head()
#get overall info of the dataset
titanic_df.info()
#count the number of passengers by gender
sns.countplot('Sex', data=titanic_df)

#count of passengers by class separated by gender
sns.countplot("Pclass", data=titanic_df, hue="Sex")
#create a function to split between males, females and children
def person_cat(passenger):
    if passenger.Age<16:
        return "child"
    else:
        return passenger.Sex
titanic_df["Person"] = titanic_df.apply(person_cat, axis=1)
titanic_df[0:10]

#count of passengers by class separated by person
sns.countplot("Pclass", data=titanic_df, hue="Person")
#histogram to show distribution of age of passengers
titanic_df["Age"].hist(bins=70)
plt.xlabel("Age")
#get the mean age of passengers
titanic_df["Age"].mean()

#get the count of passengers
titanic_df.Person.value_counts()
g = sns.FacetGrid(titanic_df, hue="Sex", size=5, aspect=3)
g = g.map(sns.kdeplot ,"Age")
g.set(xlim=(0, titanic_df["Age"].max()))
g.add_legend()
#drop the missing values from "Cabin" column
deck = titanic_df['Cabin'].dropna()

#create a loop to extract the first letter from Cabin 
levels = []
deck_array = deck.values
for i in deck_array:
    levels.append(i[0][0])
levels_df = pd.DataFrame(levels, columns=["CabinLevel"])
levels_df.sort_values(by=["CabinLevel"], inplace=True)

#count the passengers by Cabin
sns.countplot("CabinLevel", data=levels_df)

    
  

#create a new dataframe with "Survived" and "Cabin" columns and extract the first letter of the cabin  
cabin_df = titanic_df.loc[:, ("Survived", "Cabin")]
cabin_df.dropna(inplace=True)
cabin_df["DeckLevel"] = cabin_df["Cabin"].str[0]
cabin_df.head()
#group the dataframe by the new column of cabin and survived and plot an unstacked bar chart 
cabin_df.groupby(["Survived", "DeckLevel"]).size().unstack().plot(kind="bar")
#get the unique values for Embarked
titanic_df.Embarked.unique()

#make a countplot and check the results
sns.countplot("Embarked", data=titanic_df, hue='Pclass')
# create a new column to determine how many passengers are alone or with family
titanic_df["PassengerFamily"] = titanic_df["Parch"] + titanic_df["SibSp"]
titanic_df["PassengerFamily"].head(10)
# subset the dataset and assign "Family" and "Alone" for PassengerFamily ==0 & >0 respectively
titanic_df.loc[titanic_df.PassengerFamily>0, "PassengerFamily"] = "Family"
titanic_df.loc[titanic_df.PassengerFamily==0, "PassengerFamily"] = "Alone"


# display the dataset with new column
titanic_df.head()
#create a countplot to count who were Alone and with Family
sns.countplot("PassengerFamily", data=titanic_df)
