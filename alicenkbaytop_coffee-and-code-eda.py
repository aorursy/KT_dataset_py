import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization

import matplotlib.pyplot as plt # data visualization



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/coffee-and-code/CoffeeAndCodeLT2018.csv")
df.columns
df.head()
df.describe()
df.info()
def count_plot(variable):

    """

        input: variable example: "CoffeTime"

        output: count plot and value count

    """

    # get feature

    var = df[variable]

    

    #visualization

    plt.figure(figsize=(10,4))

    sns.countplot(x=var, palette="dark", order=var.value_counts().index)

    plt.xticks(rotation=45)

    plt.ylabel("Frequency")

    plt.title(variable)

    print("{}".format(var.value_counts()))

    plt.show()
categorical = ["CoffeeTime", "CodingWithoutCoffee", "CoffeeType", "CoffeeSolveBugs", "Gender", "Country", "AgeRange"]

for i in categorical:

    count_plot(i)
numerical = ["CodingHours", "CoffeeCupsPerDay"]

for i in numerical:

    count_plot(i)
plt.figure(figsize=(12,8))

sns.countplot("Gender", data=df, hue="CoffeeType", palette="dark")

plt.legend(loc="center")
df[["AgeRange","CoffeeCupsPerDay"]].groupby(["AgeRange"]).mean().sort_values(by="CoffeeCupsPerDay", ascending=False)
sns.barplot(x="AgeRange", y="CoffeeCupsPerDay",data=df, palette="dark")
df[["Gender","CoffeeCupsPerDay"]].groupby(["Gender"]).mean().sort_values(by="CoffeeCupsPerDay", ascending=False)
sns.barplot(x="Gender", y="CoffeeCupsPerDay", data=df, palette="dark")
df[["CodingHours","CoffeeCupsPerDay"]].groupby(["CodingHours"]).mean().sort_values(by="CoffeeCupsPerDay", ascending=False)
sns.barplot(x="CodingHours", y="CoffeeCupsPerDay", data=df, palette="dark")
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
df[["CoffeeSolveBugs","CoffeeCupsPerDay"]].groupby(["CoffeeSolveBugs"]).mean().sort_values(by="CoffeeCupsPerDay", ascending=False)
sns.countplot("CoffeeSolveBugs", hue="CoffeeCupsPerDay", palette="dark", data=df)

plt.legend(loc="upper right")
sns.countplot("CodingWithoutCoffee", hue="CoffeeSolveBugs",data=df, palette="dark")
df.columns[df.isnull().any()]
df.isnull().sum()
# Missing Value Table

def missing_value_table(df): 

    missing_value = df.isnull().sum()

    missing_value_percent = 100 * df.isnull().sum()/len(df)

    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)

    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})

    return missing_value_table_return

  

missing_value_table(df)
df[df.CoffeeType.isnull()]
df[df.AgeRange.isnull()]
df["CoffeeType"] = df["CoffeeType"].fillna("Nescafe") 
df["AgeRange"] = df["AgeRange"].fillna("18 to 29") 
df.isnull().sum()