import pandas as pd
df = pd.read_csv("../input/train.csv")
df.head()
df.head(10)
df.tail(10)
df1 = pd.read_csv("../input/train.csv", usecols= ["PassengerId", "Survived", "Pclass"])

df1.head() #and add this to see if our code worked
df.describe()
df.info()
df.dtypes
df.sort_values("Fare").head(10)
df.sort_values("Fare", ascending = False).head(10)
df = df.sort_values("Fare", ascending = False)
df.sort_values("Fare", ascending = False, inplace = True)
df.head() #lets check the new version of the data
df.sort_values("Cabin", ascending = True, na_position ='last').head(10)
df.tail(10) #lets check if the nan values are at the bottom of our dataset
df.Sex.head()

df["Sex"].head()
df.Sex.value_counts()
df.nunique()
df["Embarked"].nunique() #we can also specify one or more column name too
df[["Embarked" , "Sex"]].nunique() #by putting a comma between two different columns, we can see number of the unique records in both columns
df["Embarked"] = df["Embarked"].astype("category") #we are changing the data type of the Embarked column to category here

df["Embarked"].dtype #we are checking to see if our code worked here
df["Embarked"] == "C"
df[df["Embarked"] == "C"]
embarked_c_mask = df["Embarked"] == "C"

df[embarked_c_mask]
df_fare_mask = df["Fare"] < 100

df_sex_mask = df["Sex"] == "female"

df[df_fare_mask & df_sex_mask]
df_fare_mask2 = df["Fare"] > 500

df_age_mask = df["Age"] > 70

df[df_fare_mask2 | df_age_mask]
null_mask = df["Cabin"].isnull() #With this code, we are saying that “Show me the passengers whose cabin is unknown”

df[null_mask]
df.isnull().sum()
df.drop(labels = ["Cabin"], axis=1).head()
df.drop(labels = ["Cabin", "Name"], axis=1).head()
df["Age"].fillna(0, inplace = True) #with inplace argument, we don't have to write it as

df["Age"] = df["Age"].fillna(0, inplace = True) #fill the missing values with zero

df['Age'] = df['Age'].fillna((df['Age'].median())) #fill the missing values with the median of Age column