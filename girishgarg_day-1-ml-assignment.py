import pandas as pd
data = pd.read_csv("../input/titanic/train_and_test2.csv")

data
data.head(3)
data.tail(3)
data.shape
data.info()
data.isnull().sum()
# Frequency distribution

data.Age.value_counts()
data.Age.value_counts().plot(kind="bar")
data.describe()
data.Fare.hist()
data.Fare.hist(bins=20)