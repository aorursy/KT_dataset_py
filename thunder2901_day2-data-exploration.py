import pandas as pd

df = pd.read_csv("../input/autompg-dataset/auto-mpg.csv", na_values='?')

df.head(5)
df.describe()
df.columns
df.isnull().any()
horsepower_median = df['horsepower'].median()
df['horsepower'].fillna(horsepower_median,inplace=True)
df['horsepower'].isnull().any()
df.boxplot(column = [df.columns[0],df.columns[1]])
df.boxplot(column = [df.columns[2],df.columns[3]])
df.boxplot(column = [df.columns[4]])
df.boxplot(column = [df.columns[5]])
df.boxplot(column = [df.columns[6]])
df.boxplot(column = [df.columns[7]])
#Do outlier detection

            