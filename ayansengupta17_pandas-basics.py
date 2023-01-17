import pandas as pd
df = pd.read_csv("../input/train.csv")
df.head(5)
df_ = pd.read_csv("../input/train.csv", index_col=['PassengerId'])
df_.head(5)
df.columns
df_.columns
df.shape
df.iloc[4]
df.iloc[[4,6,8]]
df[:2]
df[4:10]
df['Name'][0:6]
df.columns = [c.replace(' ', '_') for c in df.columns]
print(df.columns)
df.Name[0:6]
df['Sex'].head(5)
df[['Name', 'Sex']][:3]
df.Name.iloc[4]
df.Name.iloc[[4]]
(df.Sex=='male').head(5)
df[df.Sex=='male'].head(7)
df[(df.Age > 20) & (df.Survived == 1)].head(10)
df[df['Name'].str.split().apply(lambda x: len(x) == 3)].head(3)
df[df.Cabin.isin(['C85', 'C123'])].head(10)
