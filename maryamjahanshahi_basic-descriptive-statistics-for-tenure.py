import pandas as pd 
df = pd.read_csv("../input/tenure-dataset/company_tenure.csv")

df.head()
df.mean()
df.describe()
df.hist()
df.hist(sharex=True, sharey=True)
df.hist(bins=5, sharex=True, sharey=True)