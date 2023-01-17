import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/beoutbreakprepared/nCoV2019/master/latest_data/latestdata.csv"
df = pd.read_csv(url, index_col=0)
df.rename(columns={"wuhan(0)_not_wuhan(1)" : "wuhan0_not_wuhan1",},  inplace=True)
df.head(5)
df.info()
missing_per = (df.isnull().sum()/len(df))*100
missing_per = missing_per.sort_values(ascending=False)
missing_per
plt.subplots(figsize=(12,8))
plt.xticks(rotation=90)
sns.barplot(x=missing_per.index, y= missing_per)
#Drop the columns where missing values are greater than 75%
thresh = len(df)*0.25
df.dropna(thresh = thresh, axis=1, inplace=True)
#check the column
((df.isnull().sum()/len(df))*100).sort_values(ascending=False)
for c in df.columns:
    print("*********",c,"*********")
    print("## Unique number of Values: ",df[c].nunique())
    print("## Unique Values: ",df[c].unique())
    print("## Values count: ",df[c].value_counts(), "\n\n")

