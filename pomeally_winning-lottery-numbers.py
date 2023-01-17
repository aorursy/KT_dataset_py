#import the data, specify data types
import pandas as pd
df = pd.read_csv('../input/megamillion-dataset/Lottery_Mega_Millions_Winning_Numbers__Beginning_2002.csv')
df.head()
df.columns = ['Date', 'Numbers', 'Mega Ball','Multiplier']
df["AllNumbers"] = df["Numbers"].map(str) + " " + df["Mega Ball"].map(str)
df2 = df.copy()
del df2['Mega Ball']
del df2['Multiplier']
del df2['Numbers']
df3 = pd.DataFrame(df2['AllNumbers'].str.split(" ").apply(pd.Series, 0).stack())
df3.index = df3.index.droplevel(-1)
df3.head(20)
merged = pd.merge(df, df3,  how='inner', left_index=True, right_index=True)
del merged['Numbers']
del merged['AllNumbers']
del merged['Multiplier']
del merged['Mega Ball']
merged.columns = ['Date','Number']
merged.reset_index(inplace=True)
merged.head(20)
dothis = lambda x: pd.Series([i for i in reversed(x.split('/'))])
dates = merged['Date'].apply(dothis)
merged2 = pd.merge(merged, dates,  how='inner', left_index=True, right_index=True)
del merged2['index']
merged2.columns = ['Date','Number','Year','Day','Month']
merged2.head(20)
merged2.info()
merged2['Number'] = merged2['Number'].astype(int)
merged2.info()
import seaborn as sns
import matplotlib as mpl
mpl.rc("figure", figsize=(12, 20))
ax = sns.countplot(y="Number", data=merged2)
sumtotal = merged2.groupby(['Date']).sum() # total up the winning combination numbers 
sumtotal.describe()
# show the distribution of winning combinations' totals
mpl.rc("figure", figsize=(12, 6))
ax = sns.distplot(sumtotal['Number'])