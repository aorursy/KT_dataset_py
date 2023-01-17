import pandas as pd
df = pd.read_csv("../input/Data.csv")
df.info()
print("_______________________")
df.head()
df['2012 [YR2012]'] = pd.to_numeric(df['2012 [YR2012]'], errors='coerce')
df['2013 [YR2013]'] = pd.to_numeric(df['2013 [YR2013]'], errors='coerce')
df['2014 [YR2014]'] = pd.to_numeric(df['2014 [YR2014]'], errors='coerce')
df['2015 [YR2015]'] = pd.to_numeric(df['2015 [YR2015]'], errors='coerce')
df['2016 [YR2016]'] = pd.to_numeric(df['2016 [YR2016]'], errors='coerce')
df.info()
pd.set_option('max_colwidth', 120)
pd.DataFrame(df['Series Name'].unique())
a = df['Series Name'] != "Data from database: Poverty and Equity"
b = df['Series Name'] != "Last Updated: 10/24/2017"
df = df[a & b]
df.dropna(subset=['Series Name'],inplace=True)
pd.DataFrame(df['Series Name'].unique())
pd.reset_option('max_colwidth')

TENpercenters = df[df['Series Name'] == 'Income share held by highest 10%']
TENpercenters.head()
TENpercenters['Average']=TENpercenters.mean(axis=1)
TENpercenters.head()
TENpercenters.sort_values(by=['Average'],ascending=False,inplace=True)
TENpercenters
import seaborn as sns
import matplotlib.pyplot as plt
no_missing = TENpercenters[TENpercenters['Average'].notnull()]
fig, ax = plt.subplots()
fig.set_size_inches(15, 8)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90,)
ax.set_title("Income Share Held by Top 10%")
ax = sns.barplot(x="Country", y="Average", data=no_missing)