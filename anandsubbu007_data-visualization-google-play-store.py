# import 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# reading files

df = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv') 

df.head(3)
df.info()
for i in df.columns:

    print(i)

    print(df[i].unique())
# a. Drop records where rating is missing since rating is our target/study variable

df['Rating'].isnull().sum()

df1 = df.dropna(subset=['Rating'])

df1.isnull().sum()
df1 = df1.loc[~df1['App'].str.contains('Life Made')]

#df.columns.str.contains('^X') returns array [True, True, False, False, False].

#True where condition meets. Otherwise False

#Sign ~ refers to negate the condition.

#df.loc[ ] is used to select columns
null_col = "Android Ver", "Current Ver"

for i in null_col: 

    df1[i] = df1[i].fillna(df1[i].mode()[0])



#[0] to choose 1st mode value
df1.isnull().sum()
'''

2.Data clean up – correcting the data types

    a.Which all variables need to be brought to numeric types?

    b.Price variable – remove $ sign and convert to float

    c.Installs – remove ‘,’ and ‘+’ sign, convert to integer

    d.Convert all other identified columns to numeric

'''
df1.dtypes
df1['Size'].mode()
#df["Installs"].replace({",":"","+":""})

df1["Installs"] = df1["Installs"].str.replace(",","")

df1["Installs"] = df1["Installs"].str.replace("+","")

df1["Installs"] = df1["Installs"].str.replace("Free","0")

df1["Price"] = df1["Price"].str.replace("$","")

df1["Price"] = df1["Price"].str.replace("Everyone",'0')

df1["Price"] = df1["Price"].str.replace("nan",'0')



df1["Size"] = df1["Size"].str.replace("M","")

df1["Size"] = df1["Size"].str.replace("k","")

df1["Size"] = df1["Size"].str.replace("Varies with device",'0')

for i in df.columns:

    print(i)

    print(df1[i].unique())
#df[["a", "b"]] = df[["a", "b"]].apply(pd.to_numeric)

#df["Installs"] = df["Installs"].astype(int)

#df["Price"] = df["Price"].astype(float)



df1[["Installs","Price","Size","Rating","Reviews"]] = df1[["Installs","Price","Size","Rating","Reviews"]].apply(pd.to_numeric)

df1.dtypes
df1.head()
df1["Rating"] = df1["Rating"].dropna()

df1["Rating"].unique()

df1.shape
df1.info()
df1.shape
df2 = df1[df1['Reviews']<=df1['Installs']]

df2.shape
df2.head()
'''

4.Identify and handle outliers – 

    a.Price column

        i.Make suitable plot to identify outliers in price

        ii.Do you expect apps on the play store to cost $200? Check out these cases

        iii.After dropping the useless records, make the suitable plot again to identify outliers

        iv.Limit data to records with price < $30

    b.Reviews column

        i.Make suitable plot

        ii.Limit data to apps with < 1 Million reviews

    c.Installs

        i.What is the 95th percentile of the installs?

        ii.Drop records having a value more than the 95th percentile

'''
#i.Make suitable plot to identify outliers in price

sns.boxplot(df2["Price"])
q1 = df2["Price"].quantile(.25)

q3 = df2["Price"].quantile(.5)

IQR = q1 - q3 

lower_bound = q1 -(1.5 * IQR) 

upper_bound = q3 +(1.5 * IQR) 

lower_bound
#ii.Do you expect apps on the play store to cost $200? Check out these cases

df3 = df2[df2["Price"]<200]

df3.shape
# After dropping the useless records, make the suitable plot again to identify outliers

sns.boxplot(df3["Price"])
#Limit data to records with price < $30

df4 = df3.drop(df3[df3["Price"] > 30].index)

df4.shape
# Reviews column  -  Make suitable plot

# Limit data to apps with < 1 Million reviews

sns.scatterplot(df4["Reviews"],df4["Rating"])
df5 = df4[df4["Reviews"]<1000000]

df5["Reviews"].max()

#df5.shape
'''    c.Installs

        i.What is the 95th percentile of the installs?

        ii.Drop records having a value more than the 95th percentile

'''



q3 = df5["Installs"].quantile(.95)



df6 = df5[df5["Installs"]<q3]

print(q3)

print("Max Value:",df6["Installs"].max())
'''

5.What is the distribution of ratings like? (use Seaborn) More skewed towards higher/lower values?

    a.How do you explain this?

    b.What is the implication of this on your analysis?

'''
sns.distplot(df6["Rating"])
df6["Content Rating"].value_counts()

# Adult only 18+ & unrated is very few in count
#so, drop this two

df6 = df6[(df6["Content Rating"]!="Adults only 18+")&(df6["Content Rating"]!="Unrated")].reset_index()

sns.jointplot(df6["Size"],df6["Rating"])



#Smaller the size higher the Rating

#these where inversely proportional
ax = sns.jointplot(df6["Price"],df6["Rating"])

sns.regplot(df6["Price"],df6["Rating"],ax = ax.ax_joint)

#sns.lmplot(df6["Price"],df6["Rating"])
#Replot the data, this time with only records with price > 0 



df7 = df6[df6["Price"]>0]

ax1 = sns.jointplot(df7["Price"],df7["Rating"])

sns.regplot(df7["Price"],df7["Rating"],ax = ax1.ax_joint)

#df7["Price"].min()
sns.pairplot(df6)
mean_df = df6.groupby("Content Rating").mean().reset_index()

median_df = df6.groupby("Content Rating").median().reset_index()

# no need for quantile
#mean

sns.barplot(mean_df["Rating"],mean_df["Content Rating"])
#median

sns.barplot(median_df["Rating"],median_df["Content Rating"])
# i prefer mean
#sns.distplot(df6["Rating"])

#sns.distplot(df6["Content Rating"])



sns.barplot(df6["Rating"],hue = df6["Content Rating"],y = df6["Size"])
df7 = df6.sort_values("Size")

x = df7["index"].count()

y = int(x*.20) + 1



k = []

for i in range(1,6):

    k.append(i)

l = [x * y for x in k]

l



j = []

for i in range(0,5):

    j.append(i)

m = [x * y for x in j]

print(m)

print(l)
data = []

for (x,y) in zip(m,l):

    d = df7[x:y]

    data.append(d)



data[2].shape
data[0]["Size"].quantile(.20)



q_20 = []



for i in range(5):

    x = data[i]["Size"].quantile(.20)

    q_20.append(x)
data[0].head(2)
x = df6[["Rating","Reviews","Size"]]

sns.heatmap(x, annot=True,cmap = "Greens")

res = []

for i in range(4):

    x = data[i][["Rating","Reviews","Size"]]

    res.append(sns.heatmap(x,annot=True,linewidths=2,cmap = "Greens"))



sns.relplot(style="Content Rating", x="Rating", y="Size", sizes=(200, 200), data=df6)