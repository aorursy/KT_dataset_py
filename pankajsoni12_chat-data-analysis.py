import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"../input//Chat_Team_CaseStudy.csv")
df.head(1)
df.drop(["Transaction Start Date","Session Name","Transaction End Date","Customer Comment"],inplace=True,axis=1)
df.head(1)
plt.figure(figsize=(16,4))
df.head(100).Agent.value_counts().plot(kind="bar")
plt.show()
df["Chat Duration"].unique().size
df["TimeTaken"] = df['Chat Duration'].str.split(":").apply(lambda x:int(x[0])*60+int(x[1])+int(x[2])/60.).round(2)
plt.figure(figsize=(25,8))
sns.boxplot(df["Agent"][:500],df["TimeTaken"][:500])
plt.xticks(rotation=90)
plt.show()
df.head(2)
plt.figure(figsize=(16,4))
plt.scatter(df.Agent,df.TimeTaken,alpha=.1,color="red")
plt.xticks(df.Agent[::5000],rotation=45)
plt.axhline(50,color="g")
plt.show()
plt.figure(figsize=(16,4))
plt.hist(df.TimeTaken,bins=50,rwidth=.9)
plt.show()
A_name = df.Agent.unique()
plt.figure(figsize=(16,4))
for an in A_name:
    #print(an)
    tt = df[df.Agent == an]
    
    meanTime = tt.TimeTaken.mean()
    if meanTime>90:
        plt.scatter(an,meanTime,alpha=.8,marker="*",color="blue",s=.02*np.pi*meanTime**2)
        plt.text(an,meanTime+5,an)
    else:
        plt.scatter(an,meanTime,alpha=.2,marker="^")
plt.xticks(A_name[::10],rotation=90)
plt.show()
plt.figure(figsize=(16,4))
df["Chat Closed By"].value_counts().plot(kind="bar")
plt.figure(figsize=(16,4))
df["Interactive Chat"].value_counts().plot(kind="bar")
plt.figure(figsize=(16,4))
df["Browser"].head(100).value_counts().plot(kind="bar") # INcrease the size of samples
plt.figure(figsize=(16,4))
df["Operating System"].head(100).value_counts().plot(kind="bar") # INcrease the size of samples
plt.figure(figsize=(16,4))
df["Geo"].head(1000).value_counts().plot(kind="bar") # INcrease the size of samples
plt.ylim(0,10) # comment if ylimit is NOT required
plt.figure(figsize=(16,4))
df["Customer Rating"].value_counts().plot(kind="bar") # INcrease the size of samples
plt.ylim(0,1500) # comment if ylimit is NOT required
plt.figure(figsize=(16,4))
df["Transferred Chat"].value_counts().plot(kind="bar") # INcrease the size of samples
# plt.ylim(0,1500) # comment if ylimit is NOT required
plt.figure(figsize=(16,4))
plt.hist2d(df["Order Value"],df.TimeTaken,bins=20, cmap='Blues')
plt.colorbar()
plt.show()
plt.figure(figsize=(16,4))
plt.hexbin(df["Order Value"],df.TimeTaken, gridsize=30, cmap='Blues')
plt.colorbar()
plt.show()
# df["Customer Rating"] = np.array(df["Customer Rating"],dtype="int8")
# df["Customer Rating"].isna()
# df["Customer Rating"] = df["Customer Rating"].replace(" ","")
df["Customer Rating"] = df["Customer Rating"].replace(r"\s+",np.nan,regex=True)
NoRating = df["Customer Rating"].isna().sum()
print(NoRating)
plt.figure(figsize=(16,4))
df["Customer Rating"].value_counts().plot(kind="bar")
# df["Customer Rating"].value_counts().plot(kind="line")
plt.show()

df["Customer Rating"] =pd.to_numeric(df["Customer Rating"],downcast="integer")
df["Customer Rating"].dtype
df["Customer Rating"].value_counts()
TotalRatings = df[df["Customer Rating"]>=0]["Customer Rating"].value_counts().sum()
TotalRatings
pp = df[df["Customer Rating"]>=9]["Customer Rating"].value_counts().sum()
(pp/TotalRatings)*100

pp = df[df["Customer Rating"]<=6]["Customer Rating"].value_counts().sum()
(pp/TotalRatings)*100

pp = df[(df["Customer Rating"]>=7) & (df["Customer Rating"]<=8)]["Customer Rating"].value_counts().sum()
# pp
(pp/TotalRatings)*100
plt.figure(figsize=(16,4))
pp = df["Chat Closed By"].value_counts().plot(kind="bar")
pp