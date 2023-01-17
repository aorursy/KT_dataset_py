import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../input/BreadBasket_DMS.csv")
df.head()
df.isnull().any()
df.Item.unique()
df.loc[(df['Item']=="NONE")].head()
to_drop = df.loc[(df['Item']=="NONE")].index

df.drop(to_drop, axis = 0, inplace = True)

print ("Lines dropped")
df["date"] = pd.to_datetime(df['Date'])
df["dayname"] = df["date"].dt.day_name()
df.drop("date", axis=1, inplace = True)
df.head()
df["year"], df["month"], df["day"] = df["Date"].str.split('-').str
df.drop("Date", axis=1, inplace = True)

df["hour"], df["minute"], df["second"] = df["Time"].str.split(':').str
df.drop("Time", axis=1, inplace = True)

df.head()
#Season
df["month"] = df["month"].astype(int)
df.loc[(df['month']==12),'season'] = "winter"
df.loc[(df['month']>=1) &  (df['month']<=3),'season'] = "winter"
df.loc[(df['month']>3) &  (df['month']<=6),'season'] = "spring"
df.loc[(df['month']>6) &  (df['month']<=9),'season'] = "summer"
df.loc[(df['month']>9) &  (df['month']<=11),'season'] = "fall"

df.head()
sns.countplot(df["year"].astype(int))
plt.show()
sns.countplot(df["month"].astype(int))
plt.show()
sns.barplot(df["year"].astype(int), df["Transaction"].value_counts())
plt.show()
sns.barplot(df["season"], df["Transaction"].value_counts())
plt.show()
sns.countplot(df["day"].astype(int))
plt.show()
sns.barplot(df["day"].astype(int), df["Transaction"].value_counts())
plt.show()
df_month = df[(df['month']>=1) & (df['month']<=3) | (df['month']>=11) & (df['month']<=12)]
sns.countplot(df_month["day"].astype(int))
plt.show()
sns.barplot(df["dayname"], df["Transaction"].value_counts())
plt.show()
sns.barplot(df["dayname"], df["Transaction"].value_counts())
plt.show()
sns.countplot(df["hour"].astype(int))
plt.show()
sns.barplot(df["hour"].astype(int), df["Transaction"].value_counts())
plt.show()
df["Item"].value_counts()[:10].plot(kind="bar")
plt.show()
values = df.Item.loc[(df['season']== "fall")].value_counts()[:10]
labels = df.Item.loc[(df['season']== "fall")].value_counts().index[:10]

plt.pie(values, autopct='%1.1f%%', labels = labels,
        startangle=90)

plt.show()
values = df.Item.loc[(df['season']== "winter")].value_counts()[:10]
labels = df.Item.loc[(df['season']== "winter")].value_counts().index[:10]

plt.pie(values, autopct='%1.1f%%', labels = labels,
         startangle=90)

plt.show()
values = df.Item.loc[(df['season']== "spring")].value_counts()[:10]
labels = df.Item.loc[(df['season']== "spring")].value_counts().index[:10]

plt.pie(values, autopct='%1.1f%%', labels = labels, startangle=90)

plt.show()
top_spring = df.Item.loc[(df['season']== "spring")].value_counts()[:10] / sum(df.Item.loc[(df['season']== "spring")].value_counts()) *100 
top_fall = df.Item.loc[(df['season']== "fall")].value_counts()[:10] / sum(df.Item.loc[(df['season']== "fall")].value_counts()) *100 
top_winter = df.Item.loc[(df['season']== "winter")].value_counts()[:10] / sum(df.Item.loc[(df['season']== "winter")].value_counts()) *100 
top_overall = df.Item.value_counts()[:10] / sum(df.Item.value_counts()) *100 
topseller = pd.DataFrame([top_spring, top_fall, top_winter, top_overall], index = ["Spring", "Fall", "Winter", "Overall"]).transpose()
topseller
worst_spring = df.Item.loc[(df['season']== "spring")].value_counts()[-10:] / sum(df.Item.loc[(df['season']== "spring")].value_counts()) *100 
worst_fall = df.Item.loc[(df['season']== "fall")].value_counts()[-10:] / sum(df.Item.loc[(df['season']== "fall")].value_counts()) *100 
worst_winter = df.Item.loc[(df['season']== "winter")].value_counts()[-10:] / sum(df.Item.loc[(df['season']== "winter")].value_counts()) *100 
worst_overall = df.Item.value_counts()[-10:] / sum(df.Item.value_counts()) *100 
worst = pd.DataFrame([worst_spring, worst_fall, worst_winter, worst_overall], index = ["Spring", "Fall", "Winter", "Overall"]).transpose()
worst
sns.countplot(df["Transaction"].value_counts())
plt.show()
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
overall = df
fall = df[df["season"]=="fall"]
winter = df[df["season"]=="winter"]
spring = df[df["season"]=="spring"]

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def apri(data):
    encoding = data.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction').astype(int)
    encoding = encoding.applymap(encode_units)
    frequent_itemsets = apriori(encoding, min_support=0.01, use_colnames=True)
    rules_1 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    output = rules_1.sort_values(by=['confidence'], ascending=False)
    return output

print("Model ready")
apri(overall).head(10)
def compare(season_one, season_two): 
    dataframe =  pd.concat([apri(season_one).head(10), apri(season_two).head(10)]).drop_duplicates(subset = "antecedents", keep=False).sort_values(by=['confidence'], ascending=False)
    return dataframe
compare(winter, spring)
compare(winter, fall)
compare(spring, fall)