# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Load data from Kiva
df = pd.read_csv('../input/kiva_loans.csv',parse_dates=["posted_time","disbursed_time","funded_time","date"])
theme_ids = pd.read_csv('../input/loan_theme_ids.csv')
theme_regions = pd.read_csv('../input/loan_themes_by_region.csv')
mpi_region = pd.read_csv('../input/kiva_mpi_region_locations.csv')

# Vectorizer to count genders in borrower_genders
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
fit = vec.fit_transform(df.borrower_genders.astype(str))
borrower_gender_count = pd.DataFrame(fit.A, columns=vec.get_feature_names())
borrower_gender_count.rename(columns={"female":"female_borrowers","male":"male_borrowers","nan":"nan_borrower"}, inplace=True)
df = pd.concat([df,borrower_gender_count],axis=1).set_index("id")

# Extra Features
df["borrower_count"] = df["female_borrowers"] + df["male_borrowers"]
df["female_ratio"] = df["female_borrowers"]/df["borrower_count"] 
df["male_ratio"] = df["male_borrowers"]/df["borrower_count"] 

# Full date time to date only variables
df["posted_date"] = df["posted_time"].dt.normalize()
df["disbursed_date"] = df["disbursed_time"].dt.normalize()
df["funded_date"] = df["funded_time"].dt.normalize()

def custom_describe(df):
    """
    I am a non-comformist :)
    """
    unique_count = []
    for x in df.columns:
        mode = df[x].mode().iloc[0]
        unique_count.append([x,
                             df[x].nunique(),
                             df[x].isnull().sum(),
                             mode,
                             df[x][df[x]==mode].count(),
                             df[x].dtypes])
    print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))
    return pd.DataFrame(unique_count, columns=["Column","Unique","Missing","Mode","Mode Occurence","dtype"]).set_index("Column").T
pd.set_option('display.max_columns', 500)
custom_describe(df)
plt.figure(figsize=(10,4))
plt.subplot(121)
total = [df["borrower_count"].sum(),df["female_borrowers"].sum(), df["male_borrowers"].sum()]
sns.barplot(x=["Total","Female","Male"],y= total,palette="YlGn")
plt.title("Borrowers Count by Gender")

plt.subplot(122)
total = [df["borrower_count"][df["borrower_count"]==0].count(),
         df["female_borrowers"][df["female_borrowers"]==0].count(),
         df["male_borrowers"][df["male_borrowers"]==0].count()]
sns.barplot(x=["Total","Female","Male"],y= total,palette="YlGn")
plt.title("Groups without a Certain Gender")
plt.show()
f,ax = plt.subplots(2,2,figsize=(10,8))
sns.distplot(df["female_ratio"][df["female_ratio"].notnull()], hist=False,ax=ax[0,0],label="female")
sns.distplot(df["male_ratio"][df["male_ratio"].notnull()], hist=False,ax=ax[0,0],label="male")
ax[0,0].set_title("Distribution of Female and Male Ratios")
ax[0,0].set_xlabel("")
ax[0,0].set_ylabel("Density")

sns.distplot(df.loc[(df["female_ratio"] > 0) & (df["female_ratio"] < 1),"female_ratio"], hist=False,ax=ax[1,0],label="female")
sns.distplot(df.loc[(df["male_ratio"] > 0) & (df["male_ratio"] < 1),"male_ratio"], hist=False,ax=ax[1,0],label="male")
ax[1,0].set_title("Distribution of Female and Male Ratio\nof mixed gendered groups")
ax[1,0].set_xlabel("Ratio")
ax[1,0].set_ylabel("Density")

sns.distplot(df.loc[df["borrower_count"] > 0,"borrower_count"], hist=False,ax=ax[0,1],label="All")
sns.distplot(df.loc[df["female_borrowers"] > 0,"female_borrowers"], hist=False,ax=ax[0,1],label="Female")
sns.distplot(df.loc[(df["male_borrowers"] > 0),"male_borrowers"], hist=False,ax=ax[0,1],label="Male")
ax[0,1].set_title("Average Gender Count in Group Size")
ax[0,1].set_xlabel("")

sns.distplot(df.loc[(df["male_ratio"] > 0) & (df["male_ratio"] < 1) &(df["borrower_count"] > 0),"borrower_count"], hist=False,ax=ax[1,1],label="All")
sns.distplot(df.loc[(df["female_ratio"] > 0) & (df["female_ratio"] < 1)&(df["female_borrowers"] > 0),"female_borrowers"], hist=False,ax=ax[1,1],label="Female")
sns.distplot(df.loc[(df["male_ratio"] > 0)&(df["male_ratio"] < 1)& (df["male_borrowers"] > 0),"male_borrowers"], hist=False,ax=ax[1,1],label="Male")
ax[1,1].set_title("Average Gender Count in Group Size\nfor Mixed Gendered Borrowers")
ax[1,1].set_xlabel("Count")
plt.tight_layout(pad=0)
plt.show()
df.tags = df.tags.str.replace(r"#|_"," ").str.title()
tags = df.tags.str.get_dummies(sep=', ')
tags = tags.sum().reset_index()
tags.columns = ["Tags","Count"]
tags.sort_values(by="Count",ascending=False,inplace=True)
f, ax = plt.subplots(figsize=[5,8])
sns.barplot(y = tags.Tags, x=tags.Count,ax=ax, palette="rainbow")
ax.set_title("Tag Count")
plt.show()
f, ax = plt.subplots(3,1,figsize=[12,6],sharex=True)
rol = 7
for i,gen in enumerate(["borrower_count","female_borrowers","male_borrowers"]):
    for time in ["disbursed_date","posted_date","funded_date"]:
        (df[[gen,time]].groupby(time).sum().rename(columns={gen:time})
         .rolling(window = rol).mean().plot(ax=ax[i],alpha=.8))
    ax[i].set_title("Disbursed, Posted, and Funded Date by {}".format(gen.replace("_"," ").capitalize()))
ax[0].set_xlabel("")
ax[1].set_xlabel("")
ax[1].set_ylabel("Count")
ax[2].set_xlabel("All Time")
plt.tight_layout(pad=0)
doy = []
for timecol in ["disbursed_date","posted_date","funded_date"]:
    name = timecol.replace("_"," ").title()+" Date of Year"
    df[name] = df[timecol].dt.dayofyear
    doy.append(name)

f, ax = plt.subplots(3,1,figsize=[12,6],sharex=True)
rol = 2
for i,gen in enumerate(["borrower_count","female_borrowers","male_borrowers"]):
    for time in doy:
        (df[[gen,time]].groupby(time).sum().rename(columns={gen:time})
         .rolling(window = rol).mean().plot(ax=ax[i],alpha=.8))
    ax[i].set_title("Disbursed, Posted, and Funded Date by {}".format(gen.replace("_"," ").capitalize()))
ax[0].set_xlabel("")
ax[1].set_xlabel("")
ax[1].set_ylabel("Count")
ax[2].set_xlabel("Date of Year")
plt.tight_layout(pad=0)
wkd = []
for timecol in ["disbursed_date","posted_date","funded_date"]:
    name = timecol.replace("_"," ").title()+" Weekday"
    df[name] = df[timecol].dt.weekday
    wkd.append(name)

f, ax = plt.subplots(3,1,figsize=[12,6],sharex=True)
rol = 1
for i,gen in enumerate(["borrower_count","female_borrowers","male_borrowers"]):
    for time in wkd:
        (df[[gen,time]].groupby(time).sum().rename(columns={gen:time})
         .rolling(window = rol).mean().plot(ax=ax[i],alpha=.8))
    ax[i].set_title("Disbursed, Posted, and Funded Date by {}".format(gen.replace("_"," ").capitalize()))
ax[0].set_xlabel("")
ax[1].set_xlabel("")
ax[1].set_ylabel("Count")
ax[2].set_xlabel("Date of Year")
plt.tight_layout(pad=0)