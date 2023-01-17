# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# load all libraries used in later analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10, 6)
from matplotlib import style
style.use('ggplot')
import spacy
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, MeanShift, MiniBatchKMeans
import os
from pprint import pprint
import string
import re
from sklearn.decomposition import PCA
from collections import Counter
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
plt.figure(figsize=(10,4))
plt.subplot(121)
total = [df["borrower_count"].sum(),df["female_borrowers"].sum(), df["male_borrowers"].sum()]
sns.barplot(x=["Total","Female","Male"],y= total)
plt.title("Borrowers Count by Gender")

plt.subplot(122)
total = [df["borrower_count"][df["borrower_count"]==0].count(),
         df["female_borrowers"][df["female_borrowers"]==0].count(),
         df["male_borrowers"][df["male_borrowers"]==0].count()]
sns.barplot(x=["Total","Female","Male"],y= total)
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
sns.barplot(y = tags.Tags, x=tags.Count,ax=ax)
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
df.head(5)
df.describe(include = 'all')
countries = df['country'].value_counts()[df['country'].value_counts()>3400]
list_countries = list(countries.index) 
plt.figure(figsize=(15,13))
sns.barplot(y=countries.index, x=countries.values, alpha= 1)
plt.xlabel("# of borrowers", fontsize=16)
plt.ylabel("Countries", fontsize=16)
plt.show();
plt.figure(figsize=(13,8))
sectors = df['sector'].value_counts()
sns.barplot(y=sectors.index, x=sectors.values, alpha=1)
plt.xlabel('Number of loans', fontsize=16)
plt.ylabel("Sectors", fontsize=16)
plt.show();
plt.figure(figsize=(15,10))
activities = df['activity'].value_counts().head(25)
sns.barplot(y=activities.index, x=activities.values, alpha=1)
plt.ylabel("Activity", fontsize=16)
plt.xlabel('Number of loans', fontsize=16)
plt.title("Number of loans per activity", fontsize=16)
plt.show();
time=df[["country","posted_time","funded_time"]]
time=time.copy()
time["time"]=(pd.to_datetime(time.funded_time)-pd.to_datetime(time.posted_time))
time["time"]=time["time"].apply(lambda x : x.days)
time_taken=pd.DataFrame(time.groupby(["country"])["time"].agg("mean").sort_values(ascending=False))
time_taken=time_taken[:30]
plt.figure(figsize=(12,12))
sns.barplot(y=time_taken.index,x=time_taken["time"],data=time_taken)
plt.gca().set_xlabel("Average number of days")
plt.gca().set_ylabel("Country")
plt.gca().set_title("# of days for loan to be funded")
plt.show()
temp = df['loan_amount']
plt.figure(figsize=(12,8))
sns.distplot(temp[~((temp-temp.mean()).abs()>3*temp.std())]);
plt.ylabel("density", fontsize=16)
plt.xlabel('loan amount', fontsize=16)
plt.title("loan amount", fontsize=13)
plt.show();
new_loan_amount = df.loan_amount.tolist()
new_fund_amount = df.funded_amount.tolist()
new_country = df.country.tolist()

new_unique_country = []
for i in new_country:
    if not i in new_unique_country:
        new_unique_country.append(i)

new_unique_loan=np.zeros(len(new_unique_country))
new_unique_funded=np.zeros(len(new_unique_country))
for i, loan in enumerate(new_loan_amount):
    country = new_country[i]
    dex = new_unique_country.index(country)
    
    new_unique_loan[dex]=new_unique_loan[dex] + loan
    
    fund = new_fund_amount[i]
    new_unique_funded[dex]=new_unique_funded[dex] + fund

total_fund=0
total_loan=0
for i, loan in enumerate(new_loan_amount):
    total_loan=total_loan + loan
    total_fund=total_fund + new_fund_amount[i]
    

new_loan_country_dict = dict(zip(new_unique_country, new_unique_loan))
new_fund_country_dict = dict(zip(new_unique_country, new_unique_funded))
new_country_sorted = sorted(new_loan_country_dict, key=new_loan_country_dict.get, reverse=False)
new_loan_sorted = []
new_fund_sorted = []
dummy = []
for i, country in enumerate(new_country_sorted):
    new_loan_sorted.append(new_loan_country_dict.get(country))
    new_fund_sorted.append(new_fund_country_dict.get(country))
    dummy.append(i)

f, ax = plt.subplots(1,1, figsize=(12,20))
ax.set_title("Countries with Loan amounts ")
ax.set_xlabel("Amount Loaned")
ax.barh(y=dummy, width=new_loan_sorted, label='Loan')
ax.set_xlim([0, max(new_loan_sorted)*1.1]) 
plt.yticks(dummy, new_country_sorted)
ax.legend(prop={"size" : 15})

for i, v in enumerate(new_loan_sorted):
    ax.text(v + max(new_loan_sorted)/100, i-0.25, str(round(float(v)/1000000,2))+ "M", fontweight='bold')
plt.show()
number=25
bins=[]
for i in range(number):
    bins.append((i)*50)
f, ax = plt.subplots(1,1, figsize=(12,8))
ax.set_ylabel("Frequency")
ax.set_xlabel("Loan Amount")
ax.hist(new_loan_amount, bins, rwidth=0.8)
plt.show()
# aggregate "use" by country and region and combine use text
use_by_CR = df[['country', 'region', 'use']] \
    .replace(np.nan, "") \
    .groupby(['country', 'region'])['use'] \
    .apply(lambda x: "\n".join(x)) \
    .reset_index()  # normalise
use_by_CR['region'].replace("", "#other#", inplace=True)
use_by_CR['country'].replace("", "#other#", inplace=True)
# generate a combined field for aggregation purposes
use_by_CR['CR'] = use_by_CR['country'] + "_" + use_by_CR['region']
# now we use spacy to process the per-region use descriptions and obtain document vectors
nlp = spacy.load('en_core_web_lg', disable=["tagger", "parser", "ner"])
nlp.max_length =  1627787+1000
raw_use_texts = list(use_by_CR['use'].values)
processed_use_texts = [nlp(text) for text in raw_use_texts]
processed_use_vectors = np.array([text.vector for text in processed_use_texts])
processed_use_vectors.shape
tsne = TSNE(n_components=2, metric='cosine', random_state=7777)
fitted = tsne.fit(processed_use_vectors)
fitted_components = fitted.embedding_
fitted_components.shape
use_by_CR['cx'] = fitted_components[:, 0]
use_by_CR['cy'] = fitted_components[:, 1]
use_by_CR.head()
country_region_cnt = use_by_CR.groupby('country').size()
selected_countries = country_region_cnt[country_region_cnt > 150]
n_selected_countries = len(selected_countries)
selected_country_pos = np.where(country_region_cnt > 150)[0]
id2country = dict(enumerate(selected_countries.index))
country2id = {v: k for k, v in id2country.items()}
selected_use_by_CR = use_by_CR.query('country in @selected_countries.index')
fig, ax = plt.subplots(figsize=(16, 12))
plt.scatter(selected_use_by_CR['cx'], selected_use_by_CR['cy'], s=15,
            c=[country2id[x] for x in selected_use_by_CR['country']],
            cmap=plt.cm.get_cmap('tab20', 19))
formatter = plt.FuncFormatter(lambda val, loc: id2country[val])
plt.colorbar(ticks=np.arange(19), format=formatter);
plt.show()
