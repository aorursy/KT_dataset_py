import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime



df = pd.read_csv("../input/all-space-missions-from-1957/Space_Corrected.csv",parse_dates=True)

df = df.iloc[:,2:].copy() #deleting the unnamed columns

df.head()
df.rename(columns={" Rocket":"Price"},inplace=True) #renaming the rocket column to Price



#Function to extract the launch location from the location address

def country_extract(s):

    s = s.split(",")

    return s[len(s)-1].strip()



df["Country"]=df["Location"].map(country_extract)

df["Status Rocket"] = df["Status Rocket"].str.replace("Status","")



#date extractor from Datum

def extract_date(s):

    s = s.split(" ")

    s="-".join(s[1:4])

    s=s.replace(",","")

    s = datetime.datetime.strptime(s,"%b-%d-%Y")

    return s



df["Datum"] = df["Datum"].map(extract_date)

df["Year"] = df["Datum"].dt.year
ax=df["Year"].value_counts().sort_index().plot(figsize=(15,6),marker="x",color="black")

ax.set_axisbelow(True)

ax.set_xticks(df["Year"].unique())

ax.yaxis.grid(color='lightgray', linestyle='dashed')

ax.xaxis.grid(color='lightgray', linestyle='dashed')

plt.title("Number of missions across years")

plt.ylabel("Number of Missions")

plt.xticks(rotation=90)

plt.show()
st = df[["Status Mission","Year"]].copy()

st["Status Mission"] = st["Status Mission"].replace(".* Failure","Failure",regex=True)

st = pd.crosstab(columns=st["Status Mission"],index=st["Year"],normalize="index")

ax = st.plot(kind="bar",stacked=True,figsize=(15,5),color=["black","gray"])

ax.set_axisbelow(True)

ax.yaxis.grid(color='gray', linestyle='dashed')

plt.title("Mission success/failure rate across years")

plt.ylabel("% of success/failure")

plt.show()
piv = df[["Year","Status Mission"]].copy()

piv["Status Mission"] = piv["Status Mission"].replace(".* Failure","Failure",regex=True)

piv["Year range"]=0

piv["Year range"] = pd.cut(piv["Year"],bins=[1957,1977,1997,2020],labels=["1957-1977","1978-1998","1999-2020"])

piv = pd.crosstab(columns=piv["Status Mission"],index=piv["Year range"])

piv["Success Rate"] = 100*(piv["Success"] / (piv["Success"]+piv["Failure"]))

piv["Failure Rate"] = 100*(piv["Failure"] / (piv["Success"]+piv["Failure"]))

piv.drop(columns=["Failure","Success"],inplace=True)



fig=plt.figure(figsize=(5,5))

ax=sns.heatmap(piv.T,annot=True,fmt="0.0f",cbar=False,cmap="gray",square=True,linewidths=0.1,linecolor="gray")

ax.xaxis.set_ticks_position('top')

plt.xticks(rotation=90)

plt.title("Success & failure rates across years")

plt.ylabel("")

plt.xlabel("")

plt.show()
cnt = df["Company Name"].value_counts().reset_index()[:20]



sns.catplot(y="index",x="Company Name",data=cnt,kind="bar",height=8,color="black")

plt.title("Companies by no of missions")

plt.ylabel("Company")

plt.xlabel("No of missions")



for i in range(cnt.shape[0]):

    plt.text(s=str(cnt.iloc[i,1]),y=i,x=cnt.iloc[i,1]+10)

plt.show()
cl = df.copy()

cl["Status Mission"] = cl["Status Mission"].replace(".* Failure","Failure",regex=True)

sns.catplot(data=cl.loc[cl["Company Name"].isin(cnt["index"]),:],x="Company Name",y="Year",kind="swarm",height=7,aspect=2,

            palette=sns.set_palette(["gray","black"]),hue="Status Mission")

plt.title("Yearly trend in number of missions per company?")

plt.xticks(rotation=90)

plt.show()
piv = df[["Company Name","Status Mission"]].copy()

piv["Status Mission"] = piv["Status Mission"].replace(".* Failure","Failure",regex=True)

piv = pd.crosstab(columns=piv["Status Mission"],index=piv["Company Name"])



piv["Success Rate"] = 100*(piv["Success"] / (piv["Success"]+piv["Failure"]))

piv["Failure Rate"] = 100*(piv["Failure"] / (piv["Success"]+piv["Failure"]))

piv.sort_values(by="Success Rate",ascending=False,inplace=True)



piv.drop(columns=["Failure","Success"],inplace=True)



fig,(ax1,ax2)=plt.subplots(2,1,figsize=(15,3))

ax1.set_title("Company-wise success and failure rates")

plt.ylabel("Company")

ax1.xaxis.set_ticks_position('top')

plt.xticks(rotation=90)

sns.heatmap(piv[:28].T,annot=True,fmt="0.0f",cbar=False,cmap="gray",square=True,linewidths=0.1,linecolor="lightgray",ax=ax1)

sns.heatmap(piv[27:].T,annot=True,fmt="0.0f",cbar=False,cmap="gray",square=True,linewidths=0.1,linecolor="lightgray",ax=ax2)

plt.show()
cnt_co = df.groupby(["Country"]).count()[["Location"]].copy().sort_values(by="Location",ascending=False).reset_index()

sns.catplot(y="Country",x="Location",data=cnt_co,kind="bar",color="black",height=9,aspect=1)

plt.ylabel("Launch location")

plt.xlabel("No of missions")

plt.title("Launch locations & no of missions")

for i in range(cnt_co.shape[0]):

    plt.text(s=str(cnt_co.iloc[i,1]),y=i,x=cnt_co.iloc[i,1]+10)

plt.show()
cl = df.copy()

cl["Status Mission"] = cl["Status Mission"].replace(".* Failure","Failure",regex=True)

sns.catplot(data=cl,x="Country",y="Year",kind="swarm",height=7,aspect=2,palette=sns.set_palette(["gray","black"]),hue="Status Mission")

plt.title("Yearly trend in number of missions per launch location")

plt.xlabel("Launch location")

plt.xticks(rotation=90)

plt.show()
list(df.loc[df["Country"]=="Kazakhstan","Company Name"].unique())
df.loc[(df["Country"]=="Kazakhstan") & (df["Year"]<1960),["Country","Year","Company Name"]].groupby(["Year","Company Name"]).count()["Country"]
cl1 = df[df["Country"].isin(cnt_co.loc[:4,"Country"])].copy()

cl1.rename(columns={"Country":"Launch_Location"},inplace=True)

cl1 = pd.crosstab(index=cl1["Year"],columns=cl1["Launch_Location"]).copy()

ax=cl1.plot(figsize=(15,6),cmap="Paired",marker="x")

ax.set_axisbelow(True)

ax.yaxis.grid(color='lightgray', linestyle='dashed')

ax.xaxis.grid(color='lightgray', linestyle='dashed')

plt.title("Trend in no. of missions of top 5 locations based on no. of missions")

plt.xlabel("Year")

plt.ylabel("No of launches")

plt.xticks(range(1957,2021))

plt.xticks(rotation=90)

plt.show()
piv = df[["Country","Status Mission"]].copy()

piv["Status Mission"] = piv["Status Mission"].replace(".* Failure","Failure",regex=True)

piv = pd.crosstab(columns=piv["Status Mission"],index=piv["Country"])



piv["Success Rate"] = 100*(piv["Success"] / (piv["Success"]+piv["Failure"]))

piv["Failure Rate"] = 100*(piv["Failure"] / (piv["Success"]+piv["Failure"]))

piv.sort_values(by="Success Rate",ascending=False,inplace=True)



piv.drop(columns=["Failure","Success"],inplace=True)



fig=plt.figure(figsize=(15,5))

ax=sns.heatmap(piv.T,annot=True,fmt="0.0f",cbar=False,cmap="gray",square=True,linewidth=0.1,linecolor="lightgray")

ax.xaxis.set_ticks_position('top')

plt.xticks(rotation=90)

plt.title("Launch location-wise success and failure rates")

plt.xlabel("Launch location")

plt.ylabel("")

plt.show()
aa=df.loc[(df["Country"]=="Kenya") | (df["Country"]=="Brazil"),["Country","Status Mission"]].reset_index(drop=True)

aa.columns=["Launch Location","Mission Status"]

aa.groupby(["Launch Location","Mission Status"]).size()
pri = df[["Status Mission","Price"]].copy()

pri.fillna(0,inplace=True)

pri["Price Missing"] = pri["Price"]==0

pri.groupby(["Status Mission","Price Missing"]).count()["Price"]
piv = df[["Price","Status Mission"]].copy()

piv["Status Mission"] = piv["Status Mission"].replace(".* Failure","Failure",regex=True)

piv.loc[piv["Price"]=="nan","Price"]=np.nan

piv["Price"] = piv["Price"].str.replace(",","",regex=False)

piv.dropna(inplace=True)

piv["Price"] = piv["Price"].astype(np.float32)

sns.boxplot(data=piv,x="Status Mission",y="Price")

plt.title("Price vs Mission status")

plt.xlabel("Mission status")

plt.ylabel("Price ($ million)")

plt.show()
piv = piv.loc[piv["Price"]<1000,:].copy()

sns.boxplot(data=piv,x="Status Mission",y="Price")

plt.title("Price vs Mission status without outliers")

plt.xlabel("Mission status")

plt.ylabel("Price ($ million)")

plt.show()