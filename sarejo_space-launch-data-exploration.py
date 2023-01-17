import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re
df = pd.read_csv("../input/spacelaunches/clean_launch_data.csv")

df.head()
df.describe()
df.info()
df["nation"] = df["nation"].fillna("unknown")
df["equipment"].dropna().value_counts().head(30)

#Best stay away from this for now!
#Fixing the type / application column

df["type"] = df["type / application"].fillna("").astype(str)  +df["type, application"].fillna("").astype(str) 

df = df.drop(["type / application","type, application"], axis = 1)

df.head()



df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month
#on "id" we have the id of the launch. We'll only keep those two columns, and drop the duplicates to get the unique launches each year.

per_year_df = df[ ["year","id"] ].drop_duplicates()

#Now, we group by year, get the size, and with that plot the number of launches.

per_year_df.groupby("year").size().plot(kind="line",figsize=(16,4))
per_year_df.hist(bins=[1960,1970,1980,1990,2000,2010,2020])
#Failure Rate, another plot in the same showing failure trend

per_year_df = df[ ["year","id","failed"] ].drop_duplicates()

per_year_df.head()
failures_per_year = per_year_df.groupby(["year","failed"]).size().unstack().reset_index();

failures_per_year["success"] = (failures_per_year[False])

failures_per_year["failures"] = (failures_per_year[True])

failures_per_year["total_launches"] = (failures_per_year[True]+failures_per_year[False])

failures_per_year["success_rate"] = 100*failures_per_year[False]/failures_per_year["total_launches"]

failures_per_year = failures_per_year[ ["year","success","failures","total_launches","success_rate"] ]



#failures_per_year.tail(10)



success_year= pd.melt(failures_per_year,id_vars=["year"],value_vars=["total_launches","failures","success_rate"],var_name="variable")



#success_year.tail(10)
facets = sns.FacetGrid(success_year, col="variable",sharex=False,sharey=False)

facets.map(sns.lineplot,"year","value")
sites = pd.read_json("../input/spacelaunches/sites.json")

sites.head(25)
df["site_code"] = df["site"].apply(lambda x: x.split(" ")[0]).astype("string")

print("Has null values? ", len(df["site_code"])!=len(df["site_code"].dropna()))

sites.set_index("code",inplace=True)
#Check different values for the site codes.

launch_site_codes = set(df["site_code"].unique())

site_codes = set(sites.index.unique())

unmatched_codes = list(launch_site_codes-site_codes)

unmatched_codes
def clean_site_code(raw_site):

    return re.sub("@|,|-.*","",raw_site.split(" ")[0]).strip()



def unmatched_codes(launch_site_codes,site_codes):

    unmatched_code_list = list(set(launch_site_codes)-set(site_codes))

    return unmatched_code_list
df["site_code"] = df["site"].apply(clean_site_code).astype("string")

unmatched = unmatched_codes(df["site_code"].unique(),sites.index.unique())

df[df["site_code"].isin(unmatched)]["site_code"].value_counts()
#What payloads have incorrect launch sites (besides CCK)? With the year of the launch site, we'll revisit our source of the data.



df[ (df["site_code"].isin(unmatched)) & ( df["site_code"] != "CCK")][["year","site_code","site"]]
new_sites = [

    ["USA","NASA John F. Kennedy Space Center, Cape Canaveral, Florida, USA","orb","NASA John F. Kennedy Space Center, Cape Canaveral, Florida, USA"],

    ["China","Unknown, China","orb","Unknown, China"]

]

cck_ys = pd.DataFrame(new_sites

                   ,columns=["country","raw","details","name"],

                   index=["CCK","YS"])



sites = sites.append(cck_ys)



unmatched = unmatched_codes(df["site_code"].unique(),sites.index.unique())

df[df["site_code"].isin(unmatched)][["year","site_code"]]



#df.loc[df["site"]=="LC-1/5", "site_code" ] = "CC"

df.loc[df["site_code"]=="", "site_code" ] = "CC"

df.loc[df["site_code"]=="SLC", "site_code" ] = "CC"

df.loc[df["site_code"]=="BaS", "site_code" ] = "Ba"

#df[df["site_code"]=="SLC"]



print("Number of unmatched site codes: ", len(df[df["site_code"].isin(unmatched)][["year","site_code","site"]]))
joined = df.join(sites, on="site_code", lsuffix="site__")

joined.head()

ussr_df = df[(df["nation"] == "russia") | (df["nation"] == "ussr")][ ["year","nation"] ]

grid = sns.FacetGrid(ussr_df,col="nation")

grid.map(plt.hist,"year")
sites.reset_index().head()

#sites[sites["country"].str.contains("USSR") | sites["country"].str.contains("Uzb")]

sites.reset_index().groupby("index").filter(lambda x: len(x) > 1)

sites.reset_index(inplace=True)

sites.head()

sites.at[123,"index"] = "Sin-2"

sites.at[195,"index"] = "W-MARS"





sites.groupby("index").filter(lambda x: len(x) > 1)

sites.set_index("index",inplace=True)
sites.head()



ambiguous_sites = sites[ sites["country"]=="USSR / Russia" ]



df.loc[ (df["year"]<1992) & (df["site_code"].isin(ambiguous_sites.index) ), "site_code" ] = df["site_code"] + "-USSR"



ambiguous_sites.index = ambiguous_sites.index + "-USSR"

sites = sites.append(ambiguous_sites)



sites.tail()
joined = df.join(sites, on="site_code")

#print("Joined dataframe length: ",len(joined))

joined.head(10)
unique_launches = joined.groupby("id",as_index=False).agg({'nation': 'nunique', 

                                                           'type': 'nunique', 'year': 'first', 'month': 'first', 

                                                           'country': 'first',"site":"first", "power":"count","vehicle":"first","date":"first","failed":"first","equipment":"count"})



unique_launches.loc[ (unique_launches["year"]>=1992) & (unique_launches["country"]=="USSR / Russia" ), "country" ] = "Russia"



unique_launches = unique_launches.rename(columns={"power":"launches","equipment":"payload_total"})



agg_map = {"country":"nunique","type":"nunique","launches":"count","nation":"sum","vehicle":"nunique","failed":"sum","payload_total":"sum"}

countries_year = unique_launches.groupby("year",as_index=False).agg(agg_map)

successful_countries_year = unique_launches[unique_launches["failed"]==False].groupby("year",as_index=False).agg(agg_map)



countries_year["success_rate"] = 100 - 100*(countries_year["failed"] / countries_year["launches"])

countries_year["countries_per_launch"] = 100*(countries_year["country"] / countries_year["launches"])

countries_year["nations_per_launch"] = 100*(countries_year["nation"] / countries_year["launches"])



countries_year.head()
nations_year_month = joined.groupby(["year","month"],as_index=False).agg({'nation': 'nunique'})
#Add color map.

from matplotlib.colors import ListedColormap

cmap = ListedColormap(sns.color_palette("RdYlGn",10))
#Let's analize the error/success rate each year

fig = plt.figure(figsize=(20,6))



ax = sns.barplot(x="year",y="launches",data=countries_year,color="green")

sns.barplot(x="year",y="failed",data=countries_year,color="red",ax=ax)



#ax.legend(countries_year,["A","B"])

ax.set_xticklabels([year if year%5==0 else "" for year in countries_year["year"]])

ax.set_title("Number of space launches each year (successful and unsuccessful)", fontsize=20)
print(df["nation"].value_counts())

print(df["nation"].value_counts().tail(40))
#df[ df["nation"].fillna("unknown").str.contains("→") ].count()

multi_ownership = df[ df["nation"].str.count(",") > 0 ]

transferred = df[ df["nation"].str.count("→") > 0 ]



print("Satellites with multiple owners: ", len(multi_ownership) )

print("Transferred satellites: ", len(transferred) )





df["country_count"] = 1 + df["nation"].str.count(",")



fix, axes = plt.subplots(3,1,figsize=(10,10))



axes[0].set_title("Country count")

axes[1].set_title("Payloads with multi-ownership")

axes[2].set_title("Payloads with multi-ownership")



df.hist("country_count",ax=axes[0])

df[ df["country_count"] > 1 ].hist("country_count",ax=axes[1])

df[ df["country_count"] > 2 ].hist("country_count",ax=axes[2])
def clean_country(x):

    return x.strip();



def build_clean_map():

    words_to_remove = ["\d+","\(.*?\)","\(","\)","→.*"]



    clean_map = {word:"" for word in words_to_remove}

    return clean_map



clean_map = build_clean_map();



#joined["second_nation"] = joined["nation"].apply(lambda x: x.split(",")[1] if (x.count(",") > 0) else None)

#joined["first_nation"] = joined["nation"].apply(lambda x: x.split(",")[0]);



joined["second_nation"] = joined["nation"].apply(lambda x: x.strip()).replace(clean_map,regex=True).apply(lambda x: x.strip());

joined["first_nation"] = joined["second_nation"].apply(lambda x: x.split(",")[0])

joined["second_nation"] = joined["second_nation"].apply(lambda x: x.split(",")[1] if (x.count(",") > 0) else None)



nation_payloads_year = joined.groupby("year")["first_nation","second_nation"].apply(lambda x: len(pd.unique(x.values.ravel()).tolist()))



#joined["first_nation"].concat(joined["second_nation"]).unique()



#clean_country("monaco")
launches_per_month = unique_launches.groupby( ["year","month"] ).agg({"country":"count"}).reset_index()



countries_per_month = unique_launches.groupby( ["year","month"] ).agg({"country":"nunique"}).reset_index()

countries_per_month.head()

country_month_pivot_table = countries_per_month.pivot("month","year","country").fillna(0)



launches_per_month = unique_launches.groupby( ["year","month"] ).agg({"country":"count"}).reset_index()

launches_month_pivot_table = countries_per_month.pivot("month","year","country").fillna(0)



#We transform both columns, first nation and second nation into a single list, and count the unique values with len. pd.unique(x.values.ravel()) 

nation_payloads_year = joined.groupby("year")["first_nation","second_nation"].apply(lambda x: len(pd.unique(x.values.ravel()).tolist())).reset_index().rename({0:"nation"},axis=1)

#nation_payloads_year = joined.groupby(["year"],as_index=False).agg({'first_nation': 'nunique'})



fig,(ax,ax2) = plt.subplots(2,1,figsize=(16,8))

sns.heatmap(country_month_pivot_table,ax=ax, square=True,yticklabels=False)

sns.lineplot(x="year",y="nation",data=nation_payloads_year[ nation_payloads_year["year"]<2020 ],ax=ax2,legend="full")



ax.set_xticklabels([year if year%5==0 else "" for year in countries_year["year"]])

ax.set_title("Number of different countries launching to space each month",fontsize=20)

ax.set_yticklabels([0, 1,2,3,4,5,6,7,8])

ax.yaxis.set_ticks([ 0, 1,2,3,4,5,6,7,8])



ax2.set_title("Different nations putting payloads in orbit each year",fontsize=20)



country_launches = unique_launches.groupby("country").agg({"year":["min", "max"],"type":"count"}).reset_index()

country_launches.columns = ['-'.join(col) for col in country_launches.columns.values]

country_launches = country_launches.rename({"type-count":"launches","country-":"country","year-min":"first_launch_year","year-max":"last_launch_year"},axis=1)

country_launches["yearly_rate_since_first"] = country_launches["launches"] / (2020-country_launches["first_launch_year"])

country_launches.sort_values("first_launch_year",ascending=False)
country_launches.sort_values("last_launch_year",ascending=False)
#Pace of launches for selected countries

#

#country_and_month = unique_launches.groupby( ["country","year","month"] ).size().reset_index()

#

#countries = []

#

#country = country_and_month[ country_and_month["country"]=="China" ].pivot("year","month",0).fillna(0)

#

#sns.heatmap(country,cmap="coolwarm")







argentina = joined[ joined["nation"]=="argentina" ].groupby(["year","month"] ,as_index=False).agg({'nation': 'count'})



argentina.pivot("year","month","nation").fillna(0)
decade_tables = []

years = [1950,1975,2000]

fig, axes = plt.subplots(len(years),1,figsize=(20,20))



for i,year in enumerate(years):

    decade_group = unique_launches[ (unique_launches["year"]>=year) & (unique_launches["year"]<year+25) ].groupby(["country","year"])

    decade_pivot_table = decade_group.agg({"month":"count"}).reset_index().pivot("country","year","month")

    decade_tables.append( decade_pivot_table.fillna(0 ) )

    sns.heatmap(decade_pivot_table.fillna(0),ax=axes[i], cmap="coolwarm")                                                                     



countries = unique_launches.groupby( ["year","country"] ).size().reset_index().rename({0:"launches"},axis=1)

top_countries = countries.groupby("country").mean().sort_values("launches", ascending=False).head(8).index



countries["is_top"] =  countries["country"].isin(top_countries)



other_countries = countries[ countries["is_top"]==False ].groupby("year").sum().reset_index()[ ["year","launches"] ]

other_countries["country"] = "Other"



all_countries = pd.concat( [ countries[ countries["country"].isin(top_countries) ][ ["year","launches","country"] ],other_countries] )



fig, axes = plt.subplots(2,1,figsize=(20,20))



axes[0].set_title("Number of launches by country",fontsize=30)

axes[1].set_title("Number of launches by country, 2010 onwards",fontsize=30)

sns.lineplot(x="year",y="launches",hue="country",data=all_countries,ax=axes[0], lw=3)

sns.lineplot(x="year",y="launches",hue="country",data=all_countries[ (all_countries["year"]>2010) & (all_countries["year"]<2020)] ,ax=axes[1], lw=3)



fig, (ax) = plt.subplots(1,1,figsize=(20,6))

#sns.scatterplot(x="year",y="country", size="country", hue="nation", data=countries_year)

ax.set_title("Number of launches per year, with total number of payloads",fontsize=30)

sns.scatterplot(x="year",y="launches", size="payload_total", color="green", palette="coolwarm", data=countries_year,ax=ax, sizes=(1,600), marker="o")

fig.show()

#Let's start plotting the number of launches we had each year, and how many different countries launched into space.

#We'll also plot how many different vehicles were used each year.

fig, (ax,ax2) = plt.subplots(2,1,figsize=(20,6))

#sns.scatterplot(x="year",y="country", size="country", hue="nation", data=countries_year)

ax.set_title("Number of satellites/payload put in orbit per year")

successful_countries_year["payload_per_launch_avg"] = successful_countries_year["payload_total"] / successful_countries_year["launches"]



chart = sns.barplot(x="year",y="payload_total", data=successful_countries_year,ax=ax,color="#8080DA")

ax.set_xticklabels([year if year%5==0 else "" for year in countries_year["year"]])



ax2.set_title("Average number of satellites/payloads per launch")

chart = sns.barplot(x="year",y="payload_per_launch_avg", data=successful_countries_year,ax=ax2,color="#8080DA")

ax2.set_xticklabels([year if year%5==0 else "" for year in countries_year["year"]])



fig.show()
