import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt 
import geopandas as geo
%matplotlib inline
# read in data
cali= geo.read_file(r"../input/covid19/CA_Counties_TIGER2016.shp")
cali.head()
# read in data
cases=pd.read_csv(r"https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv")
cali_cases= cases[cases["state"]=="California"]
cali_cases.head()
cali_cases=cali_cases.reset_index().drop("index", axis=1)
cali_cases=cali_cases.drop("fips",axis=1)
cali_cases.info()
# When I combined the data, I chose inner join since I wanted to keep the counties that were present in both datasets. 
covid=cali_cases.merge(cali, left_on="county",right_on="NAME", how="inner")
covid.head()
# Keep interested columns 
covid=covid[["date","county","cases","deaths","geometry"]]
covid.shape
covid.head()
# Calculates the new cases/deaths and growths for one county
def fun(name,covid=covid):
    county=covid[covid["county"]== name]
    county= county.reset_index()
    new_cases= [county["cases"][0]]+[county["cases"][i]-county["cases"][i-1]for i in range(1,len(county["cases"]))]
    new_deaths= [county["cases"][0]]+[county["deaths"][i]-county["deaths"][i-1]for i in range(1,len(county["cases"]))]
    county["new_cases"]=new_cases
    county["new_deaths"]= new_deaths
    g=[0]
    for i in range(1,len(county["new_cases"])): 
        x= ((county["new_cases"][i]-county["new_cases"][i-1])/county["new_cases"][i-1])*100
        g.append(x)
    county["cases_growth"]= g
    h=[0]
    for i in range(1,len(county["new_deaths"])): 
        x= ((county["new_deaths"][i]-county["new_deaths"][i-1])/county["new_deaths"][i-1])*100
        h.append(x)
    county["death_growth"]= h
    county["date"]= pd.to_datetime(county["date"])
    return county

# Implements the function on all counties and puts them into one dataframe
counties= covid["county"].unique()
gdf= fun(counties[0])
for i in counties[1:]: 
    x= fun(i)
    gdf= pd.concat([gdf,x], ignore_index=True)
gdf.shape
gdf.head(10)
# total represents all of california data
total= gdf.groupby("date").sum()
total=total.reset_index().drop("index",axis=1)
total.head()
fig,axes= plt.subplots(1,2,figsize=(20,5))
fig.suptitle("COVID New Cases")
shrunk=total[total["date"]>="2020-03-01"]
axes[0].plot("date","new_cases", data=shrunk)
axes[0].set_title("Number of New Cases")
axes[1].plot("date","new_deaths", data=shrunk, color="red")
axes[1].set_title("Number of New Deaths")
for ax in axes: 
       ax.tick_params("x",labelrotation=90)
       ax.set_xlabel("Date")
       ax.set_ylabel("Number of New Cases")
        
# Need to recalcuate growth percentages after groupby
g=[0]
for i in range(1,len(total["new_cases"])): 
        x= ((total["new_cases"][i]-total["new_cases"][i-1])/total["new_cases"][i-1])*100
        g.append(x)
total["cases_growth"]= g
h=[0]
for i in range(1,len(total["new_deaths"])): 
        x= ((total["new_deaths"][i]-total["new_deaths"][i-1])/total["new_deaths"][i-1])*100
        h.append(x)
total["death_growth"]= h
plt.figure(figsize=(8,8))
shrunk=total[total["date"]>="2020-03-01"]
plt.plot("date","cases_growth", data=shrunk)
plt.plot("date","death_growth", data=shrunk)
plt.xticks(rotation=90)
plt.legend();
plt.title("COVID Daily Growth Rate");
# Calculate the average number of people who contract the coronavirus daily
def average_new_cases_percent(table, start,end): 
    shrunk=table[(table["date"]>= start) & (table["date"]<= end) ]
    shrunk=shrunk.reset_index()
    return (np.mean(shrunk["new_cases"])/shrunk["e_totpop"][0])*100
def average_new_death_percent(table, start,end): 
    shrunk=table[(table["date"]>= start) &( table["date"]<= end)]
    shrunk= shrunk.reset_index()
    return (np.mean(shrunk["new_deaths"])/shrunk["e_totpop"][0])*100

def ave(d,start,end): 
    av_cases=[]
    av_deaths=[]
    for i in d["county"].unique(): 
        df=d[d["county"]==i]
        av_cases.append(average_new_cases_percent(df,start,end))
        av_deaths.append(average_new_death_percent(df,start,end))
    averages= pd.DataFrame()
    averages["county"]= d["county"].unique()
    averages["av_cases_daily"]= av_cases
    averages["av_deaths_daily"]= av_deaths
    print("Top 25 cases:")
    print(averages.sort_values(by="av_cases_daily", ascending=False).head(25).reset_index()[["county","av_cases_daily"]])

    return averages
    
    
    
# Combining the cases dataframe with a dataframe that contains the estimated population.(read in a later cell)#
new= gdf.merge(jo ,on="county")
new.head()
av=ave(new,"2020-03-01","2020-6-09")
Imperial= av.loc[34][1]
sb.distplot(av["av_cases_daily"]);
plt.scatter(Imperial, 0, color='red', s=100);
plt.title("Average Percent of Population contracting COVID Daily  ");
# only kept Counties with 100+ cases
covid["date"]= pd.to_datetime(covid["date"])
final= covid[covid["date"]=="2020-6-09"]
final["d/c"]=final["deaths"]/ final["cases"]
sfinal= final[final["cases"]>99]
print("Top 25 death ratio:")
print(sfinal.sort_values(by="d/c", ascending=False).head(25).reset_index().drop("index", axis=1)[["county","cases","deaths","d/c"]])
Yolo=sfinal.sort_values(by="d/c", ascending=False).head(1)["d/c"][1867]
sb.distplot(sfinal["d/c"]);
plt.scatter(Yolo, 0, color='red', s=100)
plt.title("Death/Cases ratio");
# Shapiro Wilk Test to test for normality ; no YOLO
from scipy import stats
no=sfinal.drop(1867)
print(no.skew())
p_value=stats.shapiro(no["d/c"])[1]
if p_value< .05: 
    print("Reject Null hypothesis of normality")
else: 
    print("Fail to reject Null Hypothesis")
# read in dataset with population numbers
social= pd.read_csv(r"../input/covid19/cdcs-social-vulnerability-index-svi-2016-overall-svi-county-level.csv")
social= social[social["state"]=="CALIFORNIA"]
pop= social[["county",'e_totpop']]
jo=pop.copy()
# join the dataset with COVID-19 cases 
final1=covid[covid["date"]=="2020-6-09"]
grouped=pop.merge(final1, on="county", how="inner")

# Calculate percent 
grouped["casesp"]= (grouped["cases"]/grouped["e_totpop"])*100
grouped["deathsp"]= (grouped["deaths"]/grouped["e_totpop"])*100
grouped.head()
print(grouped.sort_values("casesp", ascending=False)[["county","e_totpop","cases","casesp"]].reset_index().drop("index", axis=1).head(25))
plt.close()
changed= grouped.copy()
piv_cases=geo.GeoDataFrame(changed)
piv_cases["casesp"]= piv_cases["casesp"].replace(max(piv_cases["casesp"]),.65)
piv_cases["casesp"]= piv_cases["casesp"].replace(max(piv_cases["casesp"]),.65)
ax= piv_cases.plot(column="casesp",cmap="OrRd",figsize=(10,10),legend=True,edgecolor="black",linewidth=0.4)
ax.set_title("Percentage of People with COVID-19 up to June 9th ")
ax.set_axis_off()
# keep all features represented as a percent 
keep=["county","geometry","ep_pov","ep_unemp","ep_nohsdp",
     "ep_age65","ep_age17","ep_disabl","ep_sngpnt",
      "ep_minrty","ep_limeng",
     "ep_munit","ep_mobile","ep_crowd","ep_noveh","ep_groupq"]
social=social[keep]

# combinbe cases and SVI data
social_cases=grouped.merge(social,on="county")
social_cases.head()
# Create correlation heatmap
plt.figure(figsize=(10,10))
plt.title("Heatmap for Social Vulnerability Features")
keep1= keep[2:]+["casesp","deathsp","county"]
pplot= social_cases[keep1]
pcorr= pplot.corr()
ax=sb.heatmap(pcorr,vmin=-1, vmax=1, center=0, cmap=sb.diverging_palette(20, 220, n=200),square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');

interested=["ep_pov","ep_munit","ep_crowd","casesp"]
pplot1=pplot[interested]


# find where counties are and remove
print(np.where(pplot["county"]=="Imperial"))
print(np.where(pplot["county"]=="Kings"))
none=pplot.drop([54,55])
v=["ep_pov","ep_munit","ep_crowd"]
fig,ax=plt.subplots(1,3,figsize=(25,5))
fig.suptitle('Casesp vs Social Factors without Imperial and Kings County', fontsize=20)
for rows,j in zip(ax,v): 
    rows.scatter(x=none[j],y=none["casesp"])
    rows.set_xlabel(j)
    rows.set_ylabel("casesp")
# need to scale the variables 
def min_max(x): 
    return (x- min(x))/(max(x)-min(x))
    
none["crowd2"]= none["ep_crowd"]**2
for i in ["ep_munit","ep_crowd","crowd2"]: 
    none[i]= min_max(none[i])
# build stats models
import statsmodels.api as sm
none["intercept"]=1
mod1= sm.OLS(none["casesp"],none[["intercept","ep_munit","ep_crowd","crowd2"]])
res1=mod1.fit()
res1.summary()
mod2= sm.OLS(none["casesp"],none[["intercept","ep_munit","ep_crowd"]])
res2=mod2.fit()
res2.summary()
print(social.sort_values("ep_munit", ascending=False)[["county","ep_munit"]].reset_index().drop("index", axis=1).loc[:9])
print(social.sort_values("ep_crowd", ascending=False)[["county","ep_crowd"]].reset_index().drop("index", axis=1).loc[:9])
new=social.merge(cali, left_on="county",right_on="NAME", how="inner")[["county","geometry_y","ep_crowd","ep_munit"]]
new.rename(columns = {'geometry_y':'geometry'}, inplace = True)
plt.close()

new=geo.GeoDataFrame(new)
ax= new.plot(column="ep_munit",cmap="Greens",figsize=(10,10),legend=True,edgecolor="black",linewidth=0.4)
ax.set_title("Percent of People in Housing with 10 or more Units ")
ax.set_axis_off()
plt.close()

new=geo.GeoDataFrame(new)
ax= new.plot(column="ep_crowd",cmap="Greens",figsize=(10,10),legend=True,edgecolor="black",linewidth=0.4)
ax.set_title("Percent of Houses with More People than Rooms ")
ax.set_axis_off()
# read in Google Mobility Data
glob=pd.read_csv(r"../input/covid19/Global_Mobility_Report.csv")
usa= glob[glob["country_region"]== "United States"]
cal= usa[usa["sub_region_1"]== "California"]
cal.head()
cali_total= cal[cal["sub_region_2"].isna()==True]
cali_total["date"]=pd.to_datetime(cali_total["date"])
col=cali_total.columns[7:]
col[3:6]
cal["date"]=pd.to_datetime(cal["date"])
plt.close()
fig, axes= plt.subplots(2,3, figsize=(20,20))
fig.suptitle("California Mobility Data", size=15);
plt.subplots_adjust(wspace = 0.2,hspace = 0.2)
for i in range(2):
    x=col[:3]
    y=col[3:6]
    for j in range(3): 
        if i==0: 
            axes[i,j].plot(cali_total["date"],cali_total[x[j]])
            axes[i,j].tick_params(axis='x',rotation=90);
            axes[i,j].set_title(x[j]);
            axes[i,j].set_ylabel("Percent from Baseline");
            axes[i,j].set_xlabel("Date");
        if i==1: 
            axes[i,j].plot(cali_total["date"],cali_total[y[j]])
            axes[i,j].tick_params(axis='x',rotation=90);
            axes[i,j].set_title(y[j]);
            axes[i,j].set_ylabel("Percent from Baseline");
            axes[i,j].set_xlabel("Date");
# read in data
health=pd.read_csv(r"../input/covid19/crowd-sourced-covid-19-testing-locations.csv")
health.head()
# California represented two ways CA and California
health["location_address_region"].unique()
# Find just california and CA
cali_health=health[(health["location_address_region"]=="CA" )|(health["location_address_region"]=="California")]
cali_health.head()
cali_health.shape
cali_health["is_location_screening_patients"].value_counts()
# The following cells make sure the coordinate system is same for plotting
screening= cali_health[cali_health["is_location_screening_patients"]=="t"]
screening=geo.GeoDataFrame(screening)

cali.crs = {'init' :'epsg:4326'}
screening.crs=cali.crs
screening.drop("geometry", axis=1, inplace=True)
gdf = geo.GeoDataFrame(
    screening, geometry=geo.points_from_xy(screening["lng"], screening["lat"]))
gdf.crs=cali.crs
# Note: Because LA has such a large population, using the real population of LA creates a poor map. Therefore, I changed it to another value. The axis is still correct. 
plt.close()
piv_cases=geo.GeoDataFrame(grouped)
piv_cases.crs= cali.crs
piv_cases["e_totpop"]=piv_cases["e_totpop"].replace(max(piv_cases["e_totpop"]),3253356)

base= piv_cases.plot(column="e_totpop",cmap='BuGn',figsize=(10,10),legend=True,edgecolor="black",linewidth=0.4)
gdf = gdf.to_crs({'init': 'epsg:3857'})
ax=gdf.plot(ax=base,figsize=(10,10),zorder=2,color="red")
ax.set_title("Population vs COVID Testing Centers ")
ax.set_axis_off()
# read in Data
bed_counts=pd.read_csv(r"../input/covid19/bed_counts.csv")
bed_counts.head()
# change Counties to Lowercase
bed_counts["county"]= [i.lower() for i in bed_counts["COUNTY_NAME"]]
# Capitalize first letter of each word in County Name 
def capital(ls): 
    if len(ls)==1: 
        return ls[0].capitalize()
    else: 
        return ls[0].capitalize()+" " +capital(ls[1:])
bed_counts["county"]= [capital(i.split())for i in bed_counts["county"]]
icu_beds= bed_counts[["county","INTENSIVE CARE"]]
# Get ICU COVID data
icu_counts=pd.read_csv(r"../input/covid19/icu_cases.csv")
icu_counts.head()
icu_counts["total_icu"]= icu_counts["ICU COVID-19 Positive Patients"]+ icu_counts["ICU COVID-19 Suspected Patients"]
icu_counts["Most Recent Date"]=pd.to_datetime(icu_counts["Most Recent Date"])
icu= icu_counts[icu_counts["Most Recent Date"]=="2020-06-09"]
# combine all data sets 
k=icu.merge(icu_beds, left_on="County Name", right_on="county", how="inner").drop("county", axis=1)
cali_icu= k.merge(cali, left_on="County Name", right_on="NAME", how="inner")
cali_icu.head()

cali_icu.shape
# Calculate Percentage 
cali_icu["percent"]=(cali_icu["total_icu"]/cali_icu["INTENSIVE CARE"])*100
print(cali_icu.sort_values(by="percent", ascending=False)[["County Name","total_icu","INTENSIVE CARE","percent"]].reset_index().drop("index", axis=1).head(10))
# Imperial counnty reduced on map because it is an outlier. However, axis is still correct. 
plt.close()
co= cali_icu.copy()
co["percent"]=co["percent"].replace(max(co["percent"]),37)
piv_cases=geo.GeoDataFrame(co)
ax=piv_cases.plot(column="percent",cmap='Purples',figsize=(10,10),legend=True,edgecolor="black",linewidth=0.4)
ax.set_title("Percent of ICU Beds Used by Covid-19")
ax.set_axis_off()
