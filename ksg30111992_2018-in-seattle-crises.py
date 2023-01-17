# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
sns.set()

# Any results you write to the current directory are saved as output.
crisis_data = pd.read_csv("../input/crisis-data.csv", parse_dates=[["Reported Date", "Reported Time"], "Occurred Date / Time"], infer_datetime_format=True)
crisis_data.columns = crisis_data.columns.str.strip().str.lower().str.replace("/","_").str.replace(" ","_")
# Some reported date values are 1/1/1900. These values, which are rather few in number, will be overwritten by the occurred date/time
crisis_data.reported_date_reported_time = np.where(crisis_data.reported_date_reported_time.dt.year==1900, crisis_data.occurred_date___time,crisis_data.reported_date_reported_time)
crisis_data_years = [2015,2016,2017,2018]
crisis_data_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
crisis_data.head()
pd.concat([crisis_data[crisis_data.reported_date_reported_time.dt.year==x].call_type.value_counts().rename(x) for x in crisis_data.reported_date_reported_time.dt.year.unique()],axis=1, sort=False).fillna(0).T.plot(kind="bar", stacked=True, layout=(2,2), title="Crises reported each year",figsize=(16,4))
crisis_by_initial_call_type = pd.concat([crisis_data[crisis_data.reported_date_reported_time.dt.year==x].initial_call_type.value_counts().rename(x) for x in crisis_data.reported_date_reported_time.dt.year.unique()],axis=1, sort=False).fillna(0)
crisis_by_initial_call_type["Total"] = crisis_by_initial_call_type.sum(axis=1)
crisis_by_initial_call_type.sort_values(by="Total",ascending=False).drop("Total",axis=1).head(5)[::-1].plot(kind="barh", stacked=True, layout=(2,2), title="Types of Crises reported each year",figsize=(16,8),cmap="Reds");
plt.subplots(figsize=(16,4))
sns.heatmap(pd.concat([crisis_data[crisis_data.reported_date_reported_time.dt.year==x].reported_date_reported_time.dt.strftime("%b").value_counts().rename(x) for x in crisis_data_years[::-1]], axis=1, sort=False).fillna(0).reindex(index=crisis_data_months).T,cmap='Reds',annot=True, fmt='g').set_title("Crises reported by year by month");
ax = pd.concat([
    crisis_data[crisis_data.reported_date_reported_time.dt.year==x].reported_date_reported_time.dt.hour.value_counts().rename(x) for x in crisis_data_years
],axis=1).plot(kind="line", figsize=(16,4), title="Crises reported by the hour", xticks=np.arange(24), legend=True,x_compat=True).set_xticklabels(["{0:0=2d}:00".format(x) for x in np.arange(24)],rotation=90);
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(16,16))
sns.heatmap(pd.concat([crisis_data[(crisis_data.occurred_date___time.dt.hour==x) & (crisis_data.occurred_date___time.dt.year==2015)].reported_date_reported_time.dt.hour.value_counts().rename("{0:0=2d}:00".format(x)) for x in np.arange(24)],axis=1).fillna(0),ax=axes[0,0],yticklabels=["{0:0=2d}:00".format(x) for x in np.arange(24)]).set_title("2015")
axes[0,0].set_xlabel("Occurred Time");
axes[0,0].set_ylabel("Reported Time");
sns.heatmap(pd.concat([crisis_data[(crisis_data.occurred_date___time.dt.hour==x) & (crisis_data.occurred_date___time.dt.year==2016)].reported_date_reported_time.dt.hour.value_counts().rename("{0:0=2d}:00".format(x)) for x in np.arange(24)],axis=1).fillna(0),ax=axes[0,1],yticklabels=["{0:0=2d}:00".format(x) for x in np.arange(24)]).set_title("2016")
axes[0,1].set_xlabel("Occurred Time");
axes[0,1].set_ylabel("Reported Time");
sns.heatmap(pd.concat([crisis_data[(crisis_data.occurred_date___time.dt.hour==x) & (crisis_data.occurred_date___time.dt.year==2017)].reported_date_reported_time.dt.hour.value_counts().rename("{0:0=2d}:00".format(x)) for x in np.arange(24)],axis=1).fillna(0),ax=axes[1,0],yticklabels=["{0:0=2d}:00".format(x) for x in np.arange(24)]).set_title("2017")
axes[1,0].set_xlabel("Occurred Time");
axes[1,0].set_ylabel("Reported Time");
sns.heatmap(pd.concat([crisis_data[(crisis_data.occurred_date___time.dt.hour==x) & (crisis_data.occurred_date___time.dt.year==2018)].reported_date_reported_time.dt.hour.value_counts().rename("{0:0=2d}:00".format(x)) for x in np.arange(24)],axis=1).fillna(0),ax=axes[1,1],yticklabels=["{0:0=2d}:00".format(x) for x in np.arange(24)]).set_title("2018")
axes[1,1].set_xlabel("Occurred Time");
axes[1,1].set_ylabel("Reported Time");
sns.heatmap(pd.concat([crisis_data[crisis_data.reported_date_reported_time.dt.year==x].precinct.value_counts().rename(x) for x in crisis_data_years],axis=1,sort=False).assign(Total=crisis_data.precinct.value_counts()).sort_values(by="Total", ascending=False).drop("Total",axis=1),cmap="Reds",annot=True,fmt='g').set_title("Crises reported per Precinct");
seattle_precincts = crisis_data.precinct.unique()
# Removing nan and UNKNOWN precincts
seattle_precincts = seattle_precincts[~((seattle_precincts=="UNKNOWN")|(pd.isnull(seattle_precincts)))]
crisis_data_by_sector_by_precinct = pd.concat([crisis_data[(crisis_data.precinct==x) & (crisis_data.reported_date_reported_time.dt.year==2018)].sector.value_counts().rename("") for x in seattle_precincts],axis=1,sort=False).fillna(0)

# Pie charts in Pandas dont provide values by default. The autopct attribute provides a floating number equal to the slice percentage.
# To provide values instead, I will be indexing the numbers as illustrated here https://stackoverflow.com/questions/48299254/pandas-pie-plot-actual-values-for-multiple-graphs
# There the test is a temporary function that iterates over each element in my dataframe, and then returns the relevant value

row_index = [0]
column_index = [0]
def test(value):
    a = crisis_data_by_sector_by_precinct.iloc[row_index[0],column_index[0]]
    row_index[0]=(row_index[0]+1)%17
# The autopct sends values in a flattended format, meaning that if it's a multidimensional DataFrame, it will flatten to a single dimension and then send the values.
# This can be leveraged such that the column_index is incremented when the last row is hit
    if(row_index[0]==0):
        column_index[0] = (column_index[0]+1)%5
# Since most values in my DataFrame could be zeros, I will not be displaying these values to prevent overlapping
    return int(a) if a>0 else ""

crisis_data_by_sector_by_precinct.plot(kind="pie",subplots=True,title=["Sectors in {0} Precinct".format(x) for x in seattle_precincts],figsize=(15,10),layout=(2,3),legend=False,autopct=test,cmap="Reds_r");
fig, ax = plt.subplots()
size = 0.3
radius = 2
pctdistance = 0.9
# Total crises reported by precinct by sector by year. Summing up this frame by level will give total crises per precinct for level=0, crises per sector for level=1, crises per year for level=2
crisis_by_precinct_sector_year_alternate = pd.DataFrame(pd.concat([crisis_data.precinct,crisis_data.sector,crisis_data.reported_date_reported_time.dt.year.rename("reported_year")],axis=1).groupby(["precinct","sector"]).reported_year.value_counts())
index_level = [0,0,0]

# add labels for level0 (precinct)
def add_labels_level0(value):
    temp = crisis_by_precinct_sector_year_alternate.sum(level=0).values[index_level[0]]
    tempstr = crisis_by_precinct_sector_year_alternate.sum(level=0).index.values[index_level[0]]
    index_level[0]+=1
    return int(temp)
# add labels for level1 (sector)
def add_labels_level1(value):
    temp = crisis_by_precinct_sector_year_alternate.sum(level=1).values[index_level[1]]
    tempstr = crisis_by_precinct_sector_year_alternate.sum(level=1).index.values[index_level[1]]
    index_level[1]+=1
    return int(temp)
# add labels for level2 (year)
def add_labels_level2(value):
    temp = crisis_by_precinct_sector_year_alternate.sum().values[index_level[2]]
    tempstr = crisis_by_precinct_sector_year_alternate.sum().index.values[index_level[2]]
    index_level[2]+=1
    return int(temp)

# turns out you can't pass a dataframe or a series and expect labels to be rotated. I guess pandas ruined me that way. Anyway, the values are flattened and then plotted
ax.pie(crisis_by_precinct_sector_year_alternate.sum(level=0).values.flatten(),radius=radius,wedgeprops=dict(width=size,edgecolor='w'),autopct=add_labels_level0,pctdistance=pctdistance,labels=crisis_by_precinct_sector_year_alternate.sum(level=0).index.tolist(),rotatelabels=True,labeldistance=1)
radius=radius-size
ax.pie(crisis_by_precinct_sector_year_alternate.sum(level=1).values.flatten(),radius=radius,wedgeprops=dict(width=size,edgecolor='w'),autopct=add_labels_level1,pctdistance=pctdistance,labels=crisis_by_precinct_sector_year_alternate.sum(level=1).index.tolist(),rotatelabels=True,labeldistance=0.45)
ax.set(aspect="equal");
suicide_emotional_vs_others=pd.concat([
#     Find all records by year where initial_call_type contains suicid. This will include Suicide and Suicidal call types
    crisis_data[crisis_data.initial_call_type.str.contains("suicid",case=False,na=False) & crisis_data.reported_date_reported_time.dt.year.isin(crisis_data_years)].reported_date_reported_time.groupby(crisis_data.reported_date_reported_time.dt.year).count().rename("Suicide Related"),
#     Find all records by year where initial_call_type contains emotion. This will include Emotion or emotion related crises
    crisis_data[crisis_data.initial_call_type.str.contains("emotion",case=False,na=False) & crisis_data.reported_date_reported_time.dt.year.isin(crisis_data_years)].reported_date_reported_time.groupby(crisis_data.reported_date_reported_time.dt.year).count().rename("Emotional Related"),
#     Finally, Find all records by year that are not the above. These are all the rest
    crisis_data[~(crisis_data.initial_call_type.str.contains("suicid",case=False,na=False)|crisis_data.initial_call_type.str.contains("emotion",case=False,na=False)) & crisis_data.reported_date_reported_time.dt.year.isin(crisis_data_years)].reported_date_reported_time.groupby(crisis_data.reported_date_reported_time.dt.year).count().rename("Others")
],axis=1).T
temp_index=[0]
def add_labels(value):
    a = suicide_emotional_vs_others.T.values.flatten()[temp_index[0]]
    temp_index[0]+=1
    return ("{0}\n{1:.1f}%".format(a,value))
suicide_emotional_vs_others.plot(
    kind="pie", subplots=True, figsize=(12,12), layout=(2,2), legend=False, title="Suicide and Emotional Crises vs the Other Crises", autopct=add_labels
);
suicide_initial_vs_final_calltype = pd.concat([
    crisis_data[crisis_data.initial_call_type.str.contains("suicid",na=False,case=False)].reported_date_reported_time.dt.year.rename("year"),
    crisis_data[crisis_data.initial_call_type.str.contains("suicid",na=False,case=False)].initial_call_type,
    crisis_data[crisis_data.initial_call_type.str.contains("suicid",na=False,case=False)].final_call_type
],axis=1)
suicide_initial_vs_final_calltype_count = pd.concat([suicide_initial_vs_final_calltype[(suicide_initial_vs_final_calltype.year==x) & (~suicide_initial_vs_final_calltype.final_call_type.str.contains("GENERAL",case=False))].final_call_type.value_counts().rename(x) for x in crisis_data_years],axis=1,sort=False).fillna(0)
suicide_initial_vs_final_calltype_count.index.name="Final Call Types for Suicide Related Initial Report"
suicide_initial_vs_final_calltype_count.sort_values(by=2018,ascending=False).head(10)
cit_details_by_call_type = (crisis_data[crisis_data.reported_date_reported_time.dt.year==2018])[["template_id","initial_call_type","final_call_type","cit_officer_requested","cit_officer_dispatched","cit_officer_arrived"]]
fig,ax = plt.subplots(nrows=4,ncols=3,figsize=(16,20))

cit_details_by_call_type.cit_officer_requested.value_counts().plot(kind="pie",autopct="%.2f",title="How Often was a CIT Officer requested in 2018?",ax=ax[0,0]).set(ylabel="")
cit_details_by_call_type.cit_officer_dispatched.value_counts().plot(kind="pie",autopct="%.2f",title="How Often was a CIT Officer dispatched in 2018?",ax=ax[0,1]).set(ylabel="")
cit_details_by_call_type.cit_officer_arrived.value_counts()[::-1].plot(kind="pie",autopct="%.2f",title="How Often did the CIT Officer arrive in 2018?",ax=ax[0,2]).set(ylabel="")

cit_details_by_call_type.pivot_table(index="initial_call_type",columns=cit_details_by_call_type.cit_officer_requested.rename("CIT Officer Requested"),values="template_id",aggfunc="count").sort_values(by="N",ascending=False).head(5)[::-1].plot(kind="barh",ax=ax[1,0],sharey=True).set(ylabel="Initial Call Type");
cit_details_by_call_type.pivot_table(index="final_call_type",columns=cit_details_by_call_type.cit_officer_requested.rename("CIT Officer Requested"),values="template_id",aggfunc="count").sort_values(by="N",ascending=False).head(5)[::-1].plot(kind="barh",ax=ax[2,0]).set(ylabel="Final Call Type");
cit_details_by_call_type.pivot_table(index="initial_call_type",columns=cit_details_by_call_type.cit_officer_dispatched.rename("CIT Officer Dispatched"),values="template_id",aggfunc="count").sort_values(by="N",ascending=False).head(5)[::-1].plot(kind="barh",ax=ax[1,1]).set(ylabel="Initial Call Type");
cit_details_by_call_type.pivot_table(index="final_call_type",columns=cit_details_by_call_type.cit_officer_dispatched.rename("CIT Officer Dispatched"),values="template_id",aggfunc="count").sort_values(by="N",ascending=False).head(5)[::-1].plot(kind="barh",ax=ax[2,1]).set(ylabel="Final Call Type");
cit_details_by_call_type.pivot_table(index="initial_call_type",columns=cit_details_by_call_type.cit_officer_arrived.rename("CIT Officer Arrived"),values="template_id",aggfunc="count").sort_values(by="N",ascending=False).head(5)[::-1].plot(kind="barh",ax=ax[1,2]).set(ylabel="Initial Call Type");
cit_details_by_call_type.pivot_table(index="final_call_type",columns=cit_details_by_call_type.cit_officer_arrived.rename("CIT Officer Arrived"),values="template_id",aggfunc="count").sort_values(by="N",ascending=False).head(5)[::-1].plot(kind="barh",ax=ax[2,2]).set(ylabel="Final Call Type");

sns.heatmap(ax=plt.subplot2grid((4,2),(3,0),rowspan=1,colspan=1),data=cit_details_by_call_type[cit_details_by_call_type.cit_officer_requested=="Y"].pivot_table(index="cit_officer_dispatched",columns="cit_officer_arrived",values="template_id",aggfunc="count",fill_value=0),annot=True,fmt="g").set(xlabel="CIT Officer Arrived",ylabel="CIT Officer Dispatched",title="CIT Officer was Requested");
sns.heatmap(ax=plt.subplot2grid((4,2),(3,1),rowspan=1,colspan=1),data=cit_details_by_call_type[cit_details_by_call_type.cit_officer_requested=="N"].pivot_table(index="cit_officer_dispatched",columns="cit_officer_arrived",values="template_id",aggfunc="count",fill_value=0),annot=True,fmt="g").set(xlabel="CIT Officer Arrived",ylabel="CIT Officer Dispatched",title="CIT Officer was NOT Requested");
