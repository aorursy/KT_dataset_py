# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import math

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 20)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

ds = pd.read_csv('../input/airlinefleet/Fleet Data.csv')

ds.head()
print(ds.shape)

print(ds["Current"].isna().sum())



def clean(val):

    val = str(val)

    val = val.replace("$", "").replace(",", "")

    if val != "nan":

        val = int(val)

    return val



ds[["Current", "Future", "Historic", "Total", "Orders"]] = ds[[

    "Current", "Future", "Historic", "Total", "Orders"]].fillna(0)

ds["Unit Cost"] = ds["Unit Cost"].apply(clean)

ds["Total Cost (Current)"] = ds["Total Cost (Current)"].apply(clean)

ds.drop(columns = "Airline", inplace = True)

ds.rename(columns = {"Parent Airline" : "Airline", "Historic" : "Past", "Current" : "Present"}, inplace = True)
ds["Total"] = ds["Total"] - ds["Future"]

ds["Future"] = ds["Future"] + ds["Orders"]

#use floor for a conservative estimate

ds["Future"] = ds["Future"].apply(lambda x: math.floor(x/2))

ds.drop(columns = "Orders", inplace = True)

ds["Total"] = ds["Total"] + ds["Future"]
plane_data = ds.groupby("Aircraft Type", as_index = False)[["Past", "Present", "Future", "Total"]].sum()

plane_data = plane_data.sort_values(by = ["Total"], ascending = False)

plane_data = plane_data.merge(ds[["Aircraft Type", "Unit Cost"]], on = ["Aircraft Type"], how = "left")

plane_data.drop_duplicates(subset = ["Aircraft Type"], inplace = True)

plane_data.head()
ax = plane_data[:30].plot(kind = "bar", x = "Aircraft Type", y = ["Past", "Present", "Future"], 

                          figsize = (14,10), stacked = True)

ax.set_ylabel("Total")

ax.set_title("Most Popular Aircrafts Over Past, Present, and Future")

plt.tight_layout()
past_leaders = plane_data.sort_values(by = "Past", ascending = False)

present_leaders = plane_data.sort_values(by = "Present", ascending = False)

future_leaders = plane_data.sort_values(by = "Future", ascending = False)

fig, axes = plt.subplots(3, 1, figsize = (10,12))

ax1 = plt.subplot(311)

past_leaders[:15].plot(kind = "barh", x = "Aircraft Type", y = "Past", legend = False, ax = ax1, color = "b")

ax1.set_title("Past Leaders")

ax2 = plt.subplot(312)

present_leaders[:15].plot(kind = "barh", x = "Aircraft Type", y = "Present", legend = False, ax = ax2, color = "r")

ax2.set_title("Present Leaders")

ax3 = plt.subplot(313)

future_leaders[:15].plot(kind = "barh", x = "Aircraft Type", y = "Future", legend = False, ax = ax3, color = "g")

ax3.set_title("Future Leaders")

plt.tight_layout()
f, axes = plt.subplots(2,2, figsize = (16,12))

ax1 = plt.subplot(221)

past_leaders[:50].plot(kind = "scatter", x = "Present", y = "Past", ax = ax1)

past_leaders[["Present","Past",'Aircraft Type']][:50].apply(lambda x: ax1.text(*x),axis=1)

ax1.set_title("Past vs Present Aircraft Leaders")



ax2 = plt.subplot(222)

past_leaders[4:20].plot(kind = "scatter", x = "Present", y = "Past", ax = ax2)

past_leaders[["Present","Past",'Aircraft Type']][4:20].apply(lambda x: ax2.text(*x),axis=1)

ax2.set_title("Past vs Present Aircraft Leaders Excluding Outliers")



ax3 = plt.subplot(223)

future_leaders[:50].plot(kind = "scatter", x = "Present", y = "Future", ax = ax3)

future_leaders[["Present","Future",'Aircraft Type']][:50].apply(lambda x: ax3.text(*x),axis=1)

ax3.set_title("Present vs Future Aircraft Type Leaders")



ax4 = plt.subplot(224)

future_leaders[3:14].plot(kind = "scatter", x = "Present", y = "Future", ax = ax4)

future_leaders[["Present","Future",'Aircraft Type']][3:14].apply(lambda x: ax4.text(*x),axis=1)

ax4.set_title("Present vs Future Aircraft Type Leaders Excluding Outliers")
fig, axes = plt.subplots(1, 3, figsize = (17,9))

ax1 = plt.subplot(131)

past_leaders[:15].plot(kind = "bar", x = "Aircraft Type", y = "Unit Cost", legend = False, ax = ax1, color = "royalblue")

ax1.set_title("Unit Cost of Past Leaders ")

ax1.set_yticks(np.arange(0, 500, 50))

ax2 = plt.subplot(132)

present_leaders[:15].plot(kind = "bar", x = "Aircraft Type", y = "Unit Cost", legend = False, ax = ax2, color = "coral")

ax2.set_title("Unit Cost of Present Leaders")

ax2.set_yticks(np.arange(0, 500, 50))

ax3 = plt.subplot(133)

future_leaders[:15].plot(kind = "bar", x = "Aircraft Type", y = "Unit Cost", legend = False, ax = ax3, color = "lime")

ax3.set_title("Unit Cost of Future Leaders")

ax3.set_yticks(np.arange(0, 500, 50))

plt.tight_layout()
d = {"Past" : past_leaders["Unit Cost"][:50].values, "Present":present_leaders["Unit Cost"][:50].values,

     "Future":future_leaders["Unit Cost"][:50].values}

box_df = pd.DataFrame(data=d)

box_df['Past'] = box_df['Past'].astype(float)

box_df['Present'] = box_df['Present'].astype(float)

box_df['Future'] = box_df['Future'].astype(float)

f, ax = plt.subplots(1,2, figsize = (14,6))

ax1 = plt.subplot(121)

box_df.boxplot(ax = ax1)

ax1.set_title("Unit Cost of Top 50 Aircrafts Overtime")

ax1.set_ylabel("Unit Cost")



d2 = {"Past" : past_leaders["Unit Cost"][:15].values, "Present":present_leaders["Unit Cost"][:15].values,

     "Future":future_leaders["Unit Cost"][:15].values}

box_df2 = pd.DataFrame(data=d2)

box_df2['Past'] = box_df2['Past'].astype(float)

box_df2['Present'] = box_df2['Present'].astype(float)

box_df2['Future'] = box_df2['Future'].astype(float)

ax2 = plt.subplot(122)

box_df2.boxplot(ax = ax2)

ax2.set_title("Unit Cost of Top 15 Aircrafts Overtime")

ax2.set_ylabel("Unit Cost")

plt.tight_layout()
plane_merged = plane_data.merge(ds[["Aircraft Type", "Average Age"]], on = ["Aircraft Type"], how = "left")

plane_merged.drop_duplicates(subset = ["Aircraft Type"], inplace = True)

plane_merged.dropna(subset = ["Average Age"], inplace = True)

plane_merged.sort_values(by = ["Present"], ascending = False)

plane_merged["Unit Cost"] = plane_merged["Unit Cost"].astype(float)

plane_merged["Average Age"] = plane_merged["Average Age"].astype(float)
f, axes = plt.subplots(1,2,figsize = (14,6))

ax1 = plt.subplot(121)

plane_merged[:50].plot(kind = "scatter", x = "Unit Cost", y = "Average Age", ax = ax1)

plane_merged[["Unit Cost","Average Age",'Aircraft Type']][:50].apply(lambda x: ax1.text(*x),axis=1)



ax2 = plt.subplot(122)

plane_merged[4:20].plot(kind = "scatter", x = "Unit Cost", y = "Average Age", ax=ax2)

plane_merged[["Unit Cost","Average Age",'Aircraft Type']][4:20].apply(lambda x: ax2.text(*x),axis=1)

plt.tight_layout()
airline_data = ds.groupby(["Airline"], as_index = False)["Past", "Present", "Future", 

                                                         "Total"].sum().sort_values(by = "Total", ascending = False)

airline_data.head()
ax = airline_data[:30].plot(kind = "bar", stacked = True, x = "Airline", y = ["Past", "Present", "Future"], figsize = (13,7))

ax.set_ylabel("Total Fleet Count")

plt.tight_layout()
past_leaders = airline_data.sort_values(by = "Past", ascending = False)

present_leaders = airline_data.sort_values(by = "Present", ascending = False)

future_leaders = airline_data.sort_values(by = "Future", ascending = False)

fig, axes = plt.subplots(3, 1, figsize = (10,12))

ax1 = plt.subplot(311)

past_leaders[:15].plot(kind = "barh", x = "Airline", y = "Past", legend = False, ax = ax1, color = "blueviolet")

ax1.set_xlabel("Fleet Count")

ax1.set_title("Past Leaders")

ax2 = plt.subplot(312)

present_leaders[:15].plot(kind = "barh", x = "Airline", y = "Present", legend = False, ax = ax2, color = "crimson")

ax2.set_xlabel("Fleet Count")

ax2.set_title("Present Leaders")

ax3 = plt.subplot(313)

future_leaders[:15].plot(kind = "barh", x = "Airline", y = "Future", legend = False, ax = ax3, color = "olive")

ax3.set_xlabel("Fleet Count")

ax3.set_title("Future Leaders")

plt.tight_layout()
f, axes = plt.subplots(2,2, figsize = (16,12))

ax1 = plt.subplot(221)

past_leaders[:50].plot(kind = "scatter", x = "Present", y = "Past", ax = ax1, color = "DarkBlue")

past_leaders[["Present","Past",'Airline']][:50].apply(lambda x: ax1.text(*x),axis=1)

ax1.set_title("Past vs Present Airline Leaders")



ax2 = plt.subplot(222)

sns.regplot(x = past_leaders["Present"][6:25],y = past_leaders["Past"][6:25], ax = ax2)

past_leaders[["Present","Past",'Airline']][6:25].apply(lambda x: ax2.text(*x),axis=1)

ax2.set_title("Past vs Present Airline Leaders Excluding Outliers")



ax3 = plt.subplot(223)

future_leaders[:50].plot(kind = "scatter", x = "Present", y = "Future", ax = ax3)

future_leaders[["Present","Future",'Airline']][:50].apply(lambda x: ax3.text(*x),axis=1)

ax3.set_title("Present vs Future Airline Type Leaders")



ax4 = plt.subplot(224)

sns.regplot(x = future_leaders["Present"][3:17],y = future_leaders["Future"][3:17], ax = ax4)

future_leaders[["Present","Future",'Airline']][3:17].apply(lambda x: ax4.text(*x),axis=1)

ax4.set_title("Present vs Future Airline Type Leaders Excluding Outliers")

plt.tight_layout()
us_cols = ["American Airlines", "Southwest Airlines", "Delta Airlines", "United Airlines"]

euro_cols = ["Lufthansa", "IAG", "Air France/KLM", "Turkish Airlines"]

chinese_cols = ["Air China", "China Eastern Airlines", "China Southern Airlines"]



us_data = ds.loc[ds["Airline"].isin(us_cols)]

euro_data = ds.loc[ds["Airline"].isin(euro_cols)]

chinese_data = ds.loc[ds["Airline"].isin(chinese_cols)]



us_data = us_data.groupby(["Aircraft Type"], as_index = False)["Past", 

                                                               "Present", "Future"].sum().sort_values(by = ["Present"], ascending = False)

euro_data = euro_data.groupby(["Aircraft Type"], as_index = False)["Past", 

                                                               "Present", "Future"].sum().sort_values(by = ["Present"], ascending = False)

chinese_data = chinese_data.groupby(["Aircraft Type"], as_index = False)["Past", 

                                                               "Present", "Future"].sum().sort_values(by = ["Present"], ascending = False)
boeing_ind = us_data["Aircraft Type"].str.contains("Boeing")

boeing = us_data[boeing_ind.values]

#boeing.head()

airbus_ind = us_data["Aircraft Type"].str.contains("Airbus")

airbus = us_data[airbus_ind.values]



canadair_ind = us_data["Aircraft Type"].str.contains("Canadair")

canadair = us_data[canadair_ind.values]



embraer_ind = us_data["Aircraft Type"].str.contains("Embraer")

embraer= us_data[embraer_ind.values]



mcdonnell_ind = us_data["Aircraft Type"].str.contains("McDonnell")

mcdonnell = us_data[mcdonnell_ind.values]

us_pie_df = pd.DataFrame({"Present": [boeing["Present"].sum(), 

                                     airbus["Present"].sum(), canadair["Present"].sum(), embraer["Present"].sum()],

                         "Future": [boeing["Future"].sum(), 

                                     airbus["Future"].sum(), canadair["Future"].sum(), embraer["Future"].sum()]}, 

                        index = ["Boeing", "Airbus", "Canadair", "Embraer"])
boeing_ind = euro_data["Aircraft Type"].str.contains("Boeing")

boeing = euro_data[boeing_ind.values]



airbus_ind = euro_data["Aircraft Type"].str.contains("Airbus")

airbus = euro_data[airbus_ind.values]



canadair_ind = euro_data["Aircraft Type"].str.contains("Canadair")

canadair = euro_data[canadair_ind.values]



embraer_ind = euro_data["Aircraft Type"].str.contains("Embraer")

embraer= euro_data[embraer_ind.values]



mcdonnell_ind = euro_data["Aircraft Type"].str.contains("McDonnell")

mcdonnell = euro_data[mcdonnell_ind.values]

euro_pie_df = pd.DataFrame({"Present": [boeing["Present"].sum(), 

                                     airbus["Present"].sum(), canadair["Present"].sum(), embraer["Present"].sum()],

                         "Future": [boeing["Future"].sum(), 

                                     airbus["Future"].sum(), canadair["Future"].sum(), embraer["Future"].sum()]}, 

                        index = ["Boeing", "Airbus", "Canadair", "Embraer"])
boeing_ind = chinese_data["Aircraft Type"].str.contains("Boeing")

boeing = chinese_data[boeing_ind.values]



airbus_ind = chinese_data["Aircraft Type"].str.contains("Airbus")

airbus = chinese_data[airbus_ind.values]



canadair_ind = chinese_data["Aircraft Type"].str.contains("Canadair")

canadair = chinese_data[canadair_ind.values]



embraer_ind = chinese_data["Aircraft Type"].str.contains("Embraer")

embraer= chinese_data[embraer_ind.values]



mcdonnell_ind = chinese_data["Aircraft Type"].str.contains("McDonnell")

mcdonnell = chinese_data[mcdonnell_ind.values]

chinese_pie_df = pd.DataFrame({"Present": [boeing["Present"].sum(), 

                                     airbus["Present"].sum(), canadair["Present"].sum(), embraer["Present"].sum()],

                         "Future": [boeing["Future"].sum(), 

                                     airbus["Future"].sum(), canadair["Future"].sum(), embraer["Future"].sum()]}, 

                        index = ["Boeing", "Airbus", "Canadair", "Embraer"])
f, axes = plt.subplots(3, 2, figsize = (14,12))



ax1 = plt.subplot(321)

us_pie_df.plot(kind = "pie", y = "Present", legend = False, autopct="%1.1f%%", ax = ax1)

ax1.set_ylabel("US Present")



ax2 = plt.subplot(322)

us_pie_df.plot(kind = "pie", y = "Future", legend = False, autopct="%1.1f%%", ax = ax2)

ax2.set_ylabel("US Future")



ax3 = plt.subplot(323)

euro_pie_df.plot(kind = "pie", y = "Present", legend = False, autopct="%1.1f%%", ax = ax3)

ax3.set_ylabel("Euro Present")



ax4 = plt.subplot(324)

euro_pie_df.plot(kind = "pie", y = "Future", legend = False, autopct="%1.1f%%", ax = ax4)

ax4.set_ylabel("Euro Future")



ax5 = plt.subplot(325)

chinese_pie_df.plot(kind = "pie", y = "Present", legend = False, autopct="%1.1f%%", ax = ax5)

ax5.set_ylabel("Chinese Present")



ax6 = plt.subplot(326)

chinese_pie_df.plot(kind = "pie", y = "Future", legend = False, autopct="%1.1f%%", ax = ax6)

ax5.set_ylabel("Chinese Future")

plt.tight_layout()