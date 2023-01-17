import numpy as np        # linear algebra

import pandas as pd       # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

import matplotlib.dates as mdates
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")



# fill NaN

df = df.fillna({"Province/State": np.nan})



# Change astype

df["Date"] = pd.to_datetime(df["ObservationDate"])

df["Confirmed"] = df["Confirmed"].fillna(0).astype('int')

df["Deaths"] = df["Deaths"].fillna(0).astype('int')

df["Recovered"] =df["Recovered"].fillna(0).astype('int')



# Replace duplicates

# **Temporary measures**

# At this time, the data frames are based on daily reports,

# so we manually replaced the temporarily duplicated countries or regions.

# I think it's easier to use time series data, but we are still combining 

# daily reports at this time.

DUPRECATE_COUNTRIES = {

    'Ireland': 'Republic of Ireland',

    'Palestine': 'West Bank and Gaza',

    'occupied Palestinian territory': 'West Bank and Gaza',

    'Vietnam': 'Viet Nam',

    'South Korea': 'Korea, South',

    'Russia': 'Russian Federation',

    'Iran': 'Iran (Islamic Republic of)',

    'Moldova': 'Republic of Moldova',

    'Taiwan': 'Taiwan*',

    'Vatican City': 'Holy See',

    'UK': 'United Kingdom',

    'Ivory Coast': "Cote d'Ivoire",

    'Czech Republic': 'Czechia',

    'Taipei and environs': 'Taiwan*',

    'Republic of Korea': 'Korea, South',

    'Others': 'Diamond Princess',

    'Mainland China': 'China',

    'Macao SAR': 'China',

    'Hong Kong SAR': 'China',

    'Hong Kong': 'China',

    'Macau': 'China',

    'Congo (Brazzaville)': 'Republic of the Congo',

    'The Gambia': 'Gambia, The',

    'The Bahamas': 'Bahamas, The',

    'Cabo Verde': 'Cape Verde',

    'East Timor': 'Timor-Leste',

    'Gambia, The': 'Bahamas',

    'Bahamas, The': 'Gambia',

    'Cruise Ship': 'Diamond Princess',

    'French Guiana': 'France',

    'Martinique': 'France',

    'Reunion': 'France',

    'Guadeloupe':'France'

}



DUPRECATE_STATES = {

    'Macau':'Macao SAR',

    'Hong Kong': 'Hong Kong SAR',

    'Diamond Princess cruise ship' : np.nan,

    'Cruise Ship': np.nan,

    'Diamond Princess': np.nan

}





df["Country/Region"] = df["Country/Region"].replace(DUPRECATE_COUNTRIES)

df["Province/State"] = df["Province/State"].replace(DUPRECATE_STATES)



# Drop sno and Last Update

df = df.drop(columns=["SNo","Last Update", "ObservationDate"])



df["Active"] = df["Confirmed"] - df["Deaths"] - df["Recovered"]

df.info()
print(f"LAST UPDATE: {df.Date.max().strftime('%y-%m-%d')}")
unique_date = np.sort(df["Date"].unique())

last_confirmed = df[df["Date"] == unique_date[-1]].groupby(["Country/Region"]).sum()["Active"]

one_week_ago = df[df["Date"] == unique_date[-(1 + 7)]].groupby(["Country/Region"]).sum()["Active"]

new_active = last_confirmed.sub(one_week_ago, fill_value=0).astype(int)

print(f"In {len(new_active)} countries, new active are increasing in {len(new_active[new_active>0])} countries"

      f" and decreasing in {len(new_active[new_active<0])} countries.")
# active_p = new_active[new_active>0].sort_values(ascending=False).to_frame("New active last week")

# active_p.style.background_gradient(cmap='OrRd')
# active_m = new_active[new_active<0].sort_values().to_frame("New active last week")

# active_m.style.background_gradient(cmap='YlGn_r')
na_sorted = new_active.sort_values()

ax = na_sorted.plot.barh(title="New active last week", figsize=(10, 50), color="k", xlim=(-(na_sorted.max() + 100000), na_sorted.max() + 100000))

for i, v in enumerate(na_sorted):

    if v > 10000:

        ax.text(v + 10000, i, str(v), color='r', va='center', ha='left', fontweight='bold')

    elif v < -100:

        ax.text(v + -10000, i, str(v), color='g', va='center', ha='right', fontweight='bold')

    elif v < 0:

        ax.text(v - 10000, i, str(v), color='k', va='center', ha='right', fontweight='bold')

    else:

        ax.text(v + 10000, i, str(v), color='k', va='center', ha='left', fontweight='bold')

        

plt.savefig("New_active_last_week")

plt.show()

plt.close('all')
gdf = df.groupby(["Date", "Country/Region"]).sum()

values = gdf.columns

gdf = gdf.reset_index()



sub = []

for c in gdf["Country/Region"].unique():

    cgdf = gdf.loc[gdf["Country/Region"] == c]

    cdiff = cgdf.iloc[:, 2:].diff().fillna(0)

    cs = pd.concat([cgdf.iloc[:, :2], cdiff], axis=1)

    for co in values:

        p = cs.loc[cs[co].idxmax(), ["Date", co]].rename(index={co:"Maximum"})

        p["Elapsed_Days"] = (df.Date.max() - p.Date).days

        p["Latest"] = cs[cs["Date"] == cs["Date"].max()][co].values[0]

        p["Column_Name"] = f"New {co}"

        p["Country/Region"] = c

        sub.append(p)



peak = pd.DataFrame(sub)
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111)



for name in peak["Column_Name"].unique():

    s = peak[peak["Column_Name"] == name]["Elapsed_Days"].rename(f"{name}")

    sns.kdeplot(s, ax=ax, clip=(0, s.max()))



ax.set_xlim(left=0)

ax.set_xlabel("Elapsed Days")

plt.title("Peak-out Distribution")

plt.savefig("Peak_out.png")

plt.show()

plt.close('all')
threshold_of_zero_days = 10 # Confirmed > 10



gdf = df.groupby(["Date", "Country/Region"]).sum()

gdf = gdf[gdf["Confirmed"] > threshold_of_zero_days]

gdf = gdf.reset_index()



sub = []

for c in gdf["Country/Region"].unique():

    cgdf = gdf.loc[gdf["Country/Region"] == c].copy()

    cdiff = cgdf.iloc[:, 2:].diff().fillna(0).add_prefix('New ')

    cs = pd.concat([cgdf.iloc[:, :2], cdiff], axis=1)

    cs["Days"] = (cs["Date"] - cs["Date"].min()).dt.days

    sub.append(cs)



normal = pd.concat(sub, axis=0)

normal = normal.melt(id_vars=["Date","Country/Region", "Days"],

                     var_name="Column_Name")

normal
plt.figure(figsize=(10, 8))

ax = sns.lineplot(x="Days", y="value", hue="Column_Name",

                  hue_order=["New Confirmed", "New Deaths"],

                  data=normal)

ax.set_yscale('log')

plt.title(f"Mean trajectory (after {threshold_of_zero_days} confirmed cases)")

plt.savefig(f"Nean_trajectory.png")

plt.show()

plt.close('all')
# plt.figure(figsize=(8, 40)) 

# normal.groupby("Country/Region").last()["Days"].sort_values().plot.barh(grid=True)

# plt.title(f"List of current elapsed days after {threshold_of_zero_days} confirmed cases")

# plt.savefig("Current_elapsed_days_after_{threshold_of_zero_days}_confirmed.png")

# plt.show()

# plt.close('all')
threshold = 1.

lower_bound = 20



def group_country_sum(date):

    return df[df["Date"] == date].groupby(["Country/Region"]).sum()



unique_date = np.sort(df["Date"].unique())

last_confirmed = group_country_sum(unique_date[-1])["Confirmed"]

one_week_ago = group_country_sum(unique_date[-(7 + 1)])["Confirmed"]

two_weeks_ago = group_country_sum(unique_date[-(14 + 1)])["Confirmed"]



diff_one = last_confirmed.sub(one_week_ago, fill_value=0).astype(int)

diff_two = one_week_ago.sub(two_weeks_ago, fill_value=0).astype(int)

diff_one = diff_one[diff_one > lower_bound]

diff_two = diff_two[diff_two > 1]

diff = pd.concat([diff_one, diff_two], axis=1, sort=False,

                 keys=("New confirmed last week (A)", "New confirmed two weeks ago (B)"))

diff["Growth rate (A/B)"] = (diff["New confirmed last week (A)"] /

                             diff["New confirmed two weeks ago (B)"]).round(1)

diff["Predict next week (A/B)*A"] = ((diff["New confirmed last week (A)"] / diff["New confirmed two weeks ago (B)"]) *

                              diff["New confirmed last week (A)"]).round(0)

diff = diff[diff["Growth rate (A/B)"] > threshold].sort_values("Growth rate (A/B)", ascending=False)

# print(f"The surge has been observed in {len(diff)} countries")

diff.style.background_gradient(cmap="BuPu")
crisis = diff[(diff["New confirmed two weeks ago (B)"] > 200) & (diff["Predict next week (A/B)*A"] > 2000)]

crisis.style.background_gradient(cmap="BuPu")
# configuration dict

# display name, days

time_periods = {

    "-1d": 1,

    "-3d": 3,

    "-1w": 7,

    "-2w": 14,

    "-4w": 28

}



unique_date = np.sort(df["Date"].unique())



def hotspots(col, n=25, cmap="coolwarm"):

    last = df[df["Date"] == unique_date[-1]].groupby(["Country/Region"]).sum()[col]



    past = []

    for n_day in time_periods.values():

        n_days_ago = df[df["Date"] == unique_date[-(n_day + 1)]].groupby(["Country/Region"]).sum()[col]

        # diff

        diff = last.sub(n_days_ago, fill_value=0).astype(int)

        rank = diff.rank(method='min', ascending=False).astype(int)

        past.append(pd.concat([rank, diff], axis=1, keys=["Rank", "New"]))



    keys = time_periods.keys()

    hot = pd.concat(past, axis=1, keys=keys, sort=False).sort_values([("-1d","Rank")])

    return hot.head(n).style.background_gradient(cmap=cmap,

                                           subset=[(k, "New") for k in keys])



def history(col, n=25, cmap='coolwarm', figsize=(12, 12)):

    # top

    last = df[df["Date"] == unique_date[-1]].groupby(["Country/Region"]).sum()[col]

    top = last.sort_values(ascending=False).head(n).index

    # sum

    s = df[df["Country/Region"].isin(top)].groupby(["Date","Country/Region"]).sum()[col]

    s = s.unstack()[top]

    s = s.T.replace(0, 0.1).fillna(0.1)

    

    # change size

    plt.figure(figsize=figsize) 

    return sns.heatmap(s, norm=LogNorm(vmin=s.min().min(), vmax=s.max().max()),

                       cmap=cmap, cbar=False,

                       xticklabels=s.columns.strftime('%Y-%m-%d'))
hotspots("Confirmed", cmap="OrRd")
# history("Confirmed", cmap="OrRd")

# plt.title("Heat history (number of confirmed) Top 25")

# plt.savefig("Confirmed_top25_history.png")

# plt.show()

# plt.close('all')
hotspots("Deaths", cmap="RdPu")
# history("Deaths", cmap="RdPu")

# plt.title("Heat history (number of deaths) Top 25")

# plt.savefig("Deaths_top25_history.png")

# plt.show()

# plt.close('all')
hotspots("Recovered", cmap="YlGn")
# history("Recovered", cmap="YlGn")

# plt.title("Heat history (number of recovered) Top 25")

# plt.savefig("Recovered_top25_history.png")

# plt.show()

# plt.close('all')
hotspots("Active", cmap="RdYlGn_r")
# history("Active", cmap="RdYlGn_r")

# plt.title("Heat history (number of active) Top 25")

# plt.savefig("Active_top25_history.png")

# plt.show()

# plt.close('all')
group_all = df.groupby(["Date"]).sum()

diff = group_all.diff()

world_situation = pd.concat([group_all, diff],

                             axis=1, keys=["Count", "Diff"])

world_situation.plot(title="World situation", subplots=True, layout=(2, 4), figsize=(20, 10))

plt.savefig("World.png")

plt.show()

plt.close("all")
def world_new_case_plot(column):

    group_all = df.groupby(["Date"]).sum()[column]

    diff = group_all.diff()

    world_situation = pd.concat([diff, diff.rolling(5).mean()],

                                 axis=1, keys=["Diff", "Diff_Average_5"])

    return world_situation.plot(title=f"New {column} case daily growth",

                                figsize=(12, 8))



for co in ["Confirmed", "Deaths", "Recovered", "Active"]:

    world_new_case_plot(co)

    plt.savefig(f"{co}.png")

    plt.show()

    plt.close('all')
# Sort by Date

df = df.sort_values(["Date"]).reset_index(drop=True)



group_country = df.groupby(["Country/Region", "Date"]).sum()

group_country
gdf = group_country.copy().reset_index()

gdf["CFR(%)"] = gdf["Deaths"] * 100 / gdf["Confirmed"]



plt.figure(figsize=(12, 8)) 

ax = sns.lineplot(x="Date", y="CFR(%)", data=gdf)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

plt.title("Case Fatality Rate(CFR)")

plt.savefig("CFR.png")

plt.show()

plt.close('all')
# for name in ["Confirmed", "Deaths", "Recovered", "Active"]:

#     # select

#     gdf = group_country[name].unstack(level=0)

#     # sort

#     gdf = gdf.sort_values(gdf.index.max(), axis=1, ascending=False)

#     # plot

#     ax = gdf.plot(title=name, logy=True, colormap="jet_r", figsize=(30, 20))

#     # legend

#     ax.legend(loc='upper left', ncol=4)

#     plt.savefig(f"{name}_log.png")

#     plt.show()

#     plt.close('all') 
# %mkdir countries

# unique_country = group_country.index.unique(level=0)

# for c in unique_country:

#     count = group_country.loc[c, :]

#     diff = count.diff()

#     country_specific = pd.concat([count, diff],

#                                  axis=1, keys=["Count","Diff"])

#     if len(diff) > 1:

#         country_specific.plot(title=c, subplots=True,

#                               layout=(2, 4), sharex=True, figsize=(20, 9))

#         fname = str(c).replace(' ', '-').strip(",")

#         plt.savefig(f"./countries/{fname}.png")

#         plt.show()

#         plt.close('all')
# last_update = df['Date'].max().strftime('%Y-%m-%d')

# !tar -zcvf "output_{last_update}.tar.gz" *.png countries/*.png

# !rm -fd *.png countries/