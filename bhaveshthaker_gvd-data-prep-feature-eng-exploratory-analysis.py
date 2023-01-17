import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
mpl.style.use("seaborn")
plt.style.use("seaborn")

btui = [
    "#b2182b", "#d6604d", "#f4a582", "#92c5de", "#4393c3", "#2166ac", "#762a83",
    "#9970ab", "#c2a5cf", "#a6dba0", "#5aae61", "#1b7837", "#c51b7d", "#de77ae",
    "#f1b6da", "#8c510a", "#bf812d", "#dfc27d", "#80cdc1", "#35978f", "#01665e",
    ]
import random
btui_reversed = btui[::-1]
btui_shuffled=random.sample(btui, len(btui))

sns.set(context="notebook", style="darkgrid", font="monospace", font_scale=1.5, palette=btui)
sns.color_palette(btui)
sns.set_palette(btui)
sns.set(rc={"figure.figsize": (14, 10)})
dataset_gunviolence = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
dataset_gunviolence.head()
dataset_gunviolence.shape
dataset_gunviolence.describe()
dataset_gunviolence.info()
dataset_gunviolence.columns.values
# After executions found that isnull and isna gives same counts/values
#dataset_gunviolence.isnull().values.any()
#dataset_gunviolence.isnull().sum()
#dataset_gunviolence.isna().values.any()

missing_data_sum=dataset_gunviolence.isna().sum()
missing_data_count=dataset_gunviolence.isna().count()
percentage_missing_data=(missing_data_sum/missing_data_count) * 100
missing_data = pd.concat([missing_data_sum, percentage_missing_data], axis=1)
missing_data
del(missing_data_sum, missing_data_count, percentage_missing_data)
# https://stackoverflow.com/questions/21925114/is-there-an-implementation-of-missingmaps-in-pythons-ecosystem
from matplotlib import collections as collections
from matplotlib.patches import Rectangle
from itertools import cycle

def missmap(df, ax=None, colors=None, aspect=4, sort='descending',
            title=None, **kwargs):
    """
    Plot the missing values of df.

    Parameters
    ----------
    df : pandas DataFrame
    ax : matplotlib axes
        if None then a new figure and axes will be created
    colors : dict
        dict with {True: c1, False: c2} where the values are
        matplotlib colors.
    aspect : int
        the width to height ratio for each rectangle.
    sort : one of {'descending', 'ascending', None}
    title : str
    kwargs : dict
        matplotlib.axes.bar kwargs

    Returns
    -------
    ax : matplotlib axes

    """

    if ax is None:
        (fig, ax) = plt.subplots()

    # setup the axes

    dfn = pd.isnull(df)

    if sort in ('ascending', 'descending'):
        counts = dfn.sum()
        sort_dict = {'ascending': True, 'descending': False}
        counts = counts.sort_values(ascending=sort_dict[sort])
        dfn = dfn[counts.index]

    # Up to here

    ny = len(df)
    nx = len(df.columns)

    # each column is a stacked bar made up of ny patches.

    xgrid = np.tile(np.arange(nx), (ny, 1)).T
    ygrid = np.tile(np.arange(ny), (nx, 1))

    # xys is the lower left corner of each patch

    xys = (zip(x, y) for (x, y) in zip(xgrid, ygrid))

    if colors is None:
        colors = {True: '#EAF205', False: 'k'}

    widths = cycle([aspect])
    heights = cycle([1])

    for (xy, width, height, col) in zip(xys, widths, heights,
            dfn.columns):
        color_array = dfn[col].map(colors)

        rects = [Rectangle(xyc, width, height, **kwargs) for (xyc,
                 c) in zip(xy, color_array)]

        p_coll = collections.PatchCollection(rects, color=color_array,
                edgecolor=color_array, **kwargs)
        ax.add_collection(p_coll, autolim=False)

    # post plot aesthetics

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)

    ax.set_xticks(.5 + np.arange(nx))  # center the ticks
    ax.set_xticklabels(dfn.columns)
    
    for t in ax.get_xticklabels():
        t.set_rotation(90)

    # remove tick lines

    ax.tick_params(axis='both', which='both', bottom='off', left='off',
                   labelleft='off')
    ax.grid(False)

    if title:
        ax.set_title(title)
    return ax


colours = {True: "#fde725", False: "#440154"}
ax = missmap(dataset_gunviolence, colors = colours)
plt.show(ax)
dataset_gunviolence.drop([
    "incident_id",
    "incident_url",
    "sources",
    "source_url",
    "incident_url_fields_missing",
    "location_description",
    "participant_relationship",
    ], axis=1, inplace=True)
dataset_gunviolence["date"] = pd.to_datetime(dataset_gunviolence["date"])

dataset_gunviolence["day"] = dataset_gunviolence["date"].dt.day
dataset_gunviolence["month"] = dataset_gunviolence["date"].dt.month
dataset_gunviolence["year"] = dataset_gunviolence["date"].dt.year
dataset_gunviolence["weekday"] = dataset_gunviolence["date"].dt.weekday
dataset_gunviolence["week"] = dataset_gunviolence["date"].dt.week
dataset_gunviolence["quarter"] = dataset_gunviolence["date"].dt.quarter
dataset_gunviolence["participant_gender"] = dataset_gunviolence["participant_gender"].fillna("0::Unknown")


def clean_participant_gender(row) :
    gender_row_values = []
    gender_row = str(row).split("||")
    for x in gender_row :
        gender_row_value = str(x).split("::")
        if len(gender_row_value) > 1 :
            gender_row_values.append(gender_row_value[1])
            
    return gender_row_values


participant_genders = dataset_gunviolence.participant_gender.apply(clean_participant_gender)
dataset_gunviolence["participant_gender_total"] = participant_genders.apply(lambda x: len(x))
dataset_gunviolence["participant_gender_male"] = participant_genders.apply(lambda x: x.count("Male"))
dataset_gunviolence["participant_gender_female"] = participant_genders.apply(lambda x: x.count("Female"))
dataset_gunviolence["participant_gender_unknown"] = participant_genders.apply(lambda x: x.count("Unknown"))
del(participant_genders)
dataset_gunviolence["n_guns_involved"] = dataset_gunviolence["n_guns_involved"].fillna(0)
dataset_gunviolence["gun_stolen"] = dataset_gunviolence["gun_stolen"].fillna("0::Unknown")
# Prints a lot but gives all the unique values of a column
#dataset_gunviolence["gun_stolen"].unique()

def clean_gun_stolen(row) :
    unknownCount = 0
    stolenCount = 0
    notstolenCount = 0
    gunstolen_row_values = []
    
    gunstolen_row = str(row).split("||")
    for x in gunstolen_row :
            gunstolen_row_value = str(x).split("::")
            if len(gunstolen_row_value) > 1 :
                gunstolen_row_values.append(gunstolen_row_value[1])
                if "Stolen" in gunstolen_row_value :
                    stolenCount += 1
                elif "Not-stolen" in gunstolen_row_value :
                    notstolenCount += 1
                else :
                    unknownCount += 1
                    
    return gunstolen_row_values


gunstolenvalues = dataset_gunviolence.gun_stolen.apply(clean_gun_stolen)
dataset_gunviolence["gun_stolen_stolen"] = gunstolenvalues.apply(lambda x: x.count("Stolen"))
dataset_gunviolence["gun_stolen_notstolen"] = gunstolenvalues.apply(lambda x: x.count("Not-stolen"))
del(gunstolenvalues)
#dataset_gunviolence.participant_age_group.unique()
dataset_gunviolence["participant_age_group"] = dataset_gunviolence["participant_age_group"].fillna("0::Unknown")

def clean_participant_age_group(row) :
    unknownCount = 0
    childCount = 0
    teenCount = 0
    adultCount = 0
    agegroup_row_values = []
    
    agegroup_row = str(row).split("||")
    for x in agegroup_row :
        agegroup_row_value = str(x).split("::")
        if len(agegroup_row_value) > 1 :
            agegroup_row_values.append(agegroup_row_value[1])
            if "Child 0-11" in agegroup_row_value :
                childCount += 1
            elif "Teen 12-17" in agegroup_row_value :
                teenCount += 1
            elif "Adult 18+" in agegroup_row_value :
                adultCount += 1
            else :
                unknownCount += 1
                
    return agegroup_row_values

agegroupvalues = dataset_gunviolence.participant_age_group.apply(clean_participant_age_group)
dataset_gunviolence["agegroup_child"] = agegroupvalues.apply(lambda x: x.count("Child 0-11"))
dataset_gunviolence["agegroup_teen"] = agegroupvalues.apply(lambda x: x.count("Teen 12-17"))
dataset_gunviolence["agegroup_adult"] = agegroupvalues.apply(lambda x: x.count("Adult 18+"))
del(agegroupvalues)
sns.heatmap(dataset_gunviolence.corr(), cmap=btui, annot=True, fmt=".2f")
statewise_crime_rate = dataset_gunviolence["state"].value_counts()
statewise_crime_rate
plt.pie(statewise_crime_rate, labels=statewise_crime_rate.index, colors=btui, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("State-wise Gun Violence Percentage")
plt.axis("equal")
statewise_crime_rate = sns.countplot(x=dataset_gunviolence["state"], data=dataset_gunviolence, palette=btui, order=dataset_gunviolence["state"].value_counts().index)
statewise_crime_rate.set_xticklabels(statewise_crime_rate.get_xticklabels(), rotation=90)
statewise_crime_rate.set_title("State(s) with highest number of Gun Violence")
topcitywise_crime_rate = dataset_gunviolence["city_or_county"].value_counts().head(50)
plt.pie(topcitywise_crime_rate, labels=topcitywise_crime_rate.index, colors=btui, autopct="%1.1f%%", shadow=True, startangle=195)
plt.title("City-wise Gun Violence Percentage")
plt.axis("equal")
#Following line creates a Map but unreadable as there are lots of distint cities involved
#citywise_crime_rate = sns.countplot(x=dataset_gunviolence["city_or_county"], data=dataset_gunviolence, palette=btui, order=dataset_gunviolence["city_or_county"].value_counts().index)

topcitywise_crime_rate = dataset_gunviolence["city_or_county"].value_counts().head(50)
topcitywise_crime_rate_plot = sns.barplot(x=topcitywise_crime_rate.index, y=topcitywise_crime_rate, palette=btui)
topcitywise_crime_rate_plot.set_xticklabels(topcitywise_crime_rate_plot.get_xticklabels(), rotation=75)
topcitywise_crime_rate_plot.set_title("Cities or Counties with highest number of Gun Violence")
sns.jointplot(x=dataset_gunviolence["n_guns_involved"], y=dataset_gunviolence["gun_stolen_stolen"], kind="scatter", color="#D81B60")
sns.jointplot(x=dataset_gunviolence["n_guns_involved"], y=dataset_gunviolence["gun_stolen_notstolen"], kind="scatter", color="#1E88E5")
sns.jointplot(x=dataset_gunviolence.participant_gender_total, y=dataset_gunviolence.n_killed, data=dataset_gunviolence, space=0, dropna=True, color="#D81B60")
sns.jointplot(x=dataset_gunviolence.participant_gender_total, y=dataset_gunviolence.n_injured, data=dataset_gunviolence, space=0, dropna=True, color="#1E88E5")
sns.jointplot(x=dataset_gunviolence.n_guns_involved, y=dataset_gunviolence.n_killed, data=dataset_gunviolence, space=0, dropna=True, color="#D81B60")
# https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
for tmpYear in range(2013,2019) :
        yeargunviolence = dataset_gunviolence[dataset_gunviolence["year"] == tmpYear].groupby([dataset_gunviolence["date"], dataset_gunviolence["week"]]).agg({"n_killed" : "sum", "n_injured" : "sum", "participant_gender_total" : "sum"})
        yeargunviolence.index = yeargunviolence.index.droplevel(1)
        fig, axs = plt.subplots()
        sns.pointplot(x=yeargunviolence.index, y=yeargunviolence.n_killed, data=yeargunviolence, ax=axs, color="#D81B60", scale=0.5, markers="x", label="n_killed")
        sns.pointplot(x=yeargunviolence.index, y=yeargunviolence.n_injured, data=yeargunviolence, ax=axs, color="#FFC107", scale=0.5, markers="X", label="n_injured")
        sns.pointplot(x=yeargunviolence.index, y=yeargunviolence.participant_gender_total, data=yeargunviolence, ax=axs, color="#1E88E5", scale=0.5, label="n_people")
        axs.set_title("Year {0} Gun Violence incidents in USA".format(tmpYear))
        plt.gcf().autofmt_xdate()
        plt.show()

fig, axs = plt.subplots(ncols=2)

n_crimes_by_year_k = dataset_gunviolence.groupby(dataset_gunviolence["year"]).apply(lambda x: pd.Series(dict(total_killed_per_year = x.n_killed.sum())))
n_crimes_by_year_plot_k = sns.barplot(x=n_crimes_by_year_k.index, y=n_crimes_by_year_k.total_killed_per_year, palette=btui, ax=axs[0])
n_crimes_by_year_plot_k.set_title("Killed each year")
del(n_crimes_by_year_k)

n_crimes_by_year_i = dataset_gunviolence.groupby(dataset_gunviolence["year"]).apply(lambda x: pd.Series(dict(total_injured_per_year = x.n_injured.sum())))
n_crimes_by_year_plot_i = sns.barplot(x=n_crimes_by_year_i.index, y=n_crimes_by_year_i.total_injured_per_year, palette=btui, ax=axs[1])
n_crimes_by_year_plot_i.set_title("Injured each year")
del(n_crimes_by_year_i)
fig, axs = plt.subplots(ncols=2)

n_crimes_by_quarter_k = dataset_gunviolence.groupby(dataset_gunviolence["quarter"]).apply(lambda x: pd.Series(dict(total_killed_per_quarter = x.n_killed.sum())))
n_crimes_by_quarter_plot_k = sns.barplot(x=n_crimes_by_quarter_k.index, y=n_crimes_by_quarter_k.total_killed_per_quarter, palette=btui, ax=axs[0])
n_crimes_by_quarter_plot_k.set_title("Killed per quarter")
del(n_crimes_by_quarter_k)

n_crimes_by_quarter_i = dataset_gunviolence.groupby(dataset_gunviolence["quarter"]).apply(lambda x: pd.Series(dict(total_injured_per_quarter = x.n_injured.sum())))
n_crimes_by_quarter_plot_i = sns.barplot(x=n_crimes_by_quarter_i.index, y=n_crimes_by_quarter_i.total_injured_per_quarter, palette=btui, ax=axs[1])
n_crimes_by_quarter_plot_i.set_title("Injured per quarter")
del(n_crimes_by_quarter_i)
fig, axs = plt.subplots(nrows=2)
plt.subplots_adjust(hspace = 0.3)

n_crimes_by_yearquarter_k = dataset_gunviolence.groupby([dataset_gunviolence["year"], dataset_gunviolence["quarter"]]).apply(lambda x: pd.Series(dict(total_killed_per_yearquarter = x.n_killed.sum())))
n_crimes_by_yearquarter_plot_k = sns.barplot(x=n_crimes_by_yearquarter_k.index, y=n_crimes_by_yearquarter_k.total_killed_per_yearquarter, palette=btui, ax=axs[0])
n_crimes_by_yearquarter_plot_k.set_xticklabels(n_crimes_by_yearquarter_plot_k.get_xticklabels(),rotation=45)
n_crimes_by_yearquarter_plot_k.set_title("Killed per Year-Quarter")

del(n_crimes_by_yearquarter_k)

n_crimes_by_yearquarter_i = dataset_gunviolence.groupby([dataset_gunviolence["year"], dataset_gunviolence["quarter"]]).apply(lambda x: pd.Series(dict(total_injured_per_yearquarter = x.n_injured.sum())))
n_crimes_by_yearquarter_plot_i = sns.barplot(x=n_crimes_by_yearquarter_i.index, y=n_crimes_by_yearquarter_i.total_injured_per_yearquarter, palette=btui, ax=axs[1])
n_crimes_by_yearquarter_plot_i.set_xticklabels(n_crimes_by_yearquarter_plot_i.get_xticklabels(),rotation=45)
n_crimes_by_yearquarter_plot_i.set_title("Injured per Year-Quarter")
del(n_crimes_by_yearquarter_i)
fig, axs = plt.subplots(ncols=2)

n_crimes_by_month_k = dataset_gunviolence.groupby(dataset_gunviolence["month"]).apply(lambda x: pd.Series(dict(total_killed_per_month = x.n_killed.sum())))
n_crimes_by_month_plot_k = sns.barplot(x=n_crimes_by_month_k.index, y=n_crimes_by_month_k.total_killed_per_month, palette=btui, ax=axs[0])
n_crimes_by_month_plot_k.set_title("Killed per Month")
del(n_crimes_by_month_k)

n_crimes_by_month_i = dataset_gunviolence.groupby(dataset_gunviolence["month"]).apply(lambda x: pd.Series(dict(total_injured_per_month = x.n_injured.sum())))
n_crimes_by_month_plot_i = sns.barplot(x=n_crimes_by_month_i.index, y=n_crimes_by_month_i.total_injured_per_month, palette=btui, ax=axs[1])
n_crimes_by_month_plot_i.set_title("Injured per Month")
del(n_crimes_by_month_i)
fig, axs = plt.subplots(nrows=2)
plt.subplots_adjust(hspace = 0.3)

n_crimes_by_yearmonth_k = dataset_gunviolence.groupby([dataset_gunviolence["year"], dataset_gunviolence["month"]]).apply(lambda x: pd.Series(dict(total_killed_per_yearmonth = x.n_killed.sum())))
n_crimes_by_yearmonth_plot_k = sns.barplot(x=n_crimes_by_yearmonth_k.index, y=n_crimes_by_yearmonth_k.total_killed_per_yearmonth, palette=btui, ax=axs[0])
n_crimes_by_yearmonth_plot_k.set_xticklabels(n_crimes_by_yearmonth_plot_k.get_xticklabels(), rotation=45)
n_crimes_by_yearmonth_plot_k.set_title("Killed per Year-Month")
del(n_crimes_by_yearmonth_k)

n_crimes_by_yearmonth_i = dataset_gunviolence.groupby([dataset_gunviolence["year"], dataset_gunviolence["month"]]).apply(lambda x: pd.Series(dict(total_injured_per_yearmonth = x.n_injured.sum())))
n_crimes_by_yearmonth_plot_i = sns.barplot(x=n_crimes_by_yearmonth_i.index, y=n_crimes_by_yearmonth_i.total_injured_per_yearmonth, palette=btui, ax=axs[1])
n_crimes_by_yearmonth_plot_i.set_xticklabels(n_crimes_by_yearmonth_plot_i.get_xticklabels(), rotation=45)
n_crimes_by_yearmonth_plot_i.set_title("Injured per Year-Month")
del(n_crimes_by_yearmonth_i)
fig, axs = plt.subplots(nrows=2)
plt.subplots_adjust(hspace = 0.3)

n_crimes_by_week_k = dataset_gunviolence.groupby(dataset_gunviolence["week"]).apply(lambda x: pd.Series(dict(total_killed_per_week = x.n_killed.sum())))
n_crimes_by_week_plot_k = sns.barplot(x=n_crimes_by_week_k.index, y=n_crimes_by_week_k.total_killed_per_week, palette=btui, ax=axs[0])
n_crimes_by_week_plot_k.set_title("Killed per week over the years")
del(n_crimes_by_week_k)

n_crimes_by_week_i = dataset_gunviolence.groupby(dataset_gunviolence["week"]).apply(lambda x: pd.Series(dict(total_injured_per_week = x.n_injured.sum())))
n_crimes_by_week_plot_i = sns.barplot(x=n_crimes_by_week_i.index, y=n_crimes_by_week_i.total_injured_per_week, palette=btui, ax=axs[1])
n_crimes_by_week_plot_i.set_title("Injured per Week over the years")
del(n_crimes_by_week_i)
fig, axs = plt.subplots(nrows=2)
plt.subplots_adjust(hspace = 0.3)

n_crimes_by_monthweek_k = dataset_gunviolence.groupby([dataset_gunviolence["month"], dataset_gunviolence["week"]]).apply(lambda x: pd.Series(dict(total_killed_per_monthweek = x.n_killed.sum())))
n_crimes_by_monthweek_plot_k = sns.barplot(x=n_crimes_by_monthweek_k.index, y=n_crimes_by_monthweek_k.total_killed_per_monthweek, palette=btui, ax=axs[0])
n_crimes_by_monthweek_plot_k.set_xticklabels(n_crimes_by_monthweek_plot_k.get_xticklabels(), rotation=45)
n_crimes_by_monthweek_plot_k.set_title("Killed per Month-Week")
del(n_crimes_by_monthweek_k)

n_crimes_by_monthweek_i = dataset_gunviolence.groupby([dataset_gunviolence["month"], dataset_gunviolence["week"]]).apply(lambda x: pd.Series(dict(total_injured_per_monthweek = x.n_injured.sum())))
n_crimes_by_monthweek_plot_i = sns.barplot(x=n_crimes_by_monthweek_i.index, y=n_crimes_by_monthweek_i.total_injured_per_monthweek, palette=btui)
n_crimes_by_monthweek_plot_i.set_xticklabels(n_crimes_by_monthweek_plot_i.get_xticklabels(), rotation=45)
n_crimes_by_monthweek_plot_i.set_title("Injured per Month-Week")
del(n_crimes_by_monthweek_i)
n_crimes_by_yearweek = dataset_gunviolence.groupby([dataset_gunviolence["year"], dataset_gunviolence["week"]]).apply(lambda x: pd.Series(dict(total_killed_per_yearweek = x.n_killed.sum())))
n_crimes_by_yearweek_plot = sns.barplot(x=n_crimes_by_yearweek.index, y=n_crimes_by_yearweek.total_killed_per_yearweek, palette=btui)
n_crimes_by_yearweek_plot.set_xticklabels(n_crimes_by_yearweek_plot.get_xticklabels(), rotation=45)
n_crimes_by_yearweek_plot.set_title("Killed per Year-Week")
del(n_crimes_by_yearweek)
n_crimes_by_yearweek = dataset_gunviolence.groupby([dataset_gunviolence["year"], dataset_gunviolence["week"]]).apply(lambda x: pd.Series(dict(total_injured_per_yearweek = x.n_injured.sum())))
n_crimes_by_yearweek_plot = sns.barplot(x=n_crimes_by_yearweek.index, y=n_crimes_by_yearweek.total_injured_per_yearweek, palette=btui)
n_crimes_by_yearweek_plot.set_xticklabels(n_crimes_by_yearweek_plot.get_xticklabels(), rotation=45)
n_crimes_by_yearweek_plot.set_title("Injured per Year-Week")
del(n_crimes_by_yearweek)
participant_genders_sum = dataset_gunviolence[["state", "participant_gender_total", "participant_gender_male", "participant_gender_female", "participant_gender_unknown"]].groupby(dataset_gunviolence["state"]).sum()
sns.boxplot(data=participant_genders_sum, palette=btui)
g=sns.barplot(x=participant_genders_sum.index,y=participant_genders_sum.participant_gender_total,data=participant_genders_sum,color="#ffc266")
g=sns.barplot(x=participant_genders_sum.index,y=participant_genders_sum.participant_gender_male,data=participant_genders_sum,color="#8784db")
g=sns.barplot(x=participant_genders_sum.index,y=participant_genders_sum.participant_gender_female,data=participant_genders_sum,color="#d284bd")
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.show()
state_house_district_plot = sns.countplot(x=dataset_gunviolence.state_house_district, data=dataset_gunviolence, order=dataset_gunviolence.state_house_district.value_counts().index, palette=btui)
state_house_district_plot.set_xticklabels(state_house_district_plot.get_xticklabels(), rotation=90)
state_house_district_plot.set_title("Gun Violence per State-House District")
plt.show()
state_senate_district_plot = sns.countplot(x=dataset_gunviolence.state_senate_district, data=dataset_gunviolence, order=dataset_gunviolence.state_senate_district.value_counts().index, palette=btui)
state_senate_district_plot.set_xticklabels(state_senate_district_plot.get_xticklabels(), rotation=90)
state_senate_district_plot.set_title("Gun Violence per State Senate District")
plt.show()
agegroup_sum = dataset_gunviolence[["state","agegroup_child", "agegroup_teen", "agegroup_adult"]].groupby(dataset_gunviolence["state"]).sum()
g=sns.pointplot(x=agegroup_sum.index, y=agegroup_sum.agegroup_child, data=agegroup_sum, color="#c51b7d", scale=0.5, dodge=True, capsize=.2, label="agegroup_adult")
g=sns.pointplot(x=agegroup_sum.index, y=agegroup_sum.agegroup_teen, data=agegroup_sum, color="#1b7837", scale=0.5, dodge=True, capsize=.2, linestyles="--", markers="x", label="agegroup_adult")
g=sns.pointplot(x=agegroup_sum.index, y=agegroup_sum.agegroup_adult, data=agegroup_sum, color="#2166ac", scale=0.5, dodge=True, capsize=.2, linestyles="-", markers="X", label="agegroup_adult")
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_title("Gun Violence per various Age Group")
plt.show()