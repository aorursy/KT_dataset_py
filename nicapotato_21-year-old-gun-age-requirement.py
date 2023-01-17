# Gen
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

# Modeling
import statsmodels.api as sm

# Warnings
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../input/Mass Shootings Dataset Ver 5.csv", encoding = "ISO-8859-1", parse_dates=["Date"])
print("\nData has {} Rows, {} Columns".format(*df.shape))

print("Remove Missing Values..")
df=df[df.notnull()]

print("\nRace Pre-Processing")
# Unkown Race Variable
print("Create Variable to Capture Race Ambiguity -> Race_Part_Unknown")
print(df.loc[df.Race.str.contains(r"((?i)unknown)|((?i)other)|((?i)race)", na=False), "Race"].unique())
df["Race_Part_Unknown"] =  df.Race.str.contains(r"((?i)unknown)|((?i)other)|((?i)race)", na=False)

# Black American or African American
print("\nCollapse the Multiple written versions of 'Black American or African American' into Black")
print(df.loc[df.Race.str.contains(r"((?i)black)",na=False), "Race"].unique())
df.loc[df.Race.str.contains(r"((?i)black)",na=False), "Race"]= "Black"

# White American or European American 
print("\nCollapse the Multiple written versions of 'White American or European American' into 'White")
print(df.loc[df.Race.str.contains(r"((?i)white)",na=False), "Race"].unique())
df.loc[df.Race.str.contains(r"((?i)white)",na=False), "Race"]= "White"

# Asian American
print("\nCollapse the Multiple written versions of 'Asian American'")
print(df.loc[df.Race.str.contains(r"((?i)Asian American)",na=False), "Race"].unique())
df.loc[df.Race.str.contains(r"((?i)Asian American)",na=False), "Race"]= "Asian American"

# Native American or Alaska Native
print("\nCollapse the Multiple written versions of 'Native American or Alaska Native' into 'Native'")
print(df.loc[df.Race.str.contains(r"((?i)Native)",na=False), "Race"].unique())
df.loc[df.Race.str.contains(r"((?i)Native)",na=False), "Race"]= "Native"

# Ambiguous Mix
print("\nCollapse the Multiple written versions of racially 'Ambiguous Mix'")
print(df.loc[df.Race.str.contains(r"((?i)unclear|(?i)race|(?i)other)",na=False), "Race"].unique())
df.loc[df.Race.str.contains(r"((?i)unclear|(?i)race|(?i)other)",na=False), "Race"] = "Ambiguous Mix"

print("\nGender Pre-Processing")
df.loc[df.Gender == "M", "Gender"] = "Male"
df.loc[df.Gender == "Male/Female", "Gender"] = "M/F"

print("\nMental Health Pre-Processing")
df["Mental Health Issues"]= df["Mental Health Issues"].str.strip()
df.loc[df["Mental Health Issues"] == "unknown", ["Mental Health Issues"]] = "Unknown"

# Age Variable
df.Age = df.Age.str.split(",").str[0].astype(float)
# Over 21?
df["21+"] = np.nan
df.loc[df.Age >= 21,"21+"] = "21 and Over"
df.loc[df.Age < 21,"21+"] = "Under 21"

# Fix Open/Close
df.loc[df["Open/Close Location"] == "Open+CLose","Open/Close Location"] = "Open+Close"

# Map Employment status to object
df["Employeed (Y/N)"]= df["Employeed (Y/N)"].map({1.0:"Yes",0.0:"No"})

# Time Frames of Interest
df["Year"] = df["Date"].dt.year
df["Date of Year"] = df['Date'].dt.dayofyear # Day of Year
df["Weekday"] = df['Date'].dt.weekday
df["Day of Month"] = df['Date'].dt.day
def time_slicer(df, timeframes, somevar):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize = [12,7])
    for i,x in enumerate(timeframes):
        df.loc[:,[x,somevar]].groupby([x]).count().plot(ax=ax[i])
        ax[i].set_ylabel("Incident Count")
        ax[i].set_title("Incident Count by {}".format(x))
        ax[i].set_xlabel("")
        ax[i].legend_.remove()
    ax[len(timeframes)-1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)
    
# Fast funcdtion for top occruence categories
def topcat_index(series, n=5):
    """
    Wow! 2 charcters SAVED on function length
    """
    return series.value_counts().index[:n]
def topcats(series, n=5):
    return series.isin(topcat_index(series, n=n))

def cat_time_slicer(df, slicevar, n, timeframes, somevar, normalize = False):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize = [12,7])
    top_classes = topcat_index(df[slicevar],n=n)
    for i,x in enumerate(timeframes):
        for y in top_classes:
            if normalize == True:
                total = df.loc[df[slicevar]==y,slicevar].count()
                ((df.loc[(df[slicevar]==y),[x,slicevar]]
                 .groupby([x])
                 .count()/total)
                .plot(ax=ax[i], label=y))
            if normalize == False:
                total = df.loc[df[slicevar]==y,slicevar].count()
                ((df.loc[(df[slicevar]==y),[x,slicevar]]
                 .groupby([x])
                 .count())
                .plot(ax=ax[i], label=y))
        ax[i].set_ylabel("Percent of\nCompany Incidents")
        ax[i].set_title("Percent of Incident by Company by {}".format(x))
        ax[i].set_xlabel("")
        ax[i].legend(top_classes, fontsize='large', loc='center left',bbox_to_anchor=(1, 0.5))
    ax[len(timeframes)-1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)
    plt.subplots_adjust(top=0.90)
    plt.suptitle('Normalized Time-Series for top {}s over different over {}'.format(slicevar,[x for x in timeframes]),fontsize=17)
    


# CROSS TAB
def crosstab_heat(df,x,y, size=(10, 5),cmap="binary"):
    # Heatmaps of Percentage Pivot Table
    f, ax = plt.subplots(1,2,figsize=size, sharey=True)
    sns.heatmap(pd.crosstab(df[x], df[y], normalize='columns').mul(100).round(0),
                annot=True, linewidths=.5, ax = ax[0],fmt='g', cmap=cmap,
                    cbar_kws={'label': '% Percentage'})
    ax[0].set_title('{} Count by {} - Crosstab\nHeatmap % Distribution by Columns'.format(x,y))

    sns.heatmap(pd.crosstab(df[x], df[y], normalize="index").mul(100).round(0),
                annot=True, linewidths=.5, ax=ax[1],fmt='g', cmap=cmap,
                    cbar_kws={'label': 'Percentage %'})
    ax[1].set_title('{} Count by {} - Crosstab\nHeatmap % Distribution by Index'.format(x,y))
    ax[1].set_ylabel('')
    plt.tight_layout(pad=0)
    
def norm_heat(x,y, size=(10, 5),df=df,normalize=True, cmap="coolwarm"):
    # Heatmaps of Percentage Pivot Table
    f, ax = plt.subplots(1,figsize=size, sharey=True)
    sns.heatmap(pd.crosstab(df[x], df[y], normalize=normalize).mul(100).round(0),
                annot=True, linewidths=.5, ax = ax,fmt='g', cmap=cmap,
                    cbar_kws={'label': '% Percentage'})
    if normalize == "columns":ax.set_title('{} Count by {} - Crosstab\nHeatmap % Distribution by Columns'.format(x,y))
    if normalize == "index":ax.set_title('{} Count by {} - Crosstab\nHeatmap % Distribution by Index'.format(x,y))
    if normalize == True:ax.set_title('{} Count by {} - Crosstab\nHeatmap % Distribution'.format(x,y))
    ax.set_frame_on(True)
    plt.tight_layout(pad=0)
    
# Stacked Func
def stacked(time, cat, target, df=df, size=[10,4]):
    f, ax = plt.subplots(figsize=size)
    temp = df[[time,cat,target]].pivot_table(columns=cat, index=time, values=target, aggfunc="sum").plot.area(ax=ax)
    ax.legend(fontsize='large', loc='center left',bbox_to_anchor=(1, 0.5))

# Multiple Timeframes
def multi_stacked(timevars, cat, target, df=df, size=[10,9]):
    f, ax = plt.subplots(len(timevars),figsize=size)
    for i,time in enumerate(timevars):
        temp = df[[time,cat,target]].pivot_table(columns=cat, index=time, values=target,
                                                 aggfunc="sum").plot.area(ax=ax[i])
        ax[i].legend(fontsize='large', loc='center left',bbox_to_anchor=(1, 0.5))
        ax[i].set_title("Total Fatalities by {} by Age Cutoff".format(time))
        ax[i].set_ylabel("Fatalities")
    plt.tight_layout(pad=0)
    
# Describe Function
def custom_describe(df):
    """
    I am a non-comformist :)
    """
    unique_count = []
    for x in df.columns:
        mode = df[x].mode().iloc[0]
        unique_count.append([x,
                             len(df[x].unique()),
                             df[x].isnull().sum(),
                             mode,
                             df[x][df[x]==mode].count(),
                             df[x].dtypes])
    print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))
    return pd.DataFrame(unique_count, columns=["Column","Unique","Missing","Mode","Mode Occurence","dtype"]).set_index("Column").T


print("Helper Functions ready..")
pd.set_option('display.max_columns', 500)
custom_describe(df)
with sns.color_palette("binary"):
    multi_stacked(timevars=["Date","Year","Date of Year","Day of Month"],cat="21+",target="Fatalities")
import matplotlib.gridspec as gridspec
with sns.color_palette("inferno"):
    fig = plt.figure(1, figsize=[8,6])
    gridspec.GridSpec(2,2, height_ratios=[2, 1], width_ratios=[1, 3])

    # set up subplot grid
    plt.subplot2grid((2,2), (0,0))
    sns.countplot(x=df["Gender"],order=df["Gender"].value_counts().index)
    plt.xticks(rotation=10)
    plt.ylabel("Occurrence")
    plt.title("Perpetrator Frequency Distribution by\n{}".format("Gender"))

    plt.subplot2grid((2,2), (0,1))
    sns.countplot(x=df["21+"],order=["Under 21","21 and Over"])
    plt.xticks(rotation=10)
    plt.ylabel("")
    plt.title("Perpetrator Frequency Distribution by\n{}".format("Age Cutoff"))

with sns.color_palette("inferno"):
    plt.subplot2grid((2,2), (1,0), colspan=2)
    sns.countplot(y=df["Race"],order=df["Race"].value_counts().index,palette="inferno")
    plt.title("Race of Mass Shooting Perpetrator")
    plt.xlabel("Occurrence")

    fig.tight_layout(pad=0)
    plt.show()
with sns.color_palette("Paired"):
    plt.figure(figsize=(8,6))
    plt.subplot(221)
    total_inj = [df.loc[df["21+"] == "Under 21","Injured"].sum(),df.loc[df["21+"] == "21 and Over","Injured"].sum()]
    sns.barplot(x=["Under 21","21 and Over"],y= total_inj)
    plt.title("Total Injuries by Age Group")
    plt.ylabel("Count")
    plt.xlabel("")
    plt.xticks([])

    plt.subplot(222)
    total_fatal = [df.loc[df["21+"]== "Under 21","Fatalities"].sum(),df.loc[df["21+"]== "21 and Over","Fatalities"].sum()]
    sns.barplot(x=["Under 21","21 and Over"], y = total_fatal)
    plt.title("Total Fatalities by Age Group")
    plt.xlabel("")
    plt.xticks([])
    
    plt.subplot(223)
    total_inj = [df.loc[df["21+"] == "Under 21","Injured"].mean(),df.loc[df["21+"] == "21 and Over","Injured"].mean()]
    sns.barplot(x=["Under 21","21 and Over"],y= total_inj)
    plt.title("Average Injuries by Age Group")

    plt.subplot(224)
    total_fatal = [df.loc[df["21+"]== "Under 21","Fatalities"].mean(),df.loc[df["21+"]== "21 and Over","Fatalities"].mean()]
    sns.barplot(x=["Under 21","21 and Over"], y = total_fatal)
    plt.ylabel("Average Fatalities")
    plt.title("Average Fatalities by Age Group")
    plt.xlabel("")
    
    plt.tight_layout(pad=0)
    plt.show()
xvar = "Fatalities"
slices = ["Gender","Open/Close Location","Mental Health Issues","21+"]
with sns.color_palette("Paired"):
    f, axes = plt.subplots(2,2, figsize=(7,5), sharex=False)
    row = 0
    col = 0
    for i,y in enumerate(slices):
        if col == 2:
            col = 0
            row += 1
        for x in set(df[y][df[y].notnull()]):
            sns.distplot(df.loc[df[y]==x,xvar], label=x, ax=axes[row,col],
                    hist = False, kde=True,#hist_kws=dict(edgecolor="k", linewidth=2))
                         kde_kws={"lw": 3})
        axes[row,col].set_xlabel("{}".format(xvar))
        if row == 0: axes[row,col].set_xlabel("")
        axes[row,col].set_ylabel('Occurrence')
        if col == 1:  axes[row,col].set_ylabel('')
        axes[row,col].set_title('{} Distribution\nby {}'.format(xvar, y))
        axes[row,col].legend()
        col += 1
    plt.tight_layout(pad=0)
    plt.show()
xvar = "Injured"
slices = ["Gender","Open/Close Location","Mental Health Issues","21+"]
plt.rcParams["patch.force_edgecolor"] = True
with sns.color_palette("Paired"):
    f, axes = plt.subplots(2,2, figsize=(7,5), sharex=False)
    row = 0
    col = 0
    for i,y in enumerate(slices):
        if col == 2:
            col = 0
            row += 1
        for x in set(df[y][df[y].notnull()]):
            sns.distplot(np.log(df.loc[df[y]==x,xvar]+1), label=x, ax=axes[row,col], hist = False, kde=True,
                    kde_kws={"lw": 3})
        axes[row,col].set_xlabel("{}".format(xvar))
        if row == 0: axes[row,col].set_xlabel("")
        axes[row,col].set_ylabel('Occurrence Density')
        if col == 1:  axes[row,col].set_ylabel('')
        axes[row,col].set_title('{} Density Distribution\nby {}'.format(xvar, y))
        axes[row,col].legend()
        col += 1
    plt.tight_layout(pad=0)
    plt.show()
def custom(df):
    xvar = "Age"
    slices = ["Gender","Open/Close Location","Mental Health Issues","Employeed (Y/N)"]

    f, axes = plt.subplots(2,2, figsize=(7,5), sharex=False)
    row = 0
    col = 0
    for i,y in enumerate(slices):
        if col == 2:
            col = 0
            row += 1
        for x in set(df[y][df[y].notnull()]):
            sns.distplot(df.loc[df[y]==x,xvar], label=x, ax=axes[row,col],
                         bins = 18,hist = False, kde=True,
                         kde_kws={"lw": 3})
        axes[row,col].set_xlabel("{}".format(xvar))
        if row == 0: axes[row,col].set_xlabel("")
        axes[row,col].set_ylabel('Occurrence')
        if col == 1:  axes[row,col].set_ylabel('')
        axes[row,col].set_title('{} Distribution\nby {}'.format(xvar, y))
        axes[row,col].legend()
        col += 1
    plt.tight_layout(pad=0)
with sns.color_palette("Paired"):
    custom(df[df.Age.notnull()])
with sns.color_palette("binary"):
    stacked("Year","Gender","Fatalities")
    plt.title("Fatalities by Gender")
    plt.ylabel("Fatalities")
with sns.color_palette("binary"):
    stacked("Year","Employeed (Y/N)","Fatalities")
    plt.title("Fatalities by Employment Status")
    plt.ylabel("Fatalities")
g = sns.jointplot(data = df, y= "Fatalities", x="Age", kind='reg', color='k')
g.fig.suptitle("Scatter Plot for Age and Positive Feedback Count")
plt.show()
df.loc[df["Fatalities"] > 30,["Title","Fatalities"]]
plt.figure(figsize=[8,7])
x= "Gender"
y= "21+"
normalize= "columns"
cmap="binary"
plt.subplot(221)
sns.heatmap(pd.crosstab(df[x], df[y], normalize=normalize).mul(100).round(0)[["Under 21","21 and Over"]].rename(columns={"21 and Over": "21+"}),
            annot=True, linewidths=.5,fmt='g', cmap=cmap,
                cbar_kws={'label': '% Percentage'})
if normalize == "columns": plt.title('{} Count by Age Cutoff'.format(x))
if normalize == "index": plt.title('{} Count by Age Cutoff'.format(x))
if normalize == True: plt.title('{} Count by Age Cutoff'.format(x))
plt.xlabel("")

x= "Race"
y= "21+"
normalize="columns"
plt.subplot(222)
sns.heatmap(pd.crosstab(df[x], df[y], normalize=normalize).mul(100).round(0)[["Under 21","21 and Over"]].rename(columns={"21 and Over": "21+"}),
            annot=True, linewidths=.5,fmt='g', cmap=cmap,
                cbar_kws={'label': '% Percentage'})
if normalize == "columns": plt.title('{} Count by Age Cutoff'.format(x))
if normalize == "index": plt.title('{} Count by Age Cutoff'.format(x))
if normalize == True: plt.title('{} Count by Age Cutoff'.format(x))
plt.xlabel("")
    
x= "Open/Close Location"
y= "21+"
normalize= "columns"
plt.subplot(223)
sns.heatmap(pd.crosstab(df[x], df[y], normalize=normalize).mul(100).round(0)[["Under 21","21 and Over"]].rename(columns={"21 and Over": "21+"}),
            annot=True, linewidths=.5,fmt='g', cmap=cmap,
                cbar_kws={'label': '% Percentage'})
plt.xlabel("")
if normalize == "columns": plt.title('{} Count by Age Cutoff'.format(x))
if normalize == "index": plt.title('{} Count by Age Cutoff'.format(x))
if normalize == True: plt.title('{} Count by Age Cutoff'.format(x))
    
x= "Employeed (Y/N)"
y= "21+"
normalize="columns"
plt.subplot(224)
sns.heatmap(pd.crosstab(df[x], df[y], normalize=normalize).mul(100).round(0)[["Under 21","21 and Over"]].rename(columns={"21 and Over": "21+"}),
            annot=True, linewidths=.5,fmt='g', cmap=cmap,
                cbar_kws={'label': '% Percentage'})
if normalize == "columns": plt.title('{} Count by Age Cutoff'.format(x))
if normalize == "index": plt.title('{} Count by Age Cutoff'.format(x))
if normalize == True: plt.title('{} Count by Age Cutoff'.format(x))
plt.xlabel("")
plt.tight_layout(pad=1)
plt.show()
plt.figure(figsize=[9,7])
x= "Cause"
y= "21+"
normalize= "columns"
cmap="binary"

plt.subplot(121)
sns.heatmap(pd.crosstab(df[x], df[y], normalize=normalize).mul(100).round(0)[["Under 21","21 and Over"]].rename(columns={"21 and Over": "21+"})
            ,annot=True, linewidths=.5,fmt='g', cmap=cmap,
                cbar_kws={'label': '% Percentage'})
if normalize == "columns": plt.title('{} Count by Age Cutoff'.format(x))
if normalize == "index": plt.title('{} Count by Age Cutoff'.format(x))
if normalize == True: plt.title('{} Count by Age Cutoff'.format(x))
plt.xlabel("")
    
x= "Target"
y= "21+"
normalize="columns"

plt.subplot(122)
sns.heatmap(pd.crosstab(df[x], df[y], normalize=normalize).mul(100).round(0)[["Under 21","21 and Over"]].rename(columns={"21 and Over": "21+"}),
            annot=True, linewidths=.5,fmt='g', cmap=cmap,
                cbar_kws={'label': '% Percentage'})
plt.xlabel("")
if normalize == "columns": plt.title('{} Count by Age Cutoff'.format(x))
if normalize == "index": plt.title('{} Count by Age Cutoff'.format(x))
if normalize == True: plt.title('{} Count by Age Cutoff'.format(x))

plt.tight_layout(pad=2)
plt.show()

(df.loc[(df.Cause.isin(["terrorism","anger"]))&(df["21+"] == "Under 21"),["Fatalities","Target"]]
 .groupby("Target").sum()
 .sort_values(by="Fatalities",ascending=False)
 .plot.bar())
plt.title("Fatalities by Target caused by Rage and Terrorism\nUnder 21")
plt.ylabel("Fatalities")
plt.xlabel("")
plt.xticks(rotation=85)
plt.show()