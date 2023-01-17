#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

##all variables display for cells
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"

#rendering plots
%matplotlib inline

#set deafault seaborn theme for all plots
sns.set()


df = pd.read_csv("../input/dataset.csv")
print("Dataset shape -> (rows, columns):", df.shape)
#print first few columns of the dataset
df.head()
#print data types of the columns
df.dtypes
#check statistics of int64, float64 columns
df.describe()
#check statistics of object columns
df.describe(include=["O"])
#date column reformat
df_date_reformat = df["date"].str.split("-", expand=True)

#time column reformat
df_time_reformat = df["time"].str.split(":", expand=True)

#join formatted date and time dataframes
df_date_time_reformat = pd.concat([df_date_reformat, df_time_reformat], axis=1)
df_date_time_reformat.columns = ["year","month", "day", "hour", "minute", "second", "ns"] #rename columns

#create a datetime object
df_date_time_obj = pd.to_datetime(df_date_time_reformat)

#add datetime object to a new dataframe and set it as index
df_sorted = df.copy()
df_sorted["datetime"] = df_date_time_obj
df_sorted.set_index("datetime", inplace=True)
df_sorted.drop(axis=1, columns=["username"], inplace=True) # drop "username" column
print("*** last row timestamp before sorting ***")
print(df_sorted.index[-1])
#sort df_sorted data by "datetime" index
df_sorted.sort_index(inplace=True)
print("*** last row timestamp after sorting ***")
print(df_sorted.index[-1])
print("*** dataframe with datetime index ***")
df_sorted.head()
#quick insight using pandas methods
print("Start time of data recording ->", df_sorted.index.min())
print("End time of data recording ->", df_sorted.index.max())
print("Number of days of data collection ->", df_sorted.index.day.nunique())
print("Days of data collection ->", df_sorted.date.unique())
#visualization of user activity pattern
f, ax =  plt.subplots(ncols=1, nrows=2, figsize = (14,10))

#sample count vs hour of day
arr_hr = np.unique(df_sorted.index.hour, return_counts = True)
ax[0].bar(arr_hr[0], arr_hr[1])
ax[0].set_title("Recorded sample count for different hours of the day")
ax[0].set_xlabel("Hour")
ax[0].set_ylabel("Number of samples")
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%d:00'))

#sample count vs day of week
arr_day = np.unique(df_sorted.index.dayofweek, return_counts = True)
ax[1].bar(arr_day[0], arr_day[1])
ax[1].set_title("Recorded sample count for different days of the week")
ax[1].set_xlabel("Day")
ax[1].set_ylabel("Number of samples")
ax[1].set_xticklabels(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

f.tight_layout()
f.show()
#temporary dataframe with "wrist" and "activity" column values replaced
df_sorted_viz =  df_sorted.copy()
df_sorted_viz["wrist"].replace(to_replace={0:"left", 1:"right"}, inplace=True)
df_sorted_viz["activity"].replace(to_replace={0:"walk", 1:"run"}, inplace=True)
#sanity check to see if values were updated correctly
print("Updated unique values")
for each_col in ["wrist", "activity"]:
    print(each_col,":", df_sorted_viz[each_col].unique())
df_sorted_viz.head(1)
#visualizing counts of "activity" and "wrist" features
plt_ht = 4
plt_asp = 2.5
#first plot
g_act = sns.catplot(x = "activity", kind = "count", height = plt_ht, aspect = plt_asp, data=df_sorted_viz)
g_act.ax.set_title("Recorded sample count for walk and run")
#second plot
g_wrist = sns.catplot(x = "wrist", kind = "count", height =plt_ht, aspect = plt_asp, data=df_sorted_viz)
g_wrist.ax.set_title("Recorded sample count for left & right wrists")
#third plot
g_act_wri = sns.catplot(x = "activity", kind="count", hue = "wrist", height = plt_ht, aspect = plt_asp, data=df_sorted_viz)
g_act_wri.ax.set_title("Recorded sample count for different activities, split by the wrist")
plt.show()
fig_kde, ax_kde = plt.subplots(nrows=3, ncols=2, figsize=(16, 10))
ax_num = 0
for each_col in df.columns.values[5:11]:
    g_kde = sns.kdeplot(df_sorted_viz[each_col], ax=ax_kde[ax_num % 3][ax_num // 3])
    ax_num += 1
fig_kde.suptitle("Data density")
#fig_kde.tight_layout()
fig_kde.show()
fig_str, ax_str = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
fig_str.suptitle("Device data split by activity")    
ax_num = 0
for each_col in df.columns.values[5:11]:
    g_str = sns.stripplot(x = "activity", y = each_col, hue = "activity", ax=ax_str[ax_num // 3][ax_num % 3], data = df_sorted_viz)
    ax_num += 1
    g_str.set_xlabel("")
#fig_str.tight_layout()
fig_str.show()