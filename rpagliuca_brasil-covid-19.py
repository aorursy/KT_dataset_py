import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
df_full = pd.read_csv("https://data.brasil.io/dataset/covid19/caso_full.csv.gz")

df_published = pd.read_csv("https://data.brasil.io/dataset/covid19/caso.csv.gz")

df_published.head()
#df = df_full[df_full["state"] == "SP"]

#df = df_full.loc[((df_full["state"] == "RJ") | (df_full["state"] == "SP"))].copy()

df = df_full
from datetime import datetime

df.loc[:, "datetime"] = df.loc[:, "date"].apply(lambda date: datetime.strptime(date, "%Y-%m-%d"))

df_published.loc[:, "datetime"] = df.loc[:, "date"].apply(lambda date: datetime.strptime(date, "%Y-%m-%d"))



if "confirmed" not in df:

    df.loc[:, "confirmed"] = df.loc[:, "last_available_confirmed"]



if "deaths" not in df:

    df.loc[:, "deaths"] = df.loc[:, "last_available_deaths"]

    df.loc[:, "deaths_diff"] = df.loc[:, "new_deaths"]

    df.loc[:, "cases_diff"] = df.loc[:, "new_confirmed"]

else:

    df.loc[:, "deaths_diff"] = df.loc[:, "deaths"].diff(periods=-1)  

    df.loc[:, "cases_diff"] = df.loc[:, "confirmed"].diff(periods=-1)

    

df.loc[:, "count"] = df["date"].apply(lambda x: 1)

pivot = df.loc[(df["place_type"] == "state")].pivot_table(index="datetime", columns="place_type", values=["deaths_diff"], aggfunc=np.sum)

pivot_cases_diff = df.loc[(df["place_type"] == "state")].pivot_table(index="datetime", columns="place_type", values=["cases_diff"], aggfunc=np.sum)

pivot_deaths = df.loc[(df["place_type"] == "state")].pivot_table(index="datetime", columns="place_type", values=["deaths"], aggfunc=np.sum)

pivot_cases = df.loc[(df["place_type"] == "state")].pivot_table(index="datetime", columns="place_type", values=["confirmed"], aggfunc=np.sum)

pivot_count = df.loc[(df["place_type"] == "state")].pivot_table(index="datetime", columns="place_type", values=["count"], aggfunc=np.sum)

convolution_size = 15

convolution_array = np.repeat(1/convolution_size, convolution_size)

#convolution_array = [-1, 2, -1]

pivot.loc[:, "deaths_diff_convolved"] = np.convolve(pivot["deaths_diff"]["state"], convolution_array)[int((convolution_size-1)/2):-int((convolution_size-1)/2)]

pivot_cases_diff.loc[:, "cases_diff_convolved"] = np.convolve(pivot_cases_diff["cases_diff"]["state"], convolution_array)[int((convolution_size-1)/2):-int((convolution_size-1)/2)]
def merge_legends(fig, new_labels = []):

    all_lines = []

    all_labels = []

    for ax in fig.get_axes():

        lines, labels = ax.get_legend_handles_labels()

        all_lines += lines

        all_labels += labels

    if len(new_labels):

        all_labels = new_labels

    fig.get_axes()[0].legend(all_lines, all_labels)



def colorize(ax, color, side):

    ax.spines[side].set_color(color)

    ax.tick_params(axis='y', colors=color)

    ax.yaxis.label.set_color(color)

    

def new_right_axis(ax1):

    ax = ax1.twinx()

    ax.spines["top"].set_visible(False)

    ax.spines["bottom"].set_visible(False)

    ax.spines["left"].set_visible(False)

    return ax



def new_left_axis(ax1):

    ax = ax1.twinx()

    ax.spines["top"].set_visible(False)

    ax.spines["bottom"].set_visible(False)

    ax.spines["right"].set_visible(False)

    return ax
from matplotlib.colors import ListedColormap

pivot_state_data_availability = pd.pivot_table(df_published, index="state", columns="date", values=["deaths"])

data_available = ~ np.isnan(pivot_state_data_availability)

fig = plt.figure(figsize=[30,100], dpi=70)

cmap = ListedColormap(["white", "green"])

fig.gca().matshow(data_available, cmap=cmap)

fig.gca().tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True)

plt.xticks(range(len(data_available["deaths"].columns))[::3], data_available["deaths"].columns[::3], rotation=45)

plt.yticks(range(len(data_available.index)), data_available.index)

plt.show()
#period = pivot.iloc[1:]

#period = pivot.iloc[int((convolution_size-1)/2):-int((convolution_size-1)/2)]

period = pivot.copy()

#period.iloc[:int((convolution_size-1)/2)].loc[:,"deaths_diff_convolved"] = 0

period.iloc[:int((convolution_size-1)/2), df.columns.get_indexer(["deaths_diff_convolved"])] = np.nan

period.iloc[-int((convolution_size-1)/2):, df.columns.get_indexer(["deaths_diff_convolved"])] = np.nan

#period["deaths_diff_convolved"].iloc[-int((convolution_size-1)/2):] = np.nan



plt.figure(dpi=120, figsize=[14, 7])

ax2=plt.gca()

ax = ax2.twinx()  # instantiate a second axes that shares the same x-axis

ax.set_ylabel("Count of states which reported data")

pivot_count.loc[period.index].plot(ax=ax, color="#EEE", legend=False)

ax.grid(True, linestyle="--", color="#EEE")

#pivot_count.plot(ax=ax, color="#EEE", legend=False)

period.plot(ax=ax2, legend=False)

#pivot.plot(ax=plt.gca())

merge_legends(plt.gcf(), ["New deaths", "New deaths (smoothed)", "Count of states which reported data"])

ax.set_title(np.amax(period.index).strftime("%Y-%m-%d"))

plt.show()
p = pivot_deaths.copy()

p["log"] = p["deaths"].apply(lambda x: np.log10(x))

p["derivative"] = np.convolve(p.loc[:, "log"], [0.5, 0.5, 0, -0.5, -0.5])[2:-2]

p.iloc[[0, 1, -1, -2], p.columns.get_loc("derivative")] = np.nan

plt.figure(dpi=120, figsize=[14, 7])



ax1 = plt.gca()

ax1.set_ylabel("Total deaths")

ax1.spines["right"].set_visible(False)



ax2 = new_right_axis(ax1)

ax2.set_ylabel("Derivative of log of total deaths")

colorize(ax2, "red", "right")



ax3 = new_right_axis(ax1)

ax3.set_ylabel("Count of states which reported data")

ax3.spines["right"].set_position(("axes", 1.1))

colorize(ax3, "orange", "right")



ax4 = new_right_axis(ax1)

ax4.set_ylabel("Total deaths (log)")

ax4.spines["right"].set_position(("axes", 1.2))

colorize(ax4, "green", "right")

ax4.set_yscale("log")



for ax in [ax1, ax2, ax3, ax4]:

    for s in ["top", "bottom", "left", "right"]:

        ax.spines[s].set_linewidth(1.3)



p["deaths"].plot(ax=ax1, color="blue", legend=False)

p["deaths"].plot(ax=ax4, color="green", legend=False)

p["derivative"].plot(ax=ax2, color="red", legend=False)

pivot_count.loc[period.index].plot(ax=ax3, color="orange", legend=False)

#pivot_count.plot(ax=ax3, color="orange", legend=False)

colorize(ax1, "blue", "left")



merge_legends(plt.gcf(), ["Deaths", "Derivative of log of deaths", "Count of states which reported data", "Deaths (log)"])

ax1.set_xlabel("Date")



ax1.set_title(np.amax(p.index).strftime("%Y-%m-%d"))

ax1.grid(True, linestyle="--", color="#EEE")



plt.show()
#period = pivot.iloc[1:]

#period = pivot.iloc[int((convolution_size-1)/2):-int((convolution_size-1)/2)]

period = pivot_cases_diff.copy()

#period.iloc[:int((convolution_size-1)/2)].loc[:,"deaths_diff_convolved"] = 0

period.iloc[:int((convolution_size-1)/2), df.columns.get_indexer(["cases_diff_convolved"])] = np.nan

period.iloc[-int((convolution_size-1)/2):, df.columns.get_indexer(["cases_diff_convolved"])] = np.nan

#period["deaths_diff_convolved"].iloc[-int((convolution_size-1)/2):] = np.nan



plt.figure(dpi=120, figsize=[14, 7])

ax1=plt.gca()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



ax3 = new_right_axis(ax1)

ax3.set_ylabel("New cases (log scale)")

ax3.spines["right"].set_position(("axes", 1.1))

colorize(ax3, "red", "right")

ax3.set_yscale("log")



pivot_count.loc[period.index].plot(ax=ax1, color="#EEE", legend=False)

#pivot_count.plot(ax=ax, color="#EEE", legend=False)

period.plot(ax=ax2, legend=False)

period["cases_diff"].plot(ax=ax3, legend=False, color="red")

#pivot.plot(ax=plt.gca())

merge_legends(plt.gcf(), ["Count of states which reported data", "New cases", "New cases (smoothed)", "New cases (log scale)"])

ax1.set_title(np.amax(period.index).strftime("%Y-%m-%d"))

ax1.grid(True, linestyle="--", color="#EEE")

plt.show()
p = pivot_cases.copy()

p["log"] = p["confirmed"].apply(lambda x: np.log10(x))

p["derivative"] = np.convolve(p.loc[:, "log"], [0.5, 0.5, 0, -0.5, -0.5])[2:-2]

p.iloc[[0, 1, -1, -2], p.columns.get_loc("derivative")] = np.nan

plt.figure(dpi=120, figsize=[14, 7])



ax1 = plt.gca()

ax1.set_ylabel("Total cases")

ax1.spines["right"].set_visible(False)



ax2 = new_right_axis(ax1)

ax2.set_ylabel("Derivative of log of total cases")

colorize(ax2, "red", "right")



ax3 = new_right_axis(ax1)

ax3.set_ylabel("Count of states which reported data")

ax3.spines["right"].set_position(("axes", 1.1))

colorize(ax3, "orange", "right")



ax4 = new_right_axis(ax1)

ax4.set_ylabel("Total cases (log)")

ax4.spines["right"].set_position(("axes", 1.2))

colorize(ax4, "green", "right")

ax4.set_yscale("log")



for ax in [ax1, ax2, ax3, ax4]:

    for s in ["top", "bottom", "left", "right"]:

        ax.spines[s].set_linewidth(1.3)



p["confirmed"].plot(ax=ax1, color="blue", legend=False)

p["confirmed"].plot(ax=ax4, color="green", legend=False)

p["derivative"].plot(ax=ax2, color="red", legend=False)

pivot_count.loc[period.index].plot(ax=ax3, color="orange", legend=False)

colorize(ax1, "blue", "left")



merge_legends(plt.gcf(), ["Cases", "Derivative of log of cases", "Count of states which reported data", "Cases (log)"])



ax1.set_xlabel("Date")

ax1.grid(True, linestyle="--", color="#EEE")

ax1.set_title(np.amax(p.index).strftime("%Y-%m-%d"))



plt.show()