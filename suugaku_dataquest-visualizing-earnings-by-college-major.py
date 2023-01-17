import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
# Magic command to ensure the plots are displayed inline

%matplotlib inline
recent_grads_filepath = "../input/college-earnings-by-major/recent-grads.csv"
recent_grads = pd.read_csv(recent_grads_filepath)
recent_grads.head()
recent_grads.info()
recent_grads[recent_grads.isna().any(axis = "columns")]
recent_grads.dropna(inplace = True)
recent_grads_columns = recent_grads.columns.to_series()

recent_grads_columns.replace({"ShareWomen": "share_women"}, inplace = True)

recent_grads_columns = recent_grads_columns.str.lower()

recent_grads_columns
recent_grads.columns = recent_grads_columns
recent_grads.drop(columns = ["rank", "major_code"]).describe()
fig, ax = plt.subplots(figsize = (10, 8))

sns.scatterplot(x = "total", y = "median", data = recent_grads, ax = ax)

ax.axhline(y = recent_grads["median"].median(), color = "orange", linestyle = ":", linewidth = 3,

          label = "median of \nmedian incomes")

ax.legend(loc = "best")

ax.set(xlabel = "Total number of recent graduates", ylabel = "Median income of year-round full-time workers");
fig, ax = plt.subplots(figsize = (10, 8))

sns.scatterplot(x = "total", y = "unemployment_rate", data = recent_grads, ax = ax)

ax.axhline(y = recent_grads["unemployment_rate"].median(), color = "orange", linestyle = ":", linewidth = 3,

          label = "median unemployment rate")

ax.legend(loc = "best")

ax.set(xlabel = "Total number of recent graduates", ylabel = "Unemployment rate");
fig, ax = plt.subplots(figsize = (10, 8))

sns.scatterplot(x = "share_women", y = "median", data = recent_grads, ax = ax)

ax.axhline(y = recent_grads["median"].median(), color = "orange", linestyle = ":", linewidth = 3,

          label = "median of \nmedian incomes")

ax.axvline(x = 0.5, linestyle = ":")

ax.legend(loc = "best")

ax.set(xlabel = "Proportion of female recent graduates", ylabel = "Median income of year-round full-time workers");
fig, ax = plt.subplots(figsize = (10, 8))

sns.scatterplot(x = "share_women", y = "unemployment_rate", data = recent_grads, ax = ax)

ax.axhline(y = recent_grads["unemployment_rate"].median(), color = "orange", linestyle = ":", linewidth = 3,

          label = "median unemployment rate")

ax.axvline(x = 0.5, linestyle = ":")

ax.legend(loc = "best")

ax.set(xlabel = "Proportion of female recent graduates", ylabel = "Unemployment rate");
fig, ax = plt.subplots(figsize = (10, 8))

sns.scatterplot(x = "full_time", y = "median", data = recent_grads, ax = ax)

ax.axhline(y = recent_grads["median"].median(), color = "orange", linestyle = ":", linewidth = 3,

          label = "median of \nmedian incomes")

ax.legend(loc = "best")

ax.set(xlabel = "Numer of recent graduates working full-time", ylabel = "Median income of year-round full-time workers");
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (14, 10))

sns.distplot(recent_grads["median"], ax = axes[0, 0], kde = False)

sns.distplot(recent_grads["unemployment_rate"], ax = axes[0, 1], kde = False)

sns.distplot(recent_grads["employed"], ax = axes[1, 0], kde = False)

sns.distplot(recent_grads["full_time"], ax = axes[1, 1], kde = False);
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (14, 10))

sns.distplot(recent_grads["total"], ax = axes[0, 0], kde = False)

sns.distplot(recent_grads["share_women"], ax = axes[0, 1], kde = False)

sns.distplot(recent_grads["men"], ax = axes[1, 0], kde = False)

sns.distplot(recent_grads["women"], ax = axes[1, 1], kde = False);
# Focus specifically on majors with fewer than 50,000 recent graduates

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (14, 10))

sns.distplot(recent_grads.loc[recent_grads["total"] < 50000, "total"], ax = axes[0, 0], kde = False)

sns.distplot(recent_grads.loc[recent_grads["total"] < 50000, "employed"], ax = axes[0, 1], kde = False)

sns.distplot(recent_grads.loc[recent_grads["total"] < 50000, "men"], ax = axes[1, 0], kde = False)

sns.distplot(recent_grads.loc[recent_grads["total"] < 50000, "women"], ax = axes[1, 1], kde = False);
sns.pairplot(recent_grads);
sns.pairplot(recent_grads, vars = ["sample_size", "median"], height = 3);
sns.pairplot(recent_grads, vars = ["sample_size", "median", "unemployment_rate"], height = 3);
top_bottom_10 = recent_grads.head(10).append(recent_grads.tail(10))
fig, ax = plt.subplots(figsize = (10, 10))

sns.barplot(y = "major", x = "share_women", orient = "h", ax = ax, data = top_bottom_10)

ax.axvline(x = 0.5, linestyle = ":");
fig, ax = plt.subplots(figsize = (10, 10))

sns.barplot(y = "major", x = "unemployment_rate", orient = "h", ax = ax, data = top_bottom_10);
fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 14), sharex = "col")

sns.barplot(y = "major_category", x = "men", orient = "h", ax = ax[0], 

            data = recent_grads.groupby("major_category").sum().reset_index())

sns.barplot(y = "major_category", x = "women", orient = "h", ax = ax[1], 

            data = recent_grads.groupby("major_category").sum().reset_index());
fig, ax = plt.subplots(figsize = (10, 8))

sns.barplot(y = "major_category", x = "share_women", orient = "h", order = np.sort(recent_grads["major_category"].unique()),

            capsize = 0.2, ax = ax, data = recent_grads)

ax.axvline(x = 0.5, linestyle = ":");
fig, ax = plt.subplots(figsize = (10, 8))

sns.boxplot(y = "major_category", x = "median", orient = "h", order = np.sort(recent_grads["major_category"].unique()),

            ax = ax, data = recent_grads);
fig, ax = plt.subplots(figsize = (10, 8))

sns.boxplot(y = "major_category", x = "unemployment_rate", orient = "h", order = np.sort(recent_grads["major_category"].unique()),

            ax = ax, data = recent_grads);
recent_grads["majority_women"] = recent_grads["share_women"] > 0.5
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 8))

sns.boxplot(x = "majority_women", y = "median", ax = ax[0], data = recent_grads)

sns.boxplot(x = "majority_women", y = "unemployment_rate", ax = ax[1], data = recent_grads);