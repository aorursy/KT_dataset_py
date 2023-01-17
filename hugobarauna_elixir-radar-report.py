import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import numpy as np



print("Setup Complete")
plt.rcdefaults()



curated_articles_filepath = "../input/../input/elixir-radar/Elixir Radar - Elixir Radar curated content.csv"

curated_articles_data = pd.read_csv(curated_articles_filepath, parse_dates=['Email publication date'])



curated_articles_data['Year'] =  pd.DatetimeIndex(curated_articles_data['Email publication date']).year

curated_articles_data



def show_values_on_bars(axs, h_v="v", space=0.4):

    def _show_on_single_plot(ax):

        if h_v == "v":

            for p in ax.patches:

                _x = p.get_x() + p.get_width() / 2

                _y = p.get_y() + p.get_height()

                value = int(p.get_height())

                ax.text(_x, _y, value, ha="center") 

        elif h_v == "h":

            for p in ax.patches:

                _x = p.get_x() + p.get_width() + float(space)

                _y = p.get_y() + (p.get_height() / 2)

                value = "{:,}".format(int(p.get_width()))

                ax.text(_x, _y, value, ha="left")



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)
clicks_by_author = curated_articles_data.groupby('Author').Clicks.sum().sort_values(ascending=False)

top_ten_authors = clicks_by_author.head(10)





plt.figure(figsize=(12,8))

plt.title("Top 10 Elixir Radar authors in all time")



ax = sns.barplot(x=top_ten_authors[:], y=top_ten_authors.index)

show_values_on_bars(ax, "h")



plt.xlabel("Sum of unique clicks per issue")

plt.ylabel("")

foo = 'bar'
years = [2020, 2019, 2018, 2017, 2016, 2015]

for year in years:

    clicks_by_author = curated_articles_data.loc[curated_articles_data.Year == year].groupby('Author').Clicks.sum().sort_values(ascending=False)

    top_ten_authors = clicks_by_author.head(10)

    top_ten_authors



    plt.figure(figsize=(12,8))

    plt.title("Top 10 Elixir Radar authors in " + str(year))



    ax = sns.barplot(x=top_ten_authors[:], y=top_ten_authors.index)

    show_values_on_bars(ax, "h")



    plt.ylabel("")

    plt.xlabel("Sum of unique clicks per issue")