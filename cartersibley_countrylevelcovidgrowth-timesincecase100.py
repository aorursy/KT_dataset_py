# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import plotnine as gg, datetime as dt
# Read the data and fix the dates

df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df["Country/Region"].value_counts()



dates = pd.Series(pd.NaT, index = df.index)

dates.loc[(df.ObservationDate.str.len() == 8)] = pd.to_datetime(df.ObservationDate.loc[df.ObservationDate.str.len() == 8], format = '%m/%d/%y')

dates.loc[df.ObservationDate.str.len() == 10] = pd.to_datetime(df.ObservationDate.loc[df.ObservationDate.str.len() == 10], format = '%m/%d/%Y')

df['ObservationDate'] = dates

df.head()

# Create plot data grouped by country on confirmed cases

plt_df = df.groupby(["Country/Region", "ObservationDate"]).Confirmed.sum().reset_index()



# Take the top 12 countries excluding 'Others'

plt_df = plt_df[plt_df["Country/Region"] != "Others"]

plt_df = plt_df[plt_df["Country/Region"].isin(plt_df.groupby("Country/Region").Confirmed.max().sort_values().tail(12).index)]



# Remove dates with less than 100 cases reported, and then create a new time index of 'days since 100th case'

plt_df = plt_df[plt_df.Confirmed > 100]

plt_df = plt_df.groupby("Country/Region").apply(

    lambda df: pd.DataFrame({

        'DaysSince100Cases': (df.ObservationDate - df.ObservationDate.min()).values,

        'Confirmed': df.Confirmed.values, 

    })).reset_index()



# Remove durations more than two weeks after the last non-China data available 

# (this is to make the paths for remaining countries more visiable)

longest_ex_China = plt_df[plt_df['Country/Region'] != 'Mainland China'].DaysSince100Cases.max() 

plt_df = plt_df[plt_df.DaysSince100Cases <= longest_ex_China + dt.timedelta(days = 14)]

plt_df.head()
# Make a log scale plot of countries vs. days since 100th confirmed case 

p = (

    gg.ggplot(plt_df)

    + gg.geom_line(gg.aes(y="Confirmed", x="DaysSince100Cases", group="Country/Region", color="Country/Region"))

    + gg.geom_line(gg.aes(y="Confirmed", x="DaysSince100Cases", group="Country/Region", color="Country/Region"))

    + gg.theme_minimal()

    + gg.theme(plot_background = gg.element_rect(fill="white", alpha = 1), panel_grid = gg.element_blank(), axis_text_x=gg.element_text(angle=45))

    + gg.scale_y_log10(name = "Confirmed Cases", breaks=[100, 1000, 10000, 100000], labels=[100, 1000, 10000, 100000])

    + gg.scale_x_timedelta(name="Days since country first reached or exceeded 100 confirmed cases")

)

p
# To augment the chart with labels next to each line, create a second dataset for the labels

label_df = plt_df.groupby("Country/Region").apply(

    lambda df: pd.Series({

        'x': df.DaysSince100Cases[df.DaysSince100Cases.idxmax()],

        'y': df.Confirmed[df.DaysSince100Cases.idxmax()],

    })

)

    # starting x, y point for label is the last point on the curve

    # (guessing there's a cleaner way to do this)

    

label_df['x_label'] = (label_df.x + dt.timedelta(days = 3)).clip(upper = label_df.x.max() - dt.timedelta(days = 3))

    # spacing the x point a few days after the end point (to not overlap the actual data)

    # right most label no more than three days from rightmost point on chart





label_df = label_df.sort_values('y', ascending=False)

label_df["y_label"] = np.minimum(0.75, label_df.y/label_df.y.shift(1)).fillna(1).cumprod() * label_df.y.iloc[0]    

    # spacing the y points by a ratio of 0.75 to keep labels from overlapping

    

label_df = label_df.reset_index()

label_df
# Add the new labels to the plot, with connecting dotted line segments, and droping the legend 

new_p = (

    p

    + gg.geom_label(gg.aes(x="x_label", y="y_label", label="Country/Region", fill="Country/Region"), data=label_df, size=8)

    + gg.geom_segment(gg.aes(x="x", xend="x_label", y="y", yend="y_label", color="Country/Region"), data=label_df, linetype="dotted")

    + gg.scale_fill_discrete(guide=None)

    + gg.scale_color_discrete(guide=None)

)

new_p
