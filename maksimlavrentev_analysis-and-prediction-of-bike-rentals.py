# importing libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")

%matplotlib inline

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import KMeans

from itertools import combinations



df_raw = pd.read_csv("../input/bike-sharing-dataset/hour.csv", parse_dates = ["dteday"])

df_raw.head()
print("Number of rows: {}\nNumber of columns: {}".format(*df_raw.shape))
df = df_raw.drop(["casual", "registered"], axis = 1)
print("Num of duplicates: {}".format(df[["dteday", "hr"]].duplicated().sum()))

print("Num of nulls: {}".format(df.isnull().sum().sum()))
#Adding hours to dteday datetime column

import datetime

df.dteday = df.dteday + pd.to_timedelta(df.hr, unit = "h")



print("Years classification:")

pd.concat([df.dteday.dt.year, df.yr], axis = 1).drop_duplicates()
print("Number of errors in 'mnth' labels: {}".format((df.dteday.dt.month != df.mnth).sum()))
print('Number of logs by month by season')

df.pivot_table(index = 'season', columns = 'mnth', values = 'dteday', aggfunc = 'count').fillna(0).astype(int)
#creating DataFrame to help us find the dates where change of seasons happened

seasons_shifted = pd.concat([df.season, df.season.reindex(df.index - 1).reset_index(drop = True),

                             df.season.reindex(df.index + 1).reset_index(drop = True)],

                            join = "outer", axis = 1, ignore_index = True)



#finding seasons' date intervals

seasons_intervals_filter = (seasons_shifted[0] != seasons_shifted[1]) | (seasons_shifted[0] != seasons_shifted[2])

seasons_intervals_df = df.loc[seasons_intervals_filter,['dteday', 'season']].reset_index(drop = True)



int_start_ix = pd.RangeIndex(start = 0, stop = seasons_intervals_df.shape[0], step=2)

int_end_ix = pd.RangeIndex(start = 1, stop = seasons_intervals_df.shape[0], step=2)



seasons_intervals_start_df = seasons_intervals_df.loc[int_start_ix].reset_index(drop = True)

seasons_intervals_end_df = seasons_intervals_df.loc[int_end_ix].reset_index(drop = True)



seasons_intervals = seasons_intervals_start_df.merge(seasons_intervals_end_df, left_index=True, right_index=True, suffixes = ['_start','_end'])

seasons_intervals = seasons_intervals.reindex(['dteday_start','dteday_end', 'season_start'], axis = 1)

seasons_intervals.columns = ['dteday_start','dteday_end', 'season']

print('Classification of seasons based on date intervals')

seasons_intervals
df["weekday"] = df.dteday.dt.weekday
datetimes = pd.date_range(

    start=df.dteday.min(),

    end=df.dteday.max(),

    freq="1H",

    name="Datetime")

datetimes



missing_datetimes = ~datetimes.isin(df.dteday)

print('Numer of missing hourly logs in the data: {}'.format(missing_datetimes.sum()))
print('Number of cnt <= 0: {}'.format((df.cnt <= 0).sum()))
def missing_dates_neighbors(row):

    try:

        i_left = df.dteday[df.dteday < row].index[-1]

    except:

        i_left = np.nan

    try:

        i_right = df.dteday[df.dteday > row].index[0]

    except:

        i_right = np.nan

    return pd.Series({"left" : i_left,

                     "right": i_right})

md_series = pd.Series(datetimes[missing_datetimes], name = "Miss DT")

#md_series.apply(missing_dates_neighbors, axis = 1)



md_neighbors = md_series.apply(missing_dates_neighbors)

md_neighbors_counts = md_neighbors["left"].value_counts().sort_index()

missing_hrs = pd.concat([df.dteday,md_neighbors_counts],

          axis = 1, join = "inner").reset_index()

missing_hrs.columns = ["index_log_before","datetime_since","nr missing hrs"]

missing_hrs["datetime_since"] = missing_hrs["datetime_since"] + datetime.timedelta(hours = 1)

missing_hrs = missing_hrs.set_index("datetime_since")

missing_hrs['nr missing hrs'].value_counts().sort_index(ascending = False)
missing_hrs[missing_hrs['nr missing hrs'] <= 2].index.hour.value_counts().sort_index()
missing_hrs[missing_hrs["nr missing hrs"] > 2].sort_values(by = "nr missing hrs", ascending = False)
#we'll split in 4 subplots to better see the bars on the figure.

#As a split criteria, we'll use a middle point of a dteday field of df

min_date = df.dteday.dt.date.min()

max_date = df.dteday.dt.date.max()

center_date = min_date + (max_date - min_date)/2

second_quarter_date = min_date + (center_date - min_date)/2

third_quarter_date = center_date + (max_date - center_date)/2



dt_lims = [

    [min_date,second_quarter_date],

    [second_quarter_date,center_date],

    [center_date,third_quarter_date],

    [third_quarter_date,max_date + datetime.timedelta(days = 1)]

]



#aggregating rentals cnt at day level

series_for_trend_plot = df.set_index('dteday').resample('D').cnt.sum().asfreq('D')



#subplots generation

fig, axs = plt.subplots(nrows=len(dt_lims),figsize=(17,8), sharey = True)



#pandas needs this

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



#the bar chart will be on secondary y, on primary we'll place cnt trend

#let's create list for twin axs

axs2 = [i.twinx() for i in axs]

#set shared y for secondary y axes

axs2[0].get_shared_y_axes().join(*axs2)



for i in range(len(dt_lims)):

    

    trendline = series_for_trend_plot[dt_lims[i][0]:dt_lims[i][1]]

    missing_logs = missing_hrs.loc[dt_lims[i][0]:dt_lims[i][1],'nr missing hrs']



    #adding cnt trend on the primary y

    l = axs[i].plot(trendline.index,

                trendline,

                color = 'grey',

                alpha = 0.7,

                label = 'daily bike rentals'

               )

    

    #adding missing_logs bars on secondary y

    b = axs2[i].bar(missing_logs.index,

                missing_logs,

                color = 'orange',

                alpha = 0.7,

                label = 'number of missing hours - right axis (logarithmic)')



    #make 2nd y axis on log scale to increase visibility

    axs2[i].set_yscale('log')

    

    #adding legend just on the first plot

    if i == 0:

        l_b = l+[b]

        labels = [lab.get_label() for lab in l_b]

        axs2[i].legend(l_b, labels)



plt.suptitle("Daily bike rentals and number of consecutive hours without logs")

plt.show()
#Adding weekday and month to the tabme with missing hours

missing_hrs["wk_day"] = missing_hrs.index.weekday

missing_hrs["mnth"] = missing_hrs.index.month

missing_hrs[missing_hrs["nr missing hrs"] > 2]
ix_filter = missing_hrs[missing_hrs["nr missing hrs"] > 2].index_log_before

ix_filter = sorted(ix_filter.tolist() + (ix_filter + 1).tolist())

filtered = df.iloc[ix_filter].drop(["season","yr","mnth","hr"], axis = 1)

filtered
filtered["YYYYMMHH"] = (filtered.dteday.dt.year * 10000 + filtered.dteday.dt.month*100 + filtered.dteday.dt.hour) 

df1 = df.copy()

df1["YYYYMMHH"] = (df1.dteday.dt.year * 10000 + df1.dteday.dt.month*100 + df1.dteday.dt.hour)

df1 = df1.pivot_table(values = "cnt", index = "YYYYMMHH", aggfunc = np.mean).reset_index()

filtered.merge(df1,

               left_on = "YYYYMMHH",

               right_on = "YYYYMMHH",

               how = "inner")[["dteday","cnt_x","cnt_y"]]
import calendar



#we'll write a function to return slices of df with specified days before and after

def find_avg_trend(date_instant, days_before = 2, days_after = 2, df = df):



    date_instant_year = date_instant.year

    date_instant_month = date_instant.month

    date_instant_day = date_instant.day

    date_instant_weekday = date_instant.weekday()



    left = datetime.datetime(year = date_instant_year,

                             month = date_instant_month,

                             day = date_instant_day) - datetime.timedelta(days = days_before)



    monthrange = calendar.monthrange(date_instant_year,date_instant_month)



    next_month = (datetime.datetime(year = date_instant_year, month = date_instant_month, day = monthrange[1])

                  + datetime.timedelta(days = 1))



    monthrange_series = pd.date_range(start = datetime.datetime(year = date_instant.year,

                                                month = date_instant.month,

                                                day = 1),

                                      end = next_month,

                                      closed = 'left',

                                      freq = '1D')

    

    left_similar = list(monthrange_series[monthrange_series.weekday == left.weekday()])



    df_segments = []

    df_frequency = df.reset_index().set_index('dteday').asfreq('1H').sort_index()



    for l in left_similar:



        r = l + datetime.timedelta(hours = (days_before + days_after + 1)*24-1)

        df_segment = df_frequency[l:r].reset_index().set_index('index')

        df_segments.append(df_segment)



    main_period_df = df_segments.pop(left_similar.index(left))

    return(main_period_df, df_segments)
dates_before_mis_logs = df.iloc[missing_hrs[missing_hrs["nr missing hrs"] > 2].index_log_before]["dteday"].copy()

fig, axs = plt.subplots(nrows = dates_before_mis_logs.shape[0], figsize = (17,8), sharex = True)



for i in range(dates_before_mis_logs.shape[0]):

    

    slice_main, slices_other = find_avg_trend(dates_before_mis_logs.iloc[i], days_before = 2, days_after = 2, df = df)

    main_period_cnt = slice_main.cnt.values

    other_periods_cnt_mean = np.mean([table.cnt.fillna(0).values for table in slices_other], axis=0)

    

    axs[i].plot(main_period_cnt, color = 'grey', label = 'Nr rentals')

    axs[i].plot(other_periods_cnt_mean, color = 'orange', alpha = 0.5, label = 'Mean nr rentals in similar weekdays of the month')

    axs[i].set_title('Missing logs start at: ' + str(dates_before_mis_logs.iloc[i]))

    

    if i == 0:

        axs[i].legend()

plt.suptitle("Bike rentals trend 2 days before and 2 days after the start of top interruptions")

plt.xlabel('hr since period start')

plt.show()
fig, axs = plt.subplots(nrows = dates_before_mis_logs.shape[0], figsize = (17,8), sharex = True, sharey = True)



for i in range(dates_before_mis_logs.shape[0]):

    

    slice_main, slices_other = find_avg_trend(dates_before_mis_logs.iloc[i], days_before = 2, days_after = 2, df = df)

    main_period_temp = slice_main.temp.values

    main_period_hum = slice_main.hum.values

    main_period_ws = slice_main.windspeed.values



    axs[i].plot(main_period_temp, color = 'red', alpha = 0.5, label = 'Relative temperature')

    axs[i].plot(main_period_hum, color = 'green', alpha = 0.5, label = 'Relative humidity')

    axs[i].plot(main_period_ws, color = 'blue', alpha = 0.5, label = 'Relative windspeed')

    axs[i].set_title('Missing logs start at: ' + str(dates_before_mis_logs.iloc[i]))

    

    if i == 0:

        axs[i].legend()

plt.suptitle("Temperature and humidity trend 2 days before and 2 days after the start of top interruptions")

plt.xlabel('hr since period start')

plt.show()
df['instant'] = df.set_index('dteday').asfreq('1H').reset_index().dropna().index

df
df_for_trend = df.copy().set_index("dteday")



#plotting lines

ax = df_for_trend.cnt.plot(figsize = (17,7), alpha = 0.2, color = 'grey', linewidth = 0.5, label = 'Hourly rentals')

df_for_trend.resample('D').mean().cnt.plot(ax = ax, alpha = 0.4, linewidth = 0.8, color = "red", label = 'Mean hourly rentals by day')

df_for_trend.resample('W').mean().cnt.plot(ax = ax, alpha = 0.4, linewidth = 1, color = "red", marker = '.', label = 'Mean hourly rentals by week')

df_for_trend.rolling('56D').mean().cnt.plot(ax = ax, alpha = 0.4, linewidth = 1, color = "black", label = '56D rolling avg hourly rentals')



#plotting trendline

lr_matrix = pd.concat([pd.Series(1, index=df_for_trend.index), df_for_trend.instant], axis = 1)

X = lr_matrix

Y = df_for_trend.cnt



coeffs = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T, Y))

Y1 = pd.Series(np.dot(X, coeffs.T), index = df_for_trend.index,name = 'lin_approx')

Y1.plot(ax = ax, alpha = 0.6, linestyle = "--", color = "orange",

        label = "Linear Approx.\nIntercept: {:.3f}\nSpline: {:.3f}".format(coeffs[0],coeffs[1]))



ax.legend()

plt.show()
fig, ax = plt.subplots(figsize = (17,7))

sns.boxplot(x=df_for_trend.index.to_period('M'), y = df_for_trend["cnt"], ax = ax, color = "white")

plt.xticks(rotation = 45)

plt.title("Daily bike rentals distribution by month")

ax.set_xlabel ("Month")

plt.show()
seasonality_c = (Y / Y1)



seasonality_yearly = seasonality_c.groupby(seasonality_c.index.month).mean()

seasonality_wd = seasonality_c.groupby(seasonality_c.index.weekday).mean()

seasonality_d = seasonality_c.groupby(seasonality_c.index.hour).mean()



seasonality_srs = [seasonality_yearly,seasonality_wd,seasonality_d]

lbls = ['month','weekday','hour']

fig,axs = plt.subplots(ncols = 3,figsize = (15,5), sharey = True)



for i in range(3):

    seasonality_srs[i].plot(ax = axs[i], legend = False, color = "grey", alpha = 0.5, marker = '.')

    axs[i].set_xlabel(lbls[i])

axs[0].set_ylabel("Seasonality coefficient")

plt.suptitle("Seasonalities")



plt.show()
seasonality_c1 = (Y / df_for_trend.rolling('14D').mean().cnt)

seasonality_c1.name = 'seasonality_c'

df_for_trend1 = pd.concat([df_for_trend,seasonality_c1], axis = 1)



wh_seasonalities = df_for_trend1.groupby(['season','workingday',df_for_trend1.index.hour.rename('hr')]).cnt.mean()

wh_seasonalities_pivot = wh_seasonalities.reset_index().pivot_table(values = 'cnt', columns = ['workingday', 'season'], index = 'hr')



fig,axs = plt.subplots(ncols = 4, nrows = 2, figsize = (17,7), sharey = True, sharex = True)



for i in range(2):

    wh_seasonalities_pivot[i].plot(subplots = True, ax = axs[i-1], legend = False, color = "grey", alpha = 0.5, marker = '.')

    axs[i-1, 0].set_ylabel("Workingday = {}".format(i))

for i in range (1,5):

    axs[0,i-1].set_title("Season = {}".format(i))

plt.suptitle("Mean hourly bike rentals in different seasons")

plt.show()
df1 = df_for_trend

df1['daytime_lbl'] = "[0:6)"

df1.loc[(df1.index.hour >= 6) & (df1.index.hour < 9), 'daytime_lbl'] = "[6:9)"

df1.loc[(df1.index.hour >= 9) & (df1.index.hour < 12), 'daytime_lbl'] = "[9:12)"

df1.loc[(df1.index.hour >= 12) & (df1.index.hour < 16), 'daytime_lbl'] = "[12:16)"

df1.loc[(df1.index.hour >= 16) & (df1.index.hour < 20), 'daytime_lbl'] = "[16:20)"

df1.loc[(df1.index.hour >= 20) & (df1.index.hour < 24), 'daytime_lbl'] = "[20:23]"



fig, axs = plt.subplots(ncols = 2, figsize = (12,5), sharey = True)

c = sns.color_palette("BrBG", 6)

sns.boxplot(data = df1, x = "daytime_lbl", y = "cnt", ax = axs[0], palette = c)

axs[0].set_title("Distribution of bikerental count split by daytime label")



df2 = df1.pivot_table(values = "cnt", columns = "daytime_lbl", index = df1.index.to_period('D'))

df2.plot(marker='.', alpha=0.7, linestyle='None', ax = axs[1], color = c)

axs[1].set_title("Rentals by day hour category")

plt.show()
fig,axs = plt.subplots(ncols = 2, figsize = (12,5))

c = sns.color_palette("colorblind", 4)

season_weather_ctab = pd.crosstab(df1['season'],

                                   df1['weathersit'],

                                   values = df1['cnt'],

                                   aggfunc = [np.size,np.mean]).fillna(0).astype(int)

season_weather_ctab['size'].plot.bar(alpha = 0.6, color = c, logy = True, ax = axs[0],

                                     title = 'Number of hours with different weather situations', legend = False, rot = 0)

season_weather_ctab['mean'].plot.bar(alpha = 0.6, color = c, ax = axs[1],

                                     title = 'Mean bike rentals in different weather situations', rot = 0),

plt.show()
#instantiating subplots

fig,axs = plt.subplots(ncols = 4, nrows = 2, figsize = (16,8), sharex = True)



#set shared y

axs[0][0].get_shared_y_axes().join(*axs[0])

axs[1][0].get_shared_y_axes().join(*axs[1])



#some labels for text on plots

seasons_labels = sorted(df1['season'].unique().tolist())

weathersit_labels = sorted(df1['weathersit'].unique().tolist())



for i in range(len(seasons_labels)):

    

    #weight based on share of weathersit in season

    weights = df1[df1['season'] == seasons_labels[i]]['weathersit'].value_counts(normalize = 1)

        

    for j in range(len(weathersit_labels)):

        df1_ij_indexer = ((df1['weathersit'] == weathersit_labels[j]) & 

                          (df1['season'] == seasons_labels[i]))



        #if there is no weathersit label in season

        #there's nothing to plot

        if df1_ij_indexer.sum() > 0:

            df_ij = df1[df1_ij_indexer]

            df_ij_freq = df_ij.index.hour.value_counts(normalize = True).sort_index()

            df_ij_weighted = df_ij_freq*weights[weathersit_labels[j]]

            df_ij_mean_cnt = df_ij.groupby(df_ij.index.hour)['cnt'].mean()

            

            l1 = axs[0][i].plot(df_ij_weighted,

                             label = 'Weathersit = {}'.format(weathersit_labels[j]),

                             color = c[j],

                             alpha = 0.6)

            

            axs[1][i].plot(df_ij_mean_cnt,

                             label = 'Weathersit = {}'.format(weathersit_labels[j]),

                             color = c[j],

                             alpha = 0.6)



    #adding text and formatting    

    axs[0][i].set_title('Season = {}'.format(seasons_labels[i])) 

    axs[1][i].set_xlabel('hr')



    if i != 0:

        axs[0][i].yaxis.set_visible(False)

        axs[1][i].yaxis.set_visible(False)



#some other formatting

axs[0][1].set_xlim((0,23))

axs[0][0].set_ylabel('Weathersit distribution')

axs[1][0].set_ylabel('Mean rentals')

    

# Put a legend below

labels = [lab.get_label() for lab in axs[0][0].lines]



fig.legend(axs[0][0].lines, labels, bbox_to_anchor=(0.43, 0), loc='lower center',ncol=5)



fig.suptitle('Weathersit distribution by hour vs mean rentals')

plt.show()
fig, axs = plt.subplots(ncols = 2, figsize = (17,7), sharey = True)

sns.violinplot(data = df1, x = "weathersit", y = "cnt", ax = axs[0], palette = c)



df.pivot_table(values = "cnt", columns = "weathersit", index = "dteday").plot(marker = ".",

                                                                              linestyle = "None",

                                                                              alpha = 0.3,

                                                                              figsize = (17,7),

                                                                              color = c,

                                                                              ax = axs[1])

fig.suptitle("Distributions of hourly bike rentals in different weather situations")

plt.show()
fig,axs = plt.subplots(ncols = 3, nrows = 2, figsize = (15,10))

weather_continuous_cols = ["temp", "atemp","hum","windspeed"]

weather_cols_comb = combinations(weather_continuous_cols, 2)

weather_cols_comb = [list(i) for i in list(weather_cols_comb)]



centers = df1.pivot_table(values = weather_continuous_cols, index = "weathersit").sort_index()



for i in range(len(weather_cols_comb)):

    if i > 0:

        legend = False

    else:

        legend = "full"

    sns.scatterplot(data = df1,

                    x = weather_cols_comb[i][0],

                    y = weather_cols_comb[i][1],

                    hue = "weathersit",

                    alpha = 0.3,

                    ax = axs.flatten()[i],

                    palette = c,

                    marker = "D",

                    s = 10,

                    legend = False)

    sns.scatterplot(data = centers,

                    x = weather_cols_comb[i][0],

                    y = weather_cols_comb[i][1],

                    hue = centers.index,

                    ax = axs.flatten()[i],

                    palette = c,

                    legend = legend,

                    marker = "P",

                    s = 200)

fig.suptitle("Analysis of weathersit label vs weather parameters\n('+' represents centers of clusters)")

plt.show()
fig,axs = plt.subplots(ncols = 4, figsize = (16,4), sharey = True, sharex = True)

for i in range(len(weather_continuous_cols)):

    if i == len(weather_continuous_cols) - 1:

        legend_visible = True

    else:

        legend_visible = False

    for j in range(len(weathersit_labels)):

        df_ij = df1[df1['weathersit'] == weathersit_labels[j]][weather_continuous_cols[i]]

        sns.kdeplot(df_ij,

                    ax = axs[i],

                    label = 'Weathersit = {}'.format(weathersit_labels[j]),

                    color = c[j],

                    legend = legend_visible,

                    alpha = 0.6)

        axs[i].set_xlabel(weather_continuous_cols[i])

        axs[i].set_xlim((0,1))

fig.suptitle('Weather parameters distributions in different weather situations')

plt.show()
weather_ranges = []

for i in weather_continuous_cols:

    ranges = pd.cut(df1[i], bins=np.arange(0,1.01,0.05),include_lowest = True)

    mids = pd.Series(pd.IntervalIndex(ranges).mid, name = i, index = df1.index)

    weather_ranges.append(mids)



mean_cnt_by_weather = pd.concat(weather_ranges +[df1.cnt], axis = 1)
from scipy.stats import gaussian_kde as kde

cols_num = len(weather_continuous_cols)



fig,axs = plt.subplots(ncols = cols_num, figsize = (16,4), sharex = True, sharey = True)



#setting shared secondary y

axs2 = [i.twinx() for i in axs]

axs2[0].get_shared_y_axes().join(*axs2)



#instantiate X_mash and freq for kdes

X_mash = np.mgrid[0:1:.05]

freq = df1.cnt



#start plotting

for i in range(cols_num):

    #primary axis - mean

    XY = mean_cnt_by_weather.groupby(weather_continuous_cols[i]).cnt.mean()

    XY.plot(ax = axs[i],

            color = 'grey',

            alpha = 0.8,

            label = 'Mean rentals')

    

    #2nd axis - kde

    X1_freq = mean_cnt_by_weather[weather_continuous_cols[i]].value_counts(normalize = 1).sort_index()

    axs2[i].fill_between(X1_freq.index,X1_freq,

                        color = 'silver',

                        alpha = 0.1,

                        label = 'Perameter frequency')

            

    #2nd axis - kde

    X2_freq = mean_cnt_by_weather[weather_continuous_cols[i]].repeat(mean_cnt_by_weather.cnt).value_counts(normalize = 1).sort_index()

    axs2[i].fill_between(X2_freq.index,X2_freq,

                 color = 'orange',

                 alpha = 0.1,

                 label = 'Rentals frequency')

    

    #add xlabel

    axs[i].set_xlabel(weather_continuous_cols[i])

axs[0].set_xlim((0,1))

#add ylabel

axs[0].set_ylabel('Mean rentals')

axs2[-1].set_ylabel('Density')

#add legend

handles, labels = axs[0].get_legend_handles_labels()

handles1, labels2 = axs2[0].get_legend_handles_labels()

axs[-1].legend(handles + handles1, labels + labels2, loc = 'upper right')

plt.suptitle('Mean hourly bike rentals variation with weather parameters')

plt.show()
#let's aggregate on higher level

weather_ranges = []

for i in weather_continuous_cols:

    ranges = pd.cut(df1[i], bins=np.arange(0,1.01,0.07),include_lowest = True)

    mids = pd.Series(pd.IntervalIndex(ranges).mid, name = i, index = df1.index)

    weather_ranges.append(mids)

#let's plot without atemp

mean_cnt_by_weather1 = pd.concat(weather_ranges +[df1.cnt], axis = 1)

weather_cols_wo_atemp = [i for i in weather_continuous_cols if i != 'atemp']

cols_num = len(weather_cols_wo_atemp)



weather_cols_wo_atemp_comb = combinations(weather_cols_wo_atemp, 2)

weather_cols_wo_atemp_comb = [list(i) for i in list(weather_cols_wo_atemp_comb)]

cols_comb_num = len(weather_cols_wo_atemp_comb)



fig,axs = plt.subplots(ncols = cols_comb_num, nrows = 1,

                       figsize = (15,4), sharex = True, sharey = True)



#start plotting

for i in range(cols_comb_num):

    Z = mean_cnt_by_weather1.pivot_table(index = weather_cols_wo_atemp_comb[i][1],

                               columns = weather_cols_wo_atemp_comb[i][0],

                               values = 'cnt',

                               aggfunc = sum)

    Z_flat = Z.melt()

    vmin = Z_flat[Z_flat.value != 0].value.quantile(0.3)

    axs[i].contour(Z.columns,

                 Z.index,

                 Z,

                 colors='k',

                 alpha = 0.5,

                 linewidths=0.2,

                 levels=10,

                 vmin = vmin)

    axs[i].contourf(Z.columns,

                     Z.index,

                     Z,

                     cmap="Greys",

                     alpha = 0.7,

                     levels=10,

                     vmin = vmin)

    axs[i].set_xlabel(weather_cols_wo_atemp_comb[i][0])

    axs[i].set_ylabel(weather_cols_wo_atemp_comb[i][1])

plt.suptitle("Bike rentals density at combination of weather parameters")

plt.show()
from scipy.stats import mode

print('Bike rentals weather stats:')

for i in weather_continuous_cols:

    weather_params = mean_cnt_by_weather1[i].repeat(mean_cnt_by_weather1.cnt)

    print(i,

          '-- mode: ',

          round(mode(weather_params)[0][0],3),

          'median: ',

          round(weather_params.median(),3))

print()

weather_params1 = mean_cnt_by_weather1.groupby(weather_continuous_cols).cnt.sum().sort_values(ascending = False)

weather_params_top = weather_params1[weather_params1 >= weather_params1.quantile(0.99)].reset_index()

print('Top 1% of rentals happened at:')

for i in weather_continuous_cols:

    print(i, round(weather_params_top[i].min(),3), '-', round(weather_params_top[i].max(),3))
df_clustering = df1[weather_continuous_cols + ['cnt']].copy()

# normalizing weathersit

df_clustering['weathersit'] = (df1['weathersit'] - df1['weathersit'].min())/df1['weathersit'].max()
from sklearn.metrics import silhouette_score

n_clusters = 72



#let's track the time it takes for clusterization cycles

import time

start_time = time.time()



km = KMeans(n_clusters=n_clusters, random_state = 0, algorithm = "full").fit(df_clustering[weather_continuous_cols

                                                                                           + ['weathersit']])



#print time in seconds

print("--- {:.2f} seconds ---".format(time.time() - start_time))
df_clustering["wthr_lbl"] = km.labels_



#as new labels are random, let's sort them by mean temperature of the cluster in ascending order 

#this should help in visualization

lbl_ord = list(km.cluster_centers_[:,0])

Z = [x for _,x in sorted(zip(lbl_ord,list(range(n_clusters))))]



#let's assign the ordered labels to a new column

df_clustering["new_wthr_lbl_ordered_temp"] = np.nan



for i in range(n_clusters):

    df_clustering.loc[df_clustering["wthr_lbl"] == i,"new_wthr_lbl_ordered_temp"] = Z.index(i)
c = sns.color_palette("RdBu_r", n_clusters)



fig,axs = plt.subplots(ncols = 3, nrows = 2, figsize = (15,10))



centers = pd.DataFrame(km.cluster_centers_, columns = weather_continuous_cols + ['weathersit']).sort_values(by = "temp")



for i in range(len(weather_cols_comb)):

    sns.scatterplot(data = df_clustering,

                    x = weather_cols_comb[i][0],

                    y = weather_cols_comb[i][1],

                    hue = "new_wthr_lbl_ordered_temp",

                    alpha = 0.3,

                    ax = axs.flatten()[i],

                    palette = c,

                    marker = "D",

                    s = 10,

                    legend = False)

    sns.scatterplot(data = centers,

                    x = weather_cols_comb[i][0],

                    y = weather_cols_comb[i][1],

                    hue = centers.index,

                    ax = axs.flatten()[i],

                    palette = c,

                    legend = False,

                    marker = "P",

                    s = 150)

plt.suptitle("New weather clusters analysis\n('+' represents centers of clusters)")

plt.show()
#to simplify, we will create a table with cnt field of new labels in separate columns

k1 = df_clustering.pivot_table(values = 'cnt', columns = "new_wthr_lbl_ordered_temp", index = df_clustering.index)



fig, axs = plt.subplots(ncols = 2, nrows = 2, figsize = (17,10), sharey = True, sharex = True)

for i in range(1,5):

    k1[df1['weathersit'] == i].plot(marker = ".",

                                    linestyle = "None",

                                    alpha = 0.5,

                                    ax = axs.flatten()[i-1],

                                    color = c,

                                    legend = False,

                                    rot = 0)

    axs.flatten()[i-1].set_title('Weathersit = {}'.format(i))

axs.flatten()[0].set_xlim((k1.index.min(),k1.index.max()))

plt.suptitle("New weather labels ordered by temperature on time series")

plt.show()
new_lbl_sort_cnt = df_clustering.groupby("new_wthr_lbl_ordered_temp").cnt.mean().sort_values().reset_index()["new_wthr_lbl_ordered_temp"].astype(int)

new_lbl_sort_cnt = new_lbl_sort_cnt.reset_index()

new_lbl_sort_cnt = new_lbl_sort_cnt.set_index(new_lbl_sort_cnt['new_wthr_lbl_ordered_temp']).iloc[:,0].rename('new_wthr_lbl_ordered_cnt')

new_lbl_sort_cnt.sort_index()

df_clustering = df_clustering.merge(new_lbl_sort_cnt,left_on = 'new_wthr_lbl_ordered_temp', right_index = True)
fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (17,7), sharex = True, sharey = True)



labels = ["new_wthr_lbl_ordered_temp", "new_wthr_lbl_ordered_cnt"]



for j in range(2):

    for i in range(1,5):

        df_index_test = ((df1['weathersit'] == i))

        df_test = df_clustering[df_index_test][labels[j]].value_counts()

        ax[j].bar(df_test.index, df_test,

                  color = sns.color_palette("colorblind", 4)[i-1],

                  alpha = 0.5)

    ax2 = ax[j].twinx()

    df_clustering.groupby(labels[j]).cnt.mean().plot(marker = ".",

                                                     ax = ax2,

                                                     color = 'grey')

    ax[j].set_ylim((0,600))

    ax[j].set_title(labels[j])

    ax[j].set_xlabel('label')

ax[0].set_ylabel('Number of labeled items')

ax2.set_ylabel('Mean rentals')

plt.suptitle('Reordering weather labels')

ax[0].set_xlim((-1, n_clusters))

plt.show()
c2 = sns.color_palette("summer", n_clusters)

k2 = df_clustering.pivot_table(values = 'cnt', columns = "new_wthr_lbl_ordered_cnt", index = df_clustering.index)



fig, axs = plt.subplots(ncols = 2, nrows = 2, figsize = (17,10), sharey = True, sharex = True)

for i in range(1,5):

    k1[df1['weathersit'] == i].plot(marker = ".",

                                    linestyle = "None",

                                    alpha = 0.5,

                                    ax = axs.flatten()[i-1],

                                    color = c2,

                                    legend = False,

                                    rot = 0)

    axs.flatten()[i-1].set_title('Weathersit = {}'.format(i))

axs.flatten()[0].set_xlim((k1.index.min(),k1.index.max()))

plt.suptitle("New weather labels ordered by count on time series")

plt.show()
# adding anomaly label

anomaly_dates = filtered.dteday.dt.date

anomaly_dates_index = df1[pd.Series(df1.index.date, index = df1.index).isin(anomaly_dates)].index

df1['anomaly'] = 0

df1.loc[anomaly_dates_index, 'anomaly'] = 1



# adding weather label

df1['weather_lbl'] = df_clustering["new_wthr_lbl_ordered_cnt"].astype(int)



# converting daytime label to category

df1['daytime_lbl'] = df1['daytime_lbl'].astype("category")
df2 = pd.get_dummies(df1, columns = ['weathersit'], drop_first = True, prefix = 'weathersit')
df2 = pd.get_dummies(df2, columns = ['daytime_lbl'], drop_first = True, prefix = 'daytime_lbl')
calendar_cols = ['instant', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',

       'workingday', "daytime_lbl_[12:16)", "daytime_lbl_[16:20)",

                 "daytime_lbl_[20:23]", "daytime_lbl_[6:9)", "daytime_lbl_[9:12)"]

weather_cols = ['temp', 'atemp', 'hum', 'windspeed', "weather_lbl", 'weathersit_2','weathersit_3','weathersit_4']

drop = ['anomaly']

target_col = "cnt"



columns_sorted = calendar_cols + weather_cols + drop + [target_col]

dteday = df2.index

df_for_regression = df2[columns_sorted].reset_index(drop = True)
corr = df_for_regression.corr()

f, ax = plt.subplots(figsize=(7,7))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.triu(np.ones_like(corr, dtype=np.bool))

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.1, cbar_kws={"shrink": .7}, ax = ax)

ax.set_title("Fields correlation")

plt.show()

print("correlation of bike rentals (cnt field) with other fields")

print(corr[target_col].sort_values(ascending = False))
#dropping anomalies

df_for_regression = df_for_regression.drop(df_for_regression.loc[(df_for_regression['anomaly'] == 1)].index).drop(drop, axis = 1)

#resetting index for lin reg to work

df_for_regression = df_for_regression.reset_index(drop = True)
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error



kf = KFold(n_splits=5, random_state = 0)



target = target_col

feats = df_for_regression.columns.drop(target).tolist()

len(feats)



# train = df.sample(frac = 0.8, random_state = 1)

# test = df[~df.index.isin(train.index)]

# linear regression

lr_rmses = []

for train_index, test_index in kf.split(df_for_regression):

    train_x = df_for_regression.reindex(index = train_index, columns = feats)

    train_y = df_for_regression.reindex(index = train_index, columns = [target])

    test_x = df_for_regression.reindex(index = test_index, columns = feats)

    test_y = df_for_regression.reindex(index = test_index, columns = [target])



    lr = LinearRegression()

    lr.fit(train_x,train_y)

    lr_prediction = lr.predict(test_x)



    lr_rmses.append(np.sqrt(mean_squared_error(test_y, lr_prediction)))

lr_rmse = np.mean(lr_rmses)

print(lr_rmses)

print(lr_rmse)
null_prediction_rmses = []

for train_index, test_index in kf.split(df_for_regression):

    train_y = df_for_regression.reindex(index = train_index, columns = [target])

    test_y = df_for_regression.reindex(index = test_index, columns = [target])

    null_prediction_rmses.append(np.sqrt(mean_squared_error(test_y, [train_y.mean() for i in range(len(test_y))])))

null_prediction_rmse = np.mean(null_prediction_rmses)

print(null_prediction_rmses)

print(null_prediction_rmse)
comb = []

for j in range(18, len(feats)+1):

    comb += [list(i) for i in list(combinations(feats,j))]

len(comb)
rmse_df = pd.DataFrame(index = [str(i) for i in comb], columns = ["LR"])

start_time = time.time()

for feat_comb in comb:

    lr_rmses = []

    for train_index, test_index in kf.split(df_for_regression):

        

        train_x = df_for_regression.reindex(index = train_index, columns = feat_comb)

        train_y = df_for_regression.reindex(index = train_index, columns = [target])

        test_x = df_for_regression.reindex(index = test_index, columns = feat_comb)

        test_y = df_for_regression.reindex(index = test_index, columns = [target])



        lr = LinearRegression()

        lr.fit(train_x,train_y)

        lr_prediction = lr.predict(test_x)



        lr_rmses.append(np.sqrt(mean_squared_error(test_y, lr_prediction)))

    lr_rmse = np.mean(lr_rmses)

    rmse_df.loc[str(feat_comb), "LR"] = lr_rmse

print("--- {:.2f} seconds ---".format(time.time() - start_time))
rmse_df["LR"].sort_values().head(1)
import ast

list(set(feats) - set(ast.literal_eval(rmse_df["LR"].sort_values().index[0])))
#we'll convert all datetime variables that hide seasonality to dummies

vars_seasonality = ['season', 'mnth', 'weekday','hr']

df_for_regression_dummies = pd.get_dummies(df_for_regression, columns = vars_seasonality,

               drop_first = True, prefix = vars_seasonality)

#we dont need daytime labels now

df_for_regression_dummies = df_for_regression_dummies.drop(['daytime_lbl_[12:16)', 'daytime_lbl_[16:20)', 'daytime_lbl_[20:23]', 'daytime_lbl_[6:9)', 'daytime_lbl_[9:12)'], axis = 1)

feats1 = df_for_regression_dummies.columns.drop(target)
lr_rmses = []

for train_index, test_index in kf.split(df_for_regression_dummies):

    train_x = df_for_regression_dummies.reindex(train_index)[feats1]

    train_y = df_for_regression_dummies.reindex(train_index)[target]

    test_x = df_for_regression_dummies.reindex(test_index)[feats1]

    test_y = df_for_regression_dummies.reindex(test_index)[target]



    lr = LinearRegression()

    lr.fit(train_x,train_y)

    lr_prediction = lr.predict(test_x)



    lr_rmses.append(np.sqrt(mean_squared_error(test_y, lr_prediction)))

lr_rmse = np.mean(lr_rmses)

print(lr_rmses)

print(lr_rmse)
lr_rmses1 = []

for train_index, test_index in kf.split(df_for_regression_dummies):

    

    #splitting to train and test

    train_x = df_for_regression_dummies.reindex(train_index)[feats1]

    test_x = df_for_regression_dummies.reindex(test_index)[feats1]

    train_y = df_for_regression_dummies.reindex(train_index)[target]

    test_y = df_for_regression_dummies.reindex(test_index)[target]



    #pulling the trend

    lr = LinearRegression()

    lr.fit(train_x['instant'].values.reshape(-1,1),train_y)

    

    linear_approx_s1 = lr.predict(train_x['instant'].values.reshape(-1,1))

    linear_approx_s2 = lr.predict(test_x['instant'].values.reshape(-1,1))



    

    #pulling trend-corrected prediction

    train_x1 = train_x[list(set(feats1) - set('instant'))]

    test_x1 = test_x[list(set(feats1) - set('instant'))]

    train_y1 = train_y / linear_approx_s1

    test_y1 = test_y / linear_approx_s2



    lr1 = LinearRegression()

    lr1.fit(train_x1,train_y1)

    

    trend_corrected_approx_s1 = lr1.predict(train_x1)

    trend_corrected_approx_s2 = lr1.predict(test_x1)

    

    #combining the predictions

    total_approx_s2 = linear_approx_s2 * trend_corrected_approx_s2

    

    #calculating rmses

    rmse = np.sqrt(mean_squared_error(test_y, total_approx_s2))

    lr_rmses1.append(rmse)



lr_rmse1 = np.mean(lr_rmses1)

print(lr_rmses1)

print(lr_rmse1)
#we have done several manipulations with df, and now it is not in best shape for decision tree

#let's recompile it

df_for_trees = df_for_regression.merge(df1[['weathersit','daytime_lbl']], left_index = True, right_on = df1.reset_index().index, how = 'inner')

df_for_trees = df_for_trees.drop(['daytime_lbl_[12:16)', 'daytime_lbl_[16:20)', 'daytime_lbl_[20:23]', 'daytime_lbl_[6:9)', 'daytime_lbl_[9:12)'], axis = 1)

df_for_trees = df_for_trees.drop(['key_0','weathersit_2', 'weathersit_3', 'weathersit_4'], axis = 1)

df_for_trees = df_for_trees.reset_index(drop = True)

df_for_trees['daytime_lbl'] = df_for_trees['daytime_lbl'].cat.codes

feats_dt = df_for_trees.columns.drop(target)
dt_rmses_train = []

dt_rmses_test = []

for train_index, test_index in kf.split(df_for_trees):

    train_x = df_for_trees.reindex(train_index)[feats_dt]

    train_y = df_for_trees.reindex(train_index)[target]

    test_x = df_for_trees.reindex(test_index)[feats_dt]

    test_y = df_for_trees.reindex(test_index)[target]



    dt = DecisionTreeRegressor(random_state = 0)

    dt.fit(train_x,train_y)

    dt_prediction_train = dt.predict(train_x)

    dt_prediction_test = dt.predict(test_x)



    dt_rmses_train.append(np.sqrt(mean_squared_error(train_y, dt_prediction_train)))

    dt_rmses_test.append(np.sqrt(mean_squared_error(test_y, dt_prediction_test)))

    

dt_rmse_train = np.mean(dt_rmses_train)

dt_rmse_test = np.mean(dt_rmses_test)



print('Train error:')

print(dt_rmses_train)

print(dt_rmse_train)

print()

print('Test error:')

print(dt_rmses_test)

print(dt_rmse_test)
lf = np.arange(1,51, 5)

spl = np.arange(2,51, 5)



spl_lf_df = pd.DataFrame(columns = lf, index = spl)



start_time = time.time()

for i in spl:

    for j in lf:

        dt_rmses_train = []

        dt_rmses_test = []

        for train_index, test_index in kf.split(df_for_trees):

            train_x = df_for_trees.reindex(train_index)[feats_dt]

            train_y = df_for_trees.reindex(train_index)[target]

            test_x = df_for_trees.reindex(test_index)[feats_dt]

            test_y = df_for_trees.reindex(test_index)[target]



            dt = DecisionTreeRegressor(min_samples_split = i,

                                       min_samples_leaf = j, random_state = 0)

            dt.fit(train_x,train_y)

            dt_prediction_train = dt.predict(train_x)

            dt_prediction_test = dt.predict(test_x)



            dt_rmses_train.append(np.sqrt(mean_squared_error(train_y, dt_prediction_train)))

            dt_rmses_test.append(np.sqrt(mean_squared_error(test_y, dt_prediction_test)))



        dt_rmse_train = np.mean(dt_rmses_train)

        dt_rmse_test = np.mean(dt_rmses_test)

        spl_lf_df.loc[i,j] = (round(dt_rmse_test,2),round(dt_rmse_train,2))

print("--- {:.2f} seconds ---".format(time.time() - start_time))
spl_lf = spl_lf_df.reset_index().melt(id_vars = "index")

spl_lf = pd.concat([spl_lf,pd.DataFrame(spl_lf['value'].tolist(), index=spl_lf.index)], axis = 1)

spl_lf = spl_lf.drop("value", axis = 1)

spl_lf.columns = ["min_samples_split", "min_samples_leaf", "rmse_test","rmse_train"]

spl_lf ["abs_diff"] = (spl_lf["rmse_test"] - spl_lf["rmse_train"]).abs()
print('Optimal parameters (top 3):')

print(spl_lf.sort_values(by = ["abs_diff","rmse_test",

                               "min_samples_leaf","min_samples_split"]).head(3))

print()

print('Lowest rmse for test set parameters (top 3):')

print(spl_lf.sort_values(by = ["rmse_test","min_samples_leaf","min_samples_split"]).head(3))
def combined_lin_dt(spl = 2, lf = 1):

    dt_rmses_train = []

    dt_rmses_test = []

    for train_index, test_index in kf.split(df_for_regression_dummies):



        #splitting to train and test

        train_x = df_for_regression_dummies.reindex(train_index)[feats1]

        test_x = df_for_regression_dummies.reindex(test_index)[feats1]

        train_y = df_for_regression_dummies.reindex(train_index)[target]

        test_y = df_for_regression_dummies.reindex(test_index)[target]



        #pulling the trend

        lr = LinearRegression()

        lr.fit(train_x['instant'].values.reshape(-1,1),train_y)



        linear_approx_s1 = lr.predict(train_x['instant'].values.reshape(-1,1))

        linear_approx_s2 = lr.predict(test_x['instant'].values.reshape(-1,1))





        #pulling trend-corrected prediction

        train_x1 = train_x[list(set(feats1) - set('instant'))]

        test_x1 = test_x[list(set(feats1) - set('instant'))]

        train_y1 = train_y / linear_approx_s1

        test_y1 = test_y / linear_approx_s2



        dt = DecisionTreeRegressor(min_samples_split = spl,

                                   min_samples_leaf = lf,

                                   random_state = 0)

        dt.fit(train_x1,train_y1)

        trend_corrected_approx_s1 = dt.predict(train_x1)

        trend_corrected_approx_s2 = dt.predict(test_x1)



        #combining the predictions

        total_approx_s1 = linear_approx_s1 * trend_corrected_approx_s1

        total_approx_s2 = linear_approx_s2 * trend_corrected_approx_s2





        #calculating rmses

        dt_rmses_train.append(np.sqrt(mean_squared_error(train_y, total_approx_s1)))

        dt_rmses_test.append(np.sqrt(mean_squared_error(test_y, total_approx_s2)))

    return(dt_rmses_train,dt_rmses_test)
dt_rmses_train,dt_rmses_test = combined_lin_dt(spl = 2, lf = 1)

dt_rmse_train = np.mean(dt_rmses_train)

dt_rmse_test = np.mean(dt_rmses_test)

print('Train error:')

print(dt_rmses_train)

print(dt_rmse_train)

print()

print('Test error:')

print(dt_rmses_test)

print(dt_rmse_test)
lf1 = np.arange(1,51,10)

spl1 = np.arange(2,51,10)



spl_lf_df1 = pd.DataFrame(columns = lf1, index = spl1)



start_time = time.time()

for i in spl1:

    for j in lf1:

        dt_rmses_train = []

        dt_rmses_test = []

        for train_index, test_index in kf.split(df_for_trees):

            rmses_train, rmses_test = combined_lin_dt(spl = i, lf = j)

            dt_rmses_train.append(np.mean(rmses_train))

            dt_rmses_test.append(np.mean(rmses_test))



        dt_rmse_train = np.mean(dt_rmses_train)

        dt_rmse_test = np.mean(dt_rmses_test)

        spl_lf_df1.loc[i,j] = (round(dt_rmse_test,2),round(dt_rmse_train,2))

print("--- {:.2f} seconds ---".format(time.time() - start_time))
spl_lf1 = spl_lf_df1.reset_index().melt(id_vars = "index")

spl_lf1 = pd.concat([spl_lf1,pd.DataFrame(spl_lf1['value'].tolist(), index=spl_lf1.index)], axis = 1)

spl_lf1 = spl_lf1.drop("value", axis = 1)

spl_lf1.columns = ["min_samples_split", "min_samples_leaf", "rmse_test","rmse_train"]

spl_lf1 ["abs_diff"] = (spl_lf1["rmse_test"] - spl_lf1["rmse_train"]).abs()

print('Optimal parameters (top 3):')

print(spl_lf1.sort_values(by = ["abs_diff","rmse_test",

                               "min_samples_leaf","min_samples_split"]).head(3))

print()

print('Lowest rmse for test set parameters (top 3):')

print(spl_lf1.sort_values(by = ["rmse_test","min_samples_leaf","min_samples_split"]).head(3))
rf_rmses_train = []

rf_rmses_test = []

for train_index, test_index in kf.split(df_for_trees):

    train_x = df_for_trees.reindex(train_index)[feats_dt]

    train_y = df_for_trees.reindex(train_index)[target]

    test_x = df_for_trees.reindex(test_index)[feats_dt]

    test_y = df_for_trees.reindex(test_index)[target]



    rf = RandomForestRegressor(n_estimators = 10, random_state = 0)

    rf.fit(train_x,train_y)

    rf_prediction_train = rf.predict(train_x)

    rf_prediction_test = rf.predict(test_x)



    rf_rmses_train.append(np.sqrt(mean_squared_error(train_y, rf_prediction_train)))

    rf_rmses_test.append(np.sqrt(mean_squared_error(test_y, rf_prediction_test)))

    

rf_rmse_train = np.mean(rf_rmses_train)

rf_rmse_test = np.mean(rf_rmses_test)



print('Train error:')

print(rf_rmses_train)

print(rf_rmse_train)

print()

print('Test error:')

print(rf_rmses_test)

print(rf_rmse_test)
rf_rmse_train_plot = []

rf_rmse_test_plot = []



lf = range(1,11)



for i in lf:

    rf_rmses_train = []

    rf_rmses_test = []

    for train_index, test_index in kf.split(df_for_trees):

        train_x = df_for_trees.reindex(train_index)[feats_dt]

        train_y = df_for_trees.reindex(train_index)[target]

        test_x = df_for_trees.reindex(test_index)[feats_dt]

        test_y = df_for_trees.reindex(test_index)[target]



        rf = RandomForestRegressor(n_estimators = 10, random_state = 0, min_samples_leaf = i)

        rf.fit(train_x,train_y)

        rf_prediction_train = rf.predict(train_x)

        rf_prediction_test = rf.predict(test_x)



        rf_rmses_train.append(np.sqrt(mean_squared_error(train_y, rf_prediction_train)))

        rf_rmses_test.append(np.sqrt(mean_squared_error(test_y, rf_prediction_test)))



    rf_rmse_train = np.mean(rf_rmses_train)

    rf_rmse_test = np.mean(rf_rmses_test)

    

    rf_rmse_train_plot.append(rf_rmse_train)

    rf_rmse_test_plot.append(rf_rmse_test)
fig, ax = plt.subplots(figsize = (7,5))

ax.plot(lf, rf_rmse_test_plot, color = 'grey', alpha = 0.7, label = 'test set prediction rmse')

ax.plot(lf, rf_rmse_train_plot, color = 'orange', alpha = 0.7, label = 'train set prediction rmse')

plt.suptitle('Error of Random Forest model with k-number of min samples leaf')

ax.set_xlabel('k-number of min samples leaf')

ax.set_ylabel('rmse')

ax.legend()

plt.show()
print('min rmse: {:.2f} at min_samples_leaf = {}'.format(min(rf_rmse_test_plot),

                                                        (rf_rmse_test_plot.index(min(rf_rmse_test_plot)) + 1)))
def combined_lin_rf(spl = 2, lf = 1):

    rf_rmses_train = []

    rf_rmses_test = []

    for train_index, test_index in kf.split(df_for_regression_dummies):



        #splitting to train and test

        train_x = df_for_regression_dummies.reindex(train_index)[feats1]

        test_x = df_for_regression_dummies.reindex(test_index)[feats1]

        train_y = df_for_regression_dummies.reindex(train_index)[target]

        test_y = df_for_regression_dummies.reindex(test_index)[target]



        #pulling the trend

        lr = LinearRegression()

        lr.fit(train_x['instant'].values.reshape(-1,1),train_y)



        linear_approx_s1 = lr.predict(train_x['instant'].values.reshape(-1,1))

        linear_approx_s2 = lr.predict(test_x['instant'].values.reshape(-1,1))





        #pulling trend-corrected prediction

        train_x1 = train_x[list(set(feats1) - set('instant'))]

        test_x1 = test_x[list(set(feats1) - set('instant'))]

        train_y1 = train_y / linear_approx_s1

        test_y1 = test_y / linear_approx_s2

        

        rf = RandomForestRegressor(n_estimators = 10,

                                   min_samples_split = spl,

                                   min_samples_leaf = lf,

                                   random_state = 0)

        

        rf.fit(train_x1,train_y1)

        trend_corrected_approx_s1 = rf.predict(train_x1)

        trend_corrected_approx_s2 = rf.predict(test_x1)



        #combining the predictions

        total_approx_s1 = linear_approx_s1 * trend_corrected_approx_s1

        total_approx_s2 = linear_approx_s2 * trend_corrected_approx_s2





        #calculating rmses

        rf_rmses_train.append(np.sqrt(mean_squared_error(train_y, total_approx_s1)))

        rf_rmses_test.append(np.sqrt(mean_squared_error(test_y, total_approx_s2)))

    return(rf_rmses_train,rf_rmses_test)
combined_lin_rf(spl = 2, lf = 1)



rf_rmses_train,rf_rmses_test = combined_lin_rf(spl = 2, lf = 1)

rf_rmse_train = np.mean(rf_rmses_train)

rf_rmse_test = np.mean(rf_rmses_test)

print('Train error:')

print(rf_rmses_train)

print(rf_rmse_train)

print()

print('Test error:')

print(rf_rmses_test)

print(rf_rmse_test)
rf_rmse_train_plot = []

rf_rmse_test_plot = []



lf = range(1,11)



for i in lf:

    rf_rmses_train, rf_rmses_test = combined_lin_rf(lf = i)

    

    rf_rmse_train = np.mean(rf_rmses_train)

    rf_rmse_test = np.mean(rf_rmses_test)

    

    rf_rmse_train_plot.append(rf_rmse_train)

    rf_rmse_test_plot.append(rf_rmse_test)
fig, ax = plt.subplots(figsize = (7,5))

ax.plot(lf, rf_rmse_test_plot, color = 'grey', alpha = 0.7, label = 'test set prediction rmse')

ax.plot(lf, rf_rmse_train_plot, color = 'orange', alpha = 0.7, label = 'train set prediction rmse')

plt.suptitle('Error of combined Linear Regression - Random Forest model with k-number of min samples leaf')

ax.set_xlabel('k-number of min samples leaf')

ax.set_ylabel('rmse')

ax.legend()

plt.show()
print('min rmse: {:.2f} at min_samples_leaf = {}'.format(min(rf_rmse_test_plot),

                                                        (rf_rmse_test_plot.index(min(rf_rmse_test_plot)) + 1)))