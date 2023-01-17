import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import calendar

import matplotlib.style as style

import missingno as msno



%config InlineBackend.figure_format = 'retina'

%matplotlib inline



style.use('ggplot')
df = pd.read_csv('/kaggle/input/ncei-heathrow-2010-2019/NCEI Heathrow Meteo Data 2010-2019.csv', 

                               usecols=['DATE', 'PRCP', 'TAVG'], parse_dates=['DATE'])

#df['DATE'] = df['DATE'].astype('datetime64[ns]') # needed if date format is not standard

df.columns = ['date', 'precipitation', 'avg_temp']

df.sample(3)
df.dtypes
len(df)
df.isnull().sum()
round(df.isnull().mean() * 100, 2)
_ = msno.matrix(df)
df['precipitation'].fillna(0, inplace=True)
print(f"Data Available from {df.date.min()} to {df.date.max()}")
idx = pd.date_range(df.date.min(), df.date.max())

print(f"Days present {len(df)} out of {len(idx)}")
pd.DataFrame(data=idx.difference(df.date), columns=['dates']).sample(3)
MIN_PRECIPITATION_MM_DRY = 1.0
round((len(df[df['precipitation'] < MIN_PRECIPITATION_MM_DRY]) / len(df)) * 100, 2)
df[df.precipitation == df.precipitation.max()][['date', 'precipitation']]
df[df.avg_temp == df.avg_temp.max()][['date', 'avg_temp']]
sns.distplot(df.precipitation)
df['month'] = df.date.dt.month

df['year'] = df.date.dt.year

df['day'] = df.date.dt.day

df['weekdayName'] = df.date.dt.weekday_name # df.date.dt.day_name() on Pandas 1.0

df['weekday'] = df.date.dt.weekday

df['week'] = df.date.dt.week

df['weekend'] = df.date.dt.weekday // 5 == 1
df['raining'] = df['precipitation'].gt(MIN_PRECIPITATION_MM_DRY).astype('int')
df.sample(3)
all_month_year_df = pd.pivot_table(df, values="precipitation",index=["month"],

                                   columns=["year"],

                                   fill_value=0,

                                   margins=True)

named_index = [[calendar.month_abbr[i] if isinstance(i, int) else i for i in list(all_month_year_df.index)]]

all_month_year_df = all_month_year_df.set_index(named_index)

all_month_year_df
def plot_heatmap(df, title):

    plt.figure(figsize = (14, 10))

    ax = sns.heatmap(df, cmap='RdYlGn_r',

                     robust=True,

                     fmt='.2f', annot=True,

                     linewidths=.5, annot_kws={'size':11},

                     cbar_kws={'shrink':.8, 'label':'Precipitation (mm)'})

    

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

    plt.title(title, fontdict={'fontsize':18}, pad=14);
plot_heatmap(all_month_year_df, 'Average Precipitations')
all_month_year_sum_df = pd.pivot_table(df, values="precipitation",index=["month"], columns=["year"], aggfunc=np.sum, fill_value=0)

all_month_year_sum_df = all_month_year_sum_df.set_index([[calendar.month_abbr[i] if isinstance(i, int) else i for i in list(all_month_year_sum_df.index)]])

plot_heatmap(all_month_year_sum_df, 'Total Precipitations')
all_weekday_year_df = pd.pivot_table(df, values="precipitation",index=["weekday"], columns=["year"], fill_value=0.0)

all_weekday_year_df = all_weekday_year_df.set_index([[calendar.day_name[i] for i in list(all_weekday_year_df.index)]])

plot_heatmap(all_weekday_year_df, 'Average Precipitation per weekday')
all_month_year_percentage_df = pd.pivot_table(df, values="precipitation",index=["month"], columns=["year"],

                                              aggfunc=lambda x: (x>MIN_PRECIPITATION_MM_DRY).sum()/len(x),

                                              fill_value=0,

                                              margins=True)

all_month_year_percentage_df = all_month_year_percentage_df.set_index([[calendar.month_abbr[i] if isinstance(i, int)

                                                                        else i for i in list(all_month_year_percentage_df.index)]])
plt.figure(figsize = (14, 10))

ax = sns.heatmap(all_month_year_percentage_df, cmap = 'RdYlGn_r', annot=True, fmt='.0%',

                 vmin=0, vmax=1, linewidths=.5, annot_kws={"size": 16})

cbar = ax.collections[0].colorbar

cbar.set_ticks([0, .25, .50,.75, 1])

cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 14)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 14)

ax.tick_params(rotation = 0)

plt.title('Percentage of days in the month with rain', fontdict={'fontsize':18}, pad=14);
def plot_heatmap_year(year):

    plt.figure(figsize = (16, 10))

    allByYear_df = df.loc[df['year'] == year]

    allByYear_df = pd.pivot_table(allByYear_df, values="precipitation",

                                  index=["month"], columns=["day"], fill_value=None)

    allByYear_df = allByYear_df.set_index([[calendar.month_abbr[i] for i in list(allByYear_df.index)]])

    ax = sns.heatmap(allByYear_df, cmap = 'RdYlGn_r',

                     vmin=0, vmax=20,

                     annot=False, linewidths=.1,

                     annot_kws={"size": 8}, square=True, cbar_kws={"shrink": .48, 'label': 'Rain (mm)'})

    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 12)

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 12)

    ax.tick_params(rotation = 0)

    plt.title(f'Precipitations {year}', fontdict={'fontsize':18}, pad=14);
plot_heatmap_year(2019)
plot_heatmap_year(2014)
all_days_avg_df = df.groupby([df.date.dt.month, df.date.dt.day])['precipitation'].mean()

all_days_avg_df = all_days_avg_df.unstack()

all_days_avg_df = all_days_avg_df.set_index([[calendar.month_abbr[i] for i in list(all_days_avg_df.index)]])
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors



def make_colormap(seq):

    """

    Return a LinearSegmentedColormap

    seq: list

        a sequence of floats and RGB-tuples. 

        The floats should be increasing and in the interval (0,1).

    """

    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]

    cdict = {'red': [], 'green': [], 'blue': []}

    for i, item in enumerate(seq):

        if isinstance(item, float):

            r1, g1, b1 = seq[i - 1]

            r2, g2, b2 = seq[i + 1]

            cdict['red'].append([item, r1, r2])

            cdict['green'].append([item, g1, g2])

            cdict['blue'].append([item, b1, b2])

    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
import matplotlib.colors as colors

c = colors.ColorConverter().to_rgb

gyr = make_colormap([c('green'), c('yellow'), 0.25, c('yellow'), c('red')])
plt.figure(figsize = (20, 14))

ax = sns.heatmap(all_days_avg_df, cmap = gyr, annot=True, fmt='.2f',

                 vmin=0, linewidths=.1,

                 annot_kws={"size": 8}, square=True,  # <-- square cell

                 cbar_kws={"shrink": .5, 'label': 'Rain (mm)'})

ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 12)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 12)

ax.tick_params(rotation = 0)

_ = plt.title('Precipitations Average 2010-2019', fontdict={'fontsize':18}, pad=14)
custom_palette = sns.color_palette("GnBu", 6)

custom_palette[5] = sns.color_palette("OrRd", 6)[5]
sns.palplot(custom_palette)
plt.figure(figsize = (20, 14))

ax = sns.heatmap(all_days_avg_df, cmap = custom_palette, annot=True, fmt='.2f',

                 vmin=0, linewidths=.1,

                 annot_kws={"size": 8}, square=True,

                 cbar_kws={"shrink": .5, 'label': 'Rain (mm)'})

ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 12)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 12)

ax.tick_params(rotation = 0)

_ = plt.title('Precipitations Average 2010-2019', fontdict={'fontsize':18}, pad=14)
df.groupby('year')['raining'].sum()
plt.figure(figsize = (14, 6))

ax = sns.countplot(x="year", hue="raining", data=df.sort_values(by='year'))

ax.legend(loc='upper right', frameon=True, labels=['Dry', 'Rain'])



for p in ax.patches:

    ax.annotate(format(p.get_height()),

                (p.get_x()+p.get_width()/2., p.get_height()-4),

                ha = 'center', va = 'center',

                xytext = (0, 10), textcoords = 'offset points')



_ = ax.set_title("Dry and Wet Days per Year")
df['value_grp'] = (df['raining'].diff() != 0).astype('int').cumsum()
(df['raining'].diff() != 0).astype('int')
df.head(10)[['date', 'precipitation', 'raining', 'value_grp']]
grouped_values = df.groupby('value_grp')

consecutive_df = pd.DataFrame({'BeginDate' : grouped_values.date.first(), 

              'EndDate' : grouped_values.date.last(),

              'Consecutive' : grouped_values.size(),

              'condition': grouped_values.raining.max() }).reset_index(drop=True)

consecutive_df['condition'].replace({0: 'Dry', 1: 'Rain'}, inplace=True)

consecutive_df.sort_values(by='Consecutive', ascending=False).head(10)
plt.figure(figsize = (14, 6))

ax = sns.countplot(x='Consecutive', hue='condition', data=consecutive_df.query('Consecutive >= 2'))

ax.set_title('Consecutive days on a specific condition 2012-2019 (> 2 days)', pad=14)

ax.set(xlabel='Consecutive days', ylabel='Count')

_ = plt.legend(loc='upper right')
consecutive_df['DateRange'] = consecutive_df["BeginDate"].astype(str) + ' -> ' + consecutive_df["EndDate"].astype(str)

ax = sns.barplot(x="Consecutive", y="DateRange", hue="condition", data=consecutive_df.sort_values(by='Consecutive', ascending=False).head(14))



for p in ax.patches:

 width = p.get_width()

 ax.text(width -1.6, p.get_y() + p.get_height()/2. + 0.2,'{:1.0f}'.format(width), ha="center")
df_top10_per_condition = consecutive_df.sort_values(by='Consecutive',ascending = False).groupby('condition').head(10)



d = {'color': ['g', 'r']}

g = sns.FacetGrid(df_top10_per_condition, row="condition",

                      hue='condition',

                      hue_kws=d,

                      sharey=False)



g.fig.set_figheight(8)

g.fig.set_figwidth(10)

    

_ = g.map(sns.barplot, "Consecutive", "DateRange")

_ = g.set(ylabel='')



# This is just to add the numbers inside the bars

for ax in g.axes.flat:

 for p in ax.patches:

  width = p.get_width()

  _ = ax.text(width -1.6, p.get_y() + p.get_height()/2. + 0.1,'{:1.0f}'.format(width), ha="center")
custom_palette = sns.diverging_palette(128, 240, n=10)
sns.palplot(custom_palette)
plt.figure(figsize = (20, 14))

ax = sns.heatmap(all_days_avg_df, cmap = custom_palette, annot=True, fmt='.2f',

                 vmin=0, linewidths=.1,

                 annot_kws={"size": 8}, square=True,

                 cbar_kws={"shrink": .5, 'label': 'Rain (mm)'})

ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 12)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 12)

ax.tick_params(rotation = 0)

_ = plt.title('Precipitations Average 2010-2019', fontdict={'fontsize':18}, pad=14)
plt.figure(figsize = (18, 6))

plt.title('Avg Rainfall (40 days window)', pad=14)

_ = df.set_index('date')['precipitation'].rolling(40).mean().plot()
ops_month_df= df.groupby(['month', 'year']).mean()['avg_temp'].reset_index()

plt.figure(figsize = (14, 6))

ax = sns.boxplot(x = "month", y = "avg_temp", data = ops_month_df)
df.groupby(['month', 'year']).mean()['avg_temp'].reset_index()
def plotHeatmap(df, title):

    plt.figure(figsize = (20, 8))



    ax = sns.heatmap(df, cmap = 'RdYlBu_r', fmt='.2f', annot=True,

                     linewidths=.2, annot_kws={"size": 8}, square=True,

                     cbar_kws={"shrink": .9, 'label': 'Temperature °C'})

    cbar = ax.collections[0].colorbar

    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 10)

    ax.tick_params(rotation = 0)

    plt.title(title, fontdict={'fontsize':18}, pad=14);
allMonthYear_df = pd.pivot_table(df, values="avg_temp",index=["month"], columns=["year"], fill_value=None, margins=True)

allMonthYear_df = allMonthYear_df.set_index([[calendar.month_abbr[i] if isinstance(i, int) else i for i in list(allMonthYear_df.index)]])

plotHeatmap(allMonthYear_df, 'Average Temperature')