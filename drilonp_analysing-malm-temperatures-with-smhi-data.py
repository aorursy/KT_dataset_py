import numpy as np

from scipy import stats

import itertools

import pandas as pd

pd.set_option("display.max_rows", 10000)



import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib.colors import Normalize

import matplotlib.cm as cm

import seaborn as sns



import os

import fnmatch as fn



import warnings

warnings.filterwarnings('ignore')



from pprint import pprint
file_path = '../input/smhi-malm-hourly-temperatures-19312019'

files = os.listdir(file_path)
pprint(files)
a_file = os.path.join(file_path, files[0])

b_file = os.path.join(file_path, files[1])

c_file = os.path.join(file_path, files[2])

d_file = os.path.join(file_path, files[3])
a_df = pd.read_csv(a_file, delimiter=';', usecols=[0,1,2,3], skiprows=9, parse_dates=True) #some rows above the actual headers must be skipped, as well as some columns

a_df.head()
b_df = pd.read_csv(b_file, delimiter=';', usecols=[0,1,2,3], skiprows=10, parse_dates=True)

b_df.head()
c_df = pd.read_csv(c_file, delimiter=';', usecols=[0,1,2,3], skiprows=9, parse_dates=True)

c_df.head()
d_df = pd.read_csv(d_file, delimiter=';', usecols=[0,1,2,3], skiprows=10, parse_dates=True)

d_df.head()
#Function to rename column names in dataframe

def rename_columns_map(df): 

    col_name_map = dict(zip(df.columns.values, ['Date', 'Time (UTC)', 'Air temperature', 'Quality']))

    return col_name_map



#function to add date related fields (pandas.dt attributes)

def add_date_attributes(df, datefield):

    '''Requires that datefield is a datetime object'''

    fields_to_add = ['year', 'month_name', 'month', 'weekofyear']

    obj = df[datefield].dt

    for attr in dir(df[datefield].dt):

        if attr in fields_to_add:

            if not callable(getattr(obj, attr)):

                df[str.capitalize(attr)] = getattr(obj, attr)

            else:

                df[str.capitalize(attr)] = getattr(obj, attr)()

    return df.copy()
df_list = [a_df, b_df, c_df, d_df]
#Renaming column names

for df in df_list:

    col_name_map = rename_columns_map(df)

    if set(df.columns.values) != set(col_name_map.values()): #since inplace is used, first checks whether already renamed

        df.rename(columns=col_name_map, inplace=True)
a_df.columns, b_df.columns, c_df.columns, d_df.columns
#Adding 'Datetime' from 'Date' and 'Time'

for df in df_list:

    if 'Datetime' not in df.columns.values: 

        df['Datetime'] = df['Date'] + ' ' + df['Time (UTC)']

        df.drop(columns='Time (UTC)', inplace=True)

        df['Datetime'] = pd.to_datetime(df['Datetime']) 

        df['Date'] = pd.to_datetime(df['Date'])
#Column dtype check

for df in df_list:

    print(df.dtypes, '\n--')
#Final concatenated dataframe

complete_df_total = pd.concat(df_list, ignore_index=True)    

complete_df = complete_df_total.groupby(['Date', 'Datetime']).mean()

complete_df = complete_df.reset_index()



#Checking if we have unique samples

print(f'Unique rows: {complete_df.Datetime.nunique()}; Total rows: {complete_df.shape[0]}')
#Adding date related files

complete_df = add_date_attributes(complete_df, 'Datetime')

complete_df.tail(15)
def time_plot_per_year(df, aggregation_type='average', rolling_window=5):

    

    ''' Plot both the actual aggregated and a smoothed rolling average of the values in [df], 

        with window according to [rolling_window] given as an integer; by default 5. 

        [aggregation_type] is related to the aggretation function applied to [df] 

        and affects the titles of the plot; by default 'average'. '''

    

    max_val = df.max()

    min_val = df.min()

    max_year = df.idxmax()

    min_year = df.idxmin()



    min_max_dict = {max_year: max_val, min_year: min_val}



    fig, axes = plt.subplots(2,1, figsize=(12,10), dpi=200, sharex=True);

    fig.suptitle(f'{aggregation_type.lower().capitalize()} Malmö temperatures per year', fontsize=19, y=0.95);

    fig.subplots_adjust(hspace=0.125);



    for ax in axes:

        ax.set_xticks(np.arange(df.index[0]-1, df.index[-1]+2, 5));

        ax.set_ylabel('Air temperature [$^{\circ}$C]');



    #First plot

    df.plot(ax=axes[0], title=f'Actual {aggregation_type.lower()}', linewidth=2);

    axes[0].axhline(max_val, linestyle= '--', color='r', lw=0.8);

    axes[0].axhline(min_val, linestyle= '--', color='r', lw=0.8);

    axes[0].xaxis.set_ticks_position('none');



    for i, (year, value) in enumerate(min_max_dict.items()):

        if i == 0:

            xy, xytext, position = (1,1), (5, -10), 'Max'

        else:

            xy, xytext, position = (1,0), (5, 15), 'Min'   

        axes[0].annotate(xy=xy, text=f'{position}: {year}; {round(value,2)}' + '$^{\circ}$C', 

                         xytext=xytext, va='top', xycoords='axes fraction', 

                         textcoords='offset points', alpha=0.8,

                         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="k", lw=0.8));



    #Second plot

    avg_per_year_rolling = df.rolling(rolling_window).mean()



    avg_per_year_rolling.plot(ax=axes[1], title=f'Rolling {df.index.name.lower()} average: {rolling_window}', linewidth=2);

    axes[1].grid(True, axis='y', linestyle='--');
avg_per_year = complete_df.groupby('Year')['Air temperature'].mean()

time_plot_per_year(avg_per_year)
max_per_year = complete_df.groupby('Year')['Air temperature'].max()

time_plot_per_year(max_per_year, aggregation_type='maximum')
min_per_year = complete_df.groupby('Year')['Air temperature'].min()

time_plot_per_year(min_per_year, aggregation_type='minimum')
def plot_top10_highest_and_lowest(df_highest, df_lowest, df_average):

    

    ''' Plot ordered and stylized bar charts of aggregated dataframes [df_highest], [df_lowest], [df_average]. '''



    fig, axes = plt.subplots(2,2, figsize=(15, 12), dpi=200)

    fig.suptitle('Highest and lowest temperatures', fontsize=19, y=0.95);

    fig.subplots_adjust(hspace=0.25, wspace=0.25);



    top_10_highest = df_highest.sort_values(ascending=False)[:10].sort_values()

    top_10_lowest = df_lowest.sort_values(ascending=True)[:10]

    top_10_highest_avg = df_average.sort_values(ascending=False)[:10].sort_values()

    top_10_lowest_avg = df_average.sort_values(ascending=True)[:10]

    

    frames = [top_10_highest, top_10_lowest, top_10_highest_avg, top_10_lowest_avg]

    cmaps = [cm.get_cmap('YlOrBr'), cm.get_cmap('Blues_r'), cm.get_cmap('pink_r'), cm.get_cmap('bone')]

    titles = ['Top 10 highest recorded', 'Top 10 lowest recorded', 'Top 10 highest monthly averages', 'Top 10 lowest monthly averages']

    

    for i,j in enumerate(itertools.product([0,1],[0,1])): # (0,0) (0,1) (1,0) (1,1)

        x, y = j[0], j[1] #subplotgrid indices

        ax = axes[x][y]

        ax.set_xlabel('Air temperature [$^{\circ}$C]');

        ax.set_title(titles[i])

        

        #creating customized color gradients for bar charts

        norm = Normalize(vmin=frames[i].min(), vmax=frames[i].max()+0.1)

        cmap = cmaps[i]

        gradient_color = cmap(norm(frames[i].values))

        

        frames[i].plot(kind='barh', ax=ax, color=gradient_color, alpha=0.75, linewidth=0.5, edgecolor='gray');

        

        #annotating bars with values and adding some style

        for index, value in enumerate(frames[i].values):

            if y == 1: #right side plots

                text_color = (0, 0, 0) if index != 0 else (1, 1, 1)

                text_style = 'oblique' if index == 0 else 'normal'

            else: #left side plots

                text_color = (0, 0, 0) if index != 9 else (1, 1, 1)

                text_style = 'oblique' if index == 9 else 'normal'

            shift = -value*0.1

            ax.annotate(xy=(value, index), text=str(round(value,2)), xytext=(value+shift,index-0.1), color=text_color, fontstyle=text_style)



            
max_per_year_month = complete_df.groupby(['Year', 'Month'])['Air temperature'].max()

min_per_year_month = complete_df.groupby(['Year', 'Month'])['Air temperature'].min()

avg_per_year_month = complete_df.groupby(['Year', 'Month'])['Air temperature'].mean()

plot_top10_highest_and_lowest(max_per_year_month, min_per_year_month, avg_per_year_month)
avg_per_month = complete_df.groupby('Month_name')['Air temperature'].mean().sort_values()

std_per_month = complete_df.groupby('Month_name')['Air temperature'].std()



fig, ax = plt.subplots(1,1, figsize=(8, 6), dpi=100)



norm = Normalize(vmin=avg_per_month.min(), vmax=avg_per_month.max()+0.1)

cmap = cm.get_cmap('Spectral_r')

gradient_color = cmap(norm(avg_per_month.values))

        

avg_per_month.plot(kind='barh', ax=ax,  title='Average temperatures per month', color=gradient_color, alpha=0.75,

                  linewidth=0.5, edgecolor='black', xerr=std_per_month, capsize=3);



ax.set_xlabel('Air temperature [$^{\circ}$C]');

ax.set_xticks(range(-4, 24, 1), minor=True);

for a, m in dict(zip([0.25, 0.7], ['minor', 'major'])).items():

    ax.grid(True, axis='x', linestyle=':', alpha=a, which=m);

def plot_hist_prob(frames: [], hist_titles: [], main_title: str, main_title_yloc=0.95):

    

    ''' Plot histograms and norm.dist quantile plots of aggregated dataframes [frames]. Histogram titles are given in [hist_titles],

        whereas the main title of the figure in [main_title]. Optionally adjust the vertical distance of the main title

        with [main_title_yloc].'''

    

    fig, axes = plt.subplots(len(frames),2, figsize=(8, len(frames)*4), dpi=150);

    fig.suptitle(main_title, fontsize=19, y=main_title_yloc);

    fig.subplots_adjust(hspace=0.3, wspace=0.25);

    

    for i, frame in enumerate(frames):

        mean_val = round(frame.mean(), 2)

        std_val = round(frame.std(), 2)

        

        #Histogram

        sns.distplot(frame, ax=axes[i][0]);

        

        axes[i][0].set_title(hist_titles[i]);

        

        min_xtick, max_ytick = min(axes[i][0].get_xticks()), max(axes[i][0].get_yticks())

        x_shift = 1.1 if min_xtick > 15 else 1.35 if min_xtick > 0 else 0.75

        y_shift = 0.85 

        

        axes[i][0].annotate('$\mu$: ' + str(mean_val), xy=(0, 1), xytext=(12, -12), va='top',

             xycoords='axes fraction', textcoords='offset points');

        axes[i][0].annotate('$\sigma$: ' + str(std_val), xy=(0, 1), xytext=(12, -22), va='top',

             xycoords='axes fraction', textcoords='offset points')

        axes[i][0].set_xlabel('Air temperature [$^{\circ}$C]');



        #Quantile plot

        stats.probplot(frame, sparams=(mean_val, std_val), plot=axes[i][1], rvalue=True);
hist_titles = ['Maximum yearly temperatures', 'Minimum yearly temperatures', 'Average yearly temperatures']

main_title='Temperature distributions: 1931-2019'

plot_hist_prob([max_per_year, min_per_year, avg_per_year], hist_titles, main_title)
previous = complete_df[complete_df.Year < 1990]

last_three_decades = complete_df[(complete_df.Year >= 1990) & (complete_df.Year < 2019)]

print(previous.shape, last_three_decades.shape, complete_df.shape)
previous_max_per_year = previous.groupby('Year')['Air temperature'].max()

latest_max_per_year = last_three_decades.groupby('Year')['Air temperature'].max()



hist_titles = ['Maximum yearly temperatures \'90-\'18', 'Maximum yearly temperatures \'31-\'89']

main_title = 'Temperature distributions: 1941-1989 vs 1990-2018'



plot_hist_prob([latest_max_per_year, previous_max_per_year], hist_titles, main_title, 0.97)
alpha=0.1



bartlett_stat, p_value_b = stats.bartlett(latest_max_per_year, previous_max_per_year)

levene_stat, p_value_l = stats.levene(latest_max_per_year, previous_max_per_year)



print(f'Bartlett statistic: {bartlett_stat}; Bartlet p: {p_value_b};\nLevene stat: {levene_stat}; p: {p_value_l}; \nConfidence level: {alpha}')
#stats.ttest_ind performs a two-sided test, but this can be intepreted as a one tailed 'greater-than test', when p/2 < alpha and t > 0

t_stat, p_value = stats.ttest_ind(a=latest_max_per_year, b=previous_max_per_year, equal_var=False)

print(f'Test statistic: {t_stat};\np/2: {p_value/2};\nConfidence level: {alpha}')