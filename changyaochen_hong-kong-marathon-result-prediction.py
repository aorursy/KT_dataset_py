# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import re

import fnmatch    # for later text matching

import matplotlib.pyplot as plt    # for plotting

plt.style.use('ggplot')

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "/input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train_df = pd.read_csv('../input/challenge.csv')

print(train_df.shape)

train_df.head()
train_df["Category"].unique()
# true table for male and female

Male_list = [fnmatch.fnmatch(item, '?M*') for item in train_df["Category"].tolist()]

Female_list = [fnmatch.fnmatch(item, '?F*') for item in train_df["Category"].tolist()]

# create the male and female dataframes

train_Male_df = train_df[Male_list]

train_Female_df = train_df[Female_list]
print('The total entries in Male dataset is ' + str(train_Male_df.shape[0]))

print('The Gender Position of the last entry in Male dataset is '+ str(train_Male_df["Gender Position"].iloc[-1]))

print('The total entries in Female dataset is ' + str(train_Female_df.shape[0]))

print('The Gender Position of the last entry in Female dataset is '+ str(train_Female_df["Gender Position"].iloc[-1]))
# temporay dataframe

temp_df = pd.DataFrame(columns=train_df.columns)

for i in range(1, train_df.shape[0]):

    if (abs(train_df["Overall Position"].iloc[i] - train_df["Overall Position"].iloc[i-1]) > 1

        and train_df["Official Time"].iloc[i] != train_df["Official Time"].iloc[i-1]):

        temp_df = temp_df.append(train_df.iloc[i-1])

        temp_df = temp_df.append(train_df.iloc[i])
ranked_df = train_df.copy(deep = True)

## check data types

# print(ranked_df.dtypes)

official_time_temp = pd.DatetimeIndex(ranked_df['Official Time'])

net_time_temp = pd.DatetimeIndex(ranked_df['Net Time'])

ranked_df['Official Time (s)'] = official_time_temp.hour*3600 + official_time_temp.minute*60 + official_time_temp.second

ranked_df['Net Time (s)'] = net_time_temp.hour*3600 + net_time_temp.minute*60 + net_time_temp.second

ranked_df.sort_values('Net Time (s)');

ranked_df.tail(5)
# prepare the data

groups = ranked_df['Category'].unique();

group_labels = [x[1:] for x in groups];  



# initilize the runner demographic dictionary

runner_demo = {label: 0 for label in group_labels}; 



# update the dictionary

for i in range(0, ranked_df.shape[0]):

    runner_demo[ranked_df.loc[i]['Category'][1:]] += 1



# plot!

gender_less_group = [x[1:] for x in group_labels if x[0] == 'M']

female_demo = [runner_demo['F' + x] for x in gender_less_group]

male_demo = [runner_demo['M' + x] for x in gender_less_group]

index = np.arange(len(group_labels)/2)

bar_width = 0.35

opacity = 0.4

fig, ax = plt.subplots()

rects1 = ax.bar(index, female_demo, bar_width,

                 alpha=opacity,

                 color='r',

                 label='Female')

rects2 = ax.bar(index + bar_width, male_demo, bar_width,

                 alpha=opacity,

                 color='b',

                 label='Male')

plt.xlabel('Category')

plt.ylabel('Number of runners')

plt.title('Runners by category and gender')

plt.xticks(index + bar_width, (gender_less_group))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



plt.tight_layout()

def autolabel(rects):

    # attach some text labels

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                '%d' % int(height),

                ha='center', va='bottom')



autolabel(rects1)

autolabel(rects2)

plt.show()
ranked_df['Gender'] = ranked_df['Category'].apply(lambda x: x[1])    # create a gender column

ranked_df['Gender less category'] = ranked_df['Category'].apply(lambda x: x[2:])    # create a gender less category column

gender_cat = ranked_df['Gender'].groupby(ranked_df['Gender less category']) 

gender_cat.value_counts().unstack().plot(kind = 'bar')    # unstack the multi-index series, and then plot!

plt.ylabel('Number of runners');
from bokeh.plotting import figure, output_file, show, output_notebook

from bokeh.charts import Bar, Donut, BoxPlot, Scatter, Histogram

from bokeh.models import ColumnDataSource, HoverTool, Range1d, NumeralTickFormatter

from collections import OrderedDict

output_notebook()  # this is to enable the plot the figure in notebook



p = Bar(ranked_df, values = 'Gender', label = 'Gender less category',

        agg = 'count', stack = 'Gender', tools='pan,box_zoom,reset,resize,save,hover', 

        legend='top_left', plot_width=800, plot_height=300)

hover = p.select(dict(type=HoverTool))

hover.tooltips = OrderedDict([("Gender", "@Gender"),("Num", "@height{int}")])

show(p)
gender_country = ranked_df['Gender'].groupby(ranked_df['Country ']).value_counts().unstack().fillna(0).astype(int)

gender_country['Total'] = gender_country.apply(np.sum, axis = 1)

gender_country = gender_country.sort_values('Total', ascending = False)

gender_country['Percentage'] = gender_country['Total'] / gender_country['Total'].sum() * 100

gender_country.head(5)
gender_country_2 = gender_country.iloc[0:3]

gender_country_3 = gender_country.iloc[3:]

others = gender_country_3.apply(np.sum)

others.name = 'Others'

gender_country_2 = gender_country_2.append(others)

gender_country_2['Country'] = gender_country_2.index

gender_country_3 = pd.melt(gender_country_2, id_vars = 'Country', value_vars = ['F', 'M'], 

                      var_name = 'Gender', value_name = 'Total Count')



p = Donut(gender_country_3, label = ['Country', 'Gender'], values = 'Total Count', 

          tools = 'pan,box_zoom,reset,resize,save,hover')

hover = p.select(dict(type=HoverTool))

hover.tooltips = OrderedDict([('Gender', '@Gender')])

show(p)
box1 = BoxPlot(ranked_df, values='Net Time (s)', label='Category', color = 'Category', 

               tools = 'pan,box_zoom,reset,resize,save,hover', plot_width=800, plot_height=300,

              legend=False)

hover = box1.select(dict(type=HoverTool))

hover.tooltips = OrderedDict([("Category", "@Category"), ('Time', '@height{int} s')])

# there is no marathon faster than 2 hours, and I add some cushion to the top too

box1.y_range = Range1d(7200, max(ranked_df['Net Time (s)'] + 30*60))

hover = p.select(dict(type = HoverTool))

hover.tooltips = [('Category', '@Category')]

box1.yaxis[0].formatter = NumeralTickFormatter(format='00:00:00')

box1.yaxis.axis_label = 'Net Time'

show(box1)
temp_df = ranked_df['Net Time (s)'].groupby(ranked_df['Category']).median().to_frame()

hover = HoverTool(tooltips = [('y','@y s'),])  # can be improved

# The high level Scatter function didn't work for me, likely due to the fact the x axis is categorical

p = figure(title = 'Median Time for different categories',

           x_range = [x for x in temp_df.index.values], tools=['pan,box_zoom,reset,resize,save', hover],

          x_axis_label = 'Category', y_axis_label = 'Median Time', plot_width=800, plot_height=300)

p.scatter([x for x in temp_df.index.values],

          [y for y in temp_df['Net Time (s)']], size=20, color="navy", alpha=0.5)

p.yaxis[0].formatter = NumeralTickFormatter(format='00:00:00')

show(p)
hover = HoverTool(tooltips = [('Rank', '@x'), ('Time','@y s'),])

p = figure(title = 'Gun/Net Time differece', x_axis_label = 'Overall Position', y_axis_label = 'Time Gap', 

           plot_width=800, plot_height=600, tools = ['pan,box_zoom,reset,resize,save', hover])

p.circle(ranked_df['Overall Position'], ranked_df['Official Time (s)'] - ranked_df['Net Time (s)'],

        alpha = 0.5, line_alpha = 0, size = 5)

p.yaxis[0].formatter = NumeralTickFormatter(format='00:00:00')

show(p)
ranked_df.ix[(ranked_df['Official Time (s)'] - ranked_df['Net Time (s)']).argmax()]
def time_to_sec(string):

    t = list(map(float, string.split(':')))

    return 3600*t[0] + 60*t[1] + t[2]

ranked_df['10km Time'].map(type).value_counts()

float_ixs = []

for i in range(ranked_df.shape[0]):

    if type(ranked_df['10km Time'].ix[i]) is float:

        float_ixs.append(i)

ranked_df.ix[float_ixs].head(3)
ranked_df_clean = ranked_df.dropna()
hover = HoverTool(tooltips = [('Time difference', '@x{int} s'),])

ranked_df_clean['Fisrt/Second Diff'] = (ranked_df_clean['Net Time'].map(time_to_sec) \

                                        - 2*ranked_df_clean['Half Way Time'].map(time_to_sec))

hist1 = Histogram(ranked_df_clean['Fisrt/Second Diff'],

                  title = 'First/Second Half Time differece', bins = 100,

            plot_width=800, plot_height=300, tools = ['pan,box_zoom,reset,resize,save', hover])

hist1.xaxis[0].formatter = NumeralTickFormatter(format='00:00:00')

hist1.xaxis.axis_label = 'Time Difference'

hist1.yaxis.axis_label = 'Counts'

show(hist1)
hover = HoverTool(tooltips = [('Time difference', '@x{int} s'),])

hist2 = Histogram(ranked_df_clean, values = 'Fisrt/Second Diff', color = 'Gender', bins = 100,

            plot_width=800, plot_height=300, tools = ['pan,box_zoom,reset,resize,save', hover])

hist2.xaxis[0].formatter = NumeralTickFormatter(format='00:00:00')

hist2.xaxis.axis_label = 'Time Difference'

hist2.yaxis.axis_label = 'Counts'

show(hist2)