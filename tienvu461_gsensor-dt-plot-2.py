

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from datetime import datetime

# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
lte_time_dt_list = []

cpu_time_dt_list = []

gx_dt_list = []

gy_dt_list = []

gz_dt_list = []

def print_debug(string):

    if DEBUG_MACRO:

        string = "[DEBUG]" + string

        return print(string)





def find_all(org_str, sub_str):

    start = 0

    while True:

        start = org_str.find(sub_str, start)

        if start == -1: return

        yield start

        start += len(sub_str) # use start += 1 to find overlapping matches





def find_custom(org_str, sub_str, end_char, offset = 0):

    index_list = list(find_all(org_str, sub_str))

    output_dt_list = []

    for index in index_list:

        output_dt = ""

        for i in range(index + offset, len(org_str)):

            output_dt += org_str[i]

            i += 1

            if line[i] == end_char:

                output_dt_list.append(output_dt)

                break

    return output_dt_list



with open("../input/sample001.txt", "r") as input:

    for line in input:

        if line.find("header") != -1:

            lte_time_dt_list.extend(find_custom(line, 'lte_time', 'Z', 22))

            cpu_time_dt_list.extend(find_custom(line, 'cpu_time', '\"', 11))

            gx_dt_list.extend(find_custom(line, "axis_x", ',', 9))

            gy_dt_list.extend(find_custom(line, "axis_y", ',', 9))

            gz_dt_list.extend(find_custom(line, "axis_z", '}', 9))

            

for timestamp in lte_time_dt_list:

    timestamp = datetime.strptime(timestamp, '%H:%M:%S')



data_df = pd.DataFrame(lte_time_dt_list, columns=["lte_time"])

data_df["axis_x"] = gx_dt_list

data_df["axis_y"] = gy_dt_list

data_df["axis_z"] = gz_dt_list

# data_df_plotting = data_df.set_index('lte_time')

# data_df_plotting['axis_x'] = data_df_plotting['axis_x'].astype(float)

# data_df_plotting['axis_y'] = data_df_plotting['axis_y'].astype(float)

# data_df_plotting['axis_z'] = data_df_plotting['axis_z'].astype(float)

# ax = plt.gca()

# data_df_plotting.plot(kind='line',y='axis_x',color='red', ax=ax)

# data_df_plotting.plot(kind='line',y='axis_y',color='blue', ax=ax)

# data_df_plotting.plot(kind='line',y='axis_z',color='green', ax=ax)



# plt.show()





# data_tb = ff.create_table(data_df)

# py.iplot(data_tb, filename='jupyter-table1')



data_df_clone = data_df.copy()

data_df_clone['axis_x'] = data_df_clone['axis_x'].astype(float)

data_df_clone['axis_y'] = data_df_clone['axis_y'].astype(float)

data_df_clone['axis_z'] = data_df_clone['axis_z'].astype(float)



# plot_trace0 = go.Scatter(

#     x = data_df_clone.lte_time,

#     y = data_df_clone.axis_x,

#     mode = 'lines',

#     name = 'axis_x'

# )

# plot_trace1 = go.Scatter(

#     x = data_df_clone.lte_time,

#     y = data_df_clone.axis_y,

#     mode = 'lines',

#     name = 'axis_y'

# )

# plot_trace2 = go.Scatter(

#     x = data_df_clone.lte_time,

#     y = data_df_clone.axis_z,

#     mode = 'lines',

#     name = 'axis_z'

# )

# plot_data=[plot_trace0, plot_trace1, plot_trace2]

# layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',

#               xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)

#              )

# fig = dict(data = plot_data, layout = layout)

# py.iplot(fig)
# Creating trace1

trace1 = go.Scatter(

                     x = data_df_clone.lte_time,

                    y = data_df_clone.axis_x,

                    mode = 'lines',

                    name = 'axis_x',

                    marker = dict(color = 'red'))

# Creating trace2

trace2 = go.Scatter(

                   x = data_df_clone.lte_time,

                    y = data_df_clone.axis_y,

                    mode = 'lines',

                    name = 'axis_y',

                    marker = dict(color = 'yellow'))

# Creating trace3

trace3 = go.Scatter(

                   x = data_df_clone.lte_time,

                    y = data_df_clone.axis_z,

                    mode = 'lines',

                    name = 'axis_z',

                    marker = dict(color = 'blue'))

data = [trace1, trace2, trace3]

layout = dict(title = 'Gsensor data log',

              xaxis= dict(title= 'LTE Time',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)