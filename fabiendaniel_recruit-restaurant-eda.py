import numpy as np

import datetime

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import math, warnings

from mpl_toolkits.basemap import Basemap

plt.rcParams["patch.force_edgecolor"] = True

plt.style.use('fivethirtyeight')

mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)

from IPython.core.interactiveshell import InteractiveShell

from IPython.display import display, HTML

InteractiveShell.ast_node_interactivity = "last_expr"

pd.options.display.max_columns = 50

%matplotlib inline

warnings.filterwarnings('ignore')
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

file_list = check_output(["ls", "../input"]).decode("utf8")

file_list = file_list.strip().split('\n')
def get_info(df):

    print('Shape:',df.shape)

    print('Size: {:5.2f} MB'.format(df.memory_usage().sum()/1024**2))

    tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})

    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values'}))

    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.

                         rename(index={0:'null values (%)'}))

    display(tab_info)

    display(df[:5])
#_____________________________________________________________

# Read all the .csv files and show some info on their contents

for index, file in enumerate(file_list):

    var_name = file.rstrip('.csv')

    print(file)

    locals()[var_name] = pd.read_csv('../input/'+file)

    #____________________

    # convert to datetime

    for col in locals()[var_name].columns:

        if col.endswith('datetime') or col.endswith('date'):

            locals()[var_name][col] = pd.to_datetime(locals()[var_name][col])

    #__________________

    get_info(locals()[var_name])
def draw_map(df, title):

    plt.figure(figsize=(11,6))

    map = Basemap(resolution='i',llcrnrlon=127, urcrnrlon=147,

                  llcrnrlat=29, urcrnrlat=47, lat_0=0, lon_0=0,)

    map.shadedrelief()

    map.drawcoastlines()

    map.drawcountries(linewidth = 3)

    map.drawstates(color='0.3')

    parallels = np.arange(0.,360,10.,)

    map.drawparallels(parallels, labels = [True for s in range(len(parallels))])

    meridians = np.arange(0.,360,10.,)

    map.drawmeridians(meridians, labels = [True for s in range(len(meridians))])

    #______________________

    # put restaurants on map

    for index, (y,x) in df[['latitude','longitude']].iterrows():

        x, y = map(x, y)

        map.plot(x, y, marker='o', markersize = 5, markeredgewidth = 1, color = 'red',

                 markeredgecolor='k')

    plt.title(title, y = 1.05)
#draw_map(hpg_store_info, 'hpg store restaurant locations')

draw_map(air_store_info, 'air store restaurant locations')
class Figure_style():

    #_________________________________________________________________

    def __init__(self, size_x = 11, size_y = 5, nrows = 1, ncols = 1):

        sns.set_style("white")

        sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

        self.fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(size_x,size_y,))

        #________________________________

        # convert self.axs to 2D array

        if nrows == 1 and ncols == 1:

            self.axs = np.reshape(axs, (1, -1))

        elif nrows == 1:

            self.axs = np.reshape(axs, (1, -1))

        elif ncols == 1:

            self.axs = np.reshape(axs, (-1, 1))

    #_____________________________

    def pos_update(self, ix, iy):

        self.ix, self.iy = ix, iy

    #_______________

    def style(self):

        self.axs[self.ix, self.iy].spines['right'].set_visible(False)

        self.axs[self.ix, self.iy].spines['top'].set_visible(False)

        self.axs[self.ix, self.iy].yaxis.grid(color='lightgray', linestyle=':')

        self.axs[self.ix, self.iy].xaxis.grid(color='lightgray', linestyle=':')

        self.axs[self.ix, self.iy].tick_params(axis='both', which='major',

                                               labelsize=10, size = 5)

    #________________________________________

    def draw_legend(self, location='upper right'):

        legend = self.axs[self.ix, self.iy].legend(loc = location, shadow=True,

                                        facecolor = 'g', frameon = True)

        legend.get_frame().set_facecolor('whitesmoke')

    #_________________________________________________________________________________

    def cust_plot(self, x, y, color='b', linestyle='-', linewidth=1, marker=None, label=''):

        if marker:

            markerfacecolor, marker, markersize = marker[:]

            self.axs[self.ix, self.iy].plot(x, y, color = color, linestyle = linestyle,

                                linewidth = linewidth, marker = marker, label = label,

                                markerfacecolor = markerfacecolor, markersize = markersize)

        else:

            self.axs[self.ix, self.iy].plot(x, y, color = color, linestyle = linestyle,

                                        linewidth = linewidth, label=label)

        self.fig.autofmt_xdate()

    #________________________________________________________________________

    def cust_plot_date(self, x, y, color='lightblue', linestyle='-',

                       linewidth=1, markeredge=False, label=''):

        markeredgewidth = 1 if markeredge else 0

        self.axs[self.ix, self.iy].plot_date(x, y, color='lightblue', markeredgecolor='grey',

                                  markeredgewidth = markeredgewidth, label=label)

    #________________________________________________________________________

    def cust_scatter(self, x, y, color = 'lightblue', markeredge = False, label=''):

        markeredgewidth = 1 if markeredge else 0

        self.axs[self.ix, self.iy].scatter(x, y, color=color,  edgecolor='grey',

                                  linewidths = markeredgewidth, label=label)    

    #___________________________________________

    def set_xlabel(self, label, fontsize = 14):

        self.axs[self.ix, self.iy].set_xlabel(label, fontsize = fontsize)

    #___________________________________________

    def set_ylabel(self, label, fontsize = 14):

        self.axs[self.ix, self.iy].set_ylabel(label, fontsize = fontsize)

    #____________________________________

    def set_xlim(self, lim_inf, lim_sup):

        self.axs[self.ix, self.iy].set_xlim([lim_inf, lim_sup])

    #____________________________________

    def set_ylim(self, lim_inf, lim_sup):

        self.axs[self.ix, self.iy].set_ylim([lim_inf, lim_sup])  
convert_hpg = {k:v for k,v in list(zip(store_id_relation['hpg_store_id'].values,

                                       store_id_relation['air_store_id'].values))}

hpg_reserve["hpg_store_id"].replace(convert_hpg, inplace = True)

hpg_reserve = hpg_reserve[hpg_reserve['hpg_store_id'].str.startswith('air')]
def delta_reservation(df):

    df['delta_reservation'] = df['visit_datetime'] - df['reserve_datetime']

    df['delta_2days'] = df['delta_reservation'].apply(lambda x: int(x.days < 2))

    df['delta_7days'] = df['delta_reservation'].apply(lambda x: int(2 <= x.days < 7))

    df['delta_long'] = df['delta_reservation'].apply(lambda x: int(x.days >= 7))

    return df

#______________

air_reserve = delta_reservation(air_reserve)

hpg_reserve = delta_reservation(hpg_reserve)

#__________________________________________________________________________

air_reserve.rename(columns = {'air_store_id':'store_id'}, inplace = True)

hpg_reserve.rename(columns = {'hpg_store_id':'store_id'}, inplace = True)

total_reserve = pd.concat([air_reserve, hpg_reserve])

total_reserve['date'] = total_reserve['visit_datetime'].apply(lambda x:x.date())
list_visit_ids   = air_visit_data['air_store_id'].unique()

list_reserve_ids = total_reserve['store_id'].unique()

print("nb. of restaurants visited: {}".format(len(list_visit_ids)))

print("nb. of restaurants with reservations: {}".format(len(list_reserve_ids)))

print("intersections of ids: {}".format(len(set(list_visit_ids).intersection(set(list_reserve_ids)))))
df1 = total_reserve[['date', 'reserve_visitors']].groupby('date').sum().reset_index()

df2 = air_visit_data.groupby('visit_date').sum().reset_index()



fig1 = Figure_style(11, 5, 1, 1)

fig1.pos_update(0, 0)

fig1.cust_plot(df2['visit_date'], df2['visitors'], linestyle='-', label = 'nb. of visits')

fig1.cust_plot(df1['date'], df1['reserve_visitors'], color = 'r', linestyle='-', label = 'nb. of reservations')

fig1.style() 

fig1.draw_legend(location = 'upper left')

fig1.set_ylabel('Visitors', fontsize = 14)

fig1.set_xlabel('Date', fontsize = 14)

#________

# limits

date_1 = datetime.datetime(2015,12,1)

date_2 = datetime.datetime(2017,6,1)

fig1.set_xlim(date_1, date_2)

fig1.set_ylim(-50, 25000)
restaurant_id = air_reserve['store_id'][0]
df2 = air_visit_data[air_visit_data['air_store_id'] == restaurant_id]

df0 = total_reserve[total_reserve['store_id'] == restaurant_id]

df1 = df0[['date', 'reserve_visitors']].groupby('date').sum().reset_index()
fig1 = Figure_style(11, 5, 1, 1)

fig1.pos_update(0, 0)

fig1.cust_plot(df2['visit_date'], df2['visitors'], linestyle='-', label = 'nb. of visits')

fig1.cust_plot(df1['date'], df1['reserve_visitors'], color = 'r', linestyle='-', label = 'nb. of reservations')

fig1.style() 

fig1.draw_legend(location = 'upper left')

fig1.set_ylabel('Visitors', fontsize = 14)

fig1.set_xlabel('Date', fontsize = 14)

#________

# limits

date_1 = datetime.datetime(2015,12,21)

date_2 = datetime.datetime(2017,6,1)

fig1.set_xlim(date_1, date_2)

fig1.set_ylim(-3, 39)
restaurant_id = air_reserve['store_id'][2]
df2 = air_visit_data[air_visit_data['air_store_id'] == restaurant_id]

df0 = total_reserve[total_reserve['store_id'] == restaurant_id]

df1 = df0[['date', 'reserve_visitors']].groupby('date').sum().reset_index()
fig1 = Figure_style(11, 5, 1, 1)

fig1.pos_update(0, 0)

fig1.cust_plot(df2['visit_date'], df2['visitors'], linestyle='-', label = 'nb. of visits')

fig1.cust_plot(df1['date'], df1['reserve_visitors'], color = 'r', linestyle='-',

               marker = ['r', 'o', 5], label = 'nb. of reservations')

fig1.style() 

fig1.draw_legend(location = 'upper left')

fig1.set_ylabel('Visitors', fontsize = 14)

fig1.set_xlabel('Date', fontsize = 14)

#________

# limits

date_1 = datetime.datetime(2016,11,1)

date_2 = datetime.datetime(2017,5,1)

fig1.set_xlim(date_1, date_2)

fig1.set_ylim(-3, 45)
fig1 = Figure_style(11, 5, 1, 1)

fig1.pos_update(0, 0)



color = ['r', 'b', 'g']

label = ['delay < 2 days', '2 days < delay < 7 days', 'delay > 7 days']

for j, colonne in enumerate(['delta_2days', 'delta_7days', 'delta_long']):

    df0 = total_reserve[total_reserve[colonne] == 1]

    df1 = df0[['date', 'reserve_visitors']].groupby('date').sum().reset_index()

    fig1.cust_plot(df1['date'], df1['reserve_visitors'], linestyle='-', label = label[j], color = color[j])



fig1.style() 

fig1.draw_legend(location = 'upper left')

fig1.set_ylabel('Visitors', fontsize = 14)

fig1.set_xlabel('Date', fontsize = 14)

#________

# limits

date_1 = datetime.datetime(2017,2,1)

date_2 = datetime.datetime(2017,5,31)

fig1.set_xlim(date_1, date_2)

fig1.set_ylim(-3, 3000)

plt.show()