import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.gridspec as gs

import matplotlib.ticker as ticker

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%matplotlib inline
ts_confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

ts_recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

ts_deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

#open_line = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

#covid_19 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

#line_list = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
cols_to_drop = ['Province/State', 'Lat', 'Long']

deaths_country = ts_deaths.drop(cols_to_drop, axis=1).groupby(['Country/Region']).sum()

recovered_country = ts_recovered.drop(cols_to_drop, axis=1).groupby(['Country/Region']).sum()

confirmed_country = ts_confirmed.drop(cols_to_drop, axis=1).groupby(['Country/Region']).sum()

active_country = confirmed_country - recovered_country - deaths_country
# params for annotation

bbox_params = dict(boxstyle='round, pad=0.7',fc="none")

arrow_params = dict(arrowstyle='->')



#current situation

c = confirmed_country.iloc[:,-1].sum()

d = deaths_country.iloc[:,-1].sum()

r = recovered_country.iloc[:, -1].sum()

active_case = c - d - r



fig, ax = plt.subplots()

ax.pie([d,r,active_case], labels= ['death', 'recovered', 'active case'], shadow=True, autopct='%.1f%%', explode=(0.2,0,0))

text = "it equals {:,} people".format(d)

ax.annotate(text, (1.2,0.2), (1.8,0.2), arrowprops=arrow_params, bbox=bbox_params)

ax.set_title("Total confirmed cases: {:,}".format(c))

plt.show()
# calculate number of active cases per country

c_sum = confirmed_country.sum()

d_sum = deaths_country.sum()

r_sum = recovered_country.sum()

active_sum = c_sum - d_sum - r_sum



fig, ax = plt.subplots(figsize=(12,6))

r_sum.plot()

d_sum.plot()

active_sum.plot()

legend = ['recovered', 'death', 'active case']

text = "This seemed to be \n an optimistic period"

annot_x = r_sum.index.get_loc('3/8/20')

annot_y = r_sum.iloc[annot_x]

# ax.annotate(text, xy=(annot_x, annot_y), xytext=(annot_x-15,annot_y*4), arrowprops=arrow_params, bbox=bbox_params)

ax.annotate('recovered', (r_sum.size-1,r_sum[-1]), (r_sum.size,r_sum[-1]))

ax.annotate('death', (d_sum.size-1, d_sum[-1]), (d_sum.size, d_sum[-1]))

ax.annotate('active case', (active_sum.size-1, active_sum[-1]), (active_sum.size, active_sum[-1]))

ax.set_title('Cases in Progress')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

plt.show()
fig = plt.figure(figsize=(15,6))

specs = gs.GridSpec(ncols=8, nrows=1, figure=fig)

fig_ax0 = fig.add_subplot(specs[:,0:5])

fig_ax1 = fig.add_subplot(specs[:,6:])



# matching bar's color to line plot

plt0 = confirmed_country.T.plot(legend=False, ax=fig_ax0)

top_10 = confirmed_country.iloc[:,-1].sort_values(ascending=False).head(10)

top_10_iloc = [confirmed_country.index.get_loc(i) for i in reversed(top_10.index)]

colors_top10 = [plt0.get_lines()[i].get_color() for i in top_10_iloc]

plt1 = top_10.sort_values().plot(kind='barh', color = colors_top10, ax=fig_ax1)

fig_ax1.set_ylabel('')

for i, v in enumerate(top_10.sort_values().values):

    fig_ax1.text(v, i, '{:,}'.format(v), color = 'red')



fig_ax1.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: x/1000))



fig_ax0.spines['top'].set_visible(False)

fig_ax0.spines['right'].set_visible(False)

fig_ax1.spines['right'].set_visible(False)

fig_ax1.spines['top'].set_visible(False)

fig_ax0.set_title('Confirmed Cases per Country')

fig_ax1.set_title('Current Top 10')

plt.show()
fig = plt.figure(figsize=(15,6))

specs = gs.GridSpec(ncols=8, nrows=1, figure=fig)



ax0 = fig.add_subplot(specs[0, 0:5])

ax1 = fig.add_subplot(specs[0, 6:])



plt0 = deaths_country.T.plot(legend=False, ax=ax0)

ax0.set_title('Death Cases per Country')



top_10_death = deaths_country.iloc[:,-1].sort_values(ascending=False).head(10)

top_10_death_iloc = [deaths_country.index.get_loc(i) for i in top_10_death.index]

colors_top10 = [plt0.get_lines()[i].get_color() for i in top_10_death_iloc]



plt1 = top_10_death.sort_values().plot(kind='barh', color=reversed(colors_top10), ax=ax1)

for i,v in enumerate(top_10_death.sort_values().values):

    ax1.text(v,i, str(v)) 



ax0.spines['top'].set_visible(False)

ax0.spines['right'].set_visible(False)

ax1.spines['top'].set_visible(False)

ax1.spines['right'].set_visible(False)

ax1.set_ylabel('')

ax1.set_title('Current Top 10')

plt.show()
# death cases in percent

death_percentage = deaths_country.iloc[:,-1]/confirmed_country.iloc[:,-1]*100

death_percentage.drop(confirmed_country.index[confirmed_country.iloc[:,-1]<1000], inplace=True)

fig, ax = plt.subplots(figsize=(12,6))

death_percentage.sort_values().tail(10).plot(kind='barh', color='red')

ax.set_title('Highest Death Rate (%)')

ax.set_ylabel('')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

for i, v in enumerate(death_percentage.sort_values().tail(10)):

    ax.text(v, i, '{:.1f}'.format(v))

plt.show()
top_10_active = active_country.iloc[:,-1].sort_values().tail(10)

active_percentage = active_country.iloc[:,-1]/confirmed_country.iloc[:,-1]*100

fig,ax = plt.subplots(figsize=(12,6))

top_10_active.plot(kind='barh', width=0.8, ax=ax)

for i,v in enumerate(top_10_active):

    ax.text(v, i, '{:,} / {:.1f}%'.format(v, active_percentage.loc[top_10_active.index[i]]))



ax.set_ylabel('')

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.set_title('Top 10 in Actual Number')

plt.show()
# calculate daily new cases in each country

temp_df = confirmed_country.copy()

temp_df.drop(temp_df.columns[-1], axis=1, inplace=True)

temp_df.insert(0, '1/21/20', temp_df.iloc[:,0])

temp_df.columns = confirmed_country.columns

difference_daily = confirmed_country.subtract(temp_df)





fig = plt.figure(figsize=(20,10))

specs = gs.GridSpec(ncols=1, nrows=2, figure=fig, hspace=1)

ax0 = fig.add_subplot(specs[0,0])

ax1 = fig.add_subplot(specs[1,0])

#ax2 = fig.add_subplot(specs[2,0])



difference_daily.loc[top_10.index,:].T.plot(marker='', figsize=(10,6), ax = ax0)

top_10_diff = difference_daily.loc[top_10.index,difference_daily.columns[-1]]

ax0.legend(bbox_to_anchor=(1, 1))

ax0.spines['right'].set_visible(False)

ax0.spines['top'].set_visible(False)

ax0.set_title('Daily New Cases in Current Top 10 Countries')



difference_daily.sum().plot(color='black', ax = ax1)

difference_daily.drop('US').sum().plot(color='#ffff00', ax=ax1)

difference_daily.loc['US'].plot(color='blue', ax=ax1)

ax1.spines['right'].set_visible(False)

ax1.spines['top'].set_visible(False)

ax1.set_title('Daily New Cases from around the World')



ax1.legend(['World', 'All but US', 'US'], bbox_to_anchor=(1,1))



plt.show()