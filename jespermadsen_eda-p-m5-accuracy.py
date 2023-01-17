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

import matplotlib.pyplot as plt

import plotly.express as px

%matplotlib inline
calendar_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'], usecols=['date','d'])



# udvælger kun de datoer som ligger i sales_train_validation



calendar_stv = calendar_df[:1913] 

calendar_stv.info()
sales_train_validation = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv', index_col='id')

sales_train_validation.head()
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [10, 5]

ax = sales_train_validation.groupby(['store_id'])['cat_id'].value_counts().plot(kind='bar', title="observationer i datasæt fordelt på store og kategory")

ax.set_ylabel('# observationer')

ax.set_xlabel('Store - kategory')



plt.show()
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [6, 2.5]

ax = sales_train_validation.groupby('state_id')['store_id'].nunique().plot(kind='bar', title="Butikker per stat")

ax.set_ylabel('# butikker')

ax.set_xlabel('Stat')

ax.set_ylim(bottom=2)

plt.yticks(np.arange(2,5,1))



plt.show()
aggregate_state_sum = sales_train_validation.groupby(by=['state_id'],axis=0).sum()

aggregate_state_sum.columns = calendar_stv['date']

agg_state_sum_trans = aggregate_state_sum.transpose()
from_year = '2015'

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [20, 10]

plt.rcParams['lines.linewidth'] = 2

ax = agg_state_sum_trans[from_year:].plot(title="Summeret salg per stat fra {}".format(from_year))

ax.set_ylabel('Solgte enheder')

plt.show()
aggregate_state_mean = sales_train_validation.groupby(by=['state_id'],axis=0).mean()

aggregate_state_mean.columns = calendar_stv['date']

agg_state_mean_trans = aggregate_state_mean.transpose()

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [20, 10]

plt.rcParams['lines.linewidth'] = 2

ax = agg_state_mean_trans['2015':].plot(title="Gennemsnitlig salg per stat")

ax.set_ylabel('Solgte enheder')

plt.show()


aggregate_state_mean = sales_train_validation.groupby(by=['state_id', 'store_id'],axis=0).mean()

aggregate_state_mean.columns = calendar_stv['date']

agg_state_mean_trans = aggregate_state_mean.transpose()



plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [25, 20]

plt.rcParams['lines.linewidth'] = 2

fig,ax = plt.subplots(3,1)

for i, state in enumerate(['CA','TX','WI'], start=0):

    ax[i].plot(agg_state_mean_trans['2015':][state])

    ax[i].set_title("Gennemsnitlig salg per butik i {}".format(state))

    ax[i].set_ylabel('Solgte enheder')

    i = i+1

plt.show()
aggregate_state_category = sales_train_validation.groupby(by=['state_id', 'cat_id'],axis=0).sum()



aggregate_state_category.columns = calendar_stv['date']



agg_state_trans = aggregate_state_category.transpose()

fig,ax = plt.subplots(3,1)

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [30, 25]

plt.rcParams['lines.linewidth'] = 2

#ax.legend()

ax[0].plot(agg_state_trans['CA']['2015':])

ax[0].set_title('CA')

ax[0].legend(('FOODS', 'HOBBIES', 'HOUSEHOLD'), loc='upper left')

ax[1].plot(agg_state_trans['TX']['2015':])

ax[1].set_title('TX')

ax[1].legend(('FOODS', 'HOBBIES', 'HOUSEHOLD'), loc='upper left')

ax[2].plot(agg_state_trans['WI']['2015':])

ax[2].set_title('WI')

ax[2].legend(('FOODS', 'HOBBIES', 'HOUSEHOLD'), loc='upper left')

plt.show()



def plot_state_category(sales_train_validation, calendar_dates, state, category='ALL', start_time='2015'):

    sales_state_category = sales_train_validation.loc[sales_train_validation['state_id'] == state ]

    if category != 'ALL' :

        sales_state_category = sales_state_category.loc[sales_state_category['cat_id'] == category]

    aggregate_ssc = sales_state_category.groupby(by=['dept_id'],axis=0).mean()



    aggregate_ssc.columns = calendar_dates['date']



    agg_ssc_trans = aggregate_ssc.transpose()

    plt.style.use('ggplot')

    plt.rcParams['figure.figsize'] = [25, 12]

    plt.rcParams['lines.linewidth'] = 2

    ax = agg_ssc_trans[start_time:].plot(title="MEANed numbers State: {}, Category: {}".format(state, category))

    ax.set_ylabel('Units sold')

    plt.show()

plot_state_category(sales_train_validation, calendar_stv, 'CA', category='FOODS', start_time='2013')
plot_state_category(sales_train_validation, calendar_stv, 'TX', category='FOODS',start_time='2013')
plot_state_category(sales_train_validation, calendar_stv, 'WI', category='FOODS', start_time='2013')
light_sales = sales_train_validation.drop(['item_id','dept_id','cat_id','store_id'], axis=1)

light_sales = light_sales.groupby('state_id').mean()

light_sales.columns = calendar_stv['date']

light_s_t = light_sales.transpose()
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [15, 10]

fig, ax = plt.subplots(2,2)

ax[0][0].plot(light_s_t['20-12-2012':'31-12-2012'])

ax[0][0].set_title('2012')

ax[0][1].plot(light_s_t['20-12-2013':'31-12-2013'])

ax[0][1].set_title('2013')

ax[1][0].plot(light_s_t['20-12-2014':'31-12-2014'])

ax[1][0].set_title('2014')

ax[1][1].plot(light_s_t['20-12-2015':'31-12-2015'])

ax[1][1].set_title('2015')

plt.show()
sales_mean = sales_train_validation.mean()

sales_mean.index = calendar_stv['date']

sales_mean_trans = sales_mean.transpose()
fig = px.scatter(sales_mean_trans, x=sales_mean_trans.index, y=sales_mean_trans.values, trendline='ols')

fig.show()
store_dept = sales_train_validation.groupby(by= ['cat_id'], axis=0).mean()

store_dept.columns = calendar_stv['date']

store_trans = store_dept.transpose()
weekends = ['01-03-2015','01-04-2015','01-10-2015','01-11-2015','01-17-2015', '01-18-2015','01-24-2015', '01-25-2015', '01-31-2015', 

            '02-01-2015', '02-07-2015', '02-08-2015', '02-14-2015', '02-15-2015', '02-21-2015', '02-22-2015', '02-28-2015', 

            '03-01-2015', '03-07-2015', '03-08-2015', '03-14-2015', '03-15-2015', '03-21-2015', '03-22-2015', '03-28-2015',  '03-29-2015']
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [25, 5]

ax = store_trans['01-01-2015':'04-02-2015'].plot(title="Gns. salg 3 måneder jan-mar 2015")

ax.set_ylabel('# enheder')

ax.vlines(weekends, 0, 2.5, colors=['y','c'])

plt.show()
weekends= ['06-02-2012','06-02-2012','06-09-2012','06-10-2012', '06-16-2012','06-17-2012','06-23-2012', '06-24-2012', '06-30-2012', 

           '07-01-2012','07-07-2012','07-08-2012','07-14-2012', '07-15-2012','07-21-2012','07-22-2012', '07-28-2012', '07-29-2012', 

           '08-04-2012','08-05-2012','08-11-2012', '08-12-2012','08-18-2012','08-19-2012', '08-25-2012', '08-26-2012', '09-01-2012','09-02-2012'

            ]

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [25, 5]

ax = store_trans['06-01-2012':'09-02-2012'].plot(title="Gns. salg 3 måneder jun-aug 2012")

ax.set_ylabel('# enheder')

ax.vlines(weekends, 0, 2.5, colors=['y','c'])

plt.show()
sales_train_validation.head(1)