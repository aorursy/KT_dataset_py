import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl

import plotly.graph_objects as go

from scipy import stats 



try:

    import calmap

except:

    ! pip install calmap

    import calmap



plt.style.use('ggplot')

mpl.rcParams['figure.dpi'] = 100

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

# don't use scientific notation

pd.set_option('display.float_format', lambda x: '%.3f' % x)

DATA_DIR="../input/m5-forecasting-accuracy/"
calendar = pd.read_csv(f"{DATA_DIR}calendar.csv")

sales = pd.read_csv(f"{DATA_DIR}sales_train_validation.csv")

sub = pd.read_csv(f"{DATA_DIR}sample_submission.csv")

prices = pd.read_csv(f"{DATA_DIR}sell_prices.csv")
sales.head()
calendar.head()
sub.head()
sales.shape
sales.dept_id.unique()
sales.store_id.unique()
sales.head()
days = [col for col in sales.columns if "d_" in col]
total_per_day = pd.DataFrame()

total_per_day['sales'] = sales[days].sum()

total_per_day['date'] = calendar.date[:1913].values

total_per_day['date_short'] =  total_per_day['date'].str[5:]

total_per_day['date'] = pd.to_datetime(total_per_day['date'],format='%Y-%m-%d')

total_per_day['day'] = total_per_day.date.dt.day

total_per_day['month'] = total_per_day.date.dt.month_name()

total_per_day['weekday'] = total_per_day.date.dt.weekday_name

total_per_day['year'] = total_per_day.date.dt.year

# to have dates as x-axis labels in decomposition plots

total_per_day = total_per_day.set_index("date")

total_per_day.head()
calendar['date'] = pd.to_datetime(calendar['date'],format='%Y-%m-%d')
import numpy as np                                                              

import seaborn as sns                                                           

from scipy import stats                                                         

import matplotlib.pyplot as plt                                                 



ax = sns.distplot(total_per_day.sales)                                    



mu, std = stats.norm.fit(total_per_day.sales)



# Plot the histogram.

# plt.hist(data, bins=25, density=True, alpha=0.6, color='g')



# Plot the PDF.

xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = stats.norm.pdf(x, mu, std)

ax.plot(x, p, 'g')

plt.title("Histogram of total daily sales")

plt.show()
k2, p = stats.normaltest(total_per_day.sales)

alpha = 1e-3

print("Normality test:")

if p < alpha:  # null hypothesis: x comes from a normal distribution

    print("Data is normally distributed")

else:

    print("Data is NOT normally distributed")
fig = go.Figure(layout={"title":{

    'text': "Daily total unit sales","x":0.5}})

fig.add_trace(go.Scatter(x=total_per_day.index, y=total_per_day['sales'],

                         mode='lines',marker_color='green',hovertext=total_per_day.index))

    

fig.show()
from statsmodels.tsa.seasonal import seasonal_decompose



# daily measurements, season repeats every year

decompfreq = 366

model = 'multiplicative'

mpl.rcParams['figure.figsize'] = (15, 10)

decomposition = seasonal_decompose(

    total_per_day['sales'],

    freq=decompfreq)

fig = decomposition.plot()
def values_on_bars():

    for p in ax.patches:

        ax.text(p.get_x() + p.get_width()/2., p.get_height(), str(int(p.get_height())), 

        fontsize=12, ha='center', va='bottom')
mpl.rcParams['figure.figsize'] = (15, 5)

ax = sns.barplot(x='month',y='sales',data=total_per_day,ci=None)

values_on_bars()

plt.title("Average monthly sales",fontsize=15)

plt.show()
sns.barplot(x='weekday',y='sales',data=total_per_day)

plt.title("Average sales by weekday",fontsize=15)

plt.show()
total_cats = sales.groupby("cat_id")[days].sum().T

total_cats['date'] = list(calendar.date[:1913])

total_cats['year'] = pd.to_datetime(total_cats['date'],format='%Y-%m-%d').dt.year

total_cats = total_cats.groupby("year").sum().apply(lambda x:x*100/x.sum(),axis=1)

total_cats
total_cats.plot(kind='bar', stacked=True, title="Category distribution over years (in %)")

plt.legend(title="Category",bbox_to_anchor=(1,1))

plt.show()
cats_per_store = sales.groupby(["cat_id","store_id"])[days].sum().sum(axis=1).unstack(level=0).apply(lambda x:100*x/x.sum(),axis=1)

cats_per_store
cats_per_store.plot(kind='bar', stacked=True,title="Category distribution per stores (in %)")

plt.legend(title="Category",bbox_to_anchor=(1,1))

plt.show()
days = [col for col in sales.columns if "d_" in col]

total_states = sales.groupby("state_id")[days].sum().T

total_states.index = pd.to_datetime(calendar.date[:1913])

total_states.head()
total_states.describe()
fig = go.Figure(layout={"title":{

    'text': "Daily unit sales per state","x":0.5}})

for state in total_states.columns:

    fig.add_trace(go.Scatter(x=total_states.index, y=total_states[state],

                             mode='lines',hovertext=total_states.index,name=state))

    

fig.show()
from statsmodels.tsa.seasonal import seasonal_decompose

trends_states = total_states.copy()

# daily measurements, season repeats every year

decompfreq = 366

model = 'multiplicative'

mpl.rcParams['figure.figsize'] = (15, 5)

fig = go.Figure(layout={"title":{

    'text': "Trends in sales per state","x":0.5}})

for state in total_states.columns:

    decomposition = seasonal_decompose(

        total_states[state],

        freq=decompfreq)

    fig.add_trace(go.Scatter(x=total_states.index, y=decomposition.trend,

                             mode='lines',hovertext=total_states.index,name=state))

    

fig.show()
total_stores = sales.groupby(["store_id"])[days].sum().T

total_stores.index = calendar.date[:1913]

total_stores.head()
from statsmodels.tsa.seasonal import seasonal_decompose

trends_stores = total_stores.copy()

# daily measurements, season repeats every year

decompfreq = 366

model = 'multiplicative'

mpl.rcParams['figure.figsize'] = (15, 5)

fig = go.Figure(layout={"title":{

    'text': "Trends in sales per state","x":0.5}})

for state in total_stores.columns:

    decomposition = seasonal_decompose(

        total_stores[state],

        freq=decompfreq)

    fig.add_trace(go.Scatter(x=total_stores.index, y=decomposition.trend,

                             mode='lines',hovertext=total_stores.index,name=state))

    

fig.show()
total_states = sales[sales.store_id!='CA_3'].groupby("state_id")[days].sum().T

total_states.index = calendar.date[:1913]

from statsmodels.tsa.seasonal import seasonal_decompose

trends_states = total_states.copy()

# daily measurements, season repeats every year

decompfreq = 366

model = 'multiplicative'

mpl.rcParams['figure.figsize'] = (15, 5)

fig = go.Figure(layout={"title":{

    'text': "Trends in sales per state","x":0.5}})

for state in total_states.columns:

    decomposition = seasonal_decompose(

        total_states[state],

        freq=decompfreq)

    fig.add_trace(go.Scatter(x=total_states.index, y=decomposition.trend,

                             mode='lines',hovertext=total_states.index,name=state))

    

fig.show()
for state in ["CA","WI","TX"]:

    print("State",state)

#     snaps_state = total_states[state]

    snaps_state = calendar.set_index("date")["snap_"+state]

    plt.figure(figsize=(15,5))

    calmap.yearplot(snaps_state,year=2015)

    plt.show()
total_states_with_calendar = total_states.T.reset_index().melt(id_vars='state_id',var_name='date',value_name='sales')

total_states_with_calendar = pd.merge(total_states_with_calendar,calendar,on='date')

total_states_with_calendar['date'] = pd.to_datetime(total_states_with_calendar['date'],format='%Y-%m-%d')

total_states_with_calendar['snap'] = total_states_with_calendar.apply(lambda x:int(x["snap_"+x['state_id']]==1),axis=1)

total_states_with_calendar['isHoliday'] = total_states_with_calendar['event_name_1'].notna().astype(int)

total_states_with_calendar = total_states_with_calendar.set_index("date")
from scipy.stats import pearsonr     

pearsonr(total_states_with_calendar['sales'],total_states_with_calendar['snap'])
total_states_with_calendar.groupby("snap")['sales'].describe()
sns.distplot(total_states_with_calendar.loc[total_states_with_calendar['snap']==0,'sales'])

plt.title("Distribution of sales on days without food stamps")

plt.xlim([0,30000])

plt.show()
sns.distplot(total_states_with_calendar.loc[total_states_with_calendar['snap']==1,'sales'])

plt.title('Distribution of sales on days with food stamps')

plt.xlim([0,30000])

plt.show()
most_sold_products = sales.groupby("item_id")[days].sum().stack().sum(level=0).sort_values(ascending=False)
sales.groupby("item_id")[days].sum().loc[['FOODS_3_090','FOODS_3_586','FOODS_3_252'],:]
most_sold_products.head()
prices_desc = prices.groupby("item_id")['sell_price'].describe().fillna(0)

prices_desc['max_diff'] = prices_desc['max']-prices_desc['min']

prices_desc.sort_values(by="max_diff",ascending=False).head()
prices_desc[prices_desc['max_diff']==0].shape[0]/prices_desc.shape[0]
double_holidays = calendar[calendar.event_name_2.notnull()]
double_holidays.head()
total_per_day[total_per_day.index.isin(double_holidays.date)]
total_per_day.head()
total_per_day_with_calendar = pd.DataFrame()

total_per_day_with_calendar['sales'] = sales[days].sum()

total_per_day_with_calendar['date'] = calendar.date[:1913].values

total_per_day_with_calendar = pd.merge(total_per_day_with_calendar,calendar,on='date')
holiday_sales_avg = total_per_day_with_calendar[total_per_day_with_calendar.event_name_1.notna()].groupby(['event_name_1'])['sales'].mean().sort_values(ascending=False)
calendar.event_name_1.unique()
holiday_sales_avg