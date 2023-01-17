!pip install cufflink
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os #using operating system dependent functionality

import datetime #datetime module supplies classes for manipulating dates and times.

import math # provides access to the mathematical functions

from IPython.display import display, HTML



#For Plotting

# Using plotly + cufflinks in offline mode

import plotly as py

import plotly.graph_objs as go

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.set_config_file(offline=True)

init_notebook_mode(connected=True)



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



#For time series decomposition

from matplotlib import pyplot

from statsmodels.tsa.seasonal import seasonal_decompose



#Pandas option

pd.options.display.float_format = '{:.2f}'.format
# Input data files are available in the "../input/" directory.

# Listing the available files 

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sales_data = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

price_data = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

calender_data = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

submission_data = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")
print("The sales data has '{}' rows and '{}' columns".format(sales_data.shape[0],sales_data.shape[1]))
#Let's have a look at the data

sales_data.head()
#Let's make a list of date columns date_col = [d1,d2,d3,d4...]

date_col = [col for col in sales_data if col.startswith('d_')]
#Let's look at the unique states in the sales dataset

sales_data.state_id.unique()
#Lets look at the number of rows for each state. Value_counts give you that

sales_data.state_id.value_counts()
#Let's have a look at the ratio of the number of rows. Normalize = True gives you the ratio

sales_data.state_id.value_counts(normalize =True) 
#Calcuating total sales for each row/ id by adding the sales of each of the 1913 days

sales_data['total_sales'] = sales_data[date_col].sum(axis=1)

#Adding all the sales for each state

sales_data.groupby('state_id').agg({"total_sales":"sum"}).reset_index()
#Calculating the sales ratio

state_wise_sales_data = sales_data.groupby('state_id').agg({"total_sales":"sum"})/sales_data.total_sales.sum() * 100

state_wise_sales_data = state_wise_sales_data.reset_index()

#Plotting the sales ratio

fig1, ax1 = plt.subplots()

ax1.pie(state_wise_sales_data['total_sales'],labels= state_wise_sales_data['state_id'] , autopct='%1.1f%%',

        shadow=True, startangle=90)# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.title("State Wise total sales percentage",fontweight = "bold")

plt.show()
#Let's have a look at the unique stores

print("Let's have a look at the unique stores - ",sales_data.store_id.unique())
#Caculating the sales ratio for the 10 stores

store_wise_sales_data=sales_data.groupby('store_id').agg({"total_sales":"sum"})/sales_data.total_sales.sum() * 100

#Plotting the sales ratio for the 10 stores

store_wise_sales_data = store_wise_sales_data.reset_index()

fig1, ax1 = plt.subplots()

ax1.pie(store_wise_sales_data['total_sales'],labels= store_wise_sales_data['store_id'] , autopct='%1.1f%%',

        shadow=True, startangle=90)# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.title("Store Wise total sales percentage",fontweight = "bold")

plt.show()
# Let's have a look at the unique categories 

print("Let's have a look at the unique categories -",sales_data.cat_id.unique())
#Let's have a look at the total sales from each of the 3 categries

print("Total Sales from each category")

sales_data.groupby('cat_id').agg({"total_sales":"sum"}).reset_index()
#Caculating the sales ratio for the 3 categories

cat_wise_sales_data = sales_data.groupby('cat_id').agg({"total_sales":"sum"})/sales_data.total_sales.sum() * 100

cat_wise_sales_data = cat_wise_sales_data.reset_index()

#Plotting the sales ratio for the 3 categories

fig1, ax1 = plt.subplots()

ax1.pie(cat_wise_sales_data['total_sales'],labels= cat_wise_sales_data['cat_id'] , autopct='%1.1f%%',

        shadow=True, startangle=90)# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.title("Category Wise total sales percentage",fontweight = "bold")

plt.show()
cat_state_sales =sales_data.groupby(['cat_id','state_id']).agg({"total_sales":"sum"}).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).unstack()

cat_state_sales.columns = [f'{i}_{j}' if j != '' else f'{i}' for i,j in cat_state_sales.columns]

cat_state_sales.plot(kind='bar', stacked=True)

plt.title("Sales Distrubution for each category across states",fontweight = "bold")
#Calculating sales distribution for each state 

state_cat_sales = sales_data.groupby(['state_id','cat_id']).agg({"total_sales":"sum"}).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).unstack()

#Plotting the sales distribution for each state

state_cat_sales.columns = [f'{i}_{j}' if j != '' else f'{i}' for i,j in state_cat_sales.columns]

state_cat_sales.plot(kind='bar', stacked=True)

plt.title("Sales Distrubution for each state across categories",fontweight = "bold")
#Let's look at the unique departments

print("Let's look at the unique departments - ",sales_data.dept_id.unique())
#Calculating sales distribution across departments

dept_sales = sales_data.groupby('dept_id').agg({"total_sales":"sum"})/sales_data.total_sales.sum() * 100

#Plotting

dept_sales = dept_sales.reset_index()

fig1, ax1 = plt.subplots()

ax1.pie(dept_sales['total_sales'],labels= dept_sales['dept_id'] , autopct='%1.1f%%',

        shadow=True, startangle=90)# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.title("Department Wise total sales percentage",fontweight = "bold")

plt.show()
# Calculating the sales distribution of stores

store_dept_sales = sales_data.groupby(['store_id','dept_id']).agg({"total_sales":"sum"}).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).unstack()

store_dept_sales.columns = [f'{i}_{j}' if j != '' else f'{i}' for i,j in store_dept_sales.columns]

#Plotting the sales distribution

store_dept_sales.plot(kind='bar', stacked=True)

plt.title("Sales Distrubution for each store across departments",fontweight = "bold")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
print("Sales distribution(in %) in each store accross different departments")

sales_data.groupby(['store_id','dept_id']).agg({"total_sales":"sum"}).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).unstack()
dept_store_sales = sales_data.groupby(['dept_id','store_id']).agg({"total_sales":"sum"}).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).unstack()

dept_store_sales.columns = [f'{i}_{j}' if j != '' else f'{i}' for i,j in dept_store_sales.columns]

dept_store_sales.plot(kind='bar', stacked=True)

plt.title("Sales Distrubution for each state across categories",fontweight = "bold")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), shadow=True, ncol=3)
#Let's have a look at the price data

price_data.head()
price_data["Category"] = price_data["item_id"].str.split("_",expand = True)[0]

plt.figure(figsize=(12,6))

p1=sns.kdeplot(price_data[price_data['Category']=='HOBBIES']['sell_price'], shade=True, color="b")

p2=sns.kdeplot(price_data[price_data['Category']=='FOODS']['sell_price'], shade=True, color="r")

p3=sns.kdeplot(price_data[price_data['Category']=='HOUSEHOLD']['sell_price'], shade=True, color="g")

plt.legend(labels=['HOBBIES','FOODS',"HOUSEHOLD"])

plt.xscale("log")

plt.xlabel("Log of Prices")

plt.ylabel("Density")

plt.title("Density plot of log of prices accross Categories")
#Let's look at items with the maximum price change and minimum price change over the years

item_store_prices = price_data.groupby(["item_id","store_id"]).agg({"sell_price":["max","min"]})

item_store_prices.columns = [f'{i}_{j}' if j != '' else f'{i}' for i,j in item_store_prices.columns]                                               

item_store_prices["price_change"] = item_store_prices["sell_price_max"] - item_store_prices["sell_price_min"]

item_store_prices_sorted = item_store_prices.sort_values(["price_change","item_id"],ascending=False).reset_index()

item_store_prices_sorted["category"] = item_store_prices_sorted["item_id"].str.split("_",expand = True)[0]

print("Items sorted by maximum price change over the years (top 10)")

item_store_prices_sorted.head(10)
print("Items sorted by least price changes over the years (top 10)")

item_store_prices_sorted.tail(10)
#Plotting boxplot

sns.boxplot(x="price_change", y="category", data=item_store_prices_sorted)

title = plt.title("Boxplot for maximum price change for each item over the years across all categories")
#Let's look at the calender data

calender_data.head()
print("The calender dataset has {} rows and {} columns".format(calender_data.shape[0],calender_data.shape[1]))
# Event names for each event type

events1 = calender_data[['event_type_1','event_name_1',]]

events2 = calender_data[['event_type_2','event_name_2',]]

events2.columns = ["event_type_1","event_name_1"]

events = pd.concat([events1,events2],ignore_index = True)

events = events.dropna().drop_duplicates()

events

events_dict = {k: g["event_name_1"].tolist() for k,g in events.groupby("event_type_1")}

print("Event Names across different Event Types")

pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in events_dict.items()]))
snap_days = calender_data.groupby(['year','month'])['snap_CA','snap_TX','snap_WI'].sum().reset_index()

print("SNAP days for each month across the years for all the states")

snap_days.pivot(index="month",columns = "year",values = ["snap_CA","snap_TX","snap_WI"])
#Setting the start date

base = datetime.datetime(2011,1,29)

#Calculating the total sales in a day

sales_sum = pd.DataFrame(sales_data[date_col].sum(axis =0),columns = ["sales"])

#Adding the date column

sales_sum['datum'] = [base + datetime.timedelta(days=x) for x in range(1913)]

sales_sum.set_index('datum', drop=True, inplace=True)

sales_sum.sort_index(inplace=True)
#Joining the calender data with the sales data to see the impact of events

calender_data['date'] = pd.to_datetime(calender_data['date'])

overall_sales_special = pd.merge(calender_data,sales_sum, left_on = "date", right_on = "datum",how = "right")
overall_sales_special.loc[overall_sales_special.snap_CA==1,"date"].dt.day.unique()
overall_sales_special.loc[overall_sales_special.snap_TX==1,"date"].dt.day.unique()
overall_sales_special.loc[overall_sales_special.snap_WI==1,"date"].dt.day.unique()
#Plotting daily states

sales_sum.iplot(title = "Daily Overall Sales")
result = seasonal_decompose(sales_sum, model='additive')

result.plot()

pyplot.show()
state_level = sales_data.groupby("state_id")[date_col].sum().reset_index().set_index('state_id').T

state_level['datum'] = [base + datetime.timedelta(days=x) for x in range(1913)]

state_level.set_index('datum', drop=True, inplace=True)

state_level.sort_index(inplace=True)

state_level.head()

state_month_level = state_level.groupby(pd.Grouper(freq='1M')).sum()

state_month_level.iplot(title = "Monthly Sales accross States")
#Plotting the sales time series decomposition for each state

res1 = seasonal_decompose(state_month_level["CA"], model='additive')

res2 = seasonal_decompose(state_month_level["TX"], model='additive')

res3 = seasonal_decompose(state_month_level["WI"], model='additive')

def plotseasonal(res, axes ):

    res.observed.plot(ax=axes[0], legend=False)

    axes[0].set_ylabel('Observed')

    res.trend.plot(ax=axes[1], legend=False)

    axes[1].set_ylabel('Trend')

    res.seasonal.plot(ax=axes[2], legend=False)

    axes[2].set_ylabel('Seasonal')

    res.resid.plot(ax=axes[3], legend=False)

    axes[3].set_ylabel('Residual')



fig, axes = plt.subplots(ncols=3, nrows=4, sharex=True, figsize=(12,5))



plotseasonal(res1, axes[:,0])

axes[0,0].set_title("CA")

plotseasonal(res2, axes[:,1])

axes[0,1].set_title("TX")

plotseasonal(res3, axes[:,2])

axes[0,2].set_title("WI")

plt.tight_layout()

plt.show()
store_level = sales_data.groupby("store_id")[date_col].sum().reset_index().set_index('store_id').T

store_level['datum'] = [base + datetime.timedelta(days=x) for x in range(1913)]

store_level.set_index('datum', drop=True, inplace=True)

store_level.sort_index(inplace=True)

store_level.head()

store_month_level = store_level.groupby(pd.Grouper(freq='1M')).sum()

store_month_level.head()
cf.Figure(cf.subplots([store_month_level[['CA_1','CA_2','CA_3','CA_4']].figure(),store_month_level[['TX_1','TX_2','TX_3']].figure(),store_month_level[['WI_1','WI_2','WI_3']].figure()],shape=(1,3),subplot_titles=('CA', 'TX', 'WI'))).iplot()
cat_level = sales_data.groupby("cat_id")[date_col].sum().reset_index().set_index('cat_id').T

cat_level['datum'] = [base + datetime.timedelta(days=x) for x in range(1913)]

cat_level.set_index('datum', drop=True, inplace=True)

cat_level.sort_index(inplace=True)

cat_level.head()

cat_level_level = cat_level.groupby(pd.Grouper(freq='1M')).sum()

cat_level_level.iplot(title = "Monthly Sales accross Categories")
#Plotting the sales time series decomposition for each state

res1 = seasonal_decompose(cat_level_level["FOODS"], model='additive')

res2 = seasonal_decompose(cat_level_level["HOBBIES"], model='additive')

res3 = seasonal_decompose(cat_level_level["HOUSEHOLD"], model='additive')

def plotseasonal(res, axes ):

    res.observed.plot(ax=axes[0], legend=False)

    axes[0].set_ylabel('Observed')

    res.trend.plot(ax=axes[1], legend=False)

    axes[1].set_ylabel('Trend')

    res.seasonal.plot(ax=axes[2], legend=False)

    axes[2].set_ylabel('Seasonal')

    res.resid.plot(ax=axes[3], legend=False)

    axes[3].set_ylabel('Residual')



fig, axes = plt.subplots(ncols=3, nrows=4, sharex=True, figsize=(12,5))



plotseasonal(res1, axes[:,0])

axes[0,0].set_title("FOODS")

plotseasonal(res2, axes[:,1])

axes[0,1].set_title("HOBBIES")

plotseasonal(res3, axes[:,2])

axes[0,2].set_title("HOUSEHOLD")

plt.tight_layout()

plt.show()
dept_level = sales_data.groupby("dept_id")[date_col].sum().reset_index().set_index('dept_id').T

dept_level['datum'] = [base + datetime.timedelta(days=x) for x in range(1913)]

dept_level.set_index('datum', drop=True, inplace=True)

dept_level.sort_index(inplace=True)

dept_level.head()

dept_monthly_level = dept_level.groupby(pd.Grouper(freq='1M')).sum()
cf.Figure(cf.subplots([dept_monthly_level[['FOODS_1','FOODS_2','FOODS_3']].figure(),dept_monthly_level[['HOBBIES_1','HOBBIES_2']].figure(),dept_monthly_level[['HOUSEHOLD_1','HOUSEHOLD_2']].figure()],shape=(1,3),subplot_titles=('FOODS', 'HOBBIES', 'HOUSEHOLD'))).iplot()
dept_cat_level = sales_data.groupby(["state_id","cat_id"])[date_col].sum().reset_index().set_index(["state_id","cat_id"]).T

dept_cat_level['datum'] = [base + datetime.timedelta(days=x) for x in range(1913)]

dept_cat_level.set_index('datum', drop=True, inplace=True)

dept_cat_level.sort_index(inplace=True)

dept_cat_level.columns = [f'{i}_{j}' if j != '' else f'{i}' for i,j in dept_cat_level.columns]

dept_cat_monthly_level = dept_cat_level.groupby(pd.Grouper(freq='1M')).sum()

cf.Figure(cf.subplots([dept_cat_monthly_level[['CA_FOODS','TX_FOODS','WI_FOODS']].figure(),dept_cat_monthly_level[['CA_HOBBIES','TX_HOBBIES','WI_HOBBIES']].figure(),dept_cat_monthly_level[['CA_HOUSEHOLD','TX_HOUSEHOLD','WI_HOUSEHOLD']].figure()],shape=(1,3),subplot_titles=('FOODS', 'HOBBIES', 'HOUSEHOLD'))).iplot()
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']

sales_sum_weekday = sales_sum.groupby(sales_sum.index.weekday_name).mean().reindex(days)

sns.set(rc={'figure.figsize':(15,5)})

sns.barplot(x= sales_sum_weekday.index, y='sales', data=sales_sum_weekday)

cat_level = sales_data.groupby("cat_id")[date_col].sum().reset_index().set_index('cat_id').T

cat_level['datum'] = [base + datetime.timedelta(days=x) for x in range(1913)]

cat_level.set_index('datum', drop=True, inplace=True)

cat_level.sort_index(inplace=True)

cat_level.head()

days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']

sales_cat_weekday = cat_level.groupby([cat_level.index.weekday_name]).mean().reindex(days)

sales_cat_weekday.iplot( kind="bar",title = "Avg. Sales across day of week")
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 

monthly_sales = sales_sum.groupby(sales_sum.index.strftime('%b')).mean().reindex(months)

monthly_sales.iplot( kind="bar",title = "Avg. Sales across months")
cat_level = sales_data.groupby("cat_id")[date_col].sum().reset_index().set_index('cat_id').T

cat_level['datum'] = [base + datetime.timedelta(days=x) for x in range(1913)]

cat_level.set_index('datum', drop=True, inplace=True)

cat_level.sort_index(inplace=True)

cat_level.head()

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 

monthly_sales = cat_level.groupby(cat_level.index.strftime('%b')).mean().reindex(months)

monthly_sales.iplot( kind="bar",title = "Avg. Sales across months")
monthly_sales = sales_sum.groupby(cat_level.index.strftime('%d')).mean()

sales_list = np.array(monthly_sales.values.tolist())

sales_list = np.append(sales_list, np.repeat(np.nan, 4)).reshape(5,7)

labels = range(1,32)

labels = np.append(labels, np.repeat(np.nan, 4)).reshape(5,7)

heat_map= sns.heatmap(sales_list,cmap = "YlGnBu",annot = labels, yticklabels = ("Week 1","Week 2","Week 3","Week 4","Week 5"))

plt.title("Avg. Sales on day of month over the years")

plt.show()
cat_monthly_sales = cat_level.groupby(cat_level.index.strftime('%d')).mean()

foods_list = np.array(cat_monthly_sales['FOODS'].tolist())

foods_list = np.append(foods_list, np.repeat(np.nan, 4)).reshape(5,7)

hobbies_list = np.array(cat_monthly_sales['HOBBIES'].tolist())

hobbies_list = np.append(hobbies_list, np.repeat(np.nan, 4)).reshape(5,7)

household_list = np.array(cat_monthly_sales['HOUSEHOLD'].tolist())

household_list = np.append(household_list, np.repeat(np.nan, 4)).reshape(5,7)

labels = range(1,32)

labels = np.append(labels, np.repeat(np.nan, 4)).reshape(5,7)





fig, (ax1, ax2 , ax3) = plt.subplots(1,3)

foods_map= sns.heatmap(foods_list,cmap = "YlGnBu",annot = labels, yticklabels = ("Week 1","Week 2","Week 3","Week 4","Week 5"), ax =ax1)

hobbies_map= sns.heatmap(hobbies_list,cmap = "YlGnBu",annot = labels, yticklabels = ("Week 1","Week 2","Week 3","Week 4","Week 5"), ax =ax2)

household_map= sns.heatmap(household_list,cmap = "YlGnBu",annot = labels, yticklabels = ("Week 1","Week 2","Week 3","Week 4","Week 5"), ax =ax3)

ax1.set_title('FOODS')

ax2.set_title('HOBBIES')

ax3.set_title('HOUSEHOLD')

plt.suptitle("Avg. Sales on day of month over all years across different categories ")

plt.show()
overall_sales_special.head()
overall_sales_special[overall_sales_special.year == 2012].groupby("date")["sales"].sum().iplot(title = "Daily Overall Sales")
print("Event days in 2012")

overall_sales_special[(overall_sales_special.year == 2012) & ((overall_sales_special.event_name_1.notnull()) | (overall_sales_special.event_name_2.notnull()))]
#Function for tagging events to the preceding weekend 

event_days_sales = overall_sales_special[((overall_sales_special.event_name_1.notnull()) | (overall_sales_special.event_name_2.notnull()))]

overall_sales_special["weekend_precede_event"] = np.nan



def update_weekend_precede_event(week_e,wday,e1,e2):

    e2 = '_' + e2 if type(e2) == str else ''

    drift = e1 + e2

    if wday == 1:

        overall_sales_special.loc[(overall_sales_special['wm_yr_wk']==week_e)&(overall_sales_special['wday']==1),"weekend_precede_event"] = drift

    else:

        overall_sales_special.loc[(overall_sales_special['wm_yr_wk']==week_e)&((overall_sales_special['wday']==1)|(overall_sales_special['wday']==2)),"weekend_precede_event"] = drift

        

_ = event_days_sales.apply(lambda row : update_weekend_precede_event(row['wm_yr_wk'],row['wday'],row['event_name_1'], row['event_name_2']),axis = 1)
print("Events data with added weekend_prece_event column which marks the weekend before each of the event along with the event name")

overall_sales_special.head(10)
#adding event type column

events.columns = ["weekend_precede_event_type","weekend_precede_event"]

overall_sales_special = pd.merge(overall_sales_special,events,how= "left",on="weekend_precede_event")
#Calculating sales impact of each event on preceding weekend

event_type_impact = overall_sales_special.groupby(['weekend_precede_event_type'])['sales'].mean().reset_index()

event_type_impact = event_type_impact.sort_values("sales",ascending = False)

event_type_impact.columns = ["event_type","avg_sales_preceding_weekend"]

#Plotting a bar graph of avg. sales on the weekend days before the event to see the impact

chart = sns.barplot(y= "event_type", x='avg_sales_preceding_weekend', data=event_type_impact)

chart.axvline(sales_sum.sales.mean(),label = "Avg. sales in a day",c='red', linestyle='dashed')

plt.title("Avg. Sales on preceding event of each type of event", fontweight ="bold")

leg = plt.legend()
#Calculating sales impact of each event on preceding weekend

event_impact = overall_sales_special.groupby(['weekend_precede_event'])['sales'].mean().reset_index()

event_impact = event_impact.sort_values("sales",ascending = False)

event_impact.columns = ["events","avg_sales_preceding_weekend"]

# Plotting a bar graph of avg. sales on the weekend days before the event to see the impact

sns.set(rc={'figure.figsize':(15,3)})

chart = sns.barplot(x= "events", y='avg_sales_preceding_weekend', data=event_impact)

chart.axhline(sales_sum.sales.mean(),label = "Avg. sales in a day",c='red', linestyle='dashed')

var = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

plt.title("Avg. sales on preceding weekend of each event",fontweight = "bold")

leg = plt.legend()
#Joining the state wise sales with the events table

overall_sales_special = pd.merge(overall_sales_special,state_level.reset_index(),how = "left",left_on="date",right_on="datum")

overall_sales_special.drop("datum",axis = 1,inplace =True)
#Comparing the days with and w/o SNAP for all 3 states

ca_snap = overall_sales_special.groupby("snap_CA")["CA"].mean().reset_index()

tx_snap = overall_sales_special.groupby("snap_TX")["TX"].mean().reset_index()

wi_snap = overall_sales_special.groupby("snap_WI")["WI"].mean().reset_index()

ca_snap.columns = ["Snap","CA"]

tx_snap.columns = ["Snap","TX"]

wi_snap.columns = ["Snap","WI"]

snap_impact = pd.merge(ca_snap,tx_snap,on = "Snap")

snap_impact = pd.merge(snap_impact,wi_snap,on = "Snap")

snap_impact = pd.melt(snap_impact, id_vars=['Snap'], value_vars=['CA','TX','WI'],var_name='State', value_name='Avg Sales')

#Plotting bar plots for sales comparison

sns.set(rc={'figure.figsize':(10,7)})

chart=sns.barplot(x= "State", y='Avg Sales',hue = 'Snap' ,data=snap_impact)

chart.axhline(overall_sales_special.CA.mean(),label = "Avg. sales CA",c='red', linestyle='dashed')

chart.axhline(overall_sales_special.TX.mean(),label = "Avg. sales TX",c='blue', linestyle='dashed')

chart.axhline(overall_sales_special.WI.mean(),label = "Avg. sales WI",c='black', linestyle='dashed')

var = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

plt.title("Avg. Sales in each state on SNAP days(1) and on other days(0)",fontweight="bold")

leg = plt.legend()
item_sales = sales_data.groupby("item_id")[date_col].sum().reset_index().set_index('item_id').T

#Setting the start date

base = datetime.datetime(2011,1,29)

#Adding the date column

item_sales['date'] = [base + datetime.timedelta(days=x) for x in range(1913)]

item_sales = pd.merge(item_sales,overall_sales_special[["date","wm_yr_wk"]],on = "date")

item_sales=item_sales.groupby(["wm_yr_wk"]).sum().reset_index()

item_sales['date'] = [base + datetime.timedelta(days=x) for x in range(0,1913,7)]

item_sales = item_sales.melt(id_vars=['wm_yr_wk',"date"], value_vars=item_sales.columns.drop(["wm_yr_wk","date"]), var_name='item_id', value_name='sales')

item_sales.head()
item_mean_prices = price_data.groupby("item_id")["sell_price"].median().reset_index()

item_mean_prices.describe()
labels = ["Cheap","Costly"]

#Bucketing each item as cheap or costly

item_mean_prices["item_price_bucket"] =  pd.cut(item_mean_prices.sell_price, [0,3.42,np.inf], include_lowest=True,labels = labels)

item_mean_prices.head()

#Joining with the actual table

price_data_bucketed = pd.merge(price_data,item_mean_prices[["item_id","item_price_bucket"]], on = "item_id",how = "left")

#Joining with Sales data

price_data_bucketed = pd.merge(price_data_bucketed,item_sales,on = ["wm_yr_wk","item_id"],how = "left")



base = datetime.datetime(2011,1,29)

#Adding the date column



price_data_bucketed.head()
#Creating table at category, price bucket level

mean_table = price_data_bucketed.groupby(["date","Category","item_price_bucket"]).agg({"sell_price":"mean","sales":"sum"}).reset_index()

mean_table["cat-bucket"] = mean_table["Category"].astype(str) + '-'+mean_table["item_price_bucket"].astype(str)





#PLotting the graph

sns.set(rc={'figure.figsize':(20,5)})

fig, (ax1, ax2) = plt.subplots(1,2)

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

sns.set_palette(flatui)

prices_plot = sns.lineplot(x = "date",y = "sell_price",hue = "cat-bucket",data = mean_table,ax =ax1)

sales_plot = sns.lineplot(x = "date",y = "sales",hue = "cat-bucket",data = mean_table,ax=ax2)

ax1.title.set_text("Change in avg. prices over the years for cheap and costly items of each category")

ax2.title.set_text("Change in total sales over the years for cheap and costly items of each category")

fig.suptitle('Price and sales changes over the years',fontweight="bold")

ax2.set_yscale('log')

leg = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=3)

leg = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=3)