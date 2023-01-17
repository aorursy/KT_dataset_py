# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import os

import dask

import dask.dataframe as dd



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px





# # Display all cell outputs

# from IPython.core.interactiveshell import InteractiveShell

# InteractiveShell.ast_node_interactivity = 'all'





from plotly.offline import iplot, init_notebook_mode

import cufflinks as cf

import plotly.graph_objs as go

# import chart_studio.plotly as py



init_notebook_mode(connected=True)

cf.go_offline(connected=True)



# Set global theme

cf.set_config_file(world_readable=True, theme='pearl')
# from google.colab import drive

# drive.mount('/content/gdrive')
# import os

# os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/Kaggle"

# # /content/gdrive/My Drive/Kaggle is the path where kaggle.json is present in the Google Drive
# #changing the working directory

# %cd /content/gdrive/My Drive/Kaggle

# #Check the present working directory using pwd command 
!kaggle datasets download -d mkechinov/ecommerce-events-history-in-cosmetics-shop
# #unzipping the zip files and deleting the zip files

# !unzip \2019-Dec.csv.zip  && rm *.zip
# from glob import glob

# df1 = dd.read_csv(os.path.join('/content/gdrive/My Drive/Kaggle/2019-Dec.csv'))



df1 = dd.read_csv("/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2019-Dec.csv")
df1.head()
df1.visualize()
df1.npartitions
# Using verbose displays full info

# df1.info(verbose=True)
df1.dtypes
sample_df = df1.sample(frac = 0.3, random_state=101)
sample_df.info(verbose=True)


# Create null cols and non-null columns

null_cols = sample_df.columns[sample_df.isnull().sum().compute()>0]

# Non nulls are those with same lenght as df

not_null_cols = sample_df.columns[sample_df.notnull().sum().compute()==20692840]

# Create null df and non-null df

null_df= sample_df[null_cols]

not_null_df= sample_df[not_null_cols]

df_null_sum = sample_df.isnull().sum().compute()

df_is_null = sample_df.isnull().compute()
#sample_null = df_is_null.sample(100000, random_state=101)

sample_null_sort = df_is_null.sort_values('user_id')
# Using sample to prevent kernel crash

sns.heatmap(sample_null_sort,cbar=False, cmap='magma', yticklabels=False,)

plt.title('Missing values in sample dataset')
# sample_df['category_code'].value_counts().compute()
new_df = sample_df.drop(columns = 'category_code', axis = 1).copy() #drop the category_code column
# (new_df.brand.isnull().sum()/len(new_df)).compute() #Check the proportion of null values in brand column
new_df.brand = new_df.brand.replace(np.nan,'Not Available') #replace Nan values to 'Not available' text
# new_df.brand.value_counts().compute() 
new_df['price'].describe().compute() # check statistic state of price column
# len(new_df['price'][new_df['price']<0])/len(new_df['price']) 

# % of column with negative price comparing with total amount of values in price column
new_df = new_df[new_df['price']>= 0] # take only rows with positive values in price column
# (new_df.user_session.isnull().sum()/len(new_df)).compute()
new_df.user = new_df.user_session.replace(np.nan, 'Not Available') #replace Nan values with 'Not Available' text
pd_df = new_df.compute()
pd_df.event_time[:1].str.slice(-3)
# date = pd_df.event_time.str.slice(0,10)

yr=  pd_df.event_time.str.slice(0,4)

mo = pd_df.event_time.str.slice(5,7)

da = pd_df.event_time.str.slice(8,10)

time = pd_df.event_time.str.slice(10,-3).str.strip()

time_zone = pd_df.event_time.str.slice(-3).str.strip()



hr = time.str.slice(0,2).str.strip()



min = time.str.slice(3,5).str.strip()



sec= time.str.slice(6).str.strip()

mo = mo.str.zfill(2)

da =da.str.zfill(2)
hr = hr.str.zfill(2)

min=min.str.zfill(2)

sec= sec.str.zfill(2)
date_df = pd.DataFrame({'yr':yr,'mo':mo,'da':da})

date = date_df.astype(str).apply("-".join,axis=1)

time_df = pd.DataFrame({'hr':hr,'min':min,'sec':sec})

time = time_df.astype(str).apply(":".join,axis=1)

pd_df['Date'] = date 

pd_df['time'] = time
pd_df['Date'] = pd.to_datetime(pd_df['Date'], format="%Y-%m-%d")
pd_df['hr'] = pd.to_datetime(pd_df['time'], format="%H:%M:%S").dt.hour

# pd_df['hr'] = pd.to_datetime(pd_df['time'], format="%H:%M:%S").dt.time
# Set index and drop

pd_df.set_index('Date', inplace=True, drop=True)
# drop event_type columns and time for now 

pd_df.drop(['event_time','time'], axis=1, inplace=True)
# pd_df.head()
# Just incase if we need dask df



new_df = dd.from_pandas(pd_df, npartitions=7)
# Check customer behavior on the site

custmoer_behavior_share =new_df.event_type.value_counts().compute()/len(new_df)*100
labels = custmoer_behavior_share.index.tolist()

values = custmoer_behavior_share.values.tolist()

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict( line=dict(color='#000000', width=2)),)

fig.update_layout(title="Customer Beahviour",

                 

                    font=dict(

                        family="Courier New, monospace",

                              size=18,

                              color="#7f7f7f"))

# fig.show(renderer="colab")

fig.show(renderer="kaggle")
# How many visitors every day at the site

visitor_by_date = pd_df.groupby(pd_df.index)['user_id'].nunique()
# days are numbered fromo 0-7 starting from monday.

# so 0-4 gives monday-friday

#5 and 6 gives sat and sunday

weekends_df = pd_df[pd_df.index.dayofweek>4]

weekdays_df = pd_df[pd_df.index.dayofweek<=4]
# Separate visitord by weekdays and weekends

weekends_visitors = weekends_df.groupby(weekends_df.index)['user_id'].nunique()

weekdays_visitors = weekdays_df.groupby(weekdays_df.index)['user_id'].nunique()
# Experiment of gaining size value but with same denominator, this is done to gain different markers  size in scatter plot.

print(weekdays_visitors.values//200)

print(weekends_visitors.values//200)
fig = go.Figure()

fig.add_trace(go.Scatter(

            x=weekends_visitors.index.tolist(), y=weekends_visitors.values.tolist(),mode="lines+markers", name="Weekends",marker=dict(size=weekends_visitors.values//200)))

fig.add_trace(go.Scatter(

            x=weekdays_visitors.index.tolist(), y=weekdays_visitors.values.tolist(),mode="markers",name="Weekdays",marker=dict(size=weekdays_visitors.values//200)))

fig.update_layout(title="Number of Visits Everyday", xaxis_title="December 1 - 31",yaxis_title="Frequency")

fig.show(renderer="kaggle")
event_by_date = pd_df.groupby(pd_df.index)['event_type']

y = pd.DataFrame(event_by_date.value_counts().unstack())

y_d = np.array(y[['view','cart','remove_from_cart','purchase']])
title = 'Visitor actions on Page'

labels = ['View', 'Cart', 'Remove from cart', 'Purchase']

colors = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']



mode_size = [8, 8, 12, 8]

line_size = [2, 2, 4, 2]



x_data = y.index

y_data = y_d



fig = go.Figure()

for i in range(0,4):

    fig.add_trace(go.Scatter(x=x_data, y = y_d[:,i], mode='lines',

        name=labels[i],

        line=dict(color=colors[i], width=line_size[i]),

        connectgaps=True,

    ))



fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=False,

    ),

    autosize=False,

    margin=dict(

        autoexpand=False,

        l=100,

        r=20,

        t=110,

    ),

    showlegend=False,

    plot_bgcolor='white'

)





annotations = []



# Adding labels

for y_trace, label, color in zip(y_data, labels, colors):

    # labeling the left_side of the plot

    annotations.append(dict(xref='paper', x=0.08, y=y_trace[0],

                                  xanchor='right', yanchor='middle',

                                  text=label + ' {}'.format(y_trace[0]),

                                  font=dict(family='Arial',

                                            size=10),

                                  showarrow=False))

    

# Title

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Dec 2019',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

# Source

annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,

                              xanchor='center', yanchor='top',

                              text='Visitor behavior on Page',

                              font=dict(family='Arial',

                                        size=12,

                                        color='rgb(150,150,150)'),

                              showarrow=False))



fig.update_layout(annotations=annotations)



fig.show(renderer="kaggle")
brand = new_df['brand'].value_counts().compute()

print(brand)
fig = go.Figure(data=[go.Histogram(x=brand[1:],nbinsx=10, histnorm='probability')])

fig.update_layout(

    title_text='Brand Frequency', # title of plot

    xaxis_title_text='Brand Appearance Frequency', # xaxis label

    yaxis_title_text='Percentage', # yaxis label

    bargap=0.2, # gap between bars of adjacent location coordinates

    #bargroupgap=0.1 # gap between bars of the same location coordinates

)

# fig.show(renderer="colab")

fig.show(renderer="kaggle")
# create a list of brands (different from Not available) mentioned more than 10000 times.

brand_list = brand[1:][brand >= 10000].index 

# filter out a list of rows with brands in top 6.1%

best_brands = new_df[new_df['brand'].isin(brand_list)]
len(brand_list) # count top 6.1% brands
j = new_df['event_type'].value_counts().compute()

j = j.drop('remove_from_cart')

j 
fig = go.Figure()

for i in range(len(brand_list)):

    name = brand_list[i]

    j = best_brands[best_brands['brand']==name]['event_type'].value_counts().compute()

    j = j.drop('remove_from_cart')



    fig.add_trace(go.Funnel(

        name = name,

        y = j.index,

        x = j,

        orientation = "h",

        textposition = "inside",

        textinfo = "value+percent initial"))



fig.update_layout(

    title_text='Customre behavior statistic for 15 most-popular brand', # title of plot

    yaxis_title_text='Customer behavior', # xaxis label

    xaxis_title_text='Brand performance', # yaxis label

    )



# fig.show(renderer="colab")

fig.show(renderer= "kaggle")
fig = go.Figure() 



cus_act = ['view','cart','remove_from_cart','purchase']

c = ['greenyellow', 'blue', 'violet', 'tomato']

for i in range(len(cus_act)):

    fig.add_trace(go.Scatter(x=best_brands['hr'][best_brands['event_type'] == cus_act[i]],

                             y=best_brands['price'][best_brands['event_type'] == cus_act[i]],

                             mode='markers', 

                             marker_color=c[i],

                             marker_size = 3, 

                             name=cus_act[i]))



fig.update_layout(

    title_text='Customre behavior statistic in by daily hour', # title of plot

    xaxis_title_text='Day hour', # xaxis label

    yaxis_title_text='Product price', # yaxis label

    xaxis={'categoryorder':'category ascending'},

    # bargap=0.2, # gap between bars of adjacent location coordinates

    #bargroupgap=0.1 # gap between bars of the same location coordinates

)

fig.show(renderer="kaggle")
grp_by_hr_event_type = pd_df.groupby(['hr','event_type']).count()
layout= dict(title="Hourly Store Traffic", xaxis_title="Time of Day", yaxis_title="Number of  Users")

grp_by_hr_event_type['user_id'].unstack(1).iplot(kind="bar", layout=layout)
grp_id_event = pd_df.groupby(['user_id','event_type']).hr.max() - pd_df.groupby(['user_id','event_type']).hr.min()

grp_hr_id_event = grp_id_event.unstack()


mean_per_2000_user_cart = [grp_hr_id_event.cart[0+x:2000+x].mean() for x in range(0,grp_hr_id_event.shape[0],2000) ]

mean_per_2000_user_purchase = [grp_hr_id_event.purchase[0+x:2000+x].mean() for x in range(0,grp_hr_id_event.shape[0],2000) ]

mean_per_2000_user_remove_from_cart = [grp_hr_id_event.remove_from_cart[0+x:1000+x].mean() for x in range(0,grp_hr_id_event.shape[0],2000) ]

mean_per_2000_user_view = [grp_hr_id_event.view[0+x:2000+x].mean() for x in range(0,grp_hr_id_event.shape[0],2000) ]

index = [x+1000 for x in range(0,grp_hr_id_event.shape[0],2000)]
# Create df makes plotting easier

mean_per_2000_user = pd.DataFrame({'cart':mean_per_2000_user_cart,

                                   'purchase':mean_per_2000_user_purchase,

                                   "remove_from_cart":mean_per_2000_user_remove_from_cart,

                                   "view":mean_per_2000_user_view                                  

                                  }, index=index)
layout= dict(title="Avg Hourly Activity per 2000 user", xaxis_title="Number of Users", yaxis_title="Avg Hours Spent By Users")

mean_per_2000_user.iplot(kind="scatter", layout=layout)
event_type_purchase = pd_df[pd_df['event_type']=="purchase"]
event_type_purchase.price.describe()
event_type_purchase.price.iplot(kind="hist", title="Amount Spent on Purchase Distribution", xaxis_title="Money", yaxis_title="Frequency")