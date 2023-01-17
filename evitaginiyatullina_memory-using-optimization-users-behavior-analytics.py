# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import copy

import plotly.express as px

import plotly.graph_objects as go

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable = True, theme = 'pearl', offline = True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('..//input/how-to-do-product-analytics/product.csv')
data.head()
data.info()
data.info(memory_usage = 'deep')
data = data.drop(['order_id', 'page_id'], axis = 1)

data.info(memory_usage = 'deep')
print('target column unique values: ', data.target.unique())

print('-----------------------------------------------------')

print('target column in the dataframe = 0')

print(data[data.target == 0])

print('-----------------------------------------------------')

print('target column in the dataframe = 1')

print(data[data.target == 1])
print('Is there target is 1 for banner_show and banner_click in title column')

if data[(data.target == 1)&((data.title == 'banner_show')|(data.title == 'banner_click'))].empty == True:

    print('There are not such rows in the dataframe')

else:

    data[(data.target == 1)&((data.title == 'banner_show')|(data.title == 'banner_click'))]

print('Is there target is 0 for order in title column')

if data[(data.target == 0)&(data.title == 'order')].empty == True:

    print('There are not such rows in the dataframe')

else:

    data[(data.target == 0)&(data.title == 'order')]
data = data.drop('target', axis = 1)

data.info(memory_usage = 'deep')
data.select_dtypes('object').nunique()
for col in ['product', 'site_version', 'title']:

    data[col] = data[col].astype('category')

print(data.info(memory_usage = 'deep'))

data.head()
for dtype in ['int','object']:

    selected_dtype = data.select_dtypes(include=[dtype])

    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()

    mean_usage_mb = mean_usage_b / 1024 ** 2

    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
data['time'] = pd.to_datetime(data['time'])

data['date'] = data['time'].dt.date

data['hour'] = data['time'].dt.hour

data = data.drop('time', axis = 1)

data['hour'] = pd.to_numeric(data['hour'])
data.info()
data.isnull().sum()
data.shape
data2 = copy.deepcopy(data)
print('Products on banners: ', data2['product'].unique())

print('Site versions: ', data2.site_version.unique())

print('Page events: ', data2.title.unique())
fig = go.Figure(go.Funnel(x = data2.groupby('title').user_id.nunique().reset_index().sort_values('user_id', ascending = False)['user_id'],

                          y = data2.groupby('title').user_id.nunique().reset_index().sort_values('user_id', ascending = False)['title'],

                           textinfo = "value+percent initial"))

fig.show()
px.funnel(data2.groupby('product').title.value_counts(), x='title', y = data2.groupby('product').title.value_counts().index.get_level_values('title'), 

          color = data2.groupby('product').title.value_counts().index.get_level_values('product'), title = 'Product funnel')
fig = go.Figure(go.Funnel(x = data2[data2.site_version == 'mobile'].title.value_counts().reset_index()['title'],

                          y = data2[data2.site_version == 'mobile'].title.value_counts().reset_index()['index'],

                           textinfo = "value+percent initial"))

fig.show()
mobile = data2[data2.site_version == 'mobile']

mobile['date'] = pd.to_datetime(mobile['date'])

mobile
px.line(mobile.groupby('date').title.value_counts(), x = mobile.groupby('date').title.value_counts().index.get_level_values(0),

       y = 'title', color = mobile.groupby('date').title.value_counts().index.get_level_values(1))
mobile['weekday'] = mobile['date'].dt.weekday

mobile
px.line(pd.DataFrame(mobile.groupby('hour').title.value_counts()), x = pd.DataFrame(mobile.groupby('hour').title.value_counts()).index.get_level_values(0),

        y = pd.DataFrame(mobile.groupby('hour').title.value_counts()).title, color = pd.DataFrame(mobile.groupby('hour').title.value_counts()).index.get_level_values(1),

       title = 'Mobile users actions through the day')
data[data.site_version == 'mobile'].groupby('product').title.value_counts()
fig = go.Figure()



fig.add_trace(go.Funnel(

    name = 'Accessories',

    y = ["banner_show", "banner_click", "order"],

    x = [1030951,117723,22430],

    textinfo = "value+percent initial"))



fig.add_trace(go.Funnel(

    name = 'Clothes',

    y = ["banner_show", "banner_click", "order"],

    orientation = "h",

    x = [1035058,187814,45738],

    textposition = "inside",

    textinfo = "value+percent initial"))



fig.add_trace(go.Funnel(

    name = 'Company',

    y = ["banner_show", "banner_click", "order"],

    orientation = "h",

    x = [ 1102861,116357, 0],

    textposition = "inside",

    textinfo = "value+percent initial"))



fig.add_trace(go.Funnel(

    name = 'Sneakers',

    y = ["banner_show", "banner_click", "order"],

    orientation = "h",

    x = [1041430,161177,35154],

    textposition = "inside",

    textinfo = "value+percent initial"))



fig.add_trace(go.Funnel(

    name = 'Sports_nutrition',

    y = ["banner_show", "banner_click", "order"],

    orientation = "h",

    x = [1048375,131048,12219],

    textposition = "inside",

    textinfo = "value+percent initial"))



fig.show()
clicks = mobile[mobile.user_id.isin(mobile[mobile.title == 'banner_click'].user_id.unique())]

clicks.sort_values(['user_id', 'date'])

orders = mobile[mobile.user_id.isin(mobile[mobile.title == 'order'].user_id.unique())]

orders.sort_values(['user_id', 'date'])
orders['step'] = orders['product'].astype('object')+'_'+orders['title'].astype('object')

orders = orders.sort_values(['user_id', 'date', 'hour'])
px.scatter(orders.groupby(['hour', 'weekday']).step.value_counts(), x = orders.groupby(['hour', 'weekday']).step.value_counts().index.get_level_values(0),

           y = orders.groupby(['hour', 'weekday']).step.value_counts().index.get_level_values(1), size = 'step', 

           color = orders.groupby(['hour', 'weekday']).step.value_counts().index.get_level_values(2), title = 'Lead actions')

        
a = orders.groupby('user_id').step.unique().reset_index()

a = a.merge(pd.DataFrame(a['step'].values.tolist()), left_on = pd.DataFrame(a['step'].values.tolist()).index, right_on = a.index)

a = a.drop(['key_0','step'], axis = 1)

a = a.fillna('No action')
px.parallel_categories(a, dimensions = a.columns[1:], title = 'Full customer map journey')
after_1 = a[(a[1] == 'clothes_order')|(a[1] == 'accessories_order')|(a[1] == 'sports_nutrition_order')|(a[1] == 'sneakers_order')]

px.parallel_categories(after_1, dimensions = after_1.columns[1:])
after_1 = a[(((a[1] == 'clothes_banner_click')|(a[1] == 'accessories_banner_click')|(a[1] == 'sports_nutrition_banner_click')|(a[1] == 'sneakers_banner_click')|(a[1] == 'company_banner_click'))&

             ((a[2] == 'clothes_order')|(a[2] == 'accessories_order')|(a[2] == 'sports_nutrition_order')|(a[2] == 'sneakers_order')))]

px.parallel_categories(after_1, dimensions = after_1.columns[1:])
a[(((a[1] == 'clothes_banner_click')|(a[1] == 'accessories_banner_click')|(a[1] == 'sports_nutrition_banner_click')|(a[1] == 'sneakers_banner_click')|(a[1] == 'company_banner_click'))&

             ((a[2] == 'clothes_order')|(a[2] == 'accessories_order')|(a[2] == 'sports_nutrition_order')|(a[2] == 'sneakers_order')))].groupby(2)[1].value_counts()
after_1_2 = a[((a[1] == 'clothes_order')|(a[1] == 'accessories_order')|(a[1] == 'sports_nutrition_order')|(a[1] == 'sneakers_order'))

              &((a[3] == 'clothes_banner_click')|(a[3] == 'accessories_banner_click')|(a[3] == 'sports_nutrition_banner_click')|(a[3] == 'sneakers_banner_click')|

               (a[3] == 'clothes_order')|(a[3] == 'accessories_order')|(a[3] == 'sports_nutrition_order')|(a[3] == 'sneakers_order'))]

px.parallel_categories(after_1_2, dimensions = after_1_2.columns[3:])
px.line(mobile[mobile.title == 'banner_click'].groupby('date')['product'].value_counts(), 

        x = mobile[mobile.title == 'banner_click'].groupby('date')['product'].value_counts().index.get_level_values(0),

        y = 'product',

        color = mobile[mobile.title == 'banner_click'].groupby('date')['product'].value_counts().index.get_level_values(1), title = 'Microconversion from mobile users dynamics')
px.line(mobile[mobile.title == 'order'].groupby(['date', 'weekday'])['product'].value_counts(), 

        x = mobile[mobile.title == 'order'].groupby(['date', 'weekday'])['product'].value_counts().index.get_level_values(0),

        y = 'product',

        color = mobile[mobile.title == 'order'].groupby(['date', 'weekday'])['product'].value_counts().index.get_level_values(2), 

        hover_data = [mobile[mobile.title == 'order'].groupby(['date', 'weekday'])['product'].value_counts().index.get_level_values(1)], 

        title = 'Macroconversion dynamics from mobile users')
px.bar(pd.DataFrame(mobile.groupby('weekday').title.value_counts()), x = pd.DataFrame(mobile.groupby('weekday').title.value_counts()).index.get_level_values(0),

       y = pd.DataFrame(mobile.groupby('weekday').title.value_counts()).title, 

      color = pd.DataFrame(mobile.groupby('weekday').title.value_counts()).index.get_level_values(1))
desktop = data2[data2.site_version == 'desktop']

desktop['date'] = pd.to_datetime(desktop['date'])

desktop
fig = go.Figure(go.Funnel(x = data2[data2.site_version == 'desktop'].title.value_counts().reset_index()['title'],

                          y = data2[data2.site_version == 'desktop'].title.value_counts().reset_index()['index'],

                           textinfo = "value+percent initial"))

fig.show()
px.line(desktop.groupby('date').title.value_counts(), x = desktop.groupby('date').title.value_counts().index.get_level_values(0),

       y = 'title', color = desktop.groupby('date').title.value_counts().index.get_level_values(1), title = 'Desktop users activity')
px.line(pd.DataFrame(desktop.groupby('hour').title.value_counts()), x = pd.DataFrame(desktop.groupby('hour').title.value_counts()).index.get_level_values(0),

        y = pd.DataFrame(desktop.groupby('hour').title.value_counts()).title, color = pd.DataFrame(desktop.groupby('hour').title.value_counts()).index.get_level_values(1),

       title = 'Desktop users actions through the day')
data2[data2.site_version == 'desktop'].groupby('product').title.value_counts()
fig = go.Figure()



fig.add_trace(go.Funnel(

    name = 'Accessories',

    y = ["banner_show", "banner_click", "order"],

    x = [410003, 18531, 22121],

    textinfo = "value+percent initial"))



fig.add_trace(go.Funnel(

    name = 'Clothes',

    y = ["banner_show", "banner_click", "order"],

    orientation = "h",

    x = [418070, 32781, 66977],

    textposition = "inside",

    textinfo = "value+percent initial"))



fig.add_trace(go.Funnel(

    name = 'Company',

    y = ["banner_show", "banner_click", "order"],

    orientation = "h",

    x = [477374,28464,0],

    textposition = "inside",

    textinfo = "value+percent initial"))



fig.add_trace(go.Funnel(

    name = 'Sneakers',

    y = ["banner_show", "banner_click", "order"],

    orientation = "h",

    x = [ 411597, 21419, 32565],

    textposition = "inside",

    textinfo = "value+percent initial"))



fig.add_trace(go.Funnel(

    name = 'Sports_nutrition',

    y = ["banner_show", "banner_click", "order"],

    orientation = "h",

    x = [417595,13870,11518],

    textposition = "inside",

    textinfo = "value+percent initial"))



fig.show()
desktop['weekday'] = desktop['date'].dt.weekday

desktop
clicks_desk = desktop[desktop.user_id.isin(desktop[desktop.title == 'banner_click'].user_id.unique())]

clicks_desk.sort_values(['user_id', 'date'])

orders_desk = desktop[desktop.user_id.isin(desktop[desktop.title == 'order'].user_id.unique())]

orders_desk.sort_values(['user_id', 'date'])
orders_desk['step'] = orders_desk['product'].astype('object')+'_'+orders_desk['title'].astype('object')

orders_desk = orders_desk.sort_values(['user_id', 'date', 'hour'])
px.scatter(orders_desk.groupby(['hour', 'weekday']).step.value_counts(), x = orders_desk.groupby(['hour', 'weekday']).step.value_counts().index.get_level_values(0),

           y = orders_desk.groupby(['hour', 'weekday']).step.value_counts().index.get_level_values(1), size = 'step', 

           color = orders_desk.groupby(['hour', 'weekday']).step.value_counts().index.get_level_values(2), title = 'Lead actions')
px.bar(desktop[desktop.title == 'banner_click']['product'].value_counts(), title = 'Microconversion of desktop users')
px.line(desktop[desktop.title == 'banner_click'].groupby('date')['product'].value_counts(), 

        x = desktop[desktop.title == 'banner_click'].groupby('date')['product'].value_counts().index.get_level_values(0),

        y = 'product',

        color = desktop[desktop.title == 'banner_click'].groupby('date')['product'].value_counts().index.get_level_values(1), title = 'Microconversion from desktop users')
px.line(desktop[desktop.title == 'order'].groupby('date')['product'].value_counts(), 

        x = desktop[desktop.title == 'order'].groupby('date')['product'].value_counts().index.get_level_values(0),

        y = 'product',

        color = desktop[desktop.title == 'order'].groupby('date')['product'].value_counts().index.get_level_values(1), title = 'Macroconversion from mobile users')
desktop[desktop.title == 'order'].groupby(['date', 'weekday'])['product'].value_counts()
px.line(desktop[desktop.title == 'order'].groupby(['date', 'weekday'])['product'].value_counts(), 

        x = desktop[desktop.title == 'order'].groupby(['date', 'weekday'])['product'].value_counts().index.get_level_values(0),

        y = 'product',

        color = desktop[desktop.title == 'order'].groupby(['date', 'weekday'])['product'].value_counts().index.get_level_values(2), title = 'Macroconversion from mobile users',

       hover_data = [desktop[desktop.title == 'order'].groupby(['date', 'weekday'])['product'].value_counts().index.get_level_values(1)])
clicks = desktop[desktop.user_id.isin(desktop[desktop.title == 'banner_click'].user_id.unique())]

clicks.sort_values(['user_id', 'date'])

orders = desktop[desktop.user_id.isin(desktop[desktop.title == 'order'].user_id.unique())]

orders.sort_values(['user_id', 'date'])
orders['step'] = orders['product'].astype('object')+'_'+orders['title'].astype('object')

orders = orders.sort_values(['user_id', 'date', 'hour'])
px.scatter(orders.groupby(['hour', 'weekday']).step.value_counts(), x = orders.groupby(['hour', 'weekday']).step.value_counts().index.get_level_values(0),

           y = orders.groupby(['hour', 'weekday']).step.value_counts().index.get_level_values(1), size = 'step', 

           color = orders.groupby(['hour', 'weekday']).step.value_counts().index.get_level_values(2), title = 'Lead actions')

a = orders.groupby('user_id').step.unique().reset_index()

a = a.merge(pd.DataFrame(a['step'].values.tolist()), left_on = pd.DataFrame(a['step'].values.tolist()).index, right_on = a.index)

a = a.drop(['key_0','step'], axis = 1)

a = a.fillna('No action')
px.parallel_categories(a, dimensions = a.columns[1:], title = 'Full customer map journey')
after_1 = a[(a[1] == 'clothes_order')|(a[1] == 'accessories_order')|(a[1] == 'sports_nutrition_order')|(a[1] == 'sneakers_order')]

px.parallel_categories(after_1, dimensions = after_1.columns[1:])
after_1 = a[(((a[1] == 'clothes_banner_click')|(a[1] == 'accessories_banner_click')|(a[1] == 'sports_nutrition_banner_click')|(a[1] == 'sneakers_banner_click')|(a[1] == 'company_banner_click'))&

             ((a[2] == 'clothes_order')|(a[2] == 'accessories_order')|(a[2] == 'sports_nutrition_order')|(a[2] == 'sneakers_order')))]

px.parallel_categories(after_1, dimensions = after_1.columns[1:])
a[(((a[1] == 'clothes_banner_click')|(a[1] == 'accessories_banner_click')|(a[1] == 'sports_nutrition_banner_click')|(a[1] == 'sneakers_banner_click')|(a[1] == 'company_banner_click'))&

             ((a[2] == 'clothes_order')|(a[2] == 'accessories_order')|(a[2] == 'sports_nutrition_order')|(a[2] == 'sneakers_order')))].groupby(2)[1].value_counts()
after_1_2 = a[((a[1] == 'clothes_order')|(a[1] == 'accessories_order')|(a[1] == 'sports_nutrition_order')|(a[1] == 'sneakers_order'))

              &((a[3] == 'clothes_banner_click')|(a[3] == 'accessories_banner_click')|(a[3] == 'sports_nutrition_banner_click')|(a[3] == 'sneakers_banner_click')|

               (a[3] == 'clothes_order')|(a[3] == 'accessories_order')|(a[3] == 'sports_nutrition_order')|(a[3] == 'sneakers_order'))]

px.parallel_categories(after_1_2, dimensions = after_1_2.columns[3:])