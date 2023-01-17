import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

from IPython.display import Image
file_path = '/kaggle/input'



import os

for dirname, _, filenames in os.walk(file_path):

    for filename in filenames:

        print(os.path.join(dirname, filename))
bce_data = pd.read_csv('/kaggle/input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv')
bce_data.shape
bce_data.info()
bce_data.columns = bce_data.columns.str.lower()
bce_data.columns = bce_data.columns.str.replace(' ', '_')
bce_data.head()
print('No. of missing values = {}'.format(bce_data.isna().sum().sum()))
def get_df_summary(df):

    

    '''This function is used to summarise especially unique value count and data type for variable'''

    

    unq_val_cnt_df = pd.DataFrame(df.nunique(), columns = ['unq_val_cnt'])

    unq_val_cnt_df.reset_index(inplace = True)

    unq_val_cnt_df.rename(columns = {'index':'variable'}, inplace = True)

    unq_val_cnt_df = unq_val_cnt_df.merge(df.dtypes.reset_index().rename(columns = {'index':'variable', 0:'dtype'}),

                                          on = 'variable')

    unq_val_cnt_df = unq_val_cnt_df.sort_values(by = 'unq_val_cnt', ascending = False)

    

    return unq_val_cnt_df
unq_val_cnt_df = get_df_summary(bce_data)
unq_val_cnt_df
bce_data['border'].value_counts()
bce_data['measure'].value_counts()
bce_data['state'].value_counts().sort_index()
bce_data['date'] = pd.to_datetime(bce_data['date'], format = '%m/%d/%Y %I:%M:%S %p')
bce_data.head()
bce_data['location_bkup'] = bce_data['location'].copy()
bce_data['location_bkup'] = bce_data['location_bkup'].str.lstrip('POINT (').str.rstrip(')')
tmp_df = bce_data['location_bkup'].str.split(' ', expand = True)

tmp_df.rename(columns = {0:'longitude', 1:'latitude'}, inplace = True)

tmp_df['longitude'] = tmp_df['longitude'].astype('float')

tmp_df['latitude'] = tmp_df['latitude'].astype('float')



bce_data = pd.concat([bce_data, tmp_df], axis = 1)



del tmp_df
bce_data.head()
bce_data.drop(columns = ['location', 'location_bkup'], inplace = True)
bce_data.head()
print('Years : {}'.format(bce_data['date'].dt.year.unique()))

print()

print('No. of years : {}'.format(bce_data['date'].dt.year.nunique()))
print('Months : {}'.format(sorted(bce_data['date'].dt.month.unique())))
print('Days : {}'.format(sorted(bce_data['date'].dt.day.unique())))
bce_data['year'] = bce_data['date'].apply(lambda x : x.year)
bce_data['month'] = bce_data['date'].apply(lambda x : x.month)



# Map month number to month name.

month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}



bce_data['month'] = bce_data['month'].apply(lambda x : month_dict[x])
# bce_data['month'].value_counts(dropna = False)
bce_data.drop(columns = 'date', inplace = True)
bce_data.groupby(['year', 'month'])['port_code'].count().reset_index().groupby(['year'])['month'].count()
tmp_df = bce_data[['port_name', 'port_code']].drop_duplicates()
tmp_df.groupby('port_name')['port_name'].filter(lambda x : len(x) > 1)
tmp_df.loc[tmp_df['port_name'] == 'Eastport', ]
bce_data.loc[bce_data['port_name'] == 'Eastport', ].groupby(['state', 'port_name', 'port_code'])['state'].count()
print('No. of negative values in value field : {}'.format(bce_data['value'].lt(0).sum()))
tmp_df = bce_data.groupby(['state', 'port_name'])['port_code'].count().reset_index().drop(columns = 'port_code')

tmp_df.groupby(['state', 'port_name']).filter(lambda x : len(x) > 1)
print('Given data set has data of {} years.'.format(bce_data['year'].nunique()))
print('# of obs. with invalid latitudinal values : {}'.format(sum(bce_data['latitude'].lt(-90) & bce_data['latitude'].gt(90))))

print('# of obs. with invalid longitudinal values : {}'.format(sum(bce_data['longitude'].lt(-180) & bce_data['longitude'].gt(180))))



# We do not have any obs. with invalid values in latitude and longitude variables.
tmp_df = bce_data.groupby('border')['state'].nunique().reset_index().rename(columns = {'state':'count'})



px.bar(x = 'border', 

       y = 'count', 

       data_frame = tmp_df, 

       color = 'border', 

       labels = {'border':'Border', 'count':'No. of States'},

       width = 800,

       height = 500,

       title = 'No. of States entry from each border')
tmp_df = bce_data.groupby('border')['port_code'].nunique().reset_index().rename(columns = {'port_code':'count'})



px.bar(x = 'border', 

       y = 'count', 

       data_frame = tmp_df, 

       color = 'border', 

       labels = {'border':'Border', 'count':'No. of Ports'},

       width = 800,

       height = 500,

       title = 'No. of Ports entry from each border')
tmp_df = bce_data.groupby(['border', 'year'])['value'].sum().reset_index()



fig = px.line(x = 'year', 

              y = 'value', 

              data_frame = tmp_df, 

              color = 'border',

              labels = {'border':'Border', 'value':'No. of border entries', 'year':'Year'},

              width = 800,

              height = 500,

              title = 'Frequency of Border Crossings Year-wise'

              )

fig.update_traces(mode='markers+lines')

fig
bce_data.head()
tmp_df = bce_data.groupby(['border', 'year', 'month'])['value'].sum().reset_index()



max_indices = tmp_df.groupby(['border', 'year'])['value'].idxmax().values

min_indices = tmp_df.groupby(['border', 'year'])['value'].idxmin().values



tmp_df.loc[max_indices, 'value_type'] = 'max'

tmp_df.loc[min_indices, 'value_type'] = 'min'



tmp_df.dropna(subset = ['value_type'], inplace = True)
# tmp_df.head()
tmp_df = tmp_df.merge(tmp_df.groupby(['border', 'year'])['value'].sum().reset_index().rename(columns = {'value':'total_value'}), 

                      on = ['border', 'year'])
tmp_df['prop'] = round(tmp_df['value'] * 100 / tmp_df['total_value'], 2).astype('str') + ' %'
# tmp_df.head()
fig = px.bar(x = 'year', 

             y = 'value', 

             data_frame = tmp_df.loc[tmp_df['border'] == 'US-Canada Border'], 

             color = 'month',

             labels = {'month':'Month', 'value':'No. of border entries', 'year':'Year Month', 'prop':'Proportion'},

             title = 'Min-Max Border Crossing Frequencies - For US-Canada Border',

             hover_data = ['prop'],

             text = 'month')

fig.show()
fig = px.bar(x = 'year', 

             y = 'value', 

             data_frame = tmp_df.loc[tmp_df['border'] == 'US-Mexico Border'], 

             color = 'month',

             labels = {'month':'Month', 'value':'No. of border entries', 'year':'Year Month', 'prop':'Proportion'},

             title = 'Min-Max Border Crossing Frequencies - For US-Mexico Border',

             hover_data = ['prop'],

             text = 'month')

fig.show()
tmp_df = bce_data.groupby(['border', 'port_name'])['value'].sum().reset_index()

tmp_df['rank']=tmp_df.groupby(['border'])['value'].rank(ascending = False)

tmp_df = tmp_df.loc[tmp_df['rank'] <= 5]
filter_cond_1 = (tmp_df['border'] == 'US-Canada Border')

port_name_list = tmp_df.loc[filter_cond_1, ].sort_values(by = 'value', ascending = False)['port_name'].tolist()



fig = px.bar(x = 'port_name', 

             y = 'value', 

             data_frame = tmp_df.loc[filter_cond_1], 

             labels = {'port_name':'Port Name', 'value':'No. of border entries'},

             title = 'Top-5 Border Crossing Ports - For US-Canada Border',

             category_orders = {'port_name':port_name_list},

             color_discrete_sequence = ['#11abab'])

fig.show()
filter_cond_1 = (tmp_df['border'] == 'US-Mexico Border')

port_name_list = tmp_df.loc[filter_cond_1, ].sort_values(by = 'value', ascending = False)['port_name'].tolist()



fig = px.bar(x = 'port_name', 

             y = 'value', 

             data_frame = tmp_df.loc[filter_cond_1], 

             labels = {'port_name':'Port Name', 'value':'No. of border entries'},

             title = 'Top-5 Border Crossing Ports - For US-Mexico Border',

             category_orders = {'port_name':port_name_list},

             color_discrete_sequence = ['#dc2d55'])

fig.show()
tmp_df = bce_data.groupby(['border', 'measure'])['value'].sum().reset_index()

tmp_df['rank'] = tmp_df.groupby(['border'])['value'].rank(ascending  = False)

tmp_df = tmp_df.loc[tmp_df['rank'] <= 5]

# tmp_df
fig = px.bar(x = 'border', 

             y = 'value', 

             data_frame = tmp_df, 

             labels = {'border':'Border', 'measure':'Vehicle Type', 'value':'No. of border entries'},

             title = 'Top-5 Border Crossing Vehicle Types',

             color = 'measure')

fig.show()
tmp_df = bce_data.groupby(['border', 'year', 'measure'])['value'].sum().reset_index()

tmp_df['rank'] = tmp_df.groupby(['border', 'year'])['value'].rank(ascending  = False)

tmp_df = tmp_df.loc[tmp_df['rank'] == 1, ]
tmp_df['measure'].unique()
fig = px.line(x = 'year', 

              y = 'value', 

              data_frame = tmp_df, 

              color = 'border',

              labels = {'border':'Border', 'value':'No. of border entries', 'year':'Year'},

              width = 800,

              height = 500,

              title = 'Frequency of Border Crossings by "Personal Vehicle Passengers" Year-wise'

              )

fig.update_traces(mode = 'markers + lines')

fig.show()
Image("/kaggle/input/canada-us-mexico-map/images_1.jpg")