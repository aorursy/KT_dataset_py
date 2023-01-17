import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import itertools as itr

import datetime as datetime
data_file_path = '/kaggle/input/top50spotify2019/'
# Please note that we are using python engine to parse the file data instead of c engine.

# read_csv() throws error when we use c-engine.



top50_data = pd.read_csv(data_file_path + 'top50.csv', engine = 'python')

top50_data.shape
top50_data.info()
top50_data.head()
# Rename column.

top50_data.rename(columns = {'Unnamed: 0':'id'}, inplace = True)



# Change name of columns to lower case.

top50_data.columns = top50_data.columns.str.replace('.', '').str.lower()



top50_data.columns
top50_data.head()
def get_df_summary(df):

    

    '''

    

    This function is used to summarise especially unique value count, NaN values count, 

    Blank values count and data type for variable

    

    '''

    

    unq_val_cnt_df = pd.DataFrame(df.nunique(), columns = ['unq_val_cnt'])

    unq_val_cnt_df.reset_index(inplace = True)

    unq_val_cnt_df.rename(columns = {'index':'variable'}, inplace = True)

    unq_val_cnt_df = unq_val_cnt_df.merge(df.dtypes.reset_index().rename(columns = {'index':'variable', 0:'dtype'}),

                                          on = 'variable')

    unq_val_cnt_df = unq_val_cnt_df.sort_values(by = 'unq_val_cnt', ascending = False)

    

    nans_df = pd.DataFrame(df.isna().sum(), columns = ['no_nans']).reset_index()

    nans_df.rename(columns = {'index':'variable'}, inplace = True)

    

    blank_df = pd.DataFrame((df.select_dtypes(include = 'O') == '').sum(), columns = ['no_blanks']).reset_index()

    blank_df.rename(columns = {'index':'variable'}, inplace = True)

    

    neg_df = pd.DataFrame((df.select_dtypes(include = ['int64', 'float64']).lt(0)).sum(), columns = ['no_neg']).reset_index()

    neg_df.rename(columns = {'index':'variable'}, inplace = True)

    

    summ_df = unq_val_cnt_df.merge(nans_df, how = 'outer', on = 'variable')

    summ_df = summ_df.merge(blank_df, how = 'outer', on = 'variable')

    summ_df = summ_df.merge(neg_df, how = 'outer', on = 'variable')

    

    summ_df.fillna(-1, inplace = True)

    

    return summ_df
top50_summ = get_df_summary(top50_data)

top50_summ
print('No. of variables with NaN : {}'.format(sum(top50_summ['no_nans'] > 0)))

print('No. of variables with Blanks : {}'.format(sum(top50_summ['no_blanks'] > 0)))

print('No. of variables with Neg. values : {}'.format(sum(top50_summ['no_neg'] > 0)))
# List the variable(s) containing negative values.



top50_summ.loc[top50_summ['no_neg'] > 0, 'variable']
top50_data['loudnessdb'].describe()
print('No. of human audible songs in the given data set: {}'.format(top50_summ.at[(top50_summ['variable'] == 'loudnessdb').idxmax(), 'no_neg']))
# Check duplicate records.



print('No. of duplicate obs. : {}'.format(top50_data.duplicated().sum()))
top50_data.describe()
top50_data['beatsperminute'].describe()
# Convert seconds value of length variable into minutes and seconds.



top50_data['len_min_secs'] = top50_data['length'].apply(lambda x : datetime.time(divmod(x, 60)[0], divmod(x, 60)[1]))
tmp_df = pd.DataFrame(top50_data.groupby('artistname')['id'].count())

tmp_df.reset_index(inplace = True)

tmp_df.rename(columns = {'id':'count'}, inplace = True)

tmp_df = tmp_df.sort_values(by = 'count', ascending = False).head(3)

tmp_df
fig = px.scatter(data_frame = tmp_df,

                 x = 'artistname',

                 y = 'count',

                 color = 'artistname',

                 labels = {'artistname':'Artist Name', 'count':'No. of songs'},

                 size = 'count')

fig.update_xaxes(title_text = 'Artist Name')

fig.update_yaxes(title_text = 'No. of songs', dtick = 1)

fig.update_layout(title_text = 'Top-3 artists with maximum of songs')

fig.show()
tmp_df = pd.DataFrame(top50_data.groupby('genre')['id'].count())

tmp_df.reset_index(inplace = True)

tmp_df.rename(columns = {'id':'count'}, inplace = True)

tmp_df = tmp_df.sort_values(by = 'count', ascending = False).head(3)



tmp_df
fig = px.scatter(data_frame = tmp_df,

                 x = 'genre',

                 y = 'count',

                 color = 'genre',

                 labels = {'genre':'Genre', 'count':'No. of songs'},

                 size = 'count'

                 )

fig.update_xaxes(title_text = 'Genre')

fig.update_yaxes(title_text = 'No. of songs', dtick = 1)

fig.update_layout(title_text = 'Top-3 genres with maximum of songs')

fig.show()
tmp_df = top50_data.sort_values(by = 'length', ascending = False)[['trackname', 'artistname', 'genre', 'len_min_secs', 'length']]

tmp_df = tmp_df.head(3)



tmp_df
fig = px.bar(data_frame = tmp_df,

             x = 'length',

             y = 'trackname',

             color = 'genre',

             labels = {'genre':'Genre', 'length':'Duration', 'trackname':'Track Name'},

             orientation = 'h',

             text = 'len_min_secs'

            )

fig.update_xaxes(title_text = 'Duration (in seconds)')

fig.update_yaxes(title_text = 'Track Name')

fig.update_layout(title_text = 'Top-3 lengthiest songs',

                  yaxis = {'categoryorder':'total ascending'})



fig.show()
tmp_df = top50_data.sort_values(by = 'popularity', ascending = False)[['trackname', 'artistname', 'genre', 'len_min_secs', 'popularity', 'length']]

tmp_df = tmp_df.head(10)



tmp_df
fig = px.scatter(data_frame = tmp_df,

                 x = 'genre',

                 y = 'popularity',

                 labels = {'genre':'Genre', 'popularity':'Popularity', 'length':'Duration'},

                 size = 'length'

                 )

fig.update_xaxes(title_text = 'Genre')

fig.update_yaxes(title_text = 'Popularity', dtick = 1)

fig.update_layout(title_text = 'Top-10 most popular songs')

fig.show()
cut_labels = [str(i) + '-' + str(i + 10) for i in list(range(0, 100, 10))]

cut_bins = list(range(0, 110, 10))
top50_data['energy_bin'] = pd.cut(top50_data['energy'], bins = cut_bins, labels = cut_labels)
top50_data['danceability_bin'] = pd.cut(top50_data['danceability'], bins = cut_bins, labels = cut_labels)
top50_data['liveness_bin'] = pd.cut(top50_data['liveness'], bins = cut_bins, labels = cut_labels)
top50_data['valence_bin'] = pd.cut(top50_data['valence'], bins = cut_bins, labels = cut_labels)
top50_data['acousticness_bin'] = pd.cut(top50_data['acousticness'], bins = cut_bins, labels = cut_labels)
top50_data['speechiness_bin'] = pd.cut(top50_data['speechiness'], bins = cut_bins, labels = cut_labels)
cut_labels = [str(i) + '-' + str(i + 30) for i in list(range(0, 330, 30))]

top50_data['length_bin'] = pd.cut(top50_data['length'], bins = list(range(0, 360, 30)), labels = cut_labels)
song_pop_prop_df = pd.DataFrame()

cols_to_plot = ['energy', 'danceability', 'liveness', 'valence', 'acousticness', 'speechiness', 'popularity']



for col in cols_to_plot:

    tmp_df = pd.DataFrame(top50_data.loc[:, col]).reset_index(drop = True)

    tmp_df.rename(columns = {col:'property_val'}, inplace = True)

    tmp_df['property'] = col

    song_pop_prop_df = pd.concat([song_pop_prop_df, tmp_df], sort = False)



song_pop_prop_df.fillna(value = {'property_val':0}, inplace = True)



fig = px.box(data_frame = song_pop_prop_df,

             x = 'property',

             y = 'property_val',

             labels = {'property_val':'Property Value', 'property':'Property'})

fig.update_layout(title_text = 'Property Composition of Top-50 songs')

fig.show()
fig = make_subplots(rows = 4, 

                    cols = 2, 

                    shared_yaxes = True, 

                    vertical_spacing = .095, 

                    y_title = 'Popularity', 

                    row_heights = [.8, .8, .8, .8])



rol_col_ids = list(itr.product(range(1, 5), range(1, 3)))

cols_to_plot = ['energy_bin', 'danceability_bin', 'liveness_bin', 'valence_bin', 'acousticness_bin', 

                'speechiness_bin', 'length_bin']



for idx in range(len(cols_to_plot)):

    

    col = cols_to_plot[idx]

    

    tmp_df = pd.DataFrame(top50_data.groupby(col)['popularity'].median().round())

    tmp_df = tmp_df.reset_index()

    

    label_name = col.split('_')[0].title()

    

    fig.append_trace(go.Scatter(x = tmp_df[col], y = tmp_df['popularity'], name = label_name), 

                     row = rol_col_ids[idx][0], 

                     col = rol_col_ids[idx][1]

                     )

    

    fig.update_layout(title_text = 'Song Popularity v/s Song Properties', height = 900, width = 900)



fig.show()
cut_labels = [str(i) + '-' + str(i + 10) for i in list(range(0, 100, 10))]

top50_data['popularity_bin'] = pd.cut(top50_data['popularity'], bins = list(range(0, 110, 10)), labels = cut_labels)
song_pop_prop_df = pd.DataFrame()

cols_to_plot = ['energy', 'danceability', 'liveness', 'valence', 'length', 'acousticness', 'speechiness']



for col in cols_to_plot:

    tmp_df = pd.DataFrame(top50_data.groupby('popularity_bin')[col].mean().round()).reset_index()

    tmp_df.rename(columns = {col:'property_val'}, inplace = True)

    tmp_df['property'] = col

    song_pop_prop_df = pd.concat([song_pop_prop_df, tmp_df], sort = False)



song_pop_prop_df.fillna(value = {'property_val':0}, inplace = True)



fig = px.bar(data_frame = song_pop_prop_df,

             x = 'popularity_bin',

             y = 'property_val',

             color = 'property',

             barmode = 'relative',

             labels = {'popularity_bin':'Popularity', 'property':'Property', 'property_val':'Property Value'}

             )



fig.update_xaxes(title_text = 'Popularity')

fig.update_yaxes(title_text = 'Property Value')

fig.update_layout(title_text = 'Contribution of Song Properties in popularizing songs')

fig.show()
pop_prop_df = pd.DataFrame()



for pop_bin in ['60-70', '90-100']:

    tmp_df = top50_data.loc[top50_data['popularity_bin'] == pop_bin, cols_to_plot].median().round().reset_index()

    tmp_df['pop_bin'] = pop_bin

    pop_prop_df = pd.concat([pop_prop_df, tmp_df])

    

pop_prop_df.rename(columns = {0:'prop_value', 'index':'prop'}, inplace = True)



fig = px.bar(pop_prop_df, x = 'prop', y = 'prop_value', color = 'pop_bin', barmode = 'group')



fig.update_xaxes(title_text = 'Song Property of Top-10 & Bottom-10 songs')

fig.update_yaxes(title_text = 'Song Property Value')

fig.update_layout(title_text = 'Song Properties for Lowest & Highest Popularity Songs',

                  xaxis = {'categoryorder':'total descending'})



fig.show()
tmp_df = top50_data.sort_values(by = 'popularity', ascending = False)[['danceability', 'liveness', 'valence', 'length',

                                                                       'acousticness', 'speechiness', 'popularity',

                                                                       'beatsperminute', 'loudnessdb', 'energy']]

tmp_df.reset_index(drop = True, inplace = True)

tmp_df.reset_index(inplace = True)



cut_labels = [str(i + 1) + '-' + str(i + 10) for i in list(range(0, 50, 10))]

cut_bins = list(range(-1, 51, 10))



tmp_df['bunch'] = pd.cut(tmp_df['index'], bins = cut_bins, labels = cut_labels)
# tmp_df.head()
songs_bunch_df = tmp_df.loc[:, ~tmp_df.columns.isin(['index'])].groupby('bunch').median().round().reset_index()

songs_bunch_df.reset_index(inplace = True, drop = True)
# songs_bunch_df.head(20)
trf_songs_bunch_df = pd.DataFrame()

curr_songs_bunch_df = pd.DataFrame()



for col in songs_bunch_df.columns[1:]:

    curr_songs_bunch_df['bunch'] = songs_bunch_df['bunch']

    curr_songs_bunch_df['property'] = col

    curr_songs_bunch_df['property_value'] = songs_bunch_df[col]

    

    trf_songs_bunch_df = trf_songs_bunch_df.append(curr_songs_bunch_df, ignore_index = True)
# trf_songs_bunch_df.head()
fig = make_subplots(rows = 2, 

                    cols = 3, 

                    shared_yaxes = True, 

                    vertical_spacing = .1, 

                    y_title = 'Property Value', 

                    row_heights = [.8, .8], 

                    shared_xaxes = True)



rol_col_ids = list(itr.product(range(1, 3), range(1, 4)))

bunch_to_plot = trf_songs_bunch_df['bunch'].unique().tolist()

cols_to_plot = ['energy', 'danceability', 'liveness', 'valence', 'acousticness', 'speechiness']



for idx in range(len(bunch_to_plot)):

    

    bunch = bunch_to_plot[idx]

    

    tmp_df = trf_songs_bunch_df.loc[(trf_songs_bunch_df['bunch'] == bunch) & (trf_songs_bunch_df['property'].isin(cols_to_plot)), ['property', 'property_value']]

    

    fig.append_trace(go.Scatter(x = tmp_df['property'], y = tmp_df['property_value'], name = bunch), 

                     row = rol_col_ids[idx][0], 

                     col = rol_col_ids[idx][1]

                     )



fig.update_layout(title_text = 'Song Properties of 5 bunch of songs', width = 900, height = 600)



fig.show()
fig = make_subplots(rows = 2, 

                    cols = 3, 

                    shared_yaxes = True, 

                    vertical_spacing = .1, 

                    y_title = 'Popularity', 

                    row_heights = [.8, .8],

                   shared_xaxes = True)



rol_col_ids = list(itr.product(range(1, 3), range(1, 4)))

bunch_to_plot = trf_songs_bunch_df['bunch'].unique().tolist()

cols_to_plot = ['length', 'beatsperminute', 'loudnessdb']



for idx in range(len(bunch_to_plot)):

    

    bunch = bunch_to_plot[idx]

    

    tmp_df = trf_songs_bunch_df.loc[(trf_songs_bunch_df['bunch'] == bunch) & (trf_songs_bunch_df['property'].isin(cols_to_plot)), ['property', 'property_value']]

    

    fig.append_trace(go.Scatter(x = tmp_df['property'], y = tmp_df['property_value'], name = bunch), 

                     row = rol_col_ids[idx][0], 

                     col = rol_col_ids[idx][1]

                     )



fig.update_layout(title_text = 'Song Properties of bunch of songs', width = 900, height = 600)



fig.show()