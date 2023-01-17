from pathlib import Path



import numpy as np

import pandas as pd



import altair as alt

from altair import datum
DATA_PATH = Path('../input/kaggle-survey-2019/')

DATA_ADD_PATH = Path('../input/kaggle-survery-2019-additional-files/')
multi_resp_df = pd.read_csv(DATA_PATH/'multiple_choice_responses.csv', low_memory=False, skiprows=1)

multi_resp_df['user_id'] = np.arange(len(multi_resp_df))

multi_resp_df = multi_resp_df.melt(id_vars='user_id', var_name='question', value_name='answer')

multi_resp_df = multi_resp_df.dropna(subset=['answer'])



question_hier = multi_resp_df['question'].str.split('-', 1)



multi_resp_df['question_type'] = question_hier.str[0].str.strip()

multi_resp_df['question_type'] = multi_resp_df['question_type'].str.split('(:|\?)').str[0]



multi_resp_df['question_subtype'] = question_hier.str[1].str.strip()

multi_resp_df = multi_resp_df[~multi_resp_df['question_subtype'].fillna('').str.contains('- Text')]

multi_resp_df['question_subtype'] = multi_resp_df['question_subtype'].str.split('-').str[1].str.strip()



iso_country_map = {

    'United States of America': 'United States',

    'Russia': 'Russian Federation',

    'South Korea': 'Korea, Republic of',

    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',

    'Hong Kong (S.A.R.)': 'Hong Kong',

    'Singapore': 'Singapore',

    'Taiwan': 'Taiwan, Province of China',

    'South Korea': 'Korea, Republic of',

    'Republic of Korea': 'Korea, Republic of',

    'Iran, Islamic Republic of...': 'Iran, Islamic Republic of',

    

    'Other': 'Other'

}



countries_df = pd.read_json(DATA_ADD_PATH/'world-110m-country-codes.json')

iso_countries_df = pd.read_csv(DATA_ADD_PATH/'all.csv')



add_countries_df = pd.DataFrame({'code':['HK', 'SG', 'UNK'], 'name': ['Hong Kong', 'Singapore', 'Other'], 'id': [344, 702, -1]})

all_countries_df = pd.concat([countries_df, add_countries_df], axis=0, ignore_index=True, sort=False)

all_countries_df = all_countries_df.merge(iso_countries_df, how='left', on='name')



resp_pivot_df = pd.pivot_table(multi_resp_df, values='answer', index='user_id', columns=['question_type'], aggfunc=lambda x: ';'.join(str(v) for v in x))

column_names = [

    'ds_individuals_count', 'money_spent_ml', 'has_ml_in_business', 'duration', 'ml_exp',

    'has_used_TPU',  'analytic_code_exp', 'country', 'course_platforms', 'activities',

    'job_title', 'vis_libraries', 'education', 'primary_analysis_tool', 'company_size',

    'age', 'compensation', 'gender', 'rec_prog_language', 'prog_language', 

    'automl_tools', 'ml_tools', 'cv_methods', 'ml_algorithms', 'cloud_platforms', 

    'notebook_products', 'ide', 'ml_frameworks', 'ml_products', 'nlp_methods', 

    'rdb_products', 'big_data_products', 'cloud_produts', 'hardware', 'favorite_media'  

]



resp_pivot_df.columns = column_names

resp_pivot_df['country'].update(resp_pivot_df['country'].map(iso_country_map))

resp_pivot_df = resp_pivot_df.merge(all_countries_df, how='inner', left_on='country', right_on='name')
alt.themes.enable('default')

alt.renderers.enable('kaggle')

alt.data_transformers.enable('json')



import geopandas as gpd

import json



def plot_map(area_type, width=500, height=300):

    area_col = area_type.lower()

    

    gdf = gpd.read_file(DATA_ADD_PATH/'world-110m_reduced.json')

    gdf['id'] = gdf['id'].astype('int')

    

    geo_json = json.loads(gdf.to_json())

    geo_data = alt.Data(values=geo_json['features'])

    

    country_map = alt.Chart(

        resp_pivot_df,

        title=f'# Respondents by {area_type}'

    ).transform_filter(

        col1_brush

    ).transform_filter(

        col2_brush

    ).transform_lookup(

        lookup='id',

        from_=alt.LookupData(geo_data, 'properties.id'),

        as_='geom',

        default='Other'

    ).transform_aggregate(

        counter='count()',

        groupby=['geom', area_col]

    ).transform_calculate(

        geometry ='datum.geom.geometry',

        type= 'datum.geom.type'

    ).mark_geoshape(

    ).encode(

        color='counter:Q',

        tooltip=[

            alt.Tooltip(f'{area_col}:N', title='Area'),

            alt.Tooltip('counter:Q', title='# Respondents')

        ]

    ).properties(

        width=width,

        height=width

    )



    borders = alt.Chart(geo_data).mark_geoshape(

        fill='#EEEEEE',

        stroke='gray',

        strokeWidth=1

    ).properties(

        width=width,

        height=height

    )

    

    return (borders + country_map)



def plot_bar(col_name, selection=None, width=500):

    chart = alt.Chart(

        resp_pivot_df[[col_name]],

        title=''

    ).mark_bar().encode(

        y=f'{col_name}:N',

        x='count():Q',

        tooltip=[

            alt.Tooltip('count()', title='Count')

        ],

        color = alt.condition(selection, alt.value('steelblue'), alt.value('lightgray'))

    ).add_selection(

        selection

    ).properties(

        width=width

    )

    

    return chart



def plot_dash(area_type):

    return (plot_map(area_type) & plot_bar(col1_name, col1_brush) & plot_bar(col2_name, col2_brush)).resolve_legend('independent')
area_type = 'Country'

col1_name = 'age'

col1_brush = alt.selection_multi(fields=[col1_name])

col2_name = 'gender'

col2_brush = alt.selection_multi(fields=[col2_name])



plot_dash(area_type)