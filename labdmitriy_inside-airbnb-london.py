from pathlib import Path



import requests

from tqdm import tqdm



import numpy as np

import pandas as pd

import geopandas as gpd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import altair as alt
sns.set_style('white')



alt.themes.enable('default')

alt.renderers.enable('kaggle')

alt.renderers.set_embed_options(actions=False)

alt.data_transformers.enable('json')
DATA_PATH = Path('../input/airbnb/')

listings_df = pd.read_csv(DATA_PATH/'listings_summary.csv',

                          parse_dates=['last_review'])

listings_detail_df = pd.read_csv(DATA_PATH/'listings.csv', low_memory=False,

                                 parse_dates=['host_since', 

                                              'last_scraped', 'calendar_last_scraped',

                                              'first_review', 'last_review'])



reviews_df = pd.read_csv(DATA_PATH/'reviews_summary.csv', parse_dates=['date'])

reviews_detail_df = pd.read_csv(DATA_PATH/'reviews.csv', parse_dates=['date'])



calendar_df = pd.read_csv(DATA_PATH/'calendar.csv', parse_dates=['date'])



neighbourhoods_df = pd.read_csv(DATA_PATH/'neighbourhoods.csv')

gdf = gpd.read_file(DATA_PATH/'neighbourhoods.geojson')
listings_df.info()
listings_detail_df.info()
print(listings_detail_df.columns.tolist())
calendar_df.info(null_counts=True)
reviews_df.info()
reviews_detail_df.info()
neighbourhoods_df.info()
gdf.plot();
listings_df.head(1)
print(listings_df.shape)

listings_df.loc[:, listings_df.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)
listings_df.loc[:, listings_df.nunique() <= 1].nunique().sort_values()
listings_df.describe(include='datetime')
listings_df.describe(include=['object'])
listings_df['neighbourhood'].value_counts().sort_values().plot.barh(figsize=(10, 10));

sns.despine()

plt.title('Number of listings by neighbourhood', fontsize=14);
listings_df['room_type'].value_counts(dropna=False).sort_values().plot.barh()

sns.despine()

plt.title('Number of listings by room type', fontsize=14);
listings_df.hist(figsize=(12, 10), bins=20, grid=False)

sns.despine()

plt.suptitle('Numeric features distribution', fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
listings_detail_df.head(1)
print(listings_detail_df.shape)

listings_detail_df.loc[:, listings_detail_df.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)
listings_detail_df.loc[:, listings_detail_df.nunique() <= 1].nunique().sort_values()
listings_detail_df.filter(regex='review_scores').notnull().sum(axis=1).value_counts(normalize=True)
listings_detail_df.describe(include='datetime')
listings_detail_df.describe(include='object').T
print(listings_detail_df['country_code'].value_counts())

listings_detail_df.query('country_code != "GB"')
listings_detail_df.hist(figsize=(12, 30), bins=20, grid=False, layout=(15, 3))

sns.despine()

plt.suptitle('Numeric features distribution', fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.97])
calendar_df.head(1)
print(calendar_df.shape)

calendar_df.loc[:, calendar_df.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)
calendar_df.describe(include='datetime')
calendar_df.describe(include='object')
reviews_df.head(1)
print(reviews_df.shape)

reviews_df.loc[:, reviews_df.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)
reviews_df.describe(include='datetime')
reviews_df.hist(bins=20, grid=False)

sns.despine()

plt.suptitle('Numeric features distribution', fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
reviews_detail_df.head(1)
print(reviews_detail_df.shape)

reviews_detail_df.loc[:, reviews_detail_df.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)
reviews_detail_df.describe(include='datetime')
reviews_detail_df.describe(include='object')
reviews_detail_df.hist(figsize=(8, 6), bins=20, grid=False)

sns.despine()

plt.suptitle('Numeric features distribution', fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
neighbourhoods_df.head(1)
print(neighbourhoods_df.shape)

neighbourhoods_df.loc[:, neighbourhoods_df.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)
neighbourhoods_df.describe(include='object')
gdf.head(1)
print(gdf.shape)

gdf.loc[:, gdf.isnull().sum() > 0].isnull().sum().sort_values(ascending=False)
gdf.describe(include='object')
review_cols = ['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',

               'review_scores_communication', 'review_scores_location', 'review_scores_value']

host_cols = ['host_since', 'host_response_time',

             'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']



listing_detail_cols = ['id', 'instant_bookable', 'neighbourhood_cleansed', 'room_type'] + review_cols + host_cols 





res_listings_detail_df = listings_detail_df.query('country_code == "GB"')

res_listings_detail_df = res_listings_detail_df[res_listings_detail_df['host_name'].notnull()]

res_listings_detail_df = res_listings_detail_df[res_listings_detail_df.filter(regex='review_scores').notnull().all(axis=1)]

res_listings_detail_df = res_listings_detail_df[listing_detail_cols].rename({'neighbourhood_cleansed': 'neighbourhood'}, axis=1)

res_listings_detail_df.head()
res_listings_detail_df.info()
geo_cols = ['neighbourhood', 'geometry']

res_gdf = gdf.loc[:, geo_cols]

res_gdf.head()
binary_cols = ['instant_bookable', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']

binary_map = {'f': False, 't': True}

res_listings_detail_df[binary_cols] = res_listings_detail_df[binary_cols].apply(lambda x: x.map(binary_map)).astype(bool)



cat_type = pd.api.types.CategoricalDtype(['not specified', 'within an hour', 'within a few hours', 'within a day', 'a few days or more'])

res_listings_detail_df['host_response_time'] = res_listings_detail_df['host_response_time'].fillna('not specified').astype(cat_type)
res_listings_detail_df.info()
res_gdf['area_sq_km'] = (res_gdf['geometry'].to_crs({'init': 'epsg:3395'})

                                    .map(lambda p: p.area / 10**6))



res_listings_detail_df['age'] = (pd.Timestamp('now') - pd.to_datetime(res_listings_detail_df['host_since'])).dt.days.div(365.25).round(2)
geo_listings_df = res_gdf.merge(res_listings_detail_df, how='inner', on='neighbourhood')



geo_listings_df['listings_count'] = geo_listings_df.groupby('neighbourhood')['id'].transform('count')

geo_listings_df['listings_density'] = geo_listings_df.groupby('neighbourhood')['area_sq_km'].transform(lambda x: len(x) / x)



geo_listings_df['mean_review_scores_accuracy'] = geo_listings_df.groupby('neighbourhood')['review_scores_accuracy'].transform('mean')

geo_listings_df['mean_review_scores_cleanliness'] = geo_listings_df.groupby('neighbourhood')['review_scores_cleanliness'].transform('mean')

geo_listings_df['mean_review_scores_checkin'] = geo_listings_df.groupby('neighbourhood')['review_scores_checkin'].transform('mean')

geo_listings_df['mean_review_scores_communication'] = geo_listings_df.groupby('neighbourhood')['review_scores_communication'].transform('mean')

geo_listings_df['mean_review_scores_location'] = geo_listings_df.groupby('neighbourhood')['review_scores_location'].transform('mean')

geo_listings_df['mean_review_scores_value'] = geo_listings_df.groupby('neighbourhood')['review_scores_value'].transform('mean')



geo_listings_df['mean_review_scores_all'] = geo_listings_df.filter(like='mean_review_scores').mean(axis=1)
geo_listings_df.info()
review_cols = ['mean_review_scores_accuracy', 'mean_review_scores_cleanliness', 'mean_review_scores_checkin',

               'mean_review_scores_communication', 'mean_review_scores_location', 'mean_review_scores_value']

review_titles = ['Accuracy', 'Cleanliness', 'Check-in',

                 'Communication', 'Location', 'Value']

review_map = {col: title for col, title in zip(review_cols, review_titles)}



result_df = geo_listings_df[['geometry', 'neighbourhood', 'mean_review_scores_all'] + review_cols].drop_duplicates()



def gen_map_chart(df, review_col, review_title):

    '''Generate choropleth map

    

    Generate choropleth map based on scores of specific review types

    

    :param df: DataFrame with necessary geo data and review scores for different neighbourhood

    :type df: DataFrame

    :param review_col: name of review scores type

    :type review_col: str

    :param review_title: title of review scores type

    :type review_title: str

    :return: Altair Chart for displaying 

    :rtype: Chart

    '''

    chart = alt.Chart(

        df,

        title=review_title

    ).mark_geoshape().encode(

        color=f'{review_col}:Q',

        tooltip=['neighbourhood:N', f'{review_col}:Q']

    ).properties(

        width=250, 

        height=250

    )

    

    return chart



charts = []



for review_col, review_title in zip(review_cols, review_titles):

    charts.append(gen_map_chart(result_df, review_col, review_title))



overall_map_chart = gen_map_chart(result_df, 'mean_review_scores_all', 'Overall')



((alt.vconcat(alt.concat(*charts, columns=3), overall_map_chart, 

              title='Average review scores by neighbourhood', 

              center=True)

     .configure_view(strokeWidth=0)

     .configure_title(fontSize=18)

     .configure_legend(title=None, orient='top',  labelFontSize=12)))
result_df = (geo_listings_df[review_cols].rename(review_map, axis=1)

                                         .corr()

                                         .reset_index()

                                         .melt(id_vars='index')

                                         .rename({'value': 'correlation'}, axis=1))



base = alt.Chart(

    result_df,

    title='Average Review Scores Relationship'

).properties(

    width=600, 

    height=600

)



heatmap = base.mark_rect().encode(

    x=alt.X('index:N', title=None),

    y=alt.Y('variable:N', title=None),

    color='correlation:Q'

)



text = base.mark_text(baseline='middle').encode(

    x=alt.X('index:N', title=None),

    y=alt.Y('variable:N', title=None),

    text=alt.Text('correlation:Q', format='.2f'),

    color=alt.condition(

        alt.datum.correlation < 0,

        alt.value('black'),

        alt.value('white')

    )

)



(heatmap + text).configure_axis(

    labelAngle=0,

    labelFontSize=14

).configure_legend(

    orient='top',

    titleFontSize=14,    

).configure_title(

    fontSize=18,

    offset=15,

    anchor='start',

    frame='group'

)
def gen_parallel_chart(df, class_col, class_title):

    '''Generate parallel coordinates chart

    

    Generate parallel coordinates chart based on specific class column by different review score types

    

    :param df: DataFrame with necessary data for class column calculation

    :type df: DataFrame

    :param class_col: name of class column 

    :type class_col: str

    :param class_title: title of review scores type

    :type class_title: str

    :return: Altair Chart for displaying 

    :rtype: Chart

    '''

    result_df = (df.groupby(class_col)[review_cols]

                   .mean()

                   .reset_index()

                   .melt(id_vars=class_col))

    result_df['variable'] = result_df['variable'].map(review_map)



    chart = alt.Chart(

        result_df,

        title = f'{class_title}'

    ).mark_line().encode(

        x=alt.X('variable:N',

                title=None),

        y=alt.Y('value:Q',

                scale=alt.Scale(zero=False),

                axis=None),

        color=f'{class_col}:N'

    ).properties(

        width=750, 

        height=300

    )

    

    return chart



class_cols = ['room_type', 'instant_bookable', 'host_is_superhost']

class_titles = ['Room Type', 'Listing is Instant Bookable', 'Host is Superhost']



charts = []



for class_col, class_title in zip(class_cols, class_titles):

    charts.append(gen_parallel_chart(geo_listings_df, class_col, class_title))

    

(alt.concat(*charts, columns=1, title='Average Review Scores by Host/Listing Properties')

    .configure_view(strokeWidth=0)

    .configure_legend(

        title=None, 

        orient='top', 

        columns=0,

        labelFontSize=14)

    .configure_axis(

        labelAngle=0,

        grid=False,

        labelFontSize=14)

    .configure_title(

        anchor='start',

        fontSize=18,

        offset=15)

    .resolve_scale(color='independent')

)