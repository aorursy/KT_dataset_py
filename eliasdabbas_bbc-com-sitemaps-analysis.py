import pandas as pd
pd.options.display.max_columns = None
import dataset_utilities as du
import plotly.graph_objects as go
from ipywidgets import interact
bbc = pd.read_csv('/kaggle/input/news-sitemaps/bbc_sitemaps.csv',
                  parse_dates=['lastmod'], index_col='lastmod', usecols=['lastmod', 'loc'])
bbc.sample(5)
du.value_counts_plus(bbc['loc'].rename('language').str.split('/').str[3], show_top=50)
bbc['loc'].str.contains('https://www.bbc.com/').all()
bbc['slug'] = bbc['loc'].str.replace('https://www.bbc.com/', '')
bbc['slug'] = bbc['slug'].str.replace('^news|^sport|^newsround', 'english/\g<0>')
bbc.sample(5)
bbc['lang'] = bbc['slug'].str.split('/').str[0].replace('mundo', 'spanish')
bbc.sample(5)
du.value_counts_plus(bbc['lang'], show_top=50)
bbc['slug_split_length'] = bbc['slug'].str.split('/').str.len()
bbc.sample(7)
du.value_counts_plus(bbc['slug_split_length']).hide_index()
format(bbc[bbc['slug_split_length']==2]['slug'].str.split('/').str[1].nunique(), ',')
bbc[bbc['slug_split_length']==3]['slug'].sample(15)
bbc[bbc['slug_split_length']==3]['slug'].str.split('/').str[1].value_counts()[:20]
format(bbc[bbc['slug_split_length']==3]['slug'].str.split('/').str[2].nunique(), ',')
bbc[bbc['slug_split_length']==4]['slug'].sample(10)
bbc[bbc['slug_split_length']==4]['slug'].str.contains('english/sport/|english/news').mean()
bbc[bbc['slug_split_length']==4]['slug'].str.split('/').str[0].value_counts(dropna=False)
bbc[bbc['slug_split_length']==4]['slug'].str.split('/').str[1].value_counts(dropna=False)
len_4_index_1_2 = (bbc[bbc['slug_split_length']==4]
                   ['slug'].str.split('/').str[1:3]
                   .str.join('/')
                  )
len_4_index_1_2.value_counts()[:7]
non_sports = {
    'live', 'scotland', 'northern-ireland', 'get-inspired', 'wales', 'live', 'av',
    'sports-personality', 'supermovers', 'africa', 'england', 'audiovideo', 'england',
    'video_and_audio', 'features', 'live', 'live', 'special-reports', 'in-depth',
    'in-depth', 'business', 'world', 'ultimate-performers', 'scotland',
    'move-like-never-before', 'made-more-of-their-world', 'wales', 'syndication',
    'west-bank-hitchhiking', 'headlines', 'stadium', 'trump-kim-meeting',
    'deadcities', 'wedding-mixed-race', 'northern_ireland', 'wedding-dress', 'system',
    'the-vetting-files', 'brodsky', 'syndication', 'wedding-designers', 'education',
    'world-cup-russia-hopefuls', 'wedding-guests', 'uk-scotland', 'tianshu',
    'yorkshire-and-humberside', 'west',  'west-midlands', 'uk-scotland','students_diary',
    'students_experience', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
    '12', '2015', '2016', 'articles', 'asia', 'chart_uk', 'east',  'east-mids-and-lincs',
    'election', 'europe', 'globalnews', 'in_depth', 'london-and-south-east',
    'north-east-and-cumbria', 'north-west', 'on_britain', 'south', 'south-west',
    'special_reports','politics', 'qa', 'tenglong', 'technology'
}
bbc[bbc['slug_split_length']==5]['slug'].str.split('/').str[1].value_counts()[:10]
bbc[bbc['slug_split_length']==5]['slug'].str.split('/').str[2].value_counts()[:20]
bbc[bbc['slug_split_length']==5]['slug'].str.split('/').str[3].value_counts()[:20]
format(bbc[bbc['slug_split_length']==5]['slug'].str.split('/').str[4].nunique(), ',')
bbc[bbc['slug_split_length']==6]['slug'].str.split('/').str[1].value_counts()[:10]
bbc[bbc['slug_split_length']==6]['slug'].str.split('/').str[2].value_counts()[:15]
bbc[bbc['slug_split_length']==6]['slug'].str.split('/').str[1:3].str.join('/').value_counts()[:20]
bbc[bbc['slug_split_length']==6]['slug'].str.split('/').str[3].value_counts()[:15]
bbc[bbc['slug_split_length']==6]['slug'].str.split('/').str[4].value_counts()[:15]
format(bbc[bbc['slug_split_length']==6]['slug'].str.split('/').str[5].nunique(), ',')
bbc[bbc['slug_split_length']==7]['slug'].str.split('/').str[0].value_counts()
bbc[bbc['slug_split_length']==7]['slug'].str.split('/').str[1].value_counts()[:15]
bbc[bbc['slug_split_length']==7]['slug'].str.split('/').str[2].value_counts()[:15]
bbc[bbc['slug_split_length']==7]['slug'].str.split('/').str[3].value_counts()[:15]
bbc[bbc['slug_split_length']==7]['slug'].str.split('/').str[4].value_counts()[:15]
bbc[bbc['slug_split_length']==7]['slug'].str.split('/').str[5].value_counts()[:15]
bbc['year_month'] = bbc['loc'].str.extract('/(\d{4}/\d{2})/')[0].values
bbc.dropna(subset=['year_month']).sample(7)
sport_names_4 = set(bbc[bbc['slug_split_length']==4]['slug'].str.split('/').str[2].unique())
sport_names_5 = set(bbc[bbc['slug_split_length']==5]['slug'].str.split('/').str[3].unique())
sport_names_all = sport_names_4.union(sport_names_5).difference(non_sports)
sport_regex = '/(' + '|'.join(sport_names_all) + ')/'
sport_regex
bbc['sport'] = bbc['loc'].str.extract(sport_regex)[0].values
bbc.dropna(subset=['sport', 'year_month']).sample(7)
bbc['sport'].value_counts()[:10]
import numpy as np
extracted_pub_date = (bbc['slug']
                      .str.extract('/([012][0-9][01][0-9][0123][0-9])_')[0]
                      .replace('00000[01]', np.nan, regex=True))
extracted_pub_date
bbc['pub_date'] = pd.to_datetime(extracted_pub_date, format='%y%m%d', errors='coerce', utc=True)
bbc.dropna(subset=['sport', 'pub_date']).sample(7)
bbc['pub_date'].notna().mean()
du.value_counts_plus(bbc['pub_date'].sub(bbc.index).dt.days, dropna=True)
category_indexes = [(3, 1), (5, 1), (6, 2), (7, 3)]
category_indexes
categories = set()
for length, index in category_indexes:
    temp_categories = set(bbc[bbc['slug_split_length']==length]['slug'].str.split('/').str[index].unique())
    categories = categories.union(temp_categories)
categories_regex = '/(' + '|'.join(categories) + ')/'
categories_regex
bbc['category'] = bbc['loc'].str.extract(categories_regex)[0].values
bbc.dropna(subset=['category', 'sport', 'pub_date']).sample(7)
bbc['title'] = (bbc['slug']
                .str.split('/')
                .str[-1]
                .str.replace('^\d{6}_|-\d+$|^\d+$', '')
                .str.replace('_|-', ' '))
bbc.dropna(subset=['title']).sample(7)
timeframe_key = dict(A='Year', M='Month', W='Week')
def compare_langs(lang1=None, lang2=None, lang3=None, timeframe='A', y_scale='linear'):
    title_lang = []
    fig = go.Figure()
    for lang in [lang1, lang2, lang3]:
        if lang is not None:
            df = bbc[bbc['lang']==lang].resample(timeframe)['loc'].count()
            fig.add_scatter(x=df.index, y=df.values, name=lang.title(),
                            mode='markers+lines')
            title_lang.append(lang.title())
    fig.layout.title = 'Articles per ' + timeframe_key[timeframe] + ': ' + ', '.join(title_lang)
    fig.layout.yaxis.type = y_scale
    fig.layout.paper_bgcolor = '#E5ECF6'
    return fig
        
compare_langs('english', 'russian', 'portuguese')
compare_langs('english', 'russian', 'portuguese', y_scale='log')
languages = [None] +  sorted(bbc['lang'].unique())
interact(compare_langs, lang1=languages, lang2=languages, lang3=languages,
         timeframe=dict(Year='A', Month='M', Week='W'), y_scale=['linear', 'log']);