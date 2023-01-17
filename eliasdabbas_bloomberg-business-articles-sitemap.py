import logging
from xml.etree import ElementTree
from urllib.request import urlopen

import pandas as pd


def sitemap_to_df(sitemap_url):
    xml_text = urlopen(sitemap_url)
    tree = ElementTree.parse(xml_text)
    root = tree.getroot()
        
    xml_df = pd.DataFrame()

    if root.tag.split('}')[-1] == 'sitemapindex':
        for elem in root:
            for el in elem:
                if el.text.endswith('xml'):
                    try:
                        xml_df = xml_df.append(sitemap_to_df(el.text), ignore_index=True)
                        logging.info(msg='Getting ' + el.text)
                    except Exception as e:
                        logging.warning(msg=str(e) + el.text)
                        xml_df = xml_df.append(pd.DataFrame(dict(sitemap=el.text),
                                                            index=range(1)), ignore_index=True)

    else:
        logging.info(msg='Getting ' + sitemap_url)
        for elem in root:
            d = {}
            for el in elem:
                tag = el.tag
                name = tag.split('}', 1)[1] if '}' in tag else tag

                if name == 'link':
                    if 'href' in el.attrib:
                        d.setdefault('alternate', []).append(el.get('href'))
                else:
                    d[name] = el.text.strip() if el.text else ''
            d['sitemap'] = sitemap_url
            xml_df = xml_df.append(pd.DataFrame(d, index=range(1)), ignore_index=True)
    if 'lastmod' in xml_df:
        xml_df['lastmod'] = pd.to_datetime(xml_df['lastmod'], utc=True)
    if 'priority' in xml_df:
        xml_df['priority'] = xml_df['priority'].astype(float)
    return xml_df

    
bloomberg = pd.read_csv('../input/bloomberg-business-articles-urls/bloomberg_biz_sitemap.csv',
                        parse_dates=['lastmod'], index_col='lastmod',
                        low_memory=False)
bloomberg['priority'] = bloomberg['priority'].astype(float)
print(bloomberg.shape)
bloomberg
bloomberg[bloomberg.index.isna()]['sitemap'].str.contains('video').mean()
bloomberg[bloomberg.index.notna()]['sitemap'].str.contains('video').mean()
by_year_count = bloomberg.resample('A')['loc'].count()
by_year_count
by_month_count = bloomberg.resample('M')['loc'].count()
by_month_count
import plotly.graph_objects as go
fig = go.Figure()
fig.add_bar(x=by_year_count.index, y=by_year_count.values, showlegend=False)
fig.layout.template = 'none'
fig.layout.title = 'Bloomberg Business Articles Published per Year 1991 - 2020'
fig.layout.xaxis.tickvals = by_year_count.index.date
fig.layout.xaxis.ticktext = by_year_count.index.year
fig.layout.yaxis.title = 'Number of articles'
fig
fig = go.Figure()
fig.add_bar(x=by_month_count.index, y=by_month_count.values, showlegend=False)
fig.layout.template = 'none'
fig.layout.title = 'Bloomberg Business Articles Published per Month 1991 - 2020'
fig.layout.yaxis.title = 'Number of articles'
fig
(bloomberg
 .groupby(bloomberg.index.weekday)['loc']
 .count().to_frame()
 .rename(columns=dict(loc='count'))
 .assign(day=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
 .style.bar().format(dict(count='{:,}'))
)
bloomberg[:1]['loc'].values[0], bloomberg[:1]['loc'].index
bloomberg['loc'].str.split('/').str[-3].value_counts()
bloomberg['loc'].str.split('/').str[-4].value_counts()
bloomberg['loc'].str.split('/').str[-1]
!pip install advertools
import advertools as adv
adv.word_frequency(bloomberg['loc'].dropna().str.split('/').str[-1].str.replace('-', ' ')).iloc[:20, :2]
adv.word_frequency(bloomberg['loc'].dropna().str.split('/').str[-1].str.replace('-', ' '),phrase_len=2).iloc[:20, :2]
bloomberg['loc'].dropna()[bloomberg['loc'].dropna().str.contains('-dot')].str.split('/').str[-1][:10].values
by_year_count_china = bloomberg[bloomberg['loc'].fillna('').str.contains('china', case=False)].resample('A')['loc'].count()
fig = go.Figure()
fig.add_scatter(x=by_year_count.index,
                y=by_year_count.values,
                name='All Articles',
                yaxis='y', 
                mode='lines+markers',
                marker=dict(size=10))
fig.add_scatter(x=by_year_count_china.index,
                y=by_year_count_china.values,
                name='China Articles',
                yaxis='y2',
                mode='lines+markers',
                marker=dict(size=10))
fig.layout.template = 'none'
fig.layout.title = 'BusinessWeek Articles Published per Year 2001 - 2020 (China vs All Topics)'
fig.layout.xaxis.tickvals = by_year_count.index.date
fig.layout.xaxis.ticktext = by_year_count.index.year
fig.layout.yaxis.title = 'All articles'
fig.layout.yaxis2 = dict(title='China', overlaying='y', side="right", position=1, anchor='free')


fig
def plot_topic_vs_all(*topics, include_all=True, log_y=True):
    
    fig = go.Figure()
    if include_all:
        fig.add_scatter(x=by_year_count.index,
                        y=by_year_count.values,
                        name='All Articles',
                        yaxis='y', 
                        mode='lines+markers',
                        marker=dict(size=10))
    for topic in topics:
        topic_df = bloomberg[bloomberg['loc'].fillna('').str.contains(topic, case=False)].resample('A')['loc'].count()
        fig.add_scatter(x=topic_df.index,
                        y=topic_df.values,
                        name=topic + ' Articles',
                        mode='lines+markers',
                        yaxis='y2',
                        marker=dict(size=10))
    fig.layout.template = 'none'
    all_topics = ' vs. All Topics)' if include_all else ')'
    fig.layout.title = f'BusinessWeek Articles Published per Year 2001 - 2020 ({", ".join(topics)}' + all_topics
    fig.layout.xaxis.tickvals = topic_df.index.date
    fig.layout.xaxis.ticktext = topic_df.index.year
    fig.layout.yaxis.title = 'All articles'
    fig.layout.yaxis2 = dict(title=f"'{topics}' articles", overlaying='y',
                             side="right", position=1, anchor='free')
    if log_y:
        fig.layout.yaxis.type = 'log'
        fig.layout.yaxis2.type = 'log'


    return fig
plot_topic_vs_all('oil', 'china', include_all=True, log_y=True)
plot_topic_vs_all('trump', 'china', include_all=False, log_y=True)
plot_topic_vs_all('trump', 'huawei', 'china', include_all=False)
plot_topic_vs_all('trump', 'fed', include_all=False)
bloomberg[bloomberg['loc'].str.contains('articles/1000')].iloc[:10, :1]['loc'].str[30:]
bloomberg['loc'] = bloomberg['loc'].str.replace('articles/1000-', 'articles/2000-')
bloomberg['pub_date'] = pd.to_datetime(bloomberg['loc'].str.extract('(\d{4}-\d{2}-\d{2})')[0])
bloomberg['pub_date']
bloomberg['dates_equal'] = bloomberg['pub_date'].dt.date == bloomberg.index.date
bloomberg['dates_equal'].mean()
bloomberg['days_to_update'] = bloomberg['pub_date'].dt.date - bloomberg.index.date
bloomberg.iloc[:5, [0, 1, 2, 3, 4, -3, -2, -1]]
bloomberg['days_to_update'].value_counts(normalize=True)[:15]