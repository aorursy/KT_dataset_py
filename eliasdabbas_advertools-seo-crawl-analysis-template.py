!pip install advertools
import advertools as adv

import pandas as pd

pd.options.display.max_columns = None
# robots_df = adv.robotstxt_to_df('https://www.whitehouse.gov/robots.txt')



robots_df = pd.read_csv('/kaggle/input/the-white-house-website/robotstxt_df.csv')

robots_df
# sitemap_df = adv.sitemap_to_df('https://www.whitehouse.gov/sitemap_index.xml')



sitemap_df = pd.read_csv('/kaggle/input/the-white-house-website/sitemap_df.csv', parse_dates=['lastmod'])

sitemap_df.sample(5)
sitemap_df.set_index('lastmod').resample('A')['loc'].count()
sitemap_df['sitemap'].value_counts()
selectors = {'briefing_title': '.page-header__title::text',

             'briefing_date': 'time::text',

             'briefing_body_text': '.editor p::text',

             'briefing_category': '.issue-flag a::text'}
sitemaps_unique = sitemap_df['sitemap'].drop_duplicates().tolist()

sitemaps_unique
# for i, sitemap in enumerate(sitemaps_unique):

#     df = sitemap_df[sitemap_df['sitemap']==sitemaps_unique[i]]

#     adv.crawl(df['loc'], f'crawls/wh_crawl_{i+1}.csv', css_selectors=selectors)
df = pd.DataFrame({'colors': ['blue@@green@@yellow', 'red@@blue', 'green@@blue@@pink@@orange'],

                   'months': ['Jan', 'Feb', 'Mar']})

df.style.set_caption('How do we count the colors?')
df['colors'].str.split('@@')
df['colors'].str.split('@@').explode() 
df['colors'].str.split('@@').explode().value_counts()
df.assign(colors_split=df['colors'].str.split('@@'))
df.assign(colors_split=df['colors'].str.split('@@')).explode('colors_split')
month_color =  {month: color.split('@@') for month, color in zip(df['months'], df['colors'])}

month_color
list()
from collections import defaultdict



dd = defaultdict(list)



for month, color_list in month_color.items():

    for color in color_list:

        dd[color].append(month)
dict(dd)
import os

crawl_df = pd.concat([pd.read_csv('/kaggle/input/the-white-house-website/' + file)

                      for file in os.listdir('/kaggle/input/the-white-house-website/') if 'wh_crawl' in file],

                     ignore_index=True)

crawl_df.head(2)
crawl_df.filter(regex='briefing')
crawl_df.info()
set(crawl_df['url']).difference(crawl_df['url_redirected_to'])
set(crawl_df['url_redirected_to']).difference(crawl_df['url'])
crawl_df['status'].value_counts()
pd.cut(crawl_df['size'], bins=10).value_counts()
pd.cut(crawl_df['size'], bins=10).value_counts().sort_index()
crawl_df['title'].duplicated().mean()
crawl_df['title'].str.split('@@').str.len().value_counts()[:10]
adv.emoji_search('think')
crawl_df['title'].str.split('@@', expand=True)
adv.emoji_search('scream')
(crawl_df

 ['title'].str.split('@@')

 .explode()

 .value_counts()

 .reset_index()

 .rename(columns={'index': 'title_tags', 'title': 'count'})

 [:10])
(crawl_df

 ['title'].str.replace('@@', ' ')

 .str.split()

 .explode()

 .value_counts()

 .reset_index()

 .rename(columns={'index': 'title_tags', 'title': 'count'})

 [:15])
(pd.cut(

    crawl_df['title']

    .str.split('@@')

    .str[0]

    .str.len(), [0, 40, 50, 60, 70, 80, 90, 999])

 .value_counts()

 .sort_index()

 .reset_index()

 .rename(columns={'index': 'Title text length in characters (range)', 'title': 'count'})

 .style.background_gradient('cividis')

 .format({'count': '{:,}'})

 .set_caption('Distribution of title tag lengths - thewhitehouse.com'))
crawl_df['h1'].str.split('@@').str.len().value_counts()
(crawl_df

 ['h1'].str.split('@@')

 .explode()

 .value_counts()

 .reset_index()

 .rename(columns={'index': 'h1_tags',

                  'h1': 'count'})[:15])
crawl_df['h2'].str.split('@@').str.len().value_counts()
crawl_df.filter(regex='links').apply(lambda series: series.str[:25])
from urllib.parse import urlparse



url = 'https://www.example.com/category/sub-cat/article-title;some_param?one=1&two=2&three=3#fragment'



parsed = urlparse(url)

parsed
parsed.netloc
parsed.query
urlparse('anystring/http://hello')
(crawl_df['links_url']

 .str.split('@@')

 .str.len()

 .value_counts()

 .reset_index()

 .rename(columns={'index':'links_on_page',

                  'links_url': 'count'})

 .head(15))
internal_links = (crawl_df

                  ['links_url'].str.split('@@')

                  .explode()

                  .apply(lambda s: s if 'https://www.whitehouse.gov' in s else None)

                  .dropna())

internal_links.head()
(pd.Series(urlparse(url).path

           for url in internal_links)

 .value_counts()

 .reset_index()

 .rename(columns={'index': 'path', 0: 'count'})

 .head(50))
external_links = (crawl_df

                  ['links_url'].str.split('@@')

                  .explode()

                  .apply(lambda s: s if 'https://www.whitehouse.gov' not in s else None)

                  .dropna())

external_links
external_domains = pd.Series(urlparse(url).netloc for url in external_links).value_counts()

print('external domains linked to: ', external_domains.index.nunique(), '\n')

external_domains[:10]
external_links_social = pd.Series(urlparse(url).netloc for url in external_links).value_counts()[:5]

external_links_social
external_links_nonsocial = (pd.Series(link for link in external_links

                                      if urlparse(link).netloc not in external_links_social))

print('Unique external non-social links:', external_links_nonsocial.nunique(), '\n')

external_links_nonsocial
# adv.crawl(external_links_nonsocial.drop_duplicates(), 'external_links.csv', follow_links=False)

# external_links_df = pd.read_csv('external_links.csv')

# external_links_df[external_links_df['status']>400]['url']
link_text_counts = (crawl_df

                    ['links_text'].str.split('@@')

                    .explode()

                    .str.strip()

                    .value_counts())

link_text_counts[:60].reset_index()
adv.word_frequency(crawl_df

                   ['links_text'].str.split('@@')

                   .explode()

                   .str.strip()).iloc[:20, :2]
adv.word_frequency(crawl_df                   

                   ['links_text'].str.split('@@')

                   .explode()

                   .str.strip(),

                   phrase_len=2).iloc[:20, :2]
(crawl_df['img_src']

 .str.split('@@').str.len()

 .value_counts()

 .reset_index()

 .rename(columns={'index':'images_on_page',

                  'img_src': 'count'})

 .head(15))
(crawl_df['img_src']

 .str.split('@@').str.len()

 .value_counts()

 .reset_index()

 .rename(columns={'index':'images_on_page',

                  'img_src': 'count'})

 .sort_values('images_on_page')

 .head(15)

)
crawl_df.filter(regex='resp_headers').sample(5)
crawl_df.filter(regex='request_headers').sample(5)