import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker
import datetime
import numpy as np
from matplotlib.font_manager import FontProperties

category = {0:'Film & Animation', 1:"Autos & Vehicles", 2:"Music", 3:"Pets & Animals", 4:"Sports",
           5:"Short Movies", 6:"Travel & Events", 7:"Gaming", 8:"Videoblogging", 9:"People & Blogs",
           10:"Comedy", 11: "Entertainment",12: "News & Politics",13: "Howto & Style",14: "Education",
            15: "Science & Technology",16: "Movies",17: "Anime/Animation",18: "Action/Adventure",
            19: "Classics",20: "Comedy",21: "Documentary",22: "Drama",23: "Family",24: "Foreign",
            25: "Horror",26: "Sci-Fi/Fantasy",27:"Thriller",28:"Shorts", 29:"Shows",30: "Trailers"}

drop_column = ['video_id','title', 'channel_title','publish_time', 'tags', 'comment_count',
               'thumbnail_link', 'comments_disabled', 'ratings_disabled',
               'video_error_or_removed', 'description', 'likes', 'dislikes']
def filtering(address = None):
    will_drop = []
    df = pd.read_csv(address, encoding="ISO-8859-1")
    df['trending_date'] = '20' + (df['trending_date'].str.replace('.', '-'))
    df['trending_date'] = (df['trending_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%d-%m").date())).astype(str)
    df['category_id'] = df['category_id'].map(category)
    for column in df.columns.tolist():
        if column in drop_column:
            will_drop.append(column)
    df = df.drop(will_drop, axis=1)
    df_final = df.groupby(['trending_date', 'category_id'], as_index=False).sum()
    df_final['trending_date'] = df_final['trending_date'].apply(lambda x: x[:-3])
    df_final = df_final[(df_final['trending_date'] != '2017-11') & (df_final['trending_date'] !='2017-12')]
    df_final = df_final.groupby(['trending_date', 'category_id']).sum()
    return df_final
def make_graph(df, title=None ):
    fontP = FontProperties()
    pvt = df.pivot_table(index='trending_date', columns='category_id', values='views')
    plt.style.use('fivethirtyeight')
    bar = pvt.plot(figsize=(15,8), fontsize=22, cmap="Paired" ,ylim=(0, df['views'].max()*1.05), kind='bar')
    bar.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    bar.LineWidth = 10
    plt.legend(fontsize=20, title='Description of lines', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
    plt.title(title, loc='center',fontdict={'fontsize': 40})
    plt.xlabel('Months', fontdict={'fontsize': 25})
    plt.ylabel('Views', fontdict={'fontsize': 25})
    plt.xticks(rotation=45)
    return bar
df_US = filtering(address = "../input/youtube-new/USvideos.csv")
df_CA = filtering(address = "../input/youtube-new/CAvideos.csv")
df_DE = filtering(address = "../input/youtube-new/DEvideos.csv")
df_FR = filtering(address = "../input/youtube-new/FRvideos.csv")
df_GB = filtering(address = "../input/youtube-new/GBvideos.csv")
df_IN = filtering(address = "../input/youtube-new/INvideos.csv")
df_JP = filtering(address = "../input/youtube-new/JPvideos.csv")
df_KR = filtering(address = "../input/youtube-new/KRvideos.csv")
df_MX = filtering(address = "../input/youtube-new/MXvideos.csv")                     
df_RU = filtering(address = "../input/youtube-new/RUvideos.csv")
df_US_graph = make_graph(df_US, title="Trend of US")
df_CA_graph = make_graph(df_CA, title="Trend of CA")
df_DE_graph = make_graph(df_DE, title="Trend of DE")
df_FR_graph = make_graph(df_FR, title="Trend of FR")
df_GB_graph = make_graph(df_GB, title="Trend of GB")
df_IN_graph = make_graph(df_IN, title="Trend of IN")
df_JP_graph = make_graph(df_JP, title="Trend of JP")
df_KR_graph = make_graph(df_KR, title="Trend of KR")
df_MX_graph = make_graph(df_MX, title="Trend of MX")
df_RU_graph = make_graph(df_RU, title="Trend of RU")
