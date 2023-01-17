import os

import pandas as pd

os.listdir('../input/youtube-new/')

import matplotlib.pyplot as plt

import seaborn as sns

df_mx = pd.read_csv("/kaggle/input/youtube-new/MXvideos.csv" , encoding="ISO-8859–1")

df_jp = pd.read_csv("/kaggle/input/youtube-new/JPvideos.csv" , encoding="ISO-8859–1")

df_gb = pd.read_csv("/kaggle/input/youtube-new/GBvideos.csv" , encoding="ISO-8859–1")

df_us = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv" , encoding="ISO-8859–1")

df_ru = pd.read_csv("/kaggle/input/youtube-new/RUvideos.csv" , encoding="ISO-8859–1")

df_kr = pd.read_csv("/kaggle/input/youtube-new/KRvideos.csv" , encoding="ISO-8859–1")

df_in = pd.read_csv("/kaggle/input/youtube-new/INvideos.csv" , encoding="ISO-8859–1")

df_fr = pd.read_csv("/kaggle/input/youtube-new/FRvideos.csv" , encoding="ISO-8859–1")

df_de = pd.read_csv("/kaggle/input/youtube-new/DEvideos.csv" , encoding="ISO-8859–1")

df_ca = pd.read_csv("/kaggle/input/youtube-new/CAvideos.csv" , encoding="ISO-8859–1")
# df_mx['source'] = 'mx'

g = sns.pairplot(df_mx[['views', 'likes', 'dislikes', 'comment_count', 'category_id']], hue='category_id')

_ = g.fig.suptitle("mx")
# df_jp['source'] = 'jp'

g = sns.pairplot(df_jp[['views', 'likes', 'dislikes', 'comment_count', 'category_id']], hue='category_id')

_ = g.fig.suptitle("jp")
# df_gb['source'] = 'gb'

g = sns.pairplot(df_gb[['views', 'likes', 'dislikes', 'comment_count', 'category_id']], hue='category_id')

_ = g.fig.suptitle("gb")
# df_us['source'] = 'us'

g = sns.pairplot(df_us[['views', 'likes', 'dislikes', 'comment_count', 'category_id']], hue='category_id')

_ = g.fig.suptitle("us")
# df_ru['source'] = 'ru'

g = sns.pairplot(df_ru[['views', 'likes', 'dislikes', 'comment_count', 'category_id']], hue='category_id')

_ = g.fig.suptitle("ru")
# df_kr['source'] = 'kr'

g = sns.pairplot(df_kr[['views', 'likes', 'dislikes', 'comment_count', 'category_id']], hue='category_id')

_ = g.fig.suptitle("kr")
# df_in['source'] = 'in'

g = sns.pairplot(df_in[['views', 'likes', 'dislikes', 'comment_count', 'category_id']], hue='category_id')

_ = g.fig.suptitle("in")
# df_fr['source'] = 'fr'

g = sns.pairplot(df_fr[['views', 'likes', 'dislikes', 'comment_count', 'category_id']], hue='category_id')

_ = g.fig.suptitle("fr")
# df_de['source'] = 'de'

g = sns.pairplot(df_de[['views', 'likes', 'dislikes', 'comment_count', 'category_id']], hue='category_id')

_ = g.fig.suptitle("de")


# df_ca['source'] = 'ca'

g = sns.pairplot(df_ca[['views', 'likes', 'dislikes', 'comment_count', 'category_id']], hue='category_id')

_ = g.fig.suptitle("ca")
# import plotly.graph_objs as go

# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot





# init_notebook_mode(connected=True)





# fig = go.Figure()



# fig = go.Figure(data=go.Scatter(

#     x=df_mx['views'],

#     y=df_mx['likes'],

#     mode='markers',

# #     marker=dict(size=(df_mx['comment_count']/900000)*5,

# #                 color=df_mx['dislikes']

# #                )

# ))



# fig.show()


