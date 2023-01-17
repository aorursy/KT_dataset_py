# Environment defined by docker image: https://github.com/kaggle/docker-python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.graph_objs as go
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image    # to import the image
import warnings; warnings.simplefilter('ignore')

%matplotlib inline
import os
import bq_helper
from bq_helper import BigQueryHelper
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")
usa_names.list_tables()
QUERY_thru_2013 = "SELECT year, gender, name, sum(number) as year_total FROM `bigquery-public-data.usa_names.usa_1910_2013` group by year, gender, name"
QUERY_thru_current = "SELECT year, gender, name, sum(number) as year_total FROM `bigquery-public-data.usa_names.usa_1910_current` group by year, gender, name"
usa_names.estimate_query_size(QUERY_thru_current)
all_names = usa_names.query_to_pandas(QUERY_thru_current)
all_names['gender'].describe()
all_names.loc[all_names['gender']=='F'].describe()
all_names.loc[all_names['gender']=='M'].describe()
sns.countplot(x='gender',data=all_names, linewidth=5, edgecolor=sns.color_palette("Set3"))
all_names.loc[all_names['gender']=='F'].year_total.sum()  \
/ all_names.loc[all_names['gender']=='M'].year_total.sum() * 100
generations = pd.DataFrame([], columns=['year','generation'])
generations['year'] = pd.Series(range(1910,2017))
generations.loc[(generations['year'] < 1928), 'generation'] = 'Pre-Silent'
generations.loc[(generations['year'] >= 1928) & (generations['year'] < 1946), 'generation'] = 'Silent'
generations.loc[(generations['year'] >= 1946) & (generations['year'] < 1965), 'generation'] = 'Baby Boomers'
generations.loc[(generations['year'] >= 1965) & (generations['year'] < 1982), 'generation'] = 'Generation X'
generations.loc[(generations['year'] >= 1982) & (generations['year'] < 1997), 'generation'] = 'Millenials'
generations.loc[(generations['year'] >= 1997), 'generation'] = 'Post-Millenials'
generations.sample(10)
all_names_gen = pd.merge(all_names,generations, on='year')
all_names_gen
gen_group = all_names_gen.groupby(['generation','name'])
gen_stats = gen_group['year_total'].agg([np.sum,np.std,np.mean])
gen_stats.loc['Millenials']
" ".join(gen_stats.loc['Millenials'].nlargest(100,'mean').index.tolist())
top_millenial = list(zip(gen_stats.loc['Millenials'].nlargest(100,'mean')['sum'].index,
                         gen_stats.loc['Millenials'].nlargest(100,'mean')['sum']))

norm = [float(i)/sum([x[1] for x in top_millenial]) for i in [x[1] for x in top_millenial]]

top_millenial_norm = list(zip(gen_stats.loc['Millenials'].nlargest(100,'mean')['sum'].index,
         norm))
gi_mask = np.array(Image.open("../input/millenial.png"))

wordcloud = WordCloud( max_font_size=75,
                       mask = gi_mask,
                       background_color='white',
                       width=1000, height=450
                     ).fit_words(dict(top_millenial_norm))
image_colors = ImageColorGenerator(gi_mask)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

plt.title("Wordcloud for Top Names in Millenial Generation", fontsize=35)

plt.show()
def generation_cloud(gen_id,gen_stats,png_mask) :
    top_ = list(zip(gen_stats.loc[gen_id].nlargest(250,'mean')['sum'].index,
                         gen_stats.loc[gen_id].nlargest(250,'mean')['sum']))

    norm = [float(i)/sum([x[1] for x in top_]) for i in [x[1] for x in top_]]

    top_norm = list(zip(gen_stats.loc[gen_id].nlargest(100,'mean')['sum'].index,
         norm))

    image_mask = np.array(Image.open(png_mask))

    wordcloud = WordCloud( max_font_size=120,
                       mask = image_mask,
                       background_color='black',
                       width=1600, height=400
                     ).fit_words(dict(top_norm))
    
    image_colors = ImageColorGenerator(image_mask)
    
    plt.figure(figsize=(30,10))
    plt.title("Wordcloud for Top Names in " + gen_id + " Generation" , fontsize=35)
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    plt.show()
generation_cloud('Pre-Silent',gen_stats,"../input/gi.png")
generation_cloud('Silent',gen_stats,"../input/silent.png")
generation_cloud('Baby Boomers',gen_stats,"../input/boomer.png")
generation_cloud('Generation X',gen_stats,"../input/genx.png")
generation_cloud('Post-Millenials',gen_stats,"../input/gi.png")
line_dat = pd.DataFrame(all_names_gen.loc[(all_names_gen['name'] == 'Sage') & (all_names_gen['gender'] == 'F')]).sort_values(by='year')
plt.plot( 'year','year_total',data = line_dat)
from fbprophet import Prophet

line_dat['year'] = pd.to_datetime(line_dat['year'],format = '%Y')
pd.to_numeric(line_dat['year_total'])
ts_single = line_dat[['year','year_total']]
ts_single['cap'] = 2* max(ts_single['year_total'])

sns.set(font_scale=1) 
ts_date_index = ts_single
ts_date_index = ts_date_index.set_index('year')
ts_prophet = ts_date_index.copy()
ts_prophet.reset_index(drop=False,inplace=True)
ts_single.columns = ['ds','y','cap']
ts_single.head()
model = Prophet(changepoint_prior_scale=10, growth = 'logistic').fit(ts_single)

future = model.make_future_dataframe(periods=10,freq='Y')
future['cap'] = 2* max(ts_single['y'])
forecast = model.predict(future)
fig = model.plot(forecast) 