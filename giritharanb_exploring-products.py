from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
output_notebook()

def get_product_agg(cols):
    df_train = pd.read_csv('../input/train.csv', usecols = ['Semana', 'Producto_ID'] + cols,
                           dtype  = {'Semana': 'int32',
                                     'Producto_ID':'int32',
                                     'Venta_hoy':'float32',
                                     'Venta_uni_hoy': 'int32',
                                     'Dev_uni_proxima':'int32',
                                     'Dev_proxima':'float32',
                                     'Demanda_uni_equil':'int32'})
    agg  = df_train.groupby(['Semana', 'Producto_ID'], as_index=False).agg(['count','sum', 'min', 'max','median','mean'])
    agg.columns  =  ['_'.join(col).strip() for col in agg.columns.values]
    del(df_train)
    return agg
# ### a simple product aggregate. Kaggle computing power allows us to only calculate 1-2 fields at a time.
agg1 = get_product_agg(['Demanda_uni_equil','Dev_uni_proxima'])
agg1.shape
agg1.head()
agg2 = get_product_agg(['Venta_uni_hoy'])
agg = agg1.join(agg2)
# Let's preprocess products a little bit. I borrowed some of the preprocessing from [here](https://www.kaggle.com/lyytinen/grupo-bimbo-inventory-demand/basic-preprocessing-for-products) 
products  =  pd.read_csv("../input/producto_tabla.csv")
products  =  pd.read_csv("../input/producto_tabla.csv")
products['short_name'] = products.NombreProducto.str.extract('^(\D*)', expand=False)
products['brand'] = products.NombreProducto.str.extract('^.+\s(\D+) \d+$', expand=False)
w = products.NombreProducto.str.extract('(\d+)(Kg|g)', expand=True)
products['weight'] = w[0].astype('float')*w[1].map({'Kg':1000, 'g':1})
products['pieces'] =  products.NombreProducto.str.extract('(\d+)p ', expand=False).astype('float')
products.head()
products.tail()
products.short_name.value_counts(dropna=False)
# There are some weird products that weight 42 Kilos. Check out this Exhibidor :
# 
# ![Exhibidor bimbo](https://mir-s3-cdn-cf.behance.net/project_modules/disp/55c94f24003843.5632c737c062c.jpeg)
sns.distplot(products.weight.dropna())
# distribution of pieces
sns.distplot(products.pieces.dropna())
# #### Lets clean up product names a bit, we have ~1000 unique names once we cleaned the weights, but there is much more work to be done
# #### Products have some abbreviation leftovers that I did not clean, products have similar names but different word forms, etc.
products.short_name.nunique()
# Let's clean stop words and leave only the word stems (I did not clean abbreviations, sorry)
from nltk.corpus import stopwords
print(stopwords.words("spanish"))
products['short_name_processed'] = (products['short_name']
                                        .map(lambda x: " ".join([i for i in x.lower()
                                                                 .split() if i not in stopwords.words("spanish")])))
products['short_name_processed'].nunique()
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("spanish")
print(stemmer.stem("Tortillas"))
products['short_name_processed'] = (products['short_name_processed']
                                        .map(lambda x: " ".join([stemmer.stem(i) for i in x.lower().split()])))
products.short_name_processed.nunique()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 1000) 

product_bag_words = vectorizer.fit_transform(products.short_name_processed).toarray()
product_bag_words.shape
vectorizer.get_feature_names()
product_bag_words = pd.concat([products.Producto_ID, 
                               pd.DataFrame(product_bag_words, 
                                            columns= vectorizer.get_feature_names(), index = products.index)], axis=1)
product_bag_words.head()
product_bag_words.drop('Producto_ID', axis=1).sum().sort_values(ascending=False).head(100)
# ####  let's have a look, what is the product with the biggest demand of all times
df = (pd.merge(agg1.reset_index(), products, on='Producto_ID', how='left').
      groupby('short_name')['Demanda_uni_equil_sum'].sum().sort_values(ascending=False))
plt.figure(figsize = (12,15))
df.head(50).sort_values().plot(kind='barh')
# The best seller is by far Nito. Check this out, looks yummy : 
# 
# ![](http://static.manufactura.mx/media/2013/11/20/negrito.jpg)
# #### a quick look at the distributions
sns.distplot(df)
sns.distplot(np.log1p(df))
# #### expanding the aggregate
df = (pd.merge(agg.reset_index(), products, on='Producto_ID', how='left').
      groupby('short_name')['Demanda_uni_equil_sum', 'Venta_uni_hoy_sum', 'Dev_uni_proxima_sum', 'Dev_uni_proxima_count']
      .sum().sort_values(by = 'Demanda_uni_equil_sum', ascending=False))
df.describe().T
# There are interesting things. There are products for which Demanda_uni_equil_sum = 0 and other fields are not equal to 0
df[df.Demanda_uni_equil_sum == 0].count()
df[df.Demanda_uni_equil_sum == 0]
# Similarly there are products with 0 sales and only returns
df[df.Venta_uni_hoy_sum == 0]
# there are products that were never returned
df[df.Dev_uni_proxima_sum == 0].count()
df[df.Dev_uni_proxima_sum == 0].head(20)
# Let's cut products into 10 quantiles by summary adjusted demand.
df['Q'] = pd.qcut(df.Demanda_uni_equil_sum, 10)
df.Q.value_counts()
df[df.Q == '[0, 49]'].index.values
# distribution of returns by product
sns.distplot(df.Dev_uni_proxima_sum)
sns.distplot(np.log1p(df.Dev_uni_proxima_sum))
# Distribution of counts by product
sns.distplot(df.Dev_uni_proxima_count)
sns.distplot(np.log1p(df.Dev_uni_proxima_count))
# #### lets aggregate by week and short_name now
df_hmp = (pd.merge(agg.reset_index(), products, on='Producto_ID', how='left').
      groupby(['Semana','short_name'])['Demanda_uni_equil_sum', 'Venta_uni_hoy_sum', 'Dev_uni_proxima_sum', 'Dev_uni_proxima_count'].sum().reset_index())
df_hmp.head()
# #### a quick check if demand distribution changes week to week
df_hmp['log1p_Demanda_uni_equil_sum'] = np.log1p(df_hmp.Demanda_uni_equil_sum)
g = sns.FacetGrid(df_hmp, row = 'Semana')
g = g.map(sns.distplot, 'log1p_Demanda_uni_equil_sum')
# #### Now let's look at which proucts sell by week with interactive heatmaps. Let's use our quantiles here.
from bokeh.charts import HeatMap
from bokeh.plotting import vplot

heatmaps = []
for i in df.Q.cat.categories.values:
    hm = HeatMap(df_hmp[df_hmp.short_name.isin(df[df.Q == i].index.values)],
                        x='short_name', y = 'Semana', values = 'Demanda_uni_equil_sum',
                 hover_tool = True, title = 'Products with summary demand '+ str(i), xgrid = False,
                 stat = 'sum',plot_width=950, plot_height=400, tools='hover, box_zoom, resize, save, wheel_zoom, reset',
                 )
    heatmaps.append(hm)
show(vplot(*heatmaps))
# #### Same series of charts but for returns
from bokeh.charts import HeatMap
from bokeh.plotting import vplot
df['Q_ret'] = pd.qcut(df.Dev_uni_proxima_sum, 5)
heatmaps = []
for i in df.Q_ret.cat.categories.values:
    hm = HeatMap(df_hmp[df_hmp.short_name.isin(df[df.Q_ret == i].index.values)],
                        x='short_name', y = 'Semana', values = 'Demanda_uni_equil_sum',
                 hover_tool = True, title = 'Products with summary returns '+ str(i), xgrid = False,
                 stat = 'sum',plot_width=800, plot_height=400, tools='hover, box_zoom, resize, save, wheel_zoom, reset',
                 )
    heatmaps.append(hm)
show(vplot(*heatmaps))
