#download data

!wget https://github.com/cstorm125/viztech/raw/master/data/taladrod.csv

!wget https://github.com/cstorm125/viztech/raw/master/utils.py

!ls
import pandas as pd

import numpy as np

import scipy.stats as st



#ggplot equivalent: plotnine

from plotnine import *



#scales package equivalent: mizani

from mizani.breaks import *

from mizani.formatters import *



#widgets

from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets



#utility

import utils

def clean_comma(x): return float(str(x).replace(',',''))



#suppress warnings

import warnings

warnings.filterwarnings("ignore")
'''

Snippet for plotnine with thai font by @korakot

https://gist.github.com/korakot/01d181229b21411b0a20784e0ca20d3d

'''

import matplotlib

!wget https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf -q

!cp thsarabunnew-webfont.ttf /usr/share/fonts/truetype/

matplotlib.font_manager._rebuild()

matplotlib.rc('font', family='TH Sarabun New')

theme_set(theme_minimal(11, 'TH Sarabun New'));
df = pd.read_csv('taladrod.csv')

df['sales_price'] = df.sales_price.map(clean_comma)

df['market_price'] = df.market_price.map(clean_comma)

df.head()
missing = utils.check_missing(df)

missing['over90'] = missing.per_missing.map(lambda x: True if x>0.9 else False)

missing.head()
g = (ggplot(missing,aes(x='rnk',y='per_missing',fill='over90')) + #base plot

     geom_col() + #type of plot 

     geom_text(aes(x='rnk',y='per_missing+0.1',label='round(100*per_missing,2)')) +#annotate

     scale_y_continuous(labels=percent_format()) + #y-axis tick

     theme_minimal() + coord_flip()#theme and flipping plot

    )

g
#drop columns with too many missing values

df.drop(missing[missing.over90==True].col_name,1,inplace=True)

df.head()
cat_vars = ['brand','series','gen','color','gear','contact_location']

cat_df = df[cat_vars].copy()

cat_df.head()
#otherify popular values; you can (should?) also have a mapping dict

for col in cat_vars: cat_df = utils.otherify(cat_df,col,th=0.03)
interact(utils.value_dist, df =fixed(cat_df),

         col = widgets.Dropdown(options=list(cat_df.columns),value='brand'))
def cat_plot(df,col):

    return utils.cat_plot(df,col) 

#input dataframe and column

#output histogram plot of value distribution



interact(cat_plot, df=fixed(cat_df),

         col = widgets.Dropdown(options=list(cat_df.columns),value='brand'))
#excluding others

def cat_plot_noothers(df,col):

    x = df.copy()

    x = x[x[col]!='others']

    return utils.cat_plot(x,col) + utils.thai_text(8)



interact(cat_plot_noothers, df=fixed(cat_df),

         col = widgets.Dropdown(options=list(cat_df.columns),value='gen'))
#relationship between dependent variable and categorical variable

cat_df['sales_price'] = utils.boxcox(df['sales_price'])

cat_df.head()
#relationship between sales price and color

cat_df.groupby('color').sales_price.describe()
def numcat_plot(df,num,cat, no_outliers=True, geom=geom_boxplot()):

    return utils.numcat_plot(df,num,cat, no_outliers, geom) 

#plot the summary above
interact(numcat_plot, 

         df=fixed(cat_df),

         num=fixed('sales_price'),

         no_outliers = widgets.Checkbox(value=True),

         geom=fixed(geom_boxplot()), #geom_violin, geom_jitter

         cat= widgets.Dropdown(options=list(cat_df.columns)[:-1],value='brand'))
def numdist_plot(df, num,cat, geom=geom_density(alpha=0.5), no_outliers=True):

    return utils.numdist_plot(df, num, cat, geom, no_outliers)



#either

#density: geom_density(alpha=0.5)

#histogram: geom_histogram(binwidth=0.5, position='identity',alpha=0.5) 

#position: identity or dodge

numdist_plot(cat_df,'sales_price','gear')
numdist_plot(cat_df,'sales_price','gear',

             geom=geom_histogram(binwidth=0.5, position='dodge',alpha=0.5))
def catcat_plot(df, dep_cat, ind_cat):

    return utils.catcat_plot(df,dep_cat,ind_cat)
interact(catcat_plot, 

         df=fixed(cat_df),

         dep_cat=widgets.Dropdown(options=list(cat_df.columns)[:-1],value='gear'),

         ind_cat= widgets.Dropdown(options=list(cat_df.columns)[:-1],value='color'))
#getting fancy; not necessarily the best idea

new_df = utils.remove_outliers(cat_df,'sales_price')

g = (ggplot(new_df, aes(x='gen',y='sales_price')) +

     geom_boxplot() + theme_minimal() +

     facet_grid('contact_location~color') +

     theme(axis_text_x = element_text(angle = 90, hjust = 1))

    ) + utils.thai_text()

g
import datetime

now = datetime.datetime.now()

df['nb_year'] = now.year - df['year']

num_vars = ['nb_year','sales_price','market_price','subscribers']

num_df = df[num_vars].dropna() #this is why you need to deal with missing values BEFORE exploration

num_df.describe()
import seaborn as sns

sns.pairplot(num_df) #non-normal data is a problem!
interact(utils.qq_plot, df=fixed(num_df),

         col=widgets.Dropdown(options=list(num_df.columns),value='market_price'))
def boxcox(ser,lamb=0):

    pass

#input a column from pandas dataframe

#output transformed column
#see transformation results

def what_lamb(df,col,lamb):

    sample_df = df.copy()

    former_g = utils.qq_plot(sample_df,col)

    sample_df[col] = utils.boxcox(sample_df[col],lamb)

    print(utils.qq_plot(sample_df,col),former_g)

    

interact(what_lamb, df=fixed(num_df),

         col=widgets.Dropdown(options=list(num_df.columns),value='sales_price'),

         lamb=widgets.FloatSlider(min=-3,max=3,step=0.5,value=0)

         )
lamb_df = utils.boxcox_lamb_df(num_df.subscribers)

interact(utils.boxcox_plot, df=fixed(num_df),

         col=widgets.Dropdown(options=list(num_df.columns),value='sales_price'),

         ls=fixed([i/10 for i in range(-30,31,5)])

         )
#transform sales and market prices

for col in ['sales_price','market_price']:

    num_df['new_'+col] = utils.boxcox(num_df[col], utils.boxcox_lamb(num_df[col]))
sns.pairplot(num_df[['nb_year','new_sales_price','new_market_price','subscribers']]) #a little better!
num_m = num_df.melt()

num_m.head()
def value_dist_plot(df,bins=30):

    return utils.value_dist_plot(df,bins)

#input dataframe with only numerical variables

#output distribution plot for each variable

value_dist_plot(num_df)
interact(utils.jointplot, df=fixed(num_df),

         col_x= widgets.Dropdown(options=list(num_df.columns),value='sales_price'),

         col_y=widgets.Dropdown(options=list(num_df.columns),value='market_price'),

         kind=widgets.Dropdown(options=['scatter','resid','reg','hex','kde','point'],value='hex'))
#correlation plot if you must; but it's just ONE number for the relationship

num_df.corr(method='pearson').style.background_gradient(cmap='coolwarm') 
def pearson_corr(x,y):

    sub_x = x - x.mean()

    sub_y = y - y.mean()

    return (sub_x * sub_y).sum() / np.sqrt((sub_x**2).sum() * (sub_y**2).sum())



#spearman and kendall: pearson with rank variables

pearson_corr(df.nb_year,df.sales_price)