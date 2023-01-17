# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import umap

import math

import seaborn as sns



from sklearn.cluster import DBSCAN

from sklearn import preprocessing

import bokeh.io

from bokeh.models import ColumnDataSource, Label, HoverTool

from bokeh.plotting import figure, show



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

        

bokeh.io.output_notebook()

# Any results you write to the current directory are saved as output.
df_rwb = pd.read_csv('/kaggle/input/r-w-b-2019/index_2019_-_pour_import_1_1.csv')

print(df_rwb.columns.values.tolist())

df_trst = pd.read_csv('/kaggle/input/tourism-arrival-world-bank/API_ST.INT.ARVL_DS2_en_csv_v2_613598.csv',encoding = 'unicode_escape')

print(df_trst.columns.values.tolist())
r_rk = df_rwb.loc[:, ['EN_country']]

r_sc = df_rwb.loc[:, ['Score 2019']]

f_tr = df_trst.loc[:, ['Country Name']]

tr_2017 = df_trst.loc[:, ['2017']]

r_dt = r_rk.values.tolist()

r_scr = r_sc.values.tolist()

f_trst = f_tr.values.tolist()

trst_2017 = tr_2017.values.tolist()

#print(trst_2017)

country_score = {}

country_trst = {}

for c in range(len(f_trst)):

    if math.isnan(trst_2017[c][0]):

        print('no 2017 {}'.format(f_trst[c][0]))

    else:

        country_trst[f_trst[c][0]] = trst_2017[c][0]

#print(country_trst)

for c in range(len(r_dt)):

    country_score[r_dt[c][0]] = r_scr[c][0]

#print(country_score)

r_country = [*country_score]

t_country = [*country_trst]

df = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

h_rk = df.loc[:, ['Country or region']]

h_dt = h_rk.values.tolist()



score_2019 = []

ctry = []

ctry_all = []

print('Missing countries:')

msg_ctr = {}

msg_ctr['Congo (Brazzaville)'] = ['Congo']

msg_ctr['Congo (Kinshasa)'] = ['The Democratic Republic Of The Congo']

msg_ctr['Iran'] = ['Islamic Republic of Iran']

msg_ctr['Laos'] = ['Lao People\'s Democratic Republic']

msg_ctr['North Macedonia'] = ['Macedonia']

msg_ctr['Northern Cyprus'] = ['Cyprus North']

msg_ctr['Palestinian Territories'] = ['Palestine']

msg_ctr['Russia'] = ['Russian Federation']

msg_ctr['Syria'] = ['Syrian Arab Republic']

msg_ctr['Trinidad & Tobago'] = ['Trinidad and Tobago']



for c in range(len(h_dt)):

    ctry_all.append(h_dt[c][0])

    if h_dt[c][0] not in r_country:

        

        if h_dt[c][0] in msg_ctr:

            c_s = country_score[msg_ctr[h_dt[c][0]][0]]

            score_2019.append(float(c_s.replace(',','.')))

        else:

            print(h_dt[c][0])

    else: # Let's get the score 2019

        c_s = country_score[h_dt[c][0]]

        ctry.append(h_dt[c][0])

        score_2019.append(float(c_s.replace(',','.')))

    

    if h_dt[c][0] not in t_country:

        print(h_dt[c][0])

       

print("Number of missing country: {}".format(len(h_dt)-len(score_2019)))
df['Score_2019'] = np.array(score_2019)

cols_to_norm = ['GDP per capita','Score_2019']

df[cols_to_norm] = preprocessing.StandardScaler().fit_transform(df[cols_to_norm])

X = df.drop(['Country or region','Overall rank'], axis=1)

# test with heatmap

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(10, 8))

corr = X.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)

plt.show()

#X_rwb = X.copy()

#raise ValueError() 
clusterable_embedding = umap.UMAP(

                n_neighbors=5,  

                min_dist=0.0,

                n_components=2,

                random_state=42,

            ).fit_transform(X.values)



labels = DBSCAN(

                eps=0.25,

                min_samples=5).fit_predict(clusterable_embedding)

clustered = (labels >= 0)

xtx = clusterable_embedding[clustered, 0]

ytx = clusterable_embedding[clustered, 1]



xtx_n = clusterable_embedding[~clustered, 0]

ytx_n = clusterable_embedding[~clustered, 1]





llbl = list(labels[clustered])

colormap = np.array([

            "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",

"#fbb4ae","#b3cde3","#ccebc5","#decbe4","#fed9a6","#ffffcc","#e5d8bd","#fddaec","#f2f2f2"])

col = []

for i in llbl:

    col.append(colormap[i])
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)



print("NUmber of Clusters {}".format(n_clusters_))
hov_txt = list(labels)

hover_dt = []

hover_noise =[]

i = 0

for f_t in hov_txt:

    if f_t >= 0:

        hover_dt.append(h_dt[i])

    else:

        hover_noise.append(h_dt[i])

    i += 1
sourcetx = ColumnDataSource(data=dict(xtx=xtx, ytx=ytx, hover_dt=hover_dt, col=col))

source_noise = ColumnDataSource(data=dict(xtx_n=xtx_n, ytx_n=ytx_n,hover_noise=hover_noise))



ptx = figure(plot_width=830, plot_height=600,

             title="Happiness. (hover for country name)",

             tools="pan,wheel_zoom,box_zoom,reset",

             active_scroll="wheel_zoom",

             toolbar_location="above"

             )

ptx.xgrid.grid_line_color = None

ptx.ygrid.grid_line_color = None
dt_cl = ptx.scatter('xtx', 'ytx', size=4, alpha=0.8, line_dash='solid', color="col", source=sourcetx,

                     legend_label='Clusters')

ptx.add_tools(HoverTool(renderers=[dt_cl], tooltips=[("Country", "@hover_dt")]))

dt_ns = ptx.scatter('xtx_n', 'ytx_n', size=4, alpha=0.8, line_dash='solid', color='white',line_color='#aec6cf',

                         source=source_noise, legend_label='Noise') 

ptx.add_tools(HoverTool(renderers=[dt_ns], tooltips=[("Country", "@hover_noise")]))

ptx.legend.click_policy = "hide"

ptx.legend.background_fill_alpha = 0.4

ptx.legend.location = "bottom_right"



show(ptx)