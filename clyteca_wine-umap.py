# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



import bokeh.io

from bokeh.models import ColumnDataSource, Label

from bokeh.plotting import figure, show

from bokeh.models import Legend, LegendItem

from bokeh.models import HoverTool



import umap



from sklearn.cluster import DBSCAN

from sklearn.feature_extraction.text import TfidfVectorizer



from nltk.corpus import stopwords

import re

from collections import Counter



bokeh.io.output_notebook()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
w_tst = pd.read_csv('/kaggle/input/wine-reviews//winemag-data-130k-v2.csv', delimiter=',')

des_var = w_tst.loc[:, ['description', 'variety', 'country', 'title']]

d_v = des_var.values.tolist()

list_des = list(zip(*d_v))[0]

list_var = list(zip(*d_v))[1]

n_list_var = [x if type(x) == str else 'unknown' for x in list_var]

list_ctr = list(zip(*d_v))[2]

n_list_ctr = [x if type(x) == str else 'Dreamland' for x in list_ctr]

count_ctr = Counter(n_list_ctr)

count_var = Counter(n_list_var)

frst_ctr = sorted(count_ctr, key=count_ctr.get, reverse=True)

list_title = list(zip(*d_v))[3]

l_annee = []

for ttl in list_title:

    if len(ttl) > 0:

        only_nbr = re.split("[^0-9]", ttl)

        only_year = [int(x) for x in only_nbr if len(x) == 4 and 1950 < int(x) < 2017]  # more than 1 date

        # if 2 date take the oldest one

        #print(only_year)

        if len(only_year) > 0:

            if len(only_year) >= 2:

                l_annee.append(max(only_year))

            else:

                l_annee.append(only_year[0])

        else:

            l_annee.append('na')

    else:

        l_annee.append('na')
idx_all_ctr = []

for ctr in frst_ctr:

    idx_ctr = [idx for idx, e in enumerate(n_list_ctr) if e == ctr]

    idx_all_ctr.append(idx_ctr)
list_des_clean = []

en_stpwrd = set(stopwords.words("english"))

en_stpwrd.update(['wine'])



for txt in list_des:

    let_only = re.sub("[^ãâäéèñóôüúûa-zA-Z]", " ", txt)

    low_case = let_only.lower()

    words = low_case.split()



    noword_stp = [w for w in words if w not in en_stpwrd]

    txt_cleaned = " ".join(noword_stp)

    list_des_clean.append(txt_cleaned)
vectorizer = TfidfVectorizer(min_df=4, max_features=None)

vz = vectorizer.fit_transform(list_des_clean)



clusterable_embedding = umap.UMAP(

    n_neighbors=30,

    min_dist=0.0,

    n_components=2,

    random_state=42,

    ).fit_transform(vz)



labels = DBSCAN(

                eps=0.26, 

                min_samples=35

                ).fit_predict(clusterable_embedding)
xtx_us = []

ytx_us = []

color_us = []

hover_us = []

xtx_fr = []

ytx_fr = []

color_fr = []

hover_fr = []

xtx_it = []

ytx_it = []

color_it = []

hover_it = []

xtx_sp = []

ytx_sp = []

color_sp = []

hover_sp = []

xtx_pt = []

ytx_pt = []

color_pt = []

hover_pt = []

xtx_ch = []

ytx_ch = []

color_ch = []

hover_ch = []

xtx_arg = []

ytx_arg = []

color_arg = []

hover_arg = []

xtx_atr = []

ytx_atr = []

color_atr = []

hover_atr = []

xtx_atl = []

ytx_atl = []

color_atl = []

hover_atl = []

xtx_ge = []

ytx_ge = []

color_ge = []

hover_ge = []

xtx_nz = []

ytx_nz = []

color_nz = []

hover_nz = []

xtx_sa = []

ytx_sa = []

color_sa = []

hover_sa = []

xtx_is = []

ytx_is = []

color_is = []

hover_is = []

xtx_gr = []

ytx_gr = []

color_gr = []

hover_gr = []

xtx_ca = []

ytx_ca = []

color_ca = []

hover_ca = []

xtx_hu = []

ytx_hu = []

color_hu = []

hover_hu = []

xtx_bu = []

ytx_bu = []

color_bu = []

hover_bu = []

xtx_ro = []

ytx_ro = []

color_ro = []

hover_ro = []

xtx_ur = []

ytx_ur = []

color_ur = []

hover_ur = []

xtx_tu = []

ytx_tu = []

color_tu = []

hover_tu = []

xtx_sl = []

ytx_sl = []

color_sl = []

hover_sl = []

xtx_geo = []

ytx_geo = []

color_geo = []

hover_geo = []

xtx_en = []

ytx_en = []

color_en = []

hover_en = []

xtx_cr = []

ytx_cr = []

color_cr = []

hover_cr = []

xtx_me = []

ytx_me = []

color_me = []

hover_me = []

xtx_mol = []

ytx_mol = []

color_mol = []

hover_mol = []

xtx_br = []

ytx_br = []

color_br = []

hover_br = []

xtx_le = []

ytx_le = []

color_le = []

hover_le = []

xtx_mor = []

ytx_mor = []

color_mor = []

hover_mor = []

xtx_pe = []

ytx_pe = []

color_pe = []

hover_pe = []

xtx_uk = []

ytx_uk = []

color_uk = []

hover_uk = []

xtx_se = []

ytx_se = []

color_se = []

hover_se = []

xtx_cz = []

ytx_cz = []

color_cz = []

hover_cz = []

xtx_ma = []

ytx_ma = []

color_ma = []

hover_ma = []



xtx_n = []

ytx_n = []

hover_noise = []
i = 0

count_map = 0



north = []

# square_root = []

for xy in range(len(labels)):

    if clusterable_embedding[xy, 1] > 8.0:

        north.append(list_des_clean[i])

        count_map += 1

    var_ctr = n_list_var[i] + ';' + str(l_annee[i]) + ';' + n_list_ctr[i]

    if i in idx_all_ctr[0]:

        xtx_us.append(clusterable_embedding[xy, 0])

        ytx_us.append(clusterable_embedding[xy, 1])

        color_us.append('#3C3B6E')

        hover_us.append(var_ctr)

    elif i in idx_all_ctr[1]:

        xtx_fr.append(clusterable_embedding[xy, 0])

        ytx_fr.append(clusterable_embedding[xy, 1])

        color_fr.append('#002395')

        hover_fr.append(var_ctr)

    elif i in idx_all_ctr[2]:

        xtx_it.append(clusterable_embedding[xy, 0])

        ytx_it.append(clusterable_embedding[xy, 1])

        color_it.append('#009246')

        hover_it.append(var_ctr)

    elif i in idx_all_ctr[3]:

        xtx_sp.append(clusterable_embedding[xy, 0])

        ytx_sp.append(clusterable_embedding[xy, 1])

        color_sp.append('#FFC400')

        hover_sp.append(var_ctr)

    elif i in idx_all_ctr[4]:

        xtx_pt.append(clusterable_embedding[xy, 0])

        ytx_pt.append(clusterable_embedding[xy, 1])

        color_pt.append('#FFFF00')

        hover_pt.append(var_ctr)

    elif i in idx_all_ctr[5]:

        xtx_ch.append(clusterable_embedding[xy, 0])

        ytx_ch.append(clusterable_embedding[xy, 1])

        color_ch.append('#0039A6')

        hover_ch.append(var_ctr)

    elif i in idx_all_ctr[6]:

        xtx_arg.append(clusterable_embedding[xy, 0])

        ytx_arg.append(clusterable_embedding[xy, 1])

        color_arg.append('#75AADB')

        hover_arg.append(var_ctr)

    elif i in idx_all_ctr[7]:

        xtx_atr.append(clusterable_embedding[xy, 0])

        ytx_atr.append(clusterable_embedding[xy, 1])

        color_atr.append('#ED2939')

        hover_atr.append(var_ctr)

    elif i in idx_all_ctr[8]:

        xtx_atl.append(clusterable_embedding[xy, 0])

        ytx_atl.append(clusterable_embedding[xy, 1])

        color_atl.append('#012169')

        hover_atl.append(var_ctr)

    elif i in idx_all_ctr[9]:

        xtx_ge.append(clusterable_embedding[xy, 0])

        ytx_ge.append(clusterable_embedding[xy, 1])

        color_ge.append('black')

        hover_ge.append(var_ctr)

    elif i in idx_all_ctr[10]:

        xtx_nz.append(clusterable_embedding[xy, 0])

        ytx_nz.append(clusterable_embedding[xy, 1])

        color_nz.append('#00247d')

        hover_nz.append(var_ctr)

    elif i in idx_all_ctr[11]:

        xtx_sa.append(clusterable_embedding[xy, 0])

        ytx_sa.append(clusterable_embedding[xy, 1])

        color_sa.append('#007749')

        hover_sa.append(var_ctr)

    elif i in idx_all_ctr[12]:

        xtx_is.append(clusterable_embedding[xy, 0])

        ytx_is.append(clusterable_embedding[xy, 1])

        color_is.append('#0038B8')

        hover_is.append(var_ctr)

    elif i in idx_all_ctr[13]:

        xtx_gr.append(clusterable_embedding[xy, 0])

        ytx_gr.append(clusterable_embedding[xy, 1])

        color_gr.append('#0D5EAF')

        hover_gr.append(var_ctr)

    elif i in idx_all_ctr[14]:

        xtx_ca.append(clusterable_embedding[xy, 0])

        ytx_ca.append(clusterable_embedding[xy, 1])

        color_ca.append('#FF0000')

        hover_ca.append(var_ctr)

    elif i in idx_all_ctr[15]:

        xtx_hu.append(clusterable_embedding[xy, 0])

        ytx_hu.append(clusterable_embedding[xy, 1])

        color_hu.append('#436F4D')

        hover_hu.append(var_ctr)

    elif i in idx_all_ctr[16]:

        xtx_bu.append(clusterable_embedding[xy, 0])

        ytx_bu.append(clusterable_embedding[xy, 1])

        color_bu.append('#00966E')

        hover_bu.append(var_ctr)

    elif i in idx_all_ctr[17]:

        xtx_ro.append(clusterable_embedding[xy, 0])

        ytx_ro.append(clusterable_embedding[xy, 1])

        color_ro.append('#FCD116')

        hover_ro.append(var_ctr)

    elif i in idx_all_ctr[18]:

        xtx_ur.append(clusterable_embedding[xy, 0])

        ytx_ur.append(clusterable_embedding[xy, 1])

        color_ur.append('#7b3f00')

        hover_ur.append(var_ctr)

    elif i in idx_all_ctr[19]:

        xtx_tu.append(clusterable_embedding[xy, 0])

        ytx_tu.append(clusterable_embedding[xy, 1])

        color_tu.append('#E30A17')

        hover_tu.append(var_ctr)

    elif i in idx_all_ctr[20]:

        xtx_sl.append(clusterable_embedding[xy, 0])

        ytx_sl.append(clusterable_embedding[xy, 1])

        color_sl.append('#005DA4')

        hover_sl.append(var_ctr)

    elif i in idx_all_ctr[21]:

        xtx_geo.append(clusterable_embedding[xy, 0])

        ytx_geo.append(clusterable_embedding[xy, 1])

        color_geo.append('#FF0000')

        hover_geo.append(var_ctr)

    elif i in idx_all_ctr[22]:

        xtx_en.append(clusterable_embedding[xy, 0])

        ytx_en.append(clusterable_embedding[xy, 1])

        color_en.append('#FF0000')

        hover_en.append(var_ctr)

    elif i in idx_all_ctr[23]:

        xtx_cr.append(clusterable_embedding[xy, 0])

        ytx_cr.append(clusterable_embedding[xy, 1])

        color_cr.append('#171796')

        hover_cr.append(var_ctr)

    elif i in idx_all_ctr[24]:

        xtx_me.append(clusterable_embedding[xy, 0])

        ytx_me.append(clusterable_embedding[xy, 1])

        color_me.append('#006847')

        hover_me.append(var_ctr)

        # 25 is dreamland

    elif i in idx_all_ctr[26]:

        xtx_mol.append(clusterable_embedding[xy, 0])

        ytx_mol.append(clusterable_embedding[xy, 1])

        color_mol.append('#FFD200')

        hover_mol.append(var_ctr)

    elif i in idx_all_ctr[27]:

        xtx_br.append(clusterable_embedding[xy, 0])

        ytx_br.append(clusterable_embedding[xy, 1])

        color_br.append('#FEDF00')

        hover_br.append(var_ctr)

    elif i in idx_all_ctr[28]:

        xtx_le.append(clusterable_embedding[xy, 0])

        ytx_le.append(clusterable_embedding[xy, 1])

        color_le.append('#ED1C24')

        hover_le.append(var_ctr)

    elif i in idx_all_ctr[29]:

        xtx_mor.append(clusterable_embedding[xy, 0])

        ytx_mor.append(clusterable_embedding[xy, 1])

        color_mor.append('#c1272d')

        hover_mor.append(var_ctr)

    elif i in idx_all_ctr[30]:

        xtx_pe.append(clusterable_embedding[xy, 0])

        ytx_pe.append(clusterable_embedding[xy, 1])

        color_pe.append('#D91023')

        hover_pe.append(var_ctr)

    elif i in idx_all_ctr[31]:

        xtx_uk.append(clusterable_embedding[xy, 0])

        ytx_uk.append(clusterable_embedding[xy, 1])

        color_uk.append('#FFD500')

        hover_uk.append(var_ctr)

    elif i in idx_all_ctr[32]:

        xtx_se.append(clusterable_embedding[xy, 0])

        ytx_se.append(clusterable_embedding[xy, 1])

        color_se.append('#0C4076')

        hover_se.append(var_ctr)

    elif i in idx_all_ctr[33]:

        xtx_cz.append(clusterable_embedding[xy, 0])

        ytx_cz.append(clusterable_embedding[xy, 1])

        color_cz.append('#D7141A')

        hover_cz.append(var_ctr)

    elif i in idx_all_ctr[34]:

        xtx_ma.append(clusterable_embedding[xy, 0])

        ytx_ma.append(clusterable_embedding[xy, 1])

        color_ma.append('#ffe600')

        hover_ma.append(var_ctr)

    else:

        xtx_n.append(clusterable_embedding[xy, 0])

        ytx_n.append(clusterable_embedding[xy, 1])

        hover_noise.append(var_ctr)

    i += 1
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)

print('Estimated number of noise points: %d' % n_noise_)
source_us = ColumnDataSource(data=dict(xtx_us=xtx_us, ytx_us=ytx_us, hover_us=hover_us, color_us=color_us))

source_fr = ColumnDataSource(data=dict(xtx_fr=xtx_fr, ytx_fr=ytx_fr, hover_fr=hover_fr, color_fr=color_fr))

source_it = ColumnDataSource(data=dict(xtx_it=xtx_it, ytx_it=ytx_it, hover_it=hover_it, color_it=color_it))

source_sp = ColumnDataSource(data=dict(xtx_sp=xtx_sp, ytx_sp=ytx_sp, hover_sp=hover_sp, color_sp=color_sp))

source_pt = ColumnDataSource(data=dict(xtx_pt=xtx_pt, ytx_pt=ytx_pt, hover_pt=hover_pt, color_pt=color_pt))

source_ch = ColumnDataSource(data=dict(xtx_ch=xtx_ch, ytx_ch=ytx_ch, hover_ch=hover_ch, color_ch=color_ch))

source_arg = ColumnDataSource(data=dict(xtx_arg=xtx_arg, ytx_arg=ytx_arg, hover_arg=hover_arg, color_arg=color_arg))

source_atr = ColumnDataSource(

    data=dict(xtx_atr=xtx_atr, ytx_atr=ytx_atr, hover_atr=hover_atr, color_atr=color_atr))

source_atl = ColumnDataSource(

    data=dict(xtx_atl=xtx_atl, ytx_atl=ytx_atl, hover_atl=hover_atl, color_atl=color_atl))

source_ge = ColumnDataSource(

    data=dict(xtx_ge=xtx_ge, ytx_ge=ytx_ge, hover_ge=hover_ge, color_ge=color_ge))

source_nz = ColumnDataSource(

    data=dict(xtx_nz=xtx_nz, ytx_nz=ytx_nz, hover_nz=hover_nz, color_nz=color_nz))

source_sa = ColumnDataSource(

    data=dict(xtx_sa=xtx_sa, ytx_sa=ytx_sa, hover_sa=hover_sa, color_sa=color_sa))

source_is = ColumnDataSource(

    data=dict(xtx_is=xtx_is, ytx_is=ytx_is, hover_is=hover_is, color_is=color_is))

source_gr = ColumnDataSource(

    data=dict(xtx_gr=xtx_gr, ytx_gr=ytx_gr, hover_gr=hover_gr, color_gr=color_gr))

source_ca = ColumnDataSource(

    data=dict(xtx_ca=xtx_ca, ytx_ca=ytx_ca, hover_ca=hover_ca, color_ca=color_ca))

source_hu = ColumnDataSource(

    data=dict(xtx_hu=xtx_hu, ytx_hu=ytx_hu, hover_hu=hover_hu, color_hu=color_hu))

source_bu = ColumnDataSource(

    data=dict(xtx_bu=xtx_bu, ytx_bu=ytx_bu, hover_bu=hover_bu, color_bu=color_bu))

source_ro = ColumnDataSource(

    data=dict(xtx_ro=xtx_ro, ytx_ro=ytx_ro, hover_ro=hover_ro, color_ro=color_ro))

source_ur = ColumnDataSource(

    data=dict(xtx_ur=xtx_ur, ytx_ur=ytx_ur, hover_ur=hover_ur, color_ur=color_ur))

source_tu = ColumnDataSource(

    data=dict(xtx_tu=xtx_tu, ytx_tu=ytx_tu, hover_tu=hover_tu, color_tu=color_tu))

source_sl = ColumnDataSource(

    data=dict(xtx_sl=xtx_sl, ytx_sl=ytx_sl, hover_sl=hover_sl, color_sl=color_sl))

source_geo = ColumnDataSource(

    data=dict(xtx_geo=xtx_geo, ytx_geo=ytx_geo, hover_geo=hover_geo, color_geo=color_geo))

source_en = ColumnDataSource(

    data=dict(xtx_en=xtx_en, ytx_en=ytx_en, hover_en=hover_en, color_en=color_en))

source_cr = ColumnDataSource(

    data=dict(xtx_cr=xtx_cr, ytx_cr=ytx_cr, hover_cr=hover_cr, color_cr=color_cr))

source_me = ColumnDataSource(

    data=dict(xtx_me=xtx_me, ytx_me=ytx_me, hover_me=hover_me, color_me=color_me))

source_mol = ColumnDataSource(

    data=dict(xtx_mol=xtx_mol, ytx_mol=ytx_mol, hover_mol=hover_mol, color_mol=color_mol))

source_br = ColumnDataSource(

    data=dict(xtx_br=xtx_br, ytx_br=ytx_br, hover_br=hover_br, color_br=color_br))

source_le = ColumnDataSource(

    data=dict(xtx_le=xtx_le, ytx_le=ytx_le, hover_le=hover_le, color_le=color_le))

source_mor = ColumnDataSource(

    data=dict(xtx_mor=xtx_mor, ytx_mor=ytx_mor, hover_mor=hover_mor, color_mor=color_mor))

source_pe = ColumnDataSource(

    data=dict(xtx_pe=xtx_pe, ytx_pe=ytx_pe, hover_pe=hover_pe, color_pe=color_pe))

source_uk = ColumnDataSource(

    data=dict(xtx_uk=xtx_uk, ytx_uk=ytx_uk, hover_uk=hover_uk, color_uk=color_uk))

source_se = ColumnDataSource(

    data=dict(xtx_se=xtx_se, ytx_se=ytx_se, hover_se=hover_se, color_se=color_se))

source_cz = ColumnDataSource(

    data=dict(xtx_cz=xtx_cz, ytx_cz=ytx_cz, hover_cz=hover_cz, color_cz=color_cz))

source_ma = ColumnDataSource(

    data=dict(xtx_ma=xtx_ma, ytx_ma=ytx_ma, hover_ma=hover_ma, color_ma=color_ma))



source_other = ColumnDataSource(data=dict(xtx_n=xtx_n, ytx_n=ytx_n, hover_noise=hover_noise))  # , color_ns=color_ns))

ptx = figure(plot_width=1200, plot_height=650,

                 # title="UMAP, Dbscan and wine",

                 tools="pan,wheel_zoom,,box_zoom,reset",

                 active_scroll="wheel_zoom",

                 toolbar_location="above"

                 )



ptx.xaxis.visible = False

ptx.yaxis.visible = False

ptx.xgrid.grid_line_color = None

ptx.ygrid.grid_line_color = None

sz_dt = 5



wine_us = ptx.scatter('xtx_us', 'ytx_us', size=sz_dt, alpha=0.7, line_dash='solid', color='color_us',

                     source=source_us)

wine_fr = ptx.scatter('xtx_fr', 'ytx_fr', size=sz_dt, alpha=0.7, line_dash='solid', color='color_fr',

                     line_color='#ffffff', source=source_fr)

wine_it = ptx.scatter('xtx_it', 'ytx_it', size=sz_dt, alpha=0.7, line_dash='solid', color='color_it',

                      source=source_it)

wine_sp = ptx.scatter('xtx_sp', 'ytx_sp', size=sz_dt, alpha=0.7, line_dash='solid', color='color_sp',

                      source=source_sp)

wine_pt = ptx.scatter('xtx_pt', 'ytx_pt', size=sz_dt, alpha=0.7, line_dash='solid', color='color_pt',

                      source=source_pt)

wine_ch = ptx.scatter('xtx_ch', 'ytx_ch', size=sz_dt, alpha=0.7, line_dash='solid', color='color_ch',

                      source=source_ch)

wine_arg = ptx.scatter('xtx_arg', 'ytx_arg', size=sz_dt, alpha=0.7, line_dash='solid', color='color_arg',

                      source=source_arg)

wine_atr = ptx.scatter('xtx_atr', 'ytx_atr', size=sz_dt, alpha=0.7, line_dash='solid', color='color_atr',

                       source=source_atr)

wine_atl = ptx.scatter('xtx_atl', 'ytx_atl', size=sz_dt, alpha=0.7, line_dash='solid', color='color_atl',

                       source=source_atl)

wine_ge = ptx.scatter('xtx_ge', 'ytx_ge', size=sz_dt, alpha=0.7, line_dash='solid', color='color_ge',

                       source=source_ge)

wine_nz = ptx.scatter('xtx_nz', 'ytx_nz', size=sz_dt, alpha=0.7, line_dash='solid', color='color_nz',

                      source=source_nz)

wine_sa = ptx.scatter('xtx_sa', 'ytx_sa', size=sz_dt, alpha=0.7, line_dash='solid', color='color_sa',

                      source=source_sa)

wine_is = ptx.scatter('xtx_is', 'ytx_is', size=sz_dt, alpha=0.7, line_dash='solid', color='white',

                      line_color='color_is', source=source_is)

wine_gr = ptx.scatter('xtx_gr', 'ytx_gr', size=sz_dt, alpha=0.7, line_dash='dotted', color='white',

                      line_color='color_gr', source=source_gr)

wine_ca = ptx.scatter('xtx_ca', 'ytx_ca', size=sz_dt, alpha=0.7, line_dash='solid', color='white',

                      line_color='color_ca', source=source_ca)

wine_hu = ptx.scatter('xtx_hu', 'ytx_hu', size=sz_dt, alpha=0.7, line_dash='solid', color='color_hu',

                       source=source_hu)

wine_bu = ptx.scatter('xtx_bu', 'ytx_bu', size=sz_dt, alpha=0.7, line_dash='solid', color='color_bu',

                      source=source_bu)

wine_ro = ptx.scatter('xtx_ro', 'ytx_ro', size=sz_dt, alpha=0.7, line_dash='solid', color='color_ro',

                      source=source_ro)

wine_ur = ptx.scatter('xtx_ur', 'ytx_ur', size=sz_dt, alpha=0.7, line_dash='solid', color='color_ur',

                      source=source_ur)

wine_tu = ptx.scatter('xtx_tu', 'ytx_tu', size=sz_dt, alpha=0.7, line_dash='solid', color='color_tu',

                      source=source_tu)

wine_sl = ptx.scatter('xtx_sl', 'ytx_sl', size=sz_dt, alpha=0.7, line_dash='solid', color='color_sl',

                      source=source_sl)

wine_geo = ptx.scatter('xtx_geo', 'ytx_geo', size=sz_dt, alpha=0.7, line_dash='dotted', color='white',

                      line_color='color_geo', source=source_geo)

wine_en = ptx.scatter('xtx_en', 'ytx_en', size=sz_dt, alpha=0.7, line_dash='dashed', color='color_en',

                       line_color='color_en', source=source_en)

wine_cr = ptx.scatter('xtx_cr', 'ytx_cr', size=sz_dt, alpha=0.7, line_dash='dotted', color='color_cr',

                      line_color='color_cr', source=source_cr)

wine_me = ptx.scatter('xtx_me', 'ytx_me', size=sz_dt, alpha=0.7, line_dash='dotted', color='color_me',

                      line_color='color_me', source=source_me)

wine_mol = ptx.scatter('xtx_mol', 'ytx_mol', size=sz_dt, alpha=0.7, line_dash='dotted', color='color_mol',

                      line_color='color_mol', source=source_mol)

wine_br = ptx.scatter('xtx_br', 'ytx_br', size=sz_dt, alpha=0.7, line_dash='dotted', color='color_br',

                      line_color='#009B3A', source=source_br)





ptx.add_tools(HoverTool(renderers=[wine_us], tooltips=[("Variety", "@hover_us")]))

ptx.add_tools(HoverTool(renderers=[wine_fr], tooltips=[("Variety", "@hover_fr")]))

ptx.add_tools(HoverTool(renderers=[wine_it], tooltips=[("Variety", "@hover_it")]))

ptx.add_tools(HoverTool(renderers=[wine_sp], tooltips=[("Variety", "@hover_sp")]))

ptx.add_tools(HoverTool(renderers=[wine_pt], tooltips=[("Variety", "@hover_pt")]))

ptx.add_tools(HoverTool(renderers=[wine_ch], tooltips=[("Variety", "@hover_ch")]))

ptx.add_tools(HoverTool(renderers=[wine_arg], tooltips=[("Variety", "@hover_arg")]))

ptx.add_tools(HoverTool(renderers=[wine_atr], tooltips=[("Variety", "@hover_atr")]))

ptx.add_tools(HoverTool(renderers=[wine_atl], tooltips=[("Variety", "@hover_atl")]))

ptx.add_tools(HoverTool(renderers=[wine_ge], tooltips=[("Variety", "@hover_ge")]))

ptx.add_tools(HoverTool(renderers=[wine_nz], tooltips=[("Variety", "@hover_nz")]))

ptx.add_tools(HoverTool(renderers=[wine_sa], tooltips=[("Variety", "@hover_sa")]))

ptx.add_tools(HoverTool(renderers=[wine_is], tooltips=[("Variety", "@hover_is")]))

ptx.add_tools(HoverTool(renderers=[wine_gr], tooltips=[("Variety", "@hover_gr")]))

ptx.add_tools(HoverTool(renderers=[wine_ca], tooltips=[("Variety", "@hover_ca")]))

ptx.add_tools(HoverTool(renderers=[wine_hu], tooltips=[("Variety", "@hover_hu")]))

ptx.add_tools(HoverTool(renderers=[wine_bu], tooltips=[("Variety", "@hover_bu")]))

ptx.add_tools(HoverTool(renderers=[wine_ro], tooltips=[("Variety", "@hover_ro")]))

ptx.add_tools(HoverTool(renderers=[wine_ur], tooltips=[("Variety", "@hover_ur")]))

ptx.add_tools(HoverTool(renderers=[wine_tu], tooltips=[("Variety", "@hover_tu")]))

ptx.add_tools(HoverTool(renderers=[wine_sl], tooltips=[("Variety", "@hover_sl")]))

ptx.add_tools(HoverTool(renderers=[wine_geo], tooltips=[("Variety", "@hover_geo")]))

ptx.add_tools(HoverTool(renderers=[wine_en], tooltips=[("Variety", "@hover_en")]))

ptx.add_tools(HoverTool(renderers=[wine_cr], tooltips=[("Variety", "@hover_cr")]))

ptx.add_tools(HoverTool(renderers=[wine_me], tooltips=[("Variety", "@hover_me")]))

ptx.add_tools(HoverTool(renderers=[wine_mol], tooltips=[("Variety", "@hover_mol")]))

ptx.add_tools(HoverTool(renderers=[wine_br], tooltips=[("Variety", "@hover_br")]))
wine_br = ptx.scatter('xtx_br', 'ytx_br', size=sz_dt, alpha=0.7, line_dash='dotted', color='color_br',

                              line_color='#009B3A', source=source_br, legend='Brazil')

wine_le = ptx.scatter('xtx_le', 'ytx_le', size=sz_dt, alpha=0.7, line_dash='dashed', color='color_le',

                      line_color='color_le', source=source_le, legend='lebanon')

wine_mor = ptx.scatter('xtx_mor', 'ytx_mor', size=sz_dt, alpha=0.7, line_dash='solid', color='color_mor',

                      line_color='#006233', source=source_mor, legend='Morocco')

wine_pe = ptx.scatter('xtx_pe', 'ytx_pe', size=sz_dt, alpha=0.7, line_dash='solid', color='color_pe',

                      line_color='color_pe', source=source_pe, legend='Peru')

wine_uk = ptx.scatter('xtx_uk', 'ytx_uk', size=sz_dt, alpha=0.7, line_dash='solid', color='color_uk',

                      line_color='#005BBB', source=source_uk, legend='Ukraine')

wine_se = ptx.scatter('xtx_se', 'ytx_se', size=sz_dt, alpha=0.7, line_dash='solid', color='color_se',

                      line_color='#C6363C', source=source_se, legend='Serbia')

wine_cz = ptx.scatter('xtx_cz', 'ytx_cz', size=sz_dt, alpha=0.7, line_dash='solid', color='color_cz',

                      line_color='#11457E', source=source_cz, legend='Czech Republic')

wine_ma = ptx.scatter('xtx_ma', 'ytx_ma', size=sz_dt, alpha=0.7, line_dash='solid', color='color_ma',

                      line_color='#005BBB', source=source_ma, legend='Macedonia')



data_noise = ptx.scatter('xtx_n', 'ytx_n', size=5, alpha=0.8, line_dash='solid', color='grey',

                         source=source_other, legend='9 countries')  # 'color_ns'



ptx.add_tools(HoverTool(renderers=[wine_le], tooltips=[("Variety", "@hover_le")]))

ptx.add_tools(HoverTool(renderers=[wine_mor], tooltips=[("Variety", "@hover_mor")]))

ptx.add_tools(HoverTool(renderers=[wine_pe], tooltips=[("Variety", "@hover_pe")]))

ptx.add_tools(HoverTool(renderers=[wine_uk], tooltips=[("Variety", "@hover_uk")]))

ptx.add_tools(HoverTool(renderers=[wine_se], tooltips=[("Variety", "@hover_se")]))

ptx.add_tools(HoverTool(renderers=[wine_cz], tooltips=[("Variety", "@hover_cz")]))

ptx.add_tools(HoverTool(renderers=[wine_ma], tooltips=[("Variety", "@hover_ma")]))



ptx.add_tools(HoverTool(renderers=[data_noise], tooltips=[("Variety", "@hover_noise")]))



ptx.circle(x=0, y=0, radius=5, fill_color='white', line_color='black', alpha=0.0, line_width=1, line_alpha=0.2)

ptx.circle(x=0, y=0, radius=4, fill_color='white', line_color='black', alpha=0.0, line_width=1, line_alpha=0.2)

ptx.circle(x=0, y=0, radius=3, fill_color='white', line_color='black', alpha=0.0, line_width=1, line_alpha=0.2)



ptx.legend.click_policy = "hide"

ptx.legend.background_fill_alpha = 0.4



legend_r = Legend(items=[

    LegendItem(label=frst_ctr[0], renderers=[wine_us], index=0),

    LegendItem(label=frst_ctr[1], renderers=[wine_fr], index=1),

    LegendItem(label=frst_ctr[2], renderers=[wine_it], index=2),

    LegendItem(label=frst_ctr[3], renderers=[wine_sp], index=3),

    LegendItem(label=frst_ctr[4], renderers=[wine_pt], index=4),

    LegendItem(label=frst_ctr[5], renderers=[wine_ch], index=5),

    LegendItem(label=frst_ctr[6], renderers=[wine_arg], index=6),

    LegendItem(label=frst_ctr[7], renderers=[wine_atr], index=7),

    LegendItem(label=frst_ctr[8], renderers=[wine_atl], index=8),

    LegendItem(label=frst_ctr[9], renderers=[wine_ge], index=9),

    LegendItem(label=frst_ctr[10], renderers=[wine_nz], index=10),

    LegendItem(label=frst_ctr[11], renderers=[wine_sa], index=11),

    LegendItem(label=frst_ctr[12], renderers=[wine_is], index=12),

    LegendItem(label=frst_ctr[13], renderers=[wine_gr], index=13),

    LegendItem(label=frst_ctr[14], renderers=[wine_ca], index=14),

    LegendItem(label=frst_ctr[15], renderers=[wine_hu], index=15),

    LegendItem(label=frst_ctr[16], renderers=[wine_bu], index=16),

    LegendItem(label=frst_ctr[17], renderers=[wine_ro], index=17),

    LegendItem(label=frst_ctr[18], renderers=[wine_ur], index=18),

    LegendItem(label=frst_ctr[19], renderers=[wine_tu], index=19),

    LegendItem(label=frst_ctr[20], renderers=[wine_sl], index=20),

    LegendItem(label=frst_ctr[21], renderers=[wine_geo], index=21),

    LegendItem(label=frst_ctr[22], renderers=[wine_en], index=22),

    LegendItem(label=frst_ctr[23], renderers=[wine_cr], index=23),

    LegendItem(label=frst_ctr[24], renderers=[wine_me], index=24),

    LegendItem(label=frst_ctr[26], renderers=[wine_mol], index=26),

    LegendItem(label=frst_ctr[27], renderers=[wine_br], index=27),

], click_policy='hide')



ptx.add_layout(legend_r, 'right')

show(ptx)