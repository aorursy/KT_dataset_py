from IPython.display import HTML

import warnings

warnings.filterwarnings('ignore')

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/9vqLO6McFwQ?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
show_widget = False
from clustergrammer2 import net

if show_widget == False:

    print('\n-----------------------------------------------------')

    print('>>>                                               <<<')    

    print('>>> Please set show_widget to True to see widgets <<<')

    print('>>>                                               <<<')    

    print('-----------------------------------------------------\n')    

    delattr(net, 'widget_class') 
df = pd.read_csv('../input/ccle.txt/CCLE.txt', index_col=0)

from ast import literal_eval as make_tuple

cols = df.columns.tolist()

new_cols = [make_tuple(x) for x in cols]

df.columns = new_cols

df.shape
net.load_df(df.round(2))

net.filter_N_top(inst_rc='row', N_top=1000, rank_type='var')

net.widget()
net.load_df(df)

net.filter_N_top(inst_rc='row', N_top=1000, rank_type='var')

net.normalize(axis='row', norm_type='zscore')

df = net.export_df().round(2)

net.load_df(df)

net.widget()