# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
d = pd.read_csv('../input/Data_Cortex_Nuclear.csv')
# take a peek
d.head()
# turn MouseID into a true multi-index
multi = pd.MultiIndex.from_tuples( [ tuple(s.split('_')) for s in d['MouseID'] ], names=('mouse_id','replicate_id') )
d = d.set_index(multi)
# pivot data frame to facilitate facet plotting
e = pd.DataFrame(d.stack())
e.index.names = ['mouse_id','replicate_id','protein']
e.reset_index(inplace=True)
e = e.rename( {0:'value'}, axis='columns' )
e['replicate_id'] = e['replicate_id'].astype(int)
# only keep protein expression columns
e = e[ ['_N' in s for s in e['protein']] ]
# summarize per mouse data (for reference)
d.drop([ c for c in d.columns if '_N' in c ],axis=1).groupby( level=0 ).head(1)
# add log(expression).  This pulls out more information from the low expressors
import math
e['log_ex'] = [ math.log(a+0.1) for a in e['value'] ]
i = e['mouse_id'].isin(('309','311','320','321','322','3415'))
g = sns.FacetGrid(e[i], col="protein", hue='mouse_id', col_wrap=5, size=2.5)
g = g.map(plt.plot, "replicate_id", 'log_ex', marker=".").add_legend()
i = e['protein'].isin( ('ELK_N','ADARB1_N','pP70S6_N','ERK_N','pELK_N'))
g = sns.FacetGrid(e[i], col="mouse_id", hue='protein', col_wrap=5, size=2.5)
g = g.map(plt.plot, "replicate_id", 'value', marker=".").add_legend()
