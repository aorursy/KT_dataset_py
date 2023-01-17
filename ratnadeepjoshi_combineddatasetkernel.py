# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from shutil import copyfile

# Any results you write to the current directory are saved as output.
copyfile(src='../input/myFunction.py', dst='../working/myFunction.py')
import myFunction as mf
combinedDF = mf.getCombinedDF('../input')
combinedDF.shape
combinedDF.osamt.sum()
combinedDF.state.unique()
sns.kdeplot(data=np.array(tuple(np.log10(combinedDF.osamt))))
print(combinedDF.osamt.mean())
print(combinedDF.osamt.median())
print(combinedDF.osamt.mode())
df_sorted = combinedDF.groupby('state').sum().sort_values(by='osamt',ascending=False)
g = sns.barplot(data=df_sorted,x=df_sorted.index,y='osamt',ci=None)
f = plt.gcf()
f.set_size_inches(15,6)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title('Statewise Outstanding amount in Rs. Crores')
for p in g.patches:
    g.annotate('{:5.0f}'.format(p.get_height()/100),xy=(p.get_x(),p.get_height()+1.0))
f.show();
def rotate_xlabels(fig):
    #f = plt.gcf()
    fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
    #f.show();
    return
    
rotate_xlabels(sns.barplot(data=df_sorted.reset_index().sort_values(by='osamt',ascending=False),x='state',y='osamt'))
plt.gcf().set_size_inches(10,6)
plt.title('Statewise outstanding amount in lakh(100000) Rupees')
byStatedf.info()
