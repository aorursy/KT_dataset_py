# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pylab import rcParams
rcParams['figure.figsize'] = 10, 15
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import warnings
warnings.filterwarnings('ignore')
%%javascript
IPython.OutputArea.auto_scroll_threshold = 9999;    
mushroom_df = pd.read_csv('../input/mushrooms.csv')
print(mushroom_df.shape)
pd.options.display.max_columns = None
mushroom_df.head()
mrm_dict = { 
    'class':{'e':'edible','p':'poisonous'},
    'cap-shape':{'b':'bell','c':'conical','x':'convex','f':'flat','k':'knobbed','s':'sunken'},
    'cap-surface':{'g':'grooves','y':'scaly','f':'fibrous','s':'smooth'},
    'cap-color':{'n':'brown','b':'buff','c':'cinnamon','g':'gray','r':'green','p':'pink','u':'purple','e':'red','w':'white','y':'yellow'},
    'bruises':{'t':'bruises','f':'no'},
    'odor':{'a':'almond','l':'anise','c':'creosote','y':'fishy','f':'foul','m':'musty','n':'none','p':'pungent','s':'spicy'},
    'gill-attachment':{'a':'attached','d':'descending','f':'free','n':'notched'},
    'gill-spacing':{'c':'close','w':'crowded','d':'distant'},
    'gill-size':{'b':'broad','n':'narrow'},
    'gill-color':{'k':'black','n':'brown','b':'buff','h':'chocolate','g':'gray','r':'green','o':'orange','p':'pink','u':'purple','e':'red','w':'white','y':'yellow'},
    'stalk-shape': {'e':'enlarging','t':'tapering'},
    'stalk-root':{ 'b':'bulbous','c':'club','u':'cup','e':'equal','z':'rhizomorphs','r':'rooted','?':'missing'},
    'stalk-surface-above-ring':{'f':'fibrous','y':'scaly','k':'silky','s':'smooth'},
    'stalk-surface-below-ring': {'f':'fibrous','y':'scaly','k':'silky','s':'smooth'},
    'stalk-color-above-ring': {'n':'brown','b':'buff','c':'cinnamon','g':'gray','o':'orange','p':'pink','e':'red','w':'white','y':'yellow'},
    'stalk-color-below-ring':{'n':'brown','b':'buff','c':'cinnamon','g':'gray','o':'orange','p':'pink','e':'red','w':'white','y':'yellow'},
    'veil-type':{'p':'partial','u':'universal'},
    'veil-color': {'n':'brown','o':'orange','w':'white','y':'yellow'},
    'ring-number': {'n':'none','o':'one','t':'two'},
    'ring-type': {'c':'cobwebby','e':'evanescent','f':'flaring','l':'large','n':'none','p':'pendant','s':'sheathing','z':'zone'},
    'spore-print-color':{'k':'black','n':'brown','b':'buff','h':'chocolate','r':'green','o':'orange','u':'purple','w':'white','y':'yellow'},
    'population': {'a':'abundant','c':'clustered','n':'numerous','s':'scattered','v':'several','y':'solitary'},
    'habitat': {'g':'grasses','l':'leaves','m':'meadows','p':'paths','u':'urban','w':'waste','d':'woods'} 
    }
# Funtion to get the labels for a feature, if a list of values are passed along

def get_labels(feature,abbr_values):
    labels=[]
    for i in abbr_values:
        labels.append(mrm_dict[feature][i])
    return labels
print(tuple(get_labels('habitat',['g','u','d'])))
print(get_labels('population',['a','c','s']))
mushroom_df.describe()
# Creating helper functions

#Helper function to get labels and number of instances for each label in that feature
def get_counts(feature):
    features_series = mushroom_df[feature].value_counts()
    counts = features_series.values.tolist()
    value_lbl = features_series.axes[0].tolist()
    return value_lbl,counts

#get_counts()
for feature in mushroom_df.columns:
    value_lbl,counts = get_counts(feature)
    result = list((i,j) for i,j in zip(value_lbl,counts))
    print('Occurences of  each  value in the feature : {0}\n{1}\n'.format(feature,result))
#helper function that outputs bar plot of feature
def get_plot(feature):
    values,counts = get_counts(feature)
    x_indices = np.arange(len(values))
    labels = get_labels(feature,values)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xlabel(feature,fontsize=20)
    ax.set_ylabel('Quantity',fontsize=20)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(tuple(labels),fontsize=10)
    ax.set_title('{0} Vs Quantity'.format(feature),fontsize=20)
    colors = ['#47ACB1','#FFCD33','#FFE8AF','#F9AA7B','#ADD5D7','#9EF4E6']
    mushroom_bars = ax.bar(x_indices,counts,color=colors)
    for rect in mushroom_bars:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, 1*height,'%d' % int(height),
                ha='center', va='bottom',fontsize=20)
    return mushroom_bars


get_plot('cap-shape')

get_plot('gill-color')
# Generate plots for all features and save to a pdf file

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("result.pdf")
for col in mushroom_df.columns:
    get_plot(col)
    pdf.savefig()
pdf.close()

