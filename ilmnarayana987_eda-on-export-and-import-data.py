# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
export_data = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')

import_data = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')



print(export_data.shape)

print(import_data.shape)
export_data.columns
from matplotlib import pyplot as plt

%matplotlib inline
val_counts = export_data['HSCode'].value_counts()

plt.bar(val_counts.index, val_counts.values)
val_counts[:10]
# Code took from GeeksForGeeks

# ref: https://www.geeksforgeeks.org/generating-word-cloud-python/



from wordcloud import WordCloud, STOPWORDS



stopwords = set(STOPWORDS)



wordcloud_gen = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10)
commodity_85 = export_data[export_data['HSCode']==85]['Commodity']

print(commodity_85.shape)
commodity_85_words = ' '.join(commodity_85)



comm_85_wordcloud = wordcloud_gen.generate(commodity_85_words)
# Code took from https://www.geeksforgeeks.org/generating-word-cloud-python/

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(comm_85_wordcloud) 

plt.axis("off") 

plt.title("HS Code - 85 Commodity word cloud", fontdict={'fontsize': 20})

plt.tight_layout(pad = 0) 

  

plt.show()
def plot_wordcloud(HSCode):

    commodity_col = export_data[export_data['HSCode']==HSCode]['Commodity']

    commodity_words = ' '.join(commodity_col)



    comm_wordcloud = wordcloud_gen.generate(commodity_words)



    plt.figure(figsize = (8, 8), facecolor = None) 

    plt.imshow(comm_wordcloud) 

    plt.title(f"HS Code - {HSCode} Commodity word cloud", fontdict={'fontsize': 15})

    plt.axis("off") 

    plt.tight_layout(pad = 0) 



    plt.show()
plot_wordcloud(84)
plot_wordcloud(90)
hscode_value = export_data.groupby('HSCode')['value'].sum()
hscode_value.sort_values()[:-10:-1]
plot_wordcloud(27)
plot_wordcloud(71)
plot_wordcloud(87)