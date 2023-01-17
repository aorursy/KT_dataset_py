import os, sys, email
import numpy as np 
import pandas as pd
# Plotting
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns; sns.set_style('whitegrid')
import wordcloud
from fbprophet import Prophet
from wordcloud import WordCloud, STOPWORDS
import gc

# Network analysis
import networkx as nx

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
EMAIL_NUMBER = 300000  #0...517401
data = pd.read_csv('../input/enron-mails-stock-price/enron_lcfr.csv', index_col=0)
data = data.fillna(0)
sm = 1 
c = EMAIL_NUMBER
twin = 1000

gc.enable()

data_s = data.head(c)
    
t_date = list(data_s.date)
t = t_date[-1]

f, ax = plt.subplots(figsize=(12, 12), dpi= 100) 

plt.subplot(2, 2, 1)
G = nx.from_pandas_edgelist(data_s.head(twin), 'X-From', 'X-To')
nx.draw(G, node_size = 50, node_color = 'blue', edge_color = 'black', with_labels = False) #draw_circular

plt.subplot(2, 2, 2)
wordcloud = WordCloud(background_color="white", scale=1,colormap='Blues', max_font_size=40).generate(str(data_s['content'].head(twin)))
plt.imshow(wordcloud, cmap=plt.cm.gray, interpolation="bilinear")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Email Counter: ' + str(c),fontsize=12)
sns.barplot(data_s['user'].value_counts()[0:10].values, data_s['user'].value_counts()[0:10].index, palette=('Blues_r'))
plt.xlabel('Number of emails')

plt.subplot(2, 2, 4)
plt.title('Timestamp: ' + str(t),fontsize=12)
open_series = data_s['Open']
plt.plot(open_series)
plt.xlabel('Number of emails')
plt.ylabel('Stock price USD')
 
  
plt.tight_layout()
#plt.savefig("../pics/mov_4_kpi/" + str('%010d' % sm) + ".png" , bbox_inches='tight')
gc.collect()

plt.show()
from IPython.display import YouTubeVideo
YouTubeVideo('bjtwb_3rKLE',width=640, height=480)