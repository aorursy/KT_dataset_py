import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import style
plt.style.use('ggplot')

data = pd.read_csv('../input/Sentiment.csv', index_col = 'id', parse_dates = ['tweet_created'])
dt = data[data['candidate'] == 'Donald Trump']
positive_dt = dt[dt['sentiment'] == 'Positive']
positive_dt = positive_dt.tweet_created.value_counts().sort_index()
positive_dt = positive_dt.cumsum()
positive_dt.plot.area(figsize=(16,8),
                      ylim=[0,1800],
                      color='lightgreen',
                      title='Positive Trump Sentiment over time')
negative_dt = dt[dt['sentiment'] == 'Negative']
negative_dt = negative_dt.tweet_created.value_counts().sort_index()
negative_dt = negative_dt.cumsum()
negative_dt.plot.area(figsize=(16,8),
                      color='lightcoral',
                      title='Negative Trump Sentiment over time')