import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/complete-tweet-sentiment-extraction-data/tweet_dataset.csv')

data = data[['textID', 'text', 'selected_text', 'sentiment', 'new_sentiment']]

data.head()
data_plot1 = data.copy()

data_plot1['new_sentiment'] = data_plot1['new_sentiment'].fillna('unknown')

sns.countplot(x='new_sentiment', data=data_plot1, palette=['gray', 'red', 'green', 'blue'])

plt.title('New Sentiments In The Dataset')

plt.show()
data = data.dropna().reset_index(drop=True)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

sns.countplot(x='sentiment', hue='new_sentiment', ax=ax, data=data, palette=['red', 'green', 'blue'])

plt.title('Original Sentiments and Their Splits On New Sentiment')

plt.show()
for sentiment in data['sentiment'].unique():

    print('>>>', sentiment)

    # Get sentiment data

    sentidata = data[data['sentiment'] == sentiment]

    # Print proportions

    for new_sentiment in ['negative', 'positive', 'neutral']:

        print(new_sentiment, str(round(100 * sentidata[sentidata['new_sentiment'] == new_sentiment].shape[0] / sentidata.shape[0], 5)) + '%')

    print()