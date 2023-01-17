import numpy as np

import pandas as pd



from IPython.display import display

import matplotlib.pyplot as plt



from textblob import TextBlob



%matplotlib inline
datafile = '../input/abcnews-date-text.csv'

raw_data = pd.read_csv(datafile, parse_dates=[0], infer_datetime_format=True)



reindexed_data = raw_data['headline_text']

reindexed_data.index = raw_data['publish_date']



raw_data.head()
positive_sentence = "I love cheese. Truly, I feel very strongly about it: it is the best!"

neutral_sentence = "Cheese is made from milk. Yesterday, cheese was arrested. Here is some news."

negative_sentence = "Cheese is the worst! I hate it! With every fibre of my being!"



positive_blob = TextBlob(positive_sentence)

neutral_blob = TextBlob(neutral_sentence)

negative_blob = TextBlob(negative_sentence)



print("Positive sentence: ", positive_blob.sentiment)

print("Neutral sentence: ", neutral_blob.sentiment)

print("Negative sentence: ", negative_blob.sentiment)
blobs = [TextBlob(reindexed_data[i]) for i in range(reindexed_data.shape[0])]



polarity = [blob.polarity for blob in blobs]

subjectivity = [blob.subjectivity for blob in blobs]



sentiment_analysed = pd.DataFrame({'headline_text':reindexed_data, 

                                   'polarity':polarity, 

                                   'subjectivity':subjectivity},

                                  index=reindexed_data.index)
monthly_averages = sentiment_analysed.resample('M').mean()

yearly_averages = sentiment_analysed.resample('A').mean()



fig, ax = plt.subplots(2, figsize=(18,10))

ax[0].plot(monthly_averages['subjectivity'], label='Monthly mean subjectivity');

ax[0].plot(yearly_averages['subjectivity'], 'r--', label='Yearly mean subjectivity');

ax[0].set_title('Mean subjectivity scores');

ax[0].legend(loc='upper left');

ax[1].plot(monthly_averages['polarity'], label='Monthly mean polarity');

ax[1].plot(yearly_averages['polarity'], 'r--', label='Yearly mean polarity');

ax[1].set_title('Mean polarity scores');

ax[1].legend(loc='upper left');

plt.show()