import bq_helper
import pandas as pd
nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                               dataset_name="new_york")
complaints = nyc.query_to_pandas("SELECT complaint_type FROM `bigquery-public-data.new_york.311_service_requests`")
from wordcloud import WordCloud
import matplotlib.pyplot as plt
%matplotlib inline
complaints_freq = complaints.groupby('complaint_type').size().sort_values(ascending=False)
wordcloud = WordCloud(width=1200,
                      height=600,
                      max_words=200,
                      background_color='white').generate_from_frequencies(complaints_freq)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
