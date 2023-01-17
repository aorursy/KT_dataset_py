# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests, json
from matplotlib import pyplot as plt
from wordcloud import WordCloud

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
records_response = requests.get("https://mirror-mirror.vercel.app/api/records")
records_list = json.loads(records_response.text)
records_list[:5]
records_list = [{'topic': r['topic'], 'sentiment_score': r['sentiment']['score'], 'sentiment': r['sentiment']['label'], 'date': r['date']} for r in records_list]
records = pd.DataFrame(records_list).sort_values(by=['date'], ascending=False)
records['date'] = records['date'].str[:10]
records.head()
records['sentiment_score'] = pd.to_numeric(records['sentiment_score'])
topics_str = ' '.join(records['topic'])
wordcloud = WordCloud(max_font_size=80, max_words=100, background_color='white').generate(topics_str)

plt.figure(figsize=(10,8))
plt.imshow(wordcloud) 
plt.axis('off')
plt.show()
negative = records[records['sentiment_score'].between(0, 0.33)]
negative.head()
negative.describe()
negative.describe(include='object')
neutral = records[records['sentiment_score'].between(0.34, 0.66)]
neutral.head()
neutral.describe()
neutral.describe(include='object')
positive = records[records['sentiment_score'].between(0.67, 1)]
positive.head()
positive.describe()
positive.describe(include='object')
records['date'].value_counts().sort_index().plot(xlabel='Date', ylabel='Number of searches', title='Mirror Mirror Usage Chart')
records.groupby('date')['topic'].value_counts()