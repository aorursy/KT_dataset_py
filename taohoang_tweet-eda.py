# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

train_df
train_df['sentiment'].value_counts()
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

test_df
import textblob
from textblob import TextBlob, Word, Blobber
text = TextBlob("I`d have responded, if I were going")

print(text.sentiment)
dir(text.sentiment)
text.sentiment.count
train_df[train_df['sentiment'] == 'neutral'].sample(10)[['text','selected_text']]
train_df[train_df['sentiment'] == 'neutral']
train_df[(train_df['sentiment'] == 'neutral') & (train_df['text'].str.strip() != train_df['selected_text'].str.strip())]
print(train_df.iloc[27315]['text'].strip())

print(train_df.iloc[27315]['selected_text'].strip())
train_df['selected_text_percentage'] = train_df['selected_text'].str.strip().str.len() / train_df['text'].str.strip().str.len()

train_df['selected_text_percentage'].describe()
train_df.groupby(['sentiment'])['selected_text_percentage'].mean()
train_df[train_df['sentiment'] == 'positive']['selected_text_percentage'].describe()
train_df[train_df['sentiment'] == 'positive']['selected_text_percentage'].mode()
train_df[train_df['sentiment'] == 'positive'].sample(10)
print(TextBlob('X-Men:Wolverine was hot! I say go watch it').sentiment)

print(TextBlob('was hot!').sentiment)

print(TextBlob('I say go watch it').sentiment)

print(TextBlob('X-Men:Wolverine').sentiment)
'testing'.index('ing')
train_df['selected_text_index'] = train_df.apply(lambda row: str(row['text']).strip().index(str(row['selected_text']).strip()) / len(str(row['text']).strip()), axis=1)

train_df['selected_text_index']
train_df['selected_text_index'].describe()
train_df['text_len'] = train_df['text'].str.len()

train_df['text_len'].describe()