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
pos_words = pd.read_excel('/kaggle/input/sentiment-lexicons/pos-words.xlsx')

pos_words.head()
neg_words = pd.read_excel('/kaggle/input/sentiment-lexicons/neg-words.xlsx')

neg_words.head()
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

test_df.head()
submission_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

submission_df.head()
selected_text = []
for i in range(len(test_df)):

    if test_df['sentiment'][i] == 'neutral':

        selected_text.append(test_df['text'][i])

    elif test_df['sentiment'][i] == 'positive':

        emo_words = ''

        words = test_df['text'][i].split()

        for word in words:

            if word in list(pos_words.words):

                emo_words = emo_words + word + ' '

        if emo_words:

            selected_text.append(emo_words.strip().replace(' ', ', '))

        else:

            selected_text.append(test_df['text'][i])

    elif test_df['sentiment'][i] == 'negative':

        emo_words = ''

        words = test_df['text'][i].split()

        for word in words:

            if word in list(neg_words.words):

                emo_words = emo_words + word + ' '

        if emo_words:

            selected_text.append(emo_words.strip().replace(' ', ', '))

        else:

            selected_text.append(test_df['text'][i])
len(selected_text)
len(test_df)
len(submission_df)
submission_df['selected_text'] = selected_text
submission_df.head()
submission_df.to_csv('submission.csv', index=False)