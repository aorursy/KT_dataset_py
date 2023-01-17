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
df = pd.read_csv('/kaggle/input/customer-support-on-twitter/twcs/twcs.csv')
df.info()
Spectrum_df = df.query('author_id=="Ask_Spectrum"').reset_index(drop=True)
Spectrum_df.head()
def conversation_creator(row):
    text = ''
    in_response_to_tweet_id = row['in_response_to_tweet_id']
    response_tweet_id = row['response_tweet_id']
    print(in_response_to_tweet_id, response_tweet_id)
    print(Spectrum_df.query('tweet_id==@in_response_to_tweet_id'))
    print(int(in_response_to_tweet_id))
    response = df[df['tweet_id']==int(in_response_to_tweet_id)]
    print(response.values)
#     text += Spectrum_df.query('tweet_id==@response_tweet_id')
    print(text)
    return text
conversation_creator(Spectrum_df.iloc[0])
