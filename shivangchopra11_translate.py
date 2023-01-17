# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/heot-original"))



# Any results you write to the current directory are saved as output.
!pip install googletrans

from googletrans import Translator

import pandas as pd

import time
!pip install googletrans
data = pd.read_csv('../input/heot-original/preprocessed_data_HEOT.csv')
data.head()
tweets = data['lemmas'].values
translator = Translator()
cur = '<user> i hate palak paneer paalak dekh ke mera dimag kharab ho jaata hai lol'

print(cur)

translator = Translator()

translator.translate(cur).text
trans_list = []

for tweet in data['lemmas']:

    translator = Translator()

    print(tweet)

    try:

#         print('pass')

        cur = translator.translate(str(tweet)).text

        print(cur)

        trans_list.append(cur)

    except:

        print('fail')

#         time.sleep(20)

        trans_list.append('')

#         print(tweet)
len(trans_list)
data['translated'] = trans_list
data.head()
data.to_csv('trans_tweet_data1.csv')