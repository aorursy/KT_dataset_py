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
FILEPATH = '/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv'
!pip install emoji
df = pd.read_csv(FILEPATH, delimiter = '\t')
df.sample(2)
import emoji



unique_emojis = set()



def get_emojis(string):

    

    emojis = []



    my_str = str(string)

    for each in my_str:

        if each in emoji.UNICODE_EMOJI:

            emojis.append(each)

            unique_emojis.add(each)

            

    return emojis



def get_emojis_count(string):

    

    emojis = get_emojis(string)

    

    return len(emojis)
# Show first 5 items of verified_reviews

for v in df['verified_reviews'].head(10):

    print(v)
df['emojis'] = df['verified_reviews'].apply(extract_emojis)

df['emojis_count'] = df['verified_reviews'].apply(get_emojis_count)
df.sample(3)
df_emojis = df[df['emojis_count'] > 0]
df_emojis.sample(5)
len(unique_emojis)
for item in unique_emojis:

    print(item, emoji.demojize(item))
!pip install emojis
import emojis



df['emojis'] = df['verified_reviews'].apply(lambda x: emojis.get(x))

df['emojis_count'] = df['verified_reviews'].apply(lambda x: emojis.count(x))
df.sample(2)
df['emojis'][0]
df_emojis = df[df['emojis_count'] > 0]
df_emojis.sample(2)
emojis.db.get_emoji_by_alias('taco')