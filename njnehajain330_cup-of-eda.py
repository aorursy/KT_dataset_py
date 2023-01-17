# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import re
import matplotlib.pyplot as plt
import numpy as np
os.getcwd()
os.chdir('/kaggle/input/chai-time-data-science/')
thumbnail_types = pd.read_csv('Anchor Thumbnail Types.csv')
thumbnail_types.head()
thumbnail_types.shape
description = pd.read_csv('Description.csv')
description.shape
description.head()
# Total Episodes 
description['episode_id'].nunique()
description['description'][1]
! pip install contractions
import contractions
def extracting_links(text):
    text = re.sub(r'\n',' ',text)
    # Getting URLS present in the description of the episode
    l = re.findall(r'http[s]*://[a-zA-Z0-9./-]+',text)
    return l
    
    
description['links']= description['description'].apply(lambda x : extracting_links(x))
def cleaning_description(text):
    # Substituting \n with a space 
    text = re.sub(r'\n',' ',text)
    # Cleaning URLS present in the description of the episode
    text = re.sub(r'http[s]*://[a-zA-Z0-9./-]+','',text)
    # Converting text to lower case 
    text = text.lower()
    # Expanding Contractions eg. i'll to I will 
    text = contractions.fix(text)
    return text
description['description'] = description['description'].apply(lambda x : cleaning_description(x))
description.head()
youtube_thumbnail = pd.read_csv('YouTube Thumbnail Types.csv')
youtube_thumbnail.shape
youtube_thumbnail.head()
episodes = pd.read_csv('Episodes.csv')
episodes.shape
episodes.head()
episodes.columns
episodes.isnull().sum()
episodes['heroes'].unique()
sns.countplot(episodes['heroes_gender'])
plt.figure(figsize=(15,7))
sns.countplot(episodes['heroes_nationality'])
episodes_sorted = episodes.sort_values('youtube_subscribers',ascending=False).reset_index(drop= True)
plt.figure(figsize=(12,19))
sns.barplot(episodes_sorted['youtube_subscribers'],episodes_sorted['heroes'])
episodes_sorted['heroes'][:10]
sns.countplot(episodes['category'])
sns.countplot(episodes['category'],hue=episodes['heroes_gender'])
