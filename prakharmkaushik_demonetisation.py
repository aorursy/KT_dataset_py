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


file_name = "../input/demonetization-in-india/Demonetization_data29th.csv"
df = pd.read_csv(file_name, encoding = "ISO-8859-1")

df.head()
df.shape
df.describe()
df.info()
df.isnull().sum()
df.columns
df.drop([ 'possibly_sensitive', 'coordinates', 'retweeted_status','from_user_description', 

         'from_user_location','entities_urls','entities_mentions','in_reply_to_screen_name',

       'in_reply_to_status_id','entities_expanded_urls', 'entities_media_count', 'media_expanded_url',

       'media_url', 'media_type','entities_hashtags'],axis=1,inplace=True)
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")

%matplotlib inline