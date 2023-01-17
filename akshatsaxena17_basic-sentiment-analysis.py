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
import pandas as pd
import numpy as np
import seaborn as sns
from textblob import TextBlob
import matplotlib.pyplot as plt
amz_data = pd.read_csv('/kaggle/input/indian-products-on-amazon/amazon_vfl_reviews.csv')
amz_data.head()
amz_data.name.unique()
def comp(x):
    return x.split('-')[0]
amz_data['company'] = amz_data.name.apply(comp)
amz_data.head()
amz_data.shape
amz_data.isnull().sum()
amz_data=amz_data.dropna(how='any')
amz_data.shape
import matplotlib.pyplot as plt
bloblist_desc = list()

df_amz_review = amz_data['review'].astype(str)
for row in df_amz_review:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    df_amz_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])
 
def f(df_amz_polarity_desc):
    if df_amz_polarity_desc['sentiment'] > 0:
        val = "Positive"
    elif df_amz_polarity_desc['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val

df_amz_polarity_desc['Sentiment_Type'] = df_amz_polarity_desc.apply(f, axis=1)

plt.figure(figsize = (10,10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=df_amz_polarity_desc)
amz_data.company.unique()
amz_data['company'] = amz_data['company'].str.replace('PATANJALI', 'Patanjali')
amz_data['company'] = amz_data['company'].str.replace('MYSORE', 'Mysore')
amz_data.company.unique()
amz_mamaearth = amz_data[amz_data.company == 'Mamaearth']
amz_godrej = amz_data[amz_data.company == 'Godrej']
amz_titan = amz_data[amz_data.company == 'Titan']
amz_maaza = amz_data[amz_data.company == 'Maaza']
amz_paper_boat = amz_data[amz_data.company == 'Paper']
amz_indiana = amz_data[amz_data.company == 'Indiana']
amz_coca_cola = amz_data[amz_data.company == 'Coca']
amz_natural = amz_data[amz_data.company == 'Natural']
amz_maggi = amz_data[amz_data.company == 'Maggi']
amz_glucon_d = amz_data[amz_data.company == 'Glucon']
amz_amul = amz_data[amz_data.company == 'Amul']
amz_patanjali = amz_data[amz_data.company == 'Patanjali']
amz_dettol = amz_data[amz_data.company == 'Dettol']
amz_savlon = amz_data[amz_data.company == 'Savlon']
amz_cinthol = amz_data[amz_data.company == 'Cinthol']
amz_britannia = amz_data[amz_data.company == 'Britannia']
amz_nutrichoice = amz_data[amz_data.company == 'NutriChoice']
amz_streax = amz_data[amz_data.company == 'Streax']
amz_himalaya = amz_data[amz_data.company == 'Himalaya']
amz_society = amz_data[amz_data.company == 'Society']
amz_tata = amz_data[amz_data.company == 'Tata']
amz_fastrack = amz_data[amz_data.company == 'Fastrack']
amz_reflex = amz_data[amz_data.company == 'Reflex']
amz_mysore = amz_data[amz_data.company == 'Mysore']
