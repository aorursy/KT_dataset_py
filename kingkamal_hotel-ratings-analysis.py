# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Load data
reviews = pd.read_csv('../input/hotel-reviews-dataset-enriched/hotel_reviews_enriched.csv', usecols=[0,8,9,10,17,36, 37])
#reviews = pd.read_csv('../input/hotel-reviews-dataset-enriched/hotel_reviews_enriched.csv')
reviews.info()
reviews.isnull().sum()
reviews.dropna()
f, ax = plt.subplots(figsize=(10, 8))
corr = reviews.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, ax=ax)
f= plt.figure(figsize=(12,5))

ax=f.add_subplot(221)
sns.lineplot(x=reviews["Businesses_100m"], y=reviews["Reviewer_Score"], ax=ax)
ax.set_title('>100m buisnesses on reviewer score')

ax=f.add_subplot(222)
sns.lineplot(x=reviews["Businesses_1km"], y=reviews["Reviewer_Score"], ax=ax)
ax.set_title('>1km buisnesses on reviewer score')

ax=f.add_subplot(223)
sns.lineplot(x=reviews["Businesses_5km"], y=reviews["Reviewer_Score"], ax=ax)
ax.set_title('>5km buisnesses on reviewer score')
