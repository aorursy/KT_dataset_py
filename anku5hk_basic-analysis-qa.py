# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/twitter_kabir_singh_bollywood_movie.csv")

data.head()
data.corr() 
data.nunique()
del data['re_tweeter']

del data['time']

del data['type']

del data['lang']

del data['id']
temp_data = data

p = temp_data.text_raw.apply(lambda temp_data: temp_data.find('@shahidkapoor'))

print(p.nunique())

p = temp_data.text_raw.apply(lambda temp_data: temp_data.find('@Advani_Kiara'))

print(p.nunique())





# data['hashtags'].isnull() this doesnt gives you the answer, because empty ones have atleast '[]' in them

temp_data = data

temp_data.hashtags = temp_data.hashtags.apply(lambda temp_data: temp_data.find('[]'))

print(temp_data.hashtags.value_counts())
hashtag_plt = temp_data.hashtags.astype(str)

hashtag_plt = hashtag_plt.replace('-1', 'Yes')

hashtag_plt = hashtag_plt.replace('0', 'No')

hashtag_plt.value_counts()
values = ["Yes", "No"]

index = [0,1]

plt.rcParams['figure.figsize'] = (3, 4)

plt.style.use('_classic_test')

hashtag_plt.value_counts().plot.bar(color = 'red')

plt.xlabel('Hashtag Features', fontsize = 15)

plt.ylabel('Hastag_users', fontsize = 15)

plt.xticks(index, values, rotation=0)

plt.show()


