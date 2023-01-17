# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.pyplot import figure
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_detail = pd.read_csv("../input/Popular-Baby-Names.csv")
data_detail.head(15)
data_detail.info()
data_detail.columns
data_detail.dtypes
data_detail.describe()
table_pvt = pd.DataFrame(data_detail)
table_pvt["Child's First Name"]=table_pvt["Child's First Name"].str.upper()
pd.pivot_table(table_pvt, index=["Child's First Name","Year of Birth"],columns=["Gender"],values=["Count"], aggfunc=[np.sum],fill_value=0)
# Correlation
data_detail.corr()
#Correlation Map
f,ax = plt.subplots(figsize=(15, 5))
sns.heatmap(data_detail.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# WordCloud
namecloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(data_detail["Child's First Name"]))# Generate plot
figure(figsize=(15,50))
plt.imshow(namecloud,interpolation='bilinear')
plt.axis("off")
plt.show()