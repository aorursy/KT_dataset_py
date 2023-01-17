# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns
r_code=pd.read_csv(r"/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv")

r_code.head(10)
r_code.removed_by.unique()
r_code.removed_by.value_counts()
r_code.full_link.value_counts()
# Let us remove unnecessary columns 



r_code.drop(["total_awards_received","awarders"],axis=1)
r_code.info()
r_code.describe()
# Let's take individual columns and clean them

# First column is id 



r_code.id.unique()
r_code.id.value_counts()
r_code['id'].isna().sum()
# here we see that the first two in the value_counts i.e (Data_irl) & (data_irl) are the same but just have different orientations,so let us change that



pattern=r"[Dd]ata_irl"



# Method to find rows that contain a certain character and isolate them 



bool=r_code.title.str.contains(pattern)

#print(r_code[bool])

print(r_code.title.str.contains(pattern).sum())



r_code.title=r_code.title.str.replace(pattern,"DATA_irl")
import re

pattern1=r"\[OC\]"

#r_code[r_code.title.str.contains(pattern1)]

bool1=r_code[r_code.title.str.contains(pattern1,flags=re.I,na=False)]

bool1
# Let's take the second column which is the 'title'column



r_code.title.value_counts()
r_code.title.unique()
# Here we can see that there is a part '[OC]' which is common a lot,it may be related to the author/publisher(I think!)

#Let's check the null values in this column



r_code.title.isna().sum()
# So there is one null value present,now to check its position



r_code[r_code.title.isna()]
# It is present at the 80324th position,looking at it ,it seems that there is not a lot to be achieved through it so it might be better if we removed it



r_code=r_code.drop(r_code.index[80324])

r_code[80321:].head(5)


print("The minimum and the maximum score awarded are :",r_code.score.min(),'-',r_code.score.max())
r_code.score.value_counts()
r_code.score.unique()
# Check if there is any null value present in this column



r_code.score.isna().sum()
import seaborn as sns

sns.kdeplot(r_code.score)

plt.xlabel("Score")

plt.ylabel("Freq")
print(len(r_code[r_code['score'] < 10]), 'Posts with less than 10 votes')

print(len(r_code[r_code['score'] > 10]), 'Posts with more than 10 votes')
print(len(r_code[r_code['num_comments'] < 10]), 'Posts with less than 10 comments')

print(len(r_code[r_code['num_comments'] > 10]), 'Posts with more than 10 comments')
r_code[r_code['score'] == r_code['score'].max()]['title'].iloc[0]
r_code.author.value_counts()

r_code.author_flair_text.value_counts()

r_code.author_flair_text.unique()
r_code.score.plot()
most_pop=r_code.sort_values('score',ascending=False)[['score','title']].head(10)



most_pop['score1'] = most_pop['score']/1000



most_pop.score1
import matplotlib.style as style
style.available
style.use('fivethirtyeight')
plt.figure(figsize = (17,15))



sns.barplot(data=most_pop, y = 'title', x = 'score1', color = 'c')

plt.xticks(fontsize=20, rotation=0)

plt.yticks(fontsize=21, rotation=0)

plt.xlabel('Votes in Thousands', fontsize = 21)

plt.ylabel('')

plt.title('Most popular posts', fontsize = 30)
most_com=r_code.sort_values('num_comments',ascending=False)[['author','title','num_comments']].head(12)

most_com['num_comments1'] = most_com['num_comments']/1000
most_com.head()
plt.figure(figsize = (17,15))



sns.barplot(data = most_com, y = 'title', x = 'num_comments1', color = 'y')

plt.xticks(fontsize=20, rotation=0)

plt.yticks(fontsize=21, rotation=0)

plt.xlabel('Comments in Thousands', fontsize = 21)

plt.ylabel('')

plt.title('Most commented posts', fontsize = 30)