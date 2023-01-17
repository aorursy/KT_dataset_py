# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/good_reads_final.csv')
df.info()
df.head()
df_count_author = df.groupby(df['author_id']).agg({'author_id':'count'})
df_unique_autor = df['author_id'].unique()
df_unique_autor
new_df = pd.DataFrame(df_unique_autor)
len_group_author = len(new_df.index)
len_non_group_author = len(df.index)
objects = ('Non unique', 'Unique')

y_pos = np.arange(len(objects))

performance = [len_non_group_author,len_group_author]

 

plt.bar(y_pos, performance, align='center', alpha=0.5, color='red')

plt.xticks(y_pos, objects)

plt.ylabel('Quantity')

plt.title('Author Unique By Id')

 

plt.show()
group_by_gender_non_unique = df.groupby('author_gender').agg({'author_gender':'count'})
objects = ('Female', 'Male')

y_pos = np.arange(len(objects))

performance = [group_by_gender_non_unique['author_gender']['female'],group_by_gender_non_unique['author_gender']['male']]

 

plt.bar(y_pos, performance, align='center', alpha=0.5, color='red')

plt.xticks(y_pos, objects)

plt.ylabel('Quantity')

plt.title('Author Gender')

 

plt.show()
df_group_author_by_gender = df[['author_id','author_gender']].groupby(['author_id','author_gender']).count().sort_values(['author_gender'], ascending=False)
df_group_author_by_gender.reset_index(inplace=True)
df_group_author_by_gender.head()
df_group_author_gender = df_group_author_by_gender.groupby('author_gender').agg({'author_gender':'count'})
df_group_author_gender['author_gender']
objects = ('Female', 'Male')

y_pos = np.arange(len(objects))

performance = [df_group_author_gender['author_gender']['female'],[df_group_author_gender['author_gender']['male']]]

 

plt.bar(y_pos, performance, align='center', alpha=0.5, color='red')

plt.xticks(y_pos, objects)

plt.ylabel('Quantity')

plt.title('Author Gender')

 

plt.show()
n_groups = 2

group_female = (df_group_author_gender['author_gender']['female'], group_by_gender_non_unique['author_gender']['female'])

group_male = (df_group_author_gender['author_gender']['male'], group_by_gender_non_unique['author_gender']['male'])
# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.35

opacity = 0.8



rects1 = plt.bar(index, group_female, bar_width,

alpha=opacity,

color='b',

label='Female')

 

rects2 = plt.bar(index + bar_width, group_male, bar_width,

alpha=opacity,

color='g',

label='Male')

 

plt.xlabel('Author Gender')

plt.ylabel('Quantity')

plt.title('Gender by author')

plt.xticks(index + bar_width, ('Group Data', 'Ungrouped Data'))

plt.legend()

plt.tight_layout()

plt.show()