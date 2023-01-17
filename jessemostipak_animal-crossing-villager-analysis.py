# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns  # data visualization

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
villagers = pd.read_csv("/kaggle/input/animal-crossing-new-horizons-nookplaza-dataset/villagers.csv")

wish = pd.read_table("/kaggle/input/scraped-ac-wish-rankings/ac_vil_ranking.txt", header=None, names=["Name"])
villagers.head()
wish.head()

# DO NOT RUN #

### r code for creating the rank lookup

#rank_lookup <- raw %>%  # raw is the equivalent of wish

  #as_tibble() %>%  # creates a tibble 

  #mutate(value = str_squish(value)) %>%  # removes whitespace from a string 

  #filter(value %in% vil$name) %>%  # pulls value for villagers in villagers df 

  #distinct(value) %>%  # remove duplicates 

  #mutate(wish_score = rev(1:nrow(.))) %>%  # creates a wish score based on rank 

  #rename(name = value)  # renames value (villager name) to name 
wish['Name'] = wish['Name'].str.strip().drop_duplicates()

wish[0:10]



#df1['State'] = df1['State'].str.strip()

#print (df1)



#.drop_duplicates(keep=False,inplace=True)
wish = wish.dropna()

wish[0:10]
# OK now we need to merge with villagers.

# OR remove more rows and reset index?

# could do by removing "Not in stock" and anything that doesn't start with title case

#df = df[df['EntityName'].str[0].str.isupper()]

wish = wish[wish['Name'].str[0].str.isupper()]

wish[0:10]
# gotta remove "Not in stock"

#df[df.name != 'Tina']

wish = wish[wish.Name != 'Not in stock']

wish[0:10]
wish_rank = wish.reset_index()

wish_rank[0:10]
# df.drop('reports', axis=1)

wish_rank_01 = wish_rank.drop('index', axis=1)

wish_rank_01[0:10]
wish_rank_02 = wish_rank_01.reset_index()

wish_rank_02["index"] = wish_rank_02["index"].values[::-1]

wish_rank_02[0:10]
# df.rename(columns={"A": "a", "B": "c"})

wish_rank_03 = wish_rank_02.rename(columns={"index": "wish_rank"})

wish_rank_03[0:10]
#merged_left = pd.merge(left=survey_sub, right=species_sub, how='left', left_on='species_id', right_on='species_id')

merged_wish = pd.merge(left=villagers, right=wish_rank_03, how='left', left_on='Name', right_on='Name')

merged_wish[0:10]
# df1 = df[['a','b']]

vil_ranks = merged_wish[['Name', 'Species', 'Personality', 'wish_rank', 'Gender', 'Hobby']]

vil_ranks[0:10]
# Sort based on wish ranking (ranking as of May 24, 2020)

# df.sort_values(by=['col1'])

## Most wished for Villagers

vil_ranks.sort_values(ascending=False, by=['wish_rank'])[0:10]



# 4/10 cats

# 6/10 female

# 4/10 music
# visualization: wish score and personality

# steps: group by personality, calculate mean wish_rank, plot

# separate steps? rather than tidyverse piping



# R code

# villagers %>% 

#   group_by(personality) %>% 

#   summarize(mean_wish = mean(wish_score), n = n()) %>%  

#   ggplot(data = ., 

#          aes(x = reorder(personality, mean_wish), y = mean_wish, size = n)) +

#   geom_point(color = "#1c7c24", alpha = .75) + 

#   labs(title = "Mean wish scores across villager personalities", 

#        subtitle = "Villager rankings from Nook Market on May 24, 2020",

#        x = "", 

#        y = "Wish score", 

#        caption = "data: Nook Plaza and Nook Market") +

#   coord_flip() 
# data.groupby('month')['duration'].sum()

sp_mean_rank = vil_ranks.groupby('Species')['wish_rank'].mean().to_frame().reset_index().sort_values(by=['wish_rank'])

sp_mean_rank.head()
# ax = sns.scatterplot(x="total_bill", y="tip", data=tips)

# for axes in chart.axes.flat:

#     axes.set_xticklabels(axes.get_xticklabels(), rotation=65, horizontalalignment='right')

fig = plt.gcf()

fig.set_size_inches(12, 8)

plot_01 = sns.scatterplot(x="Species", y="wish_rank", data=sp_mean_rank)

for item in plot_01.get_xticklabels():

    item.set_rotation(45)

#plot_01.set_xticklabels(rotation=45, horizontalalignment='right')
# make it prettier

fig = plt.gcf()

fig.set_size_inches(10, 8)

plot_02 = sns.scatterplot(x="Species", y="wish_rank", 

                          hue="Species", size="wish_rank",

                          sizes=(40, 800), alpha=0.75, palette="BrBG", legend=False,

                          data=sp_mean_rank)

for item in plot_02.get_xticklabels():

    item.set_rotation(45)
# Now do it for personalities

pers_mean_rank = vil_ranks.groupby('Personality')['wish_rank'].mean().to_frame().reset_index().sort_values(by=['wish_rank'])

pers_mean_rank.head()
fig = plt.gcf()

fig.set_size_inches(10, 10)

plot_02 = sns.scatterplot(x="Personality", y="wish_rank", 

                          hue="Personality", size="wish_rank",

                          sizes=(40, 1000), alpha=0.75, palette="cubehelix", legend=False,

                          data=pers_mean_rank)

for item in plot_02.get_xticklabels():

    item.set_rotation(45)