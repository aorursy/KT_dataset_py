# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import itertools

import csv

import collections

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.

sns.set_context("paper")

RES_DIR = "../input/"



def load_train_data():

    categories = ['cooking','robotics','travel','crypto','diy','biology']

    train_data = []

    for cat in categories:

        #data = pd.read_csv("{}{}.csv".format("../input/", cat), usecols=['id', 'title', 'tags'])

        data = pd.read_csv("{}{}.csv".format(RES_DIR, cat),usecols=['id','title','tags'])

        data['category'] = cat

        train_data.append(data)

        

    return pd.concat(train_data)



train_data = load_train_data()

train_data.head()



ax=train_data['category'].value_counts().plot(kind='bar')

#ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation=10)

#plt.show()



tag_lists = [t.strip().split() for t in train_data['tags'].values]

all_tags = list(itertools.chain(*tag_lists))

tag_list_size = np.array([len(x) for x in tag_lists])

print("""the corpus is composed by {} questions. overal {} tags have been used,of which {} unique ones.

average number of tags per question {:.2f} (min={}, max={}, std={:.2f})""".format(len(train_data),

len(all_tags),len(set(all_tags)),tag_list_size.mean(),min(tag_list_size),max(tag_list_size),tag_list_size.std()))



#sns.distplot(tag_list_size,kde=False)

#sns.plt.show()



def get_top_tags(df,n=None):

    tags = list(itertools.chain(*[t.strip().split() for t in df['tags'].values]))

    top_tags = collections.Counter(list(tags)).most_common(n)

    tags,count = zip(*top_tags)

    return tags,count



tags,count = get_top_tags(train_data,10)

#ax = sns.barplot(x=tags,y=count)

#ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=30)

#sns.plt.show()



#fig,axes = plt.subplots(6,sharey=True)

#fig.set_size_inches(7, 10)

#fig.tight_layout()

for i, (name,group) in enumerate(train_data.groupby('category')):

    tags,count = get_top_tags(group,10)

    norm_count = np.array(count)/len(all_tags)

    #sns.barplot(x=tags,y=norm_count,ax=axes[i])

    #sns.barplot(x=tags, y=norm_count, ax=axes[i])

    #axes[i].set_xticklabels(axes[i].xaxis.get_majorticklabels(), rotation=30)

    #axes[i].set_xlabel(name)

    

tags_df = pd.DataFrame(index=set(itertools.chain(*tag_lists)))

print(tags_df.head())

for i,(name,group) in enumerate(train_data.groupby('category')):

    tags_df[name] = 0

    #print(name)

    tmp_index,count = get_top_tags(group)

    tmp = pd.Series(count,index=tmp_index)

    #print(tmp)

    


