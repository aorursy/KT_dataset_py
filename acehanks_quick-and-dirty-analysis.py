# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))







# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/reddit_worldnews_start_to_2016-11-22.csv")
data.shape

data.columns
data.dtypes
data.head()
data['up_votes'].sort_values(ascending=False).value_counts().head()
data['down_votes'].sort_values(ascending=False).value_counts().head()
[title for title in data.sort_values('up_votes', ascending=False)['title'][:10]]
import seaborn as sns

%matplotlib inline

sns.set_style("dark")



data.groupby('date_created')['up_votes'].mean().plot()

data.groupby('date_created')['up_votes'].mean().rolling(window=120).mean().plot(figsize= (12, 4))
data.over_18.value_counts()
nsfwstory= data[data['over_18']== True]

[story for story in nsfwstory.sort_values('up_votes', ascending=False)['title'][:10]]
attf= data.author.value_counts()[:20]

attf.plot.bar(figsize= (12, 4))
taff= data[data['author'] == "davidreiss666"]

#taff.sort_values('up_votes', ascending= False)['title'].value_counts()

[story for story in taff.sort_values('up_votes', ascending=False)['title'][:10]]

#htaff= taff[taff['up_votes'] > 5]
cand= ['Obama', 'Hillary', 'Trump']

for sole in cand:

   print( data.title.str.contains(sole).value_counts(),sole )



#data.title.str.contains('Trump').value_counts()
from nltk import word_tokenize

tokens = data.title.map(word_tokenize)



def tell_me_about(x):

    x_l = x.lower()

    x_t = x.title()

    return data.loc[tokens.map(lambda sent: x_l in sent or x_t in sent).values]
tell_me_about("Obama")['title'].values.tolist()[:10]
tell_me_about("Hilary")['title'].values.tolist()[:10]
tell_me_about("Donald")['title'].values.tolist()[:10]
tell_me_about("Trump")['title'].values.tolist()[:10]
tell_me_about("Apple")['title'].values.tolist()[:10]