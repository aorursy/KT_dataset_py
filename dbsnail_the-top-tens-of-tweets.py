import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data = pd.read_csv("../input/tweets_all.csv", encoding = "ISO-8859-1")



print(data.head())
data.shape
top_10_country = data.country.value_counts()[:10]

top_10_country.plot(kind='barh')

plt.show()
top_10_city = data.full_name.value_counts()[:10]

top_10_city.plot(kind='barh')

plt.show()
top_10_in_reply_to_screen_name = data.in_reply_to_screen_name.value_counts()[:10]

by_name = sns.barplot(top_10_in_reply_to_screen_name.index,top_10_in_reply_to_screen_name.values ,\

                     color = 'purple', orient='v')

for item in by_name.get_xticklabels():

    item.set_rotation(70)



by_name.set(ylabel='Number of tweets', title = 'Top 10 Names')

plt.show()
def parse_user(row):

    usr_lst = []

    for word in row.split(' '):

        if word.startswith('@'):

            word = word.strip()

            if len(word)>1:

                usr_lst.append(word[1:])

    return usr_lst



data['user'] = data.text.map(lambda row:parse_user(row))
from collections import Counter



user_cnt = Counter()

user_list = data['user'].tolist()

usr_lst = []

for user in user_list:

    if len(user) > 0:

        usr_lst.extend(user)



for usr in usr_lst:

    user_cnt[usr] +=1

    

sorted_usr = sorted(user_cnt.items(), key=lambda x: (-x[1],x[0]))

usr = [ x[0] for x in sorted_usr[:10]]

cnt = [ x[1] for x in sorted_usr[:10]]

by_user = sns.barplot(usr,cnt, color = 'purple', orient='v')

for item in by_user.get_xticklabels():

    item.set_rotation(70)



by_user.set(ylabel='Number of tweets', title = 'Top 10 Mentioned Users')

plt.show()
def parse_source(row):



    for word in row.split(' '):

        idx_start = max(word.find("ww."), word.find("://"))

        idx_end = word.find(".", idx_start+3)

        if idx_start > 0:

            return word[idx_start+3:idx_end]

data['parsed_source'] = data.source.map(lambda row: parse_source(row))  



parsed_source = data.parsed_source.value_counts()[:10]



by_source = sns.barplot(parsed_source.index,parsed_source.values ,\

                     color = 'purple', orient='v')

for item in by_source.get_xticklabels():

    item.set_rotation(70)



by_source.set(ylabel='Number of tweets', title = 'Top 10 Sources')

plt.show()
def combinations(iterable, r):



    pool = tuple(set(iterable))

    n = len(pool)

    if r > n:

        return

    indices = list(range(r))

    yield tuple(pool[i] for i in indices)

    while True:

        for i in reversed(list(range(r))):

            if indices[i] != i + n - r:

                break

        else:

            return

        indices[i] += 1

        for j in list(range((i+1), r)):

            indices[j] = indices[j]-1 + 1

        yield tuple(pool[i] for i in indices)
co_usr_lst = []

for user in user_list:

    for item in combinations(user,2):

        co_usr_lst.append(item)



co_usr_cnt_dict = {}

for tpl in co_usr_lst:

    # avoid same pair of users was added twice, (a,b) and (b,a) should consider the smae pair

    if tpl in co_usr_cnt_dict and (tpl[1],tpl[0]) not in co_usr_cnt_dict:  

        co_usr_cnt_dict[tpl] += 1

    elif (tpl[1],tpl[0]) in co_usr_cnt_dict:

        co_usr_cnt_dict[(tpl[1],tpl[0])] += 1

    else:

         co_usr_cnt_dict[tpl] = 1
sorted_co_usr_cnt = sorted(co_usr_cnt_dict.items(), key=lambda x: (-x[1],x[0]))[:10]



co_usr_cnt_df = pd.DataFrame(sorted_co_usr_cnt, columns= ['user_pair', 'total_occurance'])

by_co_usr_cnt = sns.barplot(x='user_pair',y='total_occurance',data=co_usr_cnt_df, color = 'purple', orient='v')



for item in by_co_usr_cnt.get_xticklabels():

    item.set_rotation(70)



by_co_usr_cnt.set(ylabel='Number of Co-occurances',title = 'Top 10 Co-occurant Users')

plt.show()
from wordcloud import WordCloud, STOPWORDS



def parse_txt(row):

    txt = open('text_file.txt', 'w')

    txt.write(row)

    

data.text.map(lambda row:parse_txt(row))



# read the whole text file.

text = open('text_file.txt').read()



# Generate a word cloud image

wordcloud = WordCloud(  max_font_size=60

                      , background_color="white"

                      , stopwords=STOPWORDS).generate(text)



# Display the generated image with matplotlib 

plt.figure(figsize=(8,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()