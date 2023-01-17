#imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

import re # RegEx for string manipulaion

from matplotlib import rc

from scipy import stats

from scipy.stats import norm, skew #for some statistics

font = {'family' : 'normal',

        'weight' : 'bold',

        'size'   : 15}



rc('font', **font)



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
df = pd.read_csv("../input/app-store-apple-data-set-10k-apps/AppleStore.csv")

df_description = pd.read_csv("../input/app-store-apple-data-set-10k-apps/appleStore_description.csv")
print("Dataset: iOS App Store")

print("Columns: ", list(df.columns))

print("Top five entries: \n", df.head(5))

print("Total entries: ", len(df))

print("Dataset: iOS App Store apps discription")

print("Columns: ", list(df_description.columns))

print("Top five entries: \n", df_description.head(5))

print("Total entries: ", len(df_description))
print(df['track_name'][70])

print(df['track_name'][84])

print(df['track_name'][1497])

print(df['track_name'][1511])

print(df['track_name'][5013])
def is_English(string):

    non_ascii = 0



    for character in string:

        if ord(character) > 127:

            non_ascii += 1



    if non_ascii > 3:

        return False

    else:

        return True



print(is_English('Docs To Go™ Free Office Suite'))

print(is_English('Instachat <img draggable="false" class="emoji" alt="<img draggable="false" class="emoji" alt="<img draggable="false" class="emoji" alt="<img class="emoji" alt="😜" src="https://s.w.org/images/core/emoji/11.2.0/svg/1f61c.svg">" src="https://s.w.org/images/core/emoji/11.2.0/svg/1f61c.svg">" src="https://s.w.org/images/core/emoji/11.2.0/svg/1f61c.svg">" src="https://s.w.org/images/core/emoji/11.2.0/svg/1f61c.svg">'))

print(is_English('爱奇艺PPS -《欢乐颂2》电视剧热播'))
ios_english = []

for i in range(len(df)):

    app = df.iloc[i]

    name = df['track_name'][i]

    if is_English(name):

        ios_english.append(app)
len(ios_english)
ios_final = []

for app in ios_english:

    price = float(app[5])

    if price == 0.0:

        ios_final.append(app)
len(ios_final)
train = pd.DataFrame(ios_final, columns=df.columns)

train.columns
train.drop(["Unnamed: 0", "price", "currency"], axis=1, inplace=True)
train.dtypes
train.describe(include =['object', 'float64', 'int64'] )
train.describe(include =['float64', 'int64'] )
train.describe(include =['object'] )
from collections import Counter

freq_counter = dict(Counter(train["prime_genre"]))

freq_counter_pair = list(zip(freq_counter.keys(), freq_counter.values()))

freq_counter_pair.sort(key=lambda x: x[1], reverse = True)

x = []

y = []

Genres_set = []

i = 0

for x_, y_ in freq_counter_pair:

    x.append(i)

    i += 1

    Genres_set.append(x_)

    y.append(y_)

freq_table = pd.DataFrame(columns=["prime_genre", "Number of apps"])

freq_table["prime_genre"] = Genres_set

freq_table["Number of apps"] = y



f, ax = plt.subplots(figsize=(16, 16))

fig = sns.barplot(x=x, y=y)

ax.title.set_text("Genre vs Total Number of Apps (Most common apps by Category)")

ax.set_xlabel("prime_genres")

ax.set_ylabel("Total Number of Apps")



for tick in ax.xaxis.get_major_ticks():

    tick.label.set_rotation('vertical')

plt.xticks(range(len(Genres_set)), Genres_set)

freq_table
# xs and ys were created above and using it here

y_tmp = []

for i in y:

    y_tmp.append(i/len(train)*100)

f, ax = plt.subplots(figsize=(16, 12))

fig = sns.barplot(x=x, y=y_tmp)

ax.title.set_text("Genre vs Total Number of Apps (Most common apps by Category)")

ax.set_xlabel("prime_genres")

ax.set_ylabel("Total Number of Apps (in %)")

for tick in ax.xaxis.get_major_ticks():

    tick.label.set_rotation('vertical')

plt.xticks(range(len(Genres_set)), Genres_set)

print()
var = 'prime_genre'

data = pd.concat([train['rating_count_tot'], train[var].map(lambda x: Genres_set.index(x))], axis=1)

f, ax = plt.subplots(figsize=(16, 12))

result = data.groupby(["prime_genre"])['rating_count_tot'].aggregate(np.mean).reset_index().sort_values('rating_count_tot', ascending=False)

new_mapping = list(result['prime_genre'])

new_list = []

for i in data['prime_genre']:

    new_list.append(new_mapping.index(i))

data['prime_genre'] = new_list

Genres_set_ = []

for i in new_mapping:

    Genres_set_.append(Genres_set[i])

fig = sns.barplot(x=var, y="rating_count_tot", data=data)

ax.title.set_text("Prime Genre vs Average Number of content_rating_tot (Most popular apps by prime_genre)")

ax.set_ylabel("rating_count_total")

ax.set_xlabel("prime_genres")



for tick in ax.xaxis.get_major_ticks():

    tick.label.set_rotation('vertical')

plt.xticks(range(len(Genres_set)), Genres_set_)

print()
train[train['prime_genre'] == 'Navigation']
train[train['prime_genre'] == 'Reference']
train[(train['prime_genre'] == 'Social Networking') & (train['rating_count_tot'] > 100000)]