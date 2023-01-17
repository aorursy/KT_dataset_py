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

from collections import Counter #for making frequency tables

font = {'family' : 'normal',

        'weight' : 'bold',

        'size'   : 15}



rc('font', **font)



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
#loading data set

googleplaystore = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

googleplaystore_user_reviews = pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")
print("Dataset: Google Play Store")

print("Columns: ", list(googleplaystore.columns))

print("Top five entries: \n", googleplaystore.head(5))

print("Total entries: ", len(googleplaystore))

print("\nDataset: Google Play Store user user reviews")

print("Columns: ", list(googleplaystore_user_reviews.columns))

print("Top five entries: \n", googleplaystore_user_reviews.head(5))

print("Total entries: ", len(googleplaystore_user_reviews))
googleplaystore.iloc[10472]
googleplaystore = googleplaystore.drop(index = 10472)
for app_id in range(len(googleplaystore)):

    name = googleplaystore.iloc[app_id][0]

    if name == 'Instagram':

        print(googleplaystore.iloc[app_id])
reviews_max = {}



for app_id in range(len(googleplaystore)):

    app = googleplaystore.iloc[app_id]

    name = app[0]

    n_reviews = float(app[3])



    if name in reviews_max and reviews_max[name] < n_reviews:

        reviews_max[name] = n_reviews



    elif name not in reviews_max:

        reviews_max[name] = n_reviews
[(i, reviews_max[i]) for i in list(reviews_max.keys())[:10]]
print("Length of reviews_max dict: ", len(reviews_max))

print("Total number of duplicate entries:", len(googleplaystore) - len(reviews_max))
googleplaystore_clean = []

already_added = []



for app_id in range(len(googleplaystore)):

    app = googleplaystore.iloc[app_id]

    name = app[0]

    n_reviews = float(app[3])



    if (reviews_max[name] == n_reviews) and (name not in already_added):

        googleplaystore_clean.append(app)

        already_added.append(name)
len(googleplaystore_clean)
print(googleplaystore_clean[529][0])

print(googleplaystore_clean[1206][0])

print(googleplaystore_clean[3279][0])

print(googleplaystore_clean[4412][0])
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

print(is_English('中国語 AQリスニング'))
googleplaystore_english = []





### android ###

for app in googleplaystore_clean:

    name = app[0]

    if is_English(name):

        googleplaystore_english.append(app)

### These two lists will store the new data sets ###

googleplaystore_final = []



### android ###

for app in googleplaystore_english:

    price = app[7]

    if price == '0':

        googleplaystore_final.append(app)



### Checking number of entries left ###

print(len(googleplaystore_final))
train = pd.DataFrame(googleplaystore_final, columns = googleplaystore.columns)
train.drop('Price', axis=1, inplace=True)
train[["Reviews"]] = train[["Reviews"]].astype(np.int32)
train.columns
train.describe(include =['object', 'float', 'int32'] )
train.describe(include =['float', 'int32'] )
train.describe(include =['object'] )
Installs_set = list(set(train['Installs']))

Installs_set
'''

How to get Category_set:

Category_set = list(set(train['Category']))

#Due to change of hashing keys of set indeces, the output list is changing. 

'''

Category_set =  ['PRODUCTIVITY',

                 'SPORTS',

                 'LIFESTYLE',

                 'FOOD_AND_DRINK',

                 'NEWS_AND_MAGAZINES',

                 'HOUSE_AND_HOME',

                 'GAME',

                 'COMMUNICATION',

                 'HEALTH_AND_FITNESS',

                 'ENTERTAINMENT',

                 'BEAUTY',

                 'WEATHER',

                 'AUTO_AND_VEHICLES',

                 'SOCIAL',

                 'EVENTS',

                 'BUSINESS',

                 'SHOPPING',

                 'TOOLS',

                 'PARENTING',

                 'COMICS',

                 'BOOKS_AND_REFERENCE',

                 'MEDICAL',

                 'PHOTOGRAPHY',

                 'VIDEO_PLAYERS',

                 'ART_AND_DESIGN',

                 'FINANCE',

                 'FAMILY',

                 'PERSONALIZATION',

                 'TRAVEL_AND_LOCAL',

                 'EDUCATION',

                 'MAPS_AND_NAVIGATION',

                 'DATING',

                 'LIBRARIES_AND_DEMO']
freq_counter = dict(Counter(train["Category"]))

freq_counter_pair = list(zip(freq_counter.keys(), freq_counter.values()))

freq_counter_pair.sort(key=lambda x: x[1], reverse = True)

x = []

y = []

Category_set_ = []

i = 0

for x_, y_ in freq_counter_pair:

    x.append(i)

    i += 1

    Category_set_.append(x_)

    y.append(y_)

freq_table = pd.DataFrame(columns=["Category", "Number of apps"])

freq_table["Category"] = Category_set_

freq_table["Number of apps"] = y

freq_table
font = {'family' : 'normal',

        'weight' : 'bold',

        'size'   : 15}



rc('font', **font)

f, ax = plt.subplots(figsize=(16, 12))

fig = sns.barplot(x=x, y=y)

ax.title.set_text("Category vs Total Number of Apps (Most common apps by Category)")

ax.set_xlabel("Category")

ax.set_ylabel("Total Number of Apps")

for tick in ax.xaxis.get_major_ticks():

    tick.label.set_rotation('vertical')

plt.xticks(range(len(Category_set)), Category_set_)

print()
# xs and ys were created above and using it here

y_tmp = []

for i in y:

    y_tmp.append(i/len(train)*100)

f, ax = plt.subplots(figsize=(16, 12))

fig = sns.barplot(x=x, y=y_tmp)

ax.title.set_text("Category vs Total Number of Apps (Most common apps by Category)")

ax.set_xlabel("Category")

ax.set_ylabel("Total Number of Apps (in %)")

for tick in ax.xaxis.get_major_ticks():

    tick.label.set_rotation('vertical')

plt.xticks(range(len(Category_set)), Category_set_)

print()
var = 'Category'

data = pd.concat([train['Installs'].map(lambda x:float(re.sub('[,+]', '', x))), train[var].map(lambda x: Category_set.index(x))], axis=1)

result = data.groupby(["Category"])['Installs'].aggregate(np.mean).reset_index().sort_values('Installs', ascending=False)

new_mapping = list(result['Category'])

new_list = []

for i in data['Category']:

    new_list.append(new_mapping.index(i))

data['Category'] = new_list

Category_set_ = []

for i in new_mapping:

    Category_set_.append(Category_set[i])

f, ax = plt.subplots(figsize=(16, 12))

fig = sns.barplot(x=var, y="Installs", data=data)

ax.title.set_text("Category vs Average Number of Installers (Most popular apps by Category)")

for tick in ax.xaxis.get_major_ticks():

    tick.label.set_rotation('vertical')

plt.xticks(range(len(Category_set)), Category_set_)

print()
var = Category_set[7] #var = 'COMMUNICATION'

train[(train['Installs'] == '1,000,000,000+') & (train['Category'] == var)]
var = Category_set[7]

train[(train['Installs'] == '500,000,000+') & (train['Category'] == var)]
var = Category_set[7]

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[23]# var = "VIDEO_PLAYERS"

train[(train['Installs'] == '1,000,000,000+') & (train['Category'] == var)]
var = Category_set[23]# var = "VIDEO_PLAYERS"

train[(train['Installs'] == '500,000,000+') & (train['Category'] == var)]
var = Category_set[23]# var = "VIDEO_PLAYERS"

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[13]# var = "SOCIAL"

train[(train['Installs'] == '1,000,000,000+') & (train['Category'] == var)]
var = Category_set[13]# var = "SOCIAL"

train[(train['Installs'] == '500,000,000+') & (train['Category'] == var)]
var = Category_set[13]# var = "SOCIAL"

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[22] #var = 'PHOTOGRAPHY'

train[(train['Installs'] == '1,000,000,000+') & (train['Category'] == var)]
var = Category_set[22] #var = 'PHOTOGRAPHY'

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[0] #var = 'PRODUCTIVITY'

train[(train['Installs'] == '1,000,000,000+') & (train['Category'] == var)]
var = Category_set[0] #var = 'PRODUCTIVITY'

train[(train['Installs'] == '500,000,000+') & (train['Category'] == var)]
var = Category_set[0] #var = 'PRODUCTIVITY'

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[6] #var = 'GAME'

train[(train['Installs'] == '1,000,000,000+') & (train['Category'] == var)]
var = Category_set[6] #var = 'GAME'

train[(train['Installs'] == '500,000,000+') & (train['Category'] == var)]
var = Category_set[6] #var = 'GAME'

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[28] #var = 'TRAVEL_AND_LOCAL'

train[(train['Installs'] == '1,000,000,000+') & (train['Category'] == var)]
var = Category_set[28] #var = 'TRAVEL_AND_LOCAL'

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[28] #var = 'TRAVEL_AND_LOCAL'

train[(train['Installs'] == '50,000,000+') & (train['Category'] == var)]
var = Category_set[9] #var = 'ENTERTAINMENT'

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[17] #var = 'TOOLS'

train[(train['Installs'] == '1,000,000,000+') & (train['Category'] == var)]
var = Category_set[17] #var = 'TOOLS'

train[(train['Installs'] == '500,000,000+') & (train['Category'] == var)]
var = Category_set[17] #var = 'TOOLS'

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[4] #var = 'NEWS_AND_MAGAZINES'

train[(train['Installs'] == '1,000,000,000+') & (train['Category'] == var)]
var = Category_set[4] #var = 'NEWS_AND_MAGAZINES'

train[(train['Installs'] == '500,000,000+') & (train['Category'] == var)]
var = Category_set[4] #var = 'NEWS_AND_MAGAZINES'

train[(train['Installs'] == '50,000,000+') & (train['Category'] == var)]
var = Category_set[20] #var = 'BOOKS_AND_REFERENCE'

train[(train['Installs'] == '1,000,000,000+') & (train['Category'] == var)]
var = Category_set[20] #var = 'BOOKS_AND_REFERENCE'

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[16] #var = 'SHOPPING'

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[11] #var = 'WEATHER'

train[(train['Installs'] == '50,000,000+') & (train['Category'] == var)]
var = Category_set[26] #var = 'FAMILY'

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[26] #var = 'FAMILY'

train[(train['Installs'] == '50,000,000+') & (train['Category'] == var)]
var = Category_set[27] #var = 'PERSONALIZATION'

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[27] #var = 'PERSONALIZATION'

train[(train['Installs'] == '50,000,000+') & (train['Category'] == var)]
var = Category_set[30] #var = 'MAPS_AND_NAVIGATION'

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[30] #var = 'MAPS_AND_NAVIGATION'

train[(train['Installs'] == '50,000,000+') & (train['Category'] == var)]
var = Category_set[30] #var = 'MAPS_AND_NAVIGATION'

train[(train['Installs'] == '1,000,000+') & (train['Category'] == var)]
var = Category_set[8] #var = 'HEALTH_AND_FITNESS'

train[(train['Installs'] == '100,000,000+') & (train['Category'] == var)]
var = Category_set[8] #var = 'HEALTH_AND_FITNESS'

train[(train['Installs'] == '50,000,000+') & (train['Category'] == var)]
var = Category_set[8] #var = 'HEALTH_AND_FITNESS'

train[(train['Installs'] == '1,000,000+') & (train['Category'] == var)]
Genres_set = []

for Genre in train['Genres']:

    if Genre not in Genres_set:

        Genres_set.append(Genre)

freq_counter = dict(Counter(train["Genres"]))

freq_counter_pair = list(zip(freq_counter.keys(), freq_counter.values()))

freq_counter_pair.sort(key=lambda x: x[1], reverse = True)

x = []

y = []

Genres_set_ = []

i = 0

for x_, y_ in freq_counter_pair:

    x.append(i)

    i += 1

    Genres_set_.append(x_)

    y.append(y_)

freq_table = pd.DataFrame(columns=["Genres", "Number of apps"])

freq_table["Genres"] = Genres_set_

freq_table["Number of apps"] = y

freq_table
font2 = dict(font)

font2['size'] = 20

rc('font', **font2)

f, ax = plt.subplots(figsize=(32, 30))

fig = sns.barplot(x=x, y=y)

ax.title.set_text("Genres vs Total Number of Apps (Most common apps by Genres)")

ax.set_xlabel("Genres")

ax.set_ylabel("Total Number of Apps")

for tick in ax.xaxis.get_major_ticks():

    tick.label.set_rotation('vertical')

plt.xticks(range(len(Genres_set)), Genres_set_)

print()
# xs and ys were created above and using it here

y_tmp = []

for i in y:

    y_tmp.append(i/len(train)*100)

font2 = dict(font)

font2['size'] = 20

rc('font', **font2)

f, ax = plt.subplots(figsize=(32, 30))

fig = sns.barplot(x=x, y=y_tmp)

ax.title.set_text("Genres vs Total Number of Apps (Most common apps by Genres) (in %)")

ax.set_xlabel("Genres")

ax.set_ylabel("Total Number of Apps")

for tick in ax.xaxis.get_major_ticks():

    tick.label.set_rotation('vertical')

plt.xticks(range(len(Genres_set)), Genres_set_)

print()
font2 = dict(font)

font2['size'] = 18

rc('font', **font2)

var = 'Genres'

data = pd.concat([train['Installs'].map(lambda x:float(re.sub('[,+]', '', x))), train[var].map(lambda x: Genres_set.index(x))], axis=1)

result = data.groupby([var])['Installs'].aggregate(np.mean).reset_index().sort_values('Installs', ascending=False)

new_mapping = list(result['Genres'])

new_list = []

for i in data['Genres']:

    new_list.append(new_mapping.index(i))

data['Genres'] = new_list

Genres_set_ = []

for i in new_mapping:

    Genres_set_.append(Genres_set[i])

f, ax = plt.subplots(figsize=(32, 40))

ax.title.set_text("Genres vs Average Number of Installers (Most popular apps by Genres)")

fig = sns.barplot(x=var, y="Installs", data=data)

for tick in ax.xaxis.get_major_ticks():

    tick.label.set_rotation('vertical')

Genres_set_temp = list(map(lambda x: x +" "*(4-len(str(Genres_set.index(x)))) + str(Genres_set.index(x)), Genres_set_))

plt.xticks(range(len(Genres_set)), Genres_set_temp)

print()
print("Genres of \'FITNESS\' (HEALTH_AND_FITNESS) category:")

for i in Genres_set:

    if 'fitness' in i.lower():

        print("\t", Genres_set.index(i), i)

print("\n\nGenres of \'GAME\'  category")

for i in Genres_set:

    if 'game' in i.lower():

        print("\t", Genres_set.index(i), i)
print("Genres of \'COMMUNICATION\' category:")

for i in Genres_set:

    if 'communication' in i.lower():

        print("\t", Genres_set.index(i), i)

print("\n\nGenres of \'TOOLS\'  category")

for i in Genres_set:

    if 'tools' in i.lower():

        print("\t", Genres_set.index(i), i)
#From the graph:

selected_set_1 = [8, 28, 30, 32, 54, 57, 64, 68, 80, 82, 83, 88, 94]

selected_set_2 = [4, 26, 29, 36, 37, 39, 41, 45, 50, 62, 75, 85, 86, 97, 104, 108, 109]

selected_set_3 = [12, 14, 15, 17, 18, 22, 31, 34, 38, 40, 42, 47, 48, 53, 55, 58, 61, 63, 81, 87, 93, 94, 95, 96, 98]

selected_set_4 = [11, 27, 33, 35, 65, 66, 73, 76, 77, 78, 101]

#Set 5 is else so not included

print('Set1: Most Downloaded\n')

for i in selected_set_1:

    print(i, Genres_set[i], sep='\t')

print('\nSet2: Higher number of Downloads\n')

for i in selected_set_2:

    print(i, Genres_set[i], sep='\t')

print('\nSet3: Average number of Downloads pt.1\n')

for i in selected_set_3:

    print(i, Genres_set[i], sep='\t')

print('\nSet4: Average number of Downloads pt.2\n')

for i in selected_set_4:

    print(i, Genres_set[i], sep='\t')