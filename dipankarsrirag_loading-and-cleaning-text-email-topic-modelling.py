#------------------------------------------Libraries---------------------------------------------------------------#

####################################################################################################################

#-------------------------------------Boiler Plate Imports---------------------------------------------------------#

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

#---------------------------------------Text Processing------------------------------------------------------------#

import regex

from wordcloud import WordCloud

from nltk.corpus import stopwords 

from nltk.tokenize import WordPunctTokenizer

from string import punctuation

from nltk.stem import WordNetLemmatizer

#####################################################################################################################
names = []

base = '/kaggle/input/topic-modelling-on-emails/Data/'

with os.scandir(base) as entries:

    for entry in entries:

        if(entry.is_file() == False):

            names.append(entry.name)

names
names.sort()
files = {}

unique = []

for name in names:

    path = base + name+'/'

    x = []

    with os.scandir(path) as entries:

        for entry in entries:

            if(entry.is_file()):

                x.append(entry.name)

    files[name] = x

    files[name].sort()
for k, v in files.items():

    print(k, len(v))
names
for i in range(len(names)):

    x = files[names[i]]

    for j in x:

        for k in range(i+1, len(names)):

            key = names[k]

            if j in files[key]:

                files[key].remove(j)
for k, v in files.items():

    print(k, len(v))
data = {}

i = 0



for genre in files.keys() :

    texts = files[genre]

    for text in texts:

        if text in files[genre]:

            path = base + genre + '/' + text

            with open(path, "r", encoding = "latin1") as file:

                data[i] = file.readlines()

                i = i+1

            data[i-1] = [" ".join(data[i-1]), genre] 



data = pd.DataFrame(data).T

print(data.shape)

data.columns = ['Text', 'Class']

data.head()
data.info()
data.isna().sum()
unique = list(data.Text.unique())

len(unique)
dic = dict(data)
uni = {}

i = 0

for k in range(len(list(dic['Text']))):

    if dic['Text'][k] in unique:

        uni[i] = [dic['Text'][k], dic['Class'][k]]

        unique.remove(dic['Text'][k])

        i += 1
data = pd.DataFrame(uni).T

print(data.shape)

data.columns = ['Text', 'Class']

data.head()
plt.figure(figsize=(10,5))

ax = sns.countplot(data.Class, palette = sns.color_palette("mako"))
def make_wordcloud(words,title):

    cloud = WordCloud(width=1920, height=1080,max_font_size=200, max_words=300, background_color="white").generate(words)

    plt.figure(figsize=(20,20))

    plt.imshow(cloud, interpolation="gaussian")

    plt.axis("off") 

    plt.title(title, fontsize=60)

    plt.show()
wordnet_lemmatizer = WordNetLemmatizer()



stop = stopwords.words('english')



for punct in punctuation:

    stop.append(punct)



def filter_text(text, stop_words):

    word_tokens = WordPunctTokenizer().tokenize(text.lower())

    filtered_text = [regex.sub(u'\p{^Latin}', u'', w) for w in word_tokens if w.isalpha() and len(w) > 3]

    filtered_text = [wordnet_lemmatizer.lemmatize(w, pos="v") for w in filtered_text if not w in stop_words] 

    return " ".join(filtered_text)
data["filtered_text"] = data.Text.apply(lambda x : filter_text(x, stop)) 

data.head()
all_text = " ".join(data[data.Class == "Crime"].filtered_text) 

make_wordcloud(all_text, "Crime")
count = pd.DataFrame(all_text.split(), columns = ['words'])

top_10 = count[count['words'].isin(list(count.words.value_counts()[:10].index[:10]))]

plt.figure(figsize=(10,5))

sns.barplot(x = top_10.words.value_counts().index,

            y = top_10.words.value_counts(), palette = sns.color_palette("mako"))
all_text = " ".join(data[data.Class == "Politics"].filtered_text) 

make_wordcloud(all_text, "Politics")
count = pd.DataFrame(all_text.split(), columns = ['words'])

top_10 = count[count['words'].isin(list(count.words.value_counts()[:10].index[:10]))]

plt.figure(figsize=(10,5))

sns.barplot(x = top_10.words.value_counts().index,

            y = top_10.words.value_counts(), palette = sns.color_palette("mako"))
all_text = " ".join(data[data.Class == "Science"].filtered_text) 

make_wordcloud(all_text, "Science")
count = pd.DataFrame(all_text.split(), columns = ['words'])

top_10 = count[count['words'].isin(list(count.words.value_counts()[:10].index[:10]))]

plt.figure(figsize=(10,5))

sns.barplot(x = top_10.words.value_counts().index,

            y = top_10.words.value_counts(), palette = sns.color_palette("mako"))