import pandas as pd

import numpy as np



import nltk

from gensim import corpora, models

from nltk.stem import WordNetLemmatizer



import matplotlib.pyplot as plt

import seaborn as sns



sns.set(style="whitegrid")

sns.set_color_codes("pastel")

%matplotlib inline



from collections import Counter
data = pd.read_csv('../input/attacks.csv', encoding = 'ISO-8859-1')
data.head(2)
names = list(data.columns)

names[9] = 'Sex'

names[12] = 'Fatal'

names[14] = 'Species'

data.columns = names
def year_prettify(year):

    if year > 1000: 

        return year

    else:

        return np.nan

    

def sex_prettify(sex):

    if sex == 'M' or sex == 'F':

        return sex

    else:

        return np.nan

    

def age_prettify(age):

    try:

        age = int(age)

    except ValueError:

        age = 0

        

    if (age > 0 and age <= 100):

        return age

    else:

        return np.nan

    

def fatal_prettify(fatal):

    if fatal == 'N' or fatal == 'Y':

        return fatal

    else:

        return np.nan

    

def date_prettify(date):

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    num_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    

    date = str(date)

    if (len(date) == 11 and date[2] == '-'):

        for i in range(len(months)):

            date = date.replace(months[i], num_months[i])

        return date

    else:

        return np.nan
data['Year'] = data['Year'].apply(year_prettify)

data['Sex'] = data['Sex'].apply(sex_prettify)

data['Age'].fillna(0, inplace=True)

data['Age'] = data['Age'].apply(age_prettify)

data['Fatal'] = data['Fatal'].apply(fatal_prettify)
data['Date'] = data['Date'].apply(date_prettify)
fatal_vals = data['Fatal'].value_counts().tolist()
f, ax = plt.subplots(figsize=(5, 5))



labels = ['Not Fatal', 'Fatal']

colors = ['#75fd63', '#ff474c']



plt.pie(fatal_vals, labels=labels, colors=colors,

        autopct='%1.1f%%', startangle=90)

 

axis = plt.axis('equal')
sex_vals = data['Sex'].value_counts().tolist()
f, ax = plt.subplots(figsize=(5, 5))



labels = ['Male', 'Female']

colors = ['#a2cffe', '#ffd1df']



plt.pie(sex_vals, labels=labels, colors=colors,

        autopct='%1.1f%%', startangle=90)



axis = plt.axis('equal')
fig, ax = plt.subplots(figsize=(8, 6))



sns.distplot(data['Age'].dropna(),  

             hist_kws={"alpha": 1, "color": "#a2cffe"}, 

             kde=False, bins=15)



ax = ax.set(ylabel="Count", xlabel="Age")
most_common_activities = Counter(data['Activity'].dropna().tolist()).most_common(20)

activities = [actv_list[0] for actv_list in most_common_activities]

counts = [actv_list[1] for actv_list in most_common_activities]
fig, ax = plt.subplots(figsize=(12, 5))



sns.barplot(x=activities, y=counts, color='#a2cffe', ax=ax)

ax.set(ylabel="Attacks Count", xlabel="Activities")



ticks = plt.setp(ax.get_xticklabels(), rotation=60, fontsize=9)
most_common_species = Counter(data['Species'].dropna().tolist()).most_common(20)



species = [species_list[0] for species_list in most_common_species]

for i in species:

    print(i)
dates = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
days = dates.dropna().map(lambda x: x.day)



days_counter = Counter(days)

days_keys = list(days_counter.keys())

days_values = list(days_counter.values())
months = dates.dropna().map(lambda x: x.month)



def get_season(month):

    if month >= 3 and month <= 5:

        return 'spring'

    elif month >= 6 and month <= 8:

        return 'summer'

    elif month >= 9 and month <= 11:

        return 'autumn'

    else:

        return 'winter'



months_labels = months.apply(get_season)



months_counter = Counter(months_labels)

months_keys = list(months_counter.keys())

months_values = list(months_counter.values())
fig, ax = plt.subplots(figsize=(10, 3))

sns.barplot(x=days_keys, y=days_values, color='#a2cffe', ax=ax)

ax = ax.set(ylabel="Attacks Count", xlabel="Day")
fig, ax = plt.subplots(figsize=(6, 3))

sns.barplot(x=months_keys, y=months_values, color='#a2cffe', ax=ax)

ax = ax.set(ylabel="Attacks Count", xlabel="Month")
activity_texts = data['Injury'].tolist()
wordnet_lemmatizer = WordNetLemmatizer()
def keep_only_letters(text):

    cleaned_text = ''

    for char in text:

        if (char.isalpha() or char == ' '):

            cleaned_text += char

    return cleaned_text
lemm_activity_texts = []

for text in activity_texts:

    text = keep_only_letters(str(text)).lower()   

    lemm_text = []

    for word in text.split():

        lemm_text.append(wordnet_lemmatizer.lemmatize(word))

    lemm_activity_texts.append(lemm_text)
dictionary = corpora.Dictionary(lemm_activity_texts)
word_list = []

for key, value in dictionary.dfs.items():

    if value > 50:

        word_list.append(key)
dictionary.filter_tokens(word_list)

corpus = [dictionary.doc2bow(text) for text in lemm_activity_texts]
np.random.seed(76543)

lda = models.LdaModel(corpus, num_topics=20, id2word=dictionary, passes=5)
topics = lda.show_topics(num_topics=20, num_words=5, formatted=False)

for topic in topics:

    num = int(topic[0]) + 1

    print('Cause %d:' % num, end=' ')

    print(', '.join([pair[0] for pair in topic[1]]))