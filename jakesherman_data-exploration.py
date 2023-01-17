%matplotlib inline

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import sklearn
from wordcloud import WordCloud, STOPWORDS

train = pd.read_csv('../input/train.csv')
print('Number of training observations:', len(train.index))
train.describe()
train.isnull().sum()
sns.countplot(x = "OutcomeType", data = train)
sns.countplot(x = "AnimalType", data = train)
def rel_freq_plot(train, column):
    sns.pointplot(x = 'OutcomeType', y = 'Percent', hue = column, data = (train
        .groupby(['OutcomeType', column])
        .size()
        .reset_index()
        .rename(columns = {0: 'Count'})
        .merge(
            (train
             .groupby([column])
             .size()
             .reset_index()
             .rename(columns = {0: 'Total'})
            ), how = 'inner', on = column)
        .assign(Percent = lambda x: x.Count / x.Total)
    ))
    
rel_freq_plot(train, 'AnimalType')
sns.countplot(x = "SexuponOutcome", data = train)
def create_sex_variables(data):
    SexuponOutcome = data['SexuponOutcome'].fillna('Unknown')
    results = []
    for row in SexuponOutcome:
        row = row.split(' ')
        if len(row) == 1:
            row = ['Unknown', 'Unknown']
        results.append(row)
    NeuteredSprayed, Sex = zip(
        *[['Neutered', x[1]] if x[0] == 'Spayed' else x for x in results])
    return (data.assign(Neutered = NeuteredSprayed).assign(Sex = Sex)
            .drop(['SexuponOutcome'], axis = 1))

train = train.pipe(create_sex_variables)
sns.countplot(x = "Neutered", data = train)

rel_freq_plot(train, 'Neutered')
sns.countplot(x = "Sex", data = train)
rel_freq_plot(train, 'Sex')
def create_age_in_years(ages):
    results = []
    units = {'days': 365.0, 'weeks': 52.0, 'months': 12.0}
    for age in ages:
        if age == 'NA':
            results.append('NA')
        else:
            duration, unit = age.split(' ')
            results.append(float(duration) / units.get(unit, 1.0))
    impute = np.median([age for age in results if age != 'NA'])
    return [age if age != 'NA' else impute for age in results]

train = (train
         .assign(Age = create_age_in_years(list(train['AgeuponOutcome'].fillna('NA'))))
         .drop(['AgeuponOutcome'], axis = 1))
sns.distplot(train['Age'], bins = 22)
sns.distplot([x if x == 0 else np.log(x) for x in train['Age']], bins = 10)
sns.boxplot(x = "Age", y = "AnimalType", data = train)
sns.violinplot(x = "OutcomeType", y = "Age", hue = "AnimalType", data = train, cut = 0, split = True,
              palette = "Set3")
def time_of_day(hour):
    if hour > 4 and hour < 12:
        return 'morning'
    elif hour >= 12 and hour < 18:
        return 'afternoon'
    else:
        return 'evening/night'
    
def day_of_the_week(DateTime):
    return datetime.datetime.strptime(DateTime, '%Y-%m-%d %H:%M:%S').weekday()

train = (train
         .assign(Year = train.DateTime.map(lambda x: x[:4]))
         .assign(Month = train.DateTime.map(lambda x: x[5:7]))
         .assign(Day = train.DateTime.map(lambda x: day_of_the_week(x)))
         .assign(TimeOfDay = train.DateTime.map(lambda x: time_of_day(int(x[11:13]))))
         .drop(['DateTime'], axis = 1))
sns.countplot(x = "Day", data = train)
sns.countplot(x = "TimeOfDay", data = train)
print('Total number of breeds:', len(train['Breed'].unique()))
print('Number of cat breeds:', len(train[train['AnimalType'] == 'Cat']['Breed'].unique()))
print('Number of dog breeds:', len(train[train['AnimalType'] == 'Dog']['Breed'].unique()))

sns.distplot(np.log(train.groupby('Breed').size().values), bins = 10)
sns.barplot(x = "Count", y = "Breed", data = (
        train[train['AnimalType'] == 'Dog']
        .groupby(['Breed'])
        .size()
        .reset_index()
        .rename(columns = {0: 'Count'})
        .sort_values(['Count'], ascending = False)
        .head(n = 25)))

def wordcount_dict(wordlist):
    results = {}
    for item in wordlist:
        item = re.split('\W+', item)
        for word in item:
            try:
                results[word] += 1
            except:
                results[word] = 1
    return results

def wordcloud_string(wordcount_dict):
    final_list = []
    for word, count in wordcount_dict.items():
        final_list += [word] * count
    return ' '.join(final_list)

def display_wordcloud(wordcloud_string):
    wordcloud = (
        WordCloud(background_color = 'white', stopwords = STOPWORDS, height = 700, width = 1000)
        .generate(wordcloud_string))
    plt.imshow(wordcloud)
    plt.show()
    return None

display_wordcloud(
    wordcloud_string(
        wordcount_dict(list(train.query('AnimalType == "Dog"')['Breed']))
    )
)
def common_breeds(breeds):
    breed_counts = {}
    for breed in breeds:
        breed = breed.replace(' Mix', '').split('/')
        for subbreed in breed:
            try:
                breed_counts[subbreed] += 1
            except:
                breed_counts[subbreed] = 1
    return breed_counts

breed_counts = common_breeds(list(train[train['AnimalType'] == 'Dog']['Breed']))
len([breed for breed, count in breed_counts.items() if count >= 30])
sns.barplot(x = "Count", y = "Breed", data = (
        train[train['AnimalType'] == 'Cat']
        .groupby(['Breed'])
        .size()
        .reset_index()
        .rename(columns = {0: 'Count'})
        .sort_values(['Count'], ascending = False)
        .head(n = 20)))
display_wordcloud(
    wordcloud_string(
        wordcount_dict(list(train.query('AnimalType == "Cat"')['Breed']))
    )
)