import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#---Some visualization libraries----

from matplotlib import pyplot as plt



import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')







from wordcloud import WordCloud



#-----------------------------------



from scipy.stats import norm, skew

from scipy import stats





from sklearn.preprocessing import StandardScaler



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import eli5



from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn import model_selection

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn import linear_model



## Some other snippit of codes to get the setting right 

## This is so that the chart created by matplotlib can be shown in the jupyter notebook. 

%matplotlib inline 

%config InlineBackend.figure_format = 'retina' ## This is preferable for retina display. 



#import warnings ## importing warnings library. 

#warnings.filterwarnings('ignore') ## Ignore warning



#def ignore_warn(*args, **kwargs):

#    pass

#warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

pd.set_option('max_columns', None)



import os ## imporing os

print(os.listdir("../input/"))



#text_to_dict

import ast

from collections import Counter

import time
#feature engineering from https://www.kaggle.com/artgor/eda-feature-engineering-and-model-interpretation



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()

# from this kernel: https://www.kaggle.com/gravix/gradient-in-a-box



#this thing converts dicts to correct text, which later can be parsed

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

train = text_to_dict(train)

test = text_to_dict(test)
#first five

for i, e in enumerate(train['belongs_to_collection'][:5]):

    print(i, e)
train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0).value_counts()
train.shape , test.shape
train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

train['has_collection'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)



test['collection_name'] = test['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

test['has_collection'] = test['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)



train = train.drop(['belongs_to_collection'], axis=1)

test = test.drop(['belongs_to_collection'], axis=1)
train.head()
for i, e in enumerate(train['genres'][:5]):

    print(i, e)
print('Number of genres in films')

train['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_genres
plt.figure(figsize = (12, 8))

text = ' '.join([i for j in list_of_genres for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top genres')

plt.axis("off")

plt.show()
Counter([i for j in list_of_genres for i in j]).most_common()
train['num_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)

train['all_genres'] = train['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]

for g in top_genres:

    train['genre_' + g] = train['all_genres'].apply(lambda x: 1 if g in x else 0)

    

test['num_genres'] = test['genres'].apply(lambda x: len(x) if x != {} else 0)

test['all_genres'] = test['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_genres:

    test['genre_' + g] = test['all_genres'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['genres'], axis=1)

test = test.drop(['genres'], axis=1)
train.head()
for i, e in enumerate(train['production_companies'][:5]):

    print(i, e)
print('Number of production companies in films')

train['production_companies'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_companies = list(train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
Counter([i for j in list_of_companies for i in j]).most_common(30)
train['num_companies'] = train['production_companies'].apply(lambda x: len(x) if x != {} else 0)

train['all_production_companies'] = train['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]

for g in top_companies:

    train['production_company_' + g] = train['all_production_companies'].apply(lambda x: 1 if g in x else 0)

    

test['num_companies'] = test['production_companies'].apply(lambda x: len(x) if x != {} else 0)

test['all_production_companies'] = test['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_companies:

    test['production_company_' + g] = test['all_production_companies'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['production_companies', 'all_production_companies'], axis=1)

test = test.drop(['production_companies', 'all_production_companies'], axis=1)
train.shape,test.shape
for i, e in enumerate(train['production_countries'][:5]):

    print(i, e)
print('Number of production countries in films')

train['production_countries'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_countries = list(train['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_countries for i in j]).most_common(25)
train['num_countries'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)

train['all_countries'] = train['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(25)]

for g in top_countries:

    train['production_country_' + g] = train['all_countries'].apply(lambda x: 1 if g in x else 0)

    

test['num_countries'] = test['production_countries'].apply(lambda x: len(x) if x != {} else 0)

test['all_countries'] = test['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_countries:

    test['production_country_' + g] = test['all_countries'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['production_countries', 'all_countries'], axis=1)

test = test.drop(['production_countries', 'all_countries'], axis=1)
train.shape,test.shape
for i, e in enumerate(train['spoken_languages'][:5]):

    print(i, e)
print('Number of spoken languages in films')

train['spoken_languages'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_languages = list(train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_languages for i in j]).most_common(15)
train['num_languages'] = train['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

train['all_languages'] = train['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_languages = [m[0] for m in Counter([i for j in list_of_languages for i in j]).most_common(30)]

for g in top_languages:

    train['language_' + g] = train['all_languages'].apply(lambda x: 1 if g in x else 0)

    

test['num_languages'] = test['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

test['all_languages'] = test['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_languages:

    test['language_' + g] = test['all_languages'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['spoken_languages', 'all_languages'], axis=1)

test = test.drop(['spoken_languages', 'all_languages'], axis=1)
train.shape,test.shape
for i, e in enumerate(train['Keywords'][:5]):

    print(i, e)
print('Number of Keywords in films')

train['Keywords'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)
list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

plt.figure(figsize = (16, 12))

text = ' '.join(['_'.join(i.split(' ')) for j in list_of_keywords for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top keywords')

plt.axis("off")

plt.show()
train['num_Keywords'] = train['Keywords'].apply(lambda x: len(x) if x != {} else 0)

train['all_Keywords'] = train['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]

for g in top_keywords:

    train['keyword_' + g] = train['all_Keywords'].apply(lambda x: 1 if g in x else 0)

    

test['num_Keywords'] = test['Keywords'].apply(lambda x: len(x) if x != {} else 0)

test['all_Keywords'] = test['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_keywords:

    test['keyword_' + g] = test['all_Keywords'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['Keywords', 'all_Keywords'], axis=1)

test = test.drop(['Keywords', 'all_Keywords'], axis=1)
train.shape,test.shape
for i, e in enumerate(train['cast'][:1]):

    print(i, e)
print('Number of casted persons in films')

train['cast'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)
list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_cast_names for i in j]).most_common(15)
list_of_cast_genders = list(train['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_cast_genders for i in j]).most_common()
list_of_cast_characters = list(train['cast'].apply(lambda x: [i['character'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_cast_characters for i in j]).most_common(15)
train['num_cast'] = train['cast'].apply(lambda x: len(x) if x != {} else 0)

top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]

for g in top_cast_names:

    train['cast_name_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)

train['genders_0_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

train['genders_1_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

train['genders_2_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(15)]

for g in top_cast_characters:

    train['cast_character_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)

    

test['num_cast'] = test['cast'].apply(lambda x: len(x) if x != {} else 0)

for g in top_cast_names:

    test['cast_name_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)

test['genders_0_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

test['genders_1_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

test['genders_2_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

for g in top_cast_characters:

    test['cast_character_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)



train = train.drop(['cast'], axis=1)

test = test.drop(['cast'], axis=1)
train.shape,test.shape
for i, e in enumerate(train['crew'][:1]):

    print(i, e[:10])
print('Number of casted persons in films')

train['crew'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)
list_of_crew_names = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_crew_names for i in j]).most_common(15)
list_of_crew_jobs = list(train['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)
list_of_crew_genders = list(train['crew'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_crew_genders for i in j]).most_common(15)
list_of_crew_departments = list(train['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_crew_departments for i in j]).most_common(14)
list_of_crew_names = train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values

Counter([i for j in list_of_crew_names for i in j]).most_common(15)
train['num_crew'] = train['crew'].apply(lambda x: len(x) if x != {} else 0)

top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]

for g in top_crew_names:

    train['crew_name_' + g] = train['crew'].apply(lambda x: 1 if g in str(x) else 0)

train['genders_0_crew'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

train['genders_1_crew'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

train['genders_2_crew'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]



top_crew_jobs = [m[0] for m in Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)]

for j in top_crew_jobs:

    train['jobs_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))

top_crew_departments = [m[0] for m in Counter([i for j in list_of_crew_departments for i in j]).most_common(15)]

for j in top_crew_departments:

    train['departments_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 

    

test['num_crew'] = test['crew'].apply(lambda x: len(x) if x != {} else 0)

for g in top_crew_names:

    test['crew_name_' + g] = test['crew'].apply(lambda x: 1 if g in str(x) else 0)

test['genders_0_crew'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

test['genders_1_crew'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

test['genders_2_crew'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))



for j in top_crew_jobs:

    test['jobs_' + j] = test['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))

for j in top_crew_departments:

    test['departments_' + j] = test['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 



train = train.drop(['crew'], axis=1)

test = test.drop(['crew'], axis=1)
train.shape,test.shape
fig, ax = plt.subplots(figsize = (16, 6))

plt.subplot(1, 2, 1)

plt.hist(train['revenue']);

plt.title('Distribution of revenue');

plt.subplot(1, 2, 2)

plt.hist(np.log1p(train['revenue']));

plt.title('Distribution of log of revenue');
train['log_revenue'] = np.log1p(train['revenue'])
fig, ax = plt.subplots(figsize = (16, 6))

plt.subplot(1, 2, 1)

plt.hist(train['budget']);

plt.title('Distribution of budget');

plt.subplot(1, 2, 2)

plt.hist(np.log1p(train['budget']));

plt.title('Distribution of log of budget');
train['log_budget'] = np.log1p(train['budget'])

test['log_budget'] = np.log1p(test['budget'])
train['has_homepage'] = 0

train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1

test['has_homepage'] = 0

test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1
sns.catplot(x='has_homepage', y='revenue', data=train);

plt.title('Revenue for film with and without homepage');
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

sns.boxplot(x='original_language', y='revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)]);

plt.title('Mean revenue per language');

plt.subplot(1, 2, 2)

sns.boxplot(x='original_language', y='log_revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)]);

plt.title('Mean log revenue per language');
plt.figure(figsize = (12, 12))

text = ' '.join(train['original_title'].values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top words in titles')

plt.axis("off")

plt.show()
plt.figure(figsize = (12, 12))

text = ' '.join(train['overview'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top words in overview')

plt.axis("off")

plt.show()
vectorizer = TfidfVectorizer(

            sublinear_tf=True,

            analyzer='word',

            token_pattern=r'\w{1,}',

            ngram_range=(1, 2),

            min_df=5)



overview_text = vectorizer.fit_transform(train['overview'].fillna(''))

linreg = LinearRegression()

linreg.fit(overview_text, train['log_revenue'])

eli5.show_weights(linreg, vec=vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')
print('Target value:', train['log_revenue'][1000])

eli5.show_prediction(linreg, doc=train['overview'].values[1000], vec=vectorizer)
test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/98'
def fix_date(x):

    """

    Fixes dates which are in 20xx

    """

    year = x.split('/')[2]

    if int(year) <= 19:

        return x[:-2] + '20' + year

    else:

        return x[:-2] + '19' + year
train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))

test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))

train['release_date'] = pd.to_datetime(train['release_date'])

test['release_date'] = pd.to_datetime(test['release_date'])
# creating features based on dates

def process_date(df):

    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']

    for part in date_parts:

        part_col = 'release_date' + "_" + part

        df[part_col] = getattr(df['release_date'].dt, part).astype(int)

    

    return df



train = process_date(train)

test = process_date(test)
sns.catplot(x='release_date_month', y='revenue', data=train);

plt.title('Revenue on different number of week of release');
sns.boxplot(x='has_collection', y='revenue', data=train);
sns.violinplot(x='genre_Drama', y='revenue', data=train[:100]);
f, axes = plt.subplots(3, 5, figsize=(24, 12))

plt.suptitle('Violinplot of revenue vs genres')

for i, e in enumerate([col for col in train.columns if 'genre_' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
train = train.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status', 'log_revenue'], axis=1)

test = test.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status'], axis=1)
for col in train.columns:

    if train[col].nunique() == 1:

        print(col)

        train = train.drop([col], axis=1)

        test = test.drop([col], axis=1)
for col in ['original_language', 'collection_name', 'all_genres']:

    le = LabelEncoder()

    le.fit(list(train[col].fillna('')) + list(test[col].fillna('')))

    train[col] = le.transform(train[col].fillna('').astype(str))

    test[col] = le.transform(test[col].fillna('').astype(str))
train.shape, test.shape
train_texts = train[['title', 'tagline', 'overview', 'original_title']]

test_texts = test[['title', 'tagline', 'overview', 'original_title']]
for col in ['title', 'tagline', 'overview', 'original_title']:

    train['len_' + col] = train[col].fillna('').apply(lambda x: len(str(x)))

    train['words_' + col] = train[col].fillna('').apply(lambda x: len(str(x.split(' '))))

    train = train.drop(col, axis=1)

    test['len_' + col] = test[col].fillna('').apply(lambda x: len(str(x)))

    test['words_' + col] = test[col].fillna('').apply(lambda x: len(str(x.split(' '))))

    test = test.drop(col, axis=1)
# data fixes from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3

train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning

train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          

train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs

train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven

train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 

train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty

train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood

train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II

train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada

train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol

train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip

train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times

train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman

train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   

train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 

train.loc[train['id'] == 1542,'budget'] = 1              # All at Once

train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II

train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp

train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit

train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon

train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget

train.loc[train['id'] == 2491,'revenue'] = 6800000       # Never Talk to Strangers

train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus

train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams

train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D

train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture

test.loc[test['id'] == 3889,'budget'] = 15000000       # Colossal

test.loc[test['id'] == 6733,'budget'] = 5000000        # The Big Sick

test.loc[test['id'] == 3197,'budget'] = 8000000        # High-Rise

test.loc[test['id'] == 6683,'budget'] = 50000000       # The Pink Panther 2

test.loc[test['id'] == 5704,'budget'] = 4300000        # French Connection II

test.loc[test['id'] == 6109,'budget'] = 281756         # Dogtooth

test.loc[test['id'] == 7242,'budget'] = 10000000       # Addams Family Values

test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family

test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage

test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee



power_six = train.id[train.budget > 1000][train.revenue < 100]



for k in power_six :

    train.loc[train['id'] == k,'revenue'] =  train.loc[train['id'] == k,'revenue'] * 1000000
train.shape, test.shape
def new_features(df):

    df['budget_to_popularity'] = df['budget'] / df['popularity']

    df['budget_to_runtime'] = df['budget'] / df['runtime']

    

    # some features from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3

    df['_budget_year_ratio'] = df['budget'] / (df['release_date_year'] * df['release_date_year'])

    df['_releaseYear_popularity_ratio'] = df['release_date_year'] / df['popularity']

    df['_releaseYear_popularity_ratio2'] = df['popularity'] / df['release_date_year']

    

    df['runtime_to_mean_year'] = df['runtime'] / df.groupby("release_date_year")["runtime"].transform('mean')

    df['popularity_to_mean_year'] = df['popularity'] / df.groupby("release_date_year")["popularity"].transform('mean')

    df['budget_to_mean_year'] = df['budget'] / df.groupby("release_date_year")["budget"].transform('mean')

        

    return df
test = new_features(test)

train = new_features(train)
test['runtime'] = test['runtime'].fillna((test['runtime'].mean()))

train['runtime'] = train['runtime'].fillna((test['runtime'].mean()))



all_data = pd.concat((train, test)).reset_index(drop=True)



all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
numeric_feats_test = test.dtypes[test.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats_test = test[numeric_feats_test].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness_test = pd.DataFrame({'Skew' :skewed_feats_test})

skewness_test.head(10)
skewness_test = skewness_test[abs(skewness_test) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness_test.shape[0]))



from scipy.special import boxcox1p

skewed_features_test = skewness_test.index

lam_test = 0.15

for feat_test in skewed_features_test:

    if feat_test !='revenue':

        test[feat_test] = boxcox1p(test[feat_test], lam_test)
numeric_feats_train = train.dtypes[train.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats_train = train[numeric_feats_train].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness_train = pd.DataFrame({'Skew' :skewed_feats_train})

skewness_train.head(10)
skewness_train = skewness_train[abs(skewness_train) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness_train.shape[0]))



from scipy.special import boxcox1p

skewed_features_train = skewness_train.index

lam_train = 0.15

for feat_train in skewed_features_train:

    if feat_train !='revenue':

        train[feat_train] = boxcox1p(train[feat_train], lam_train)
X = train.drop(['id', 'revenue'], axis=1)

y = np.log1p(train['revenue']) # do not forget to un-log in the end: sub['revenue'] = np.expm1(prediction_lgb)

X_testfile = test.drop(['id'], axis=1)
test.shape,train.shape
from sklearn.model_selection import KFold, cross_val_score, train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape,X_testfile.shape
import lightgbm as lgb

params = {'num_leaves': 30,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 5,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2,

         "verbosity": -1}

model_lgb = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

model_lgb.fit(X_train, y_train

        ,eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)
eli5.show_weights(model_lgb, feature_filter=lambda x: x != '<BIAS>')
prediction_lgb_train=model_lgb.predict(X_train)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



#have to use np.expm1() as y was np.log1p() earlier

print(rmsle(y_train, prediction_lgb_train))
prediction_lgb_test=model_lgb.predict(X_test)
print(rmsle(y_test, prediction_lgb_test))
prediction_lgb_testfile=model_lgb.predict(X_testfile)
sub = pd.read_csv('../input/sample_submission.csv')

sub['revenue'] = np.expm1(prediction_lgb_testfile)

sub.to_csv("lgb.csv", index=False)