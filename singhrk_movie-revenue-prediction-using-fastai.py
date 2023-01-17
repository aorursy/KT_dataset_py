# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Fastai Library

from fastai import *

from fastai.tabular import *



# visualization library

from wordcloud import WordCloud

import seaborn as sb

sb.set(rc={'figure.figsize':(11.7,8.27)})
df_train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

df_test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.expand_frame_repr', True)
df_train.head(1).T
df_train.isnull().sum()
df_test.isnull().sum()
df_train.info()
df_train.columns = df_train.columns.str.strip().str.lower().str.replace(' ', '_')

df_test.columns = df_test.columns.str.strip().str.lower().str.replace(' ', '_')
df_train.revenue.min()
df_train.budget.min()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

sb.distplot(df_train.revenue, ax=ax1)

sb.distplot(np.log1p(df_train.revenue), ax=ax2)

ax1.set_title('Distribution of revenue')

ax2.set_title('Distribution of log of revenue')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

sb.distplot(df_train.budget, ax=ax1)

sb.distplot(np.log1p(df_train.budget), ax=ax2)

ax1.set_title('Distribution of Budget')

ax2.set_title('Distribution of log of Budget')
df_train['log_revenue'] = np.log1p(df_train.revenue)

df_train['log_budget'] = np.log1p(df_train.budget)



df_test['log_budget'] = np.log1p(df_test.budget)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

sb.distplot(df_train.popularity, ax=ax1)

sb.distplot(np.log1p(df_train.popularity), ax=ax2)

ax1.set_title('Distribution of popularity')

ax2.set_title('Distribution of log of popularity')
df_train.fillna({'runtime': df_train['runtime'].mean()}, inplace=True)

df_test.fillna({'runtime': df_test['runtime'].mean()}, inplace=True)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

sb.distplot(df_train.runtime, ax=ax1)

sb.distplot(np.log1p(df_train.runtime), ax=ax2)

ax1.set_title('Distribution of runtime')

ax2.set_title('Distribution of log of runtime')
df_train['log_popularity'] = np.log1p(df_train.popularity)

df_train['log_runtime'] = np.log1p(df_train.runtime)



df_test['log_popularity'] = np.log1p(df_test.popularity)

df_test['log_runtime'] = np.log1p(df_test.runtime)
nan_percentage = pd.DataFrame({'ColName':df_train.columns, 'NaN%':df_train.isnull().mean()})

plt.figure(figsize=(16, 8))

chart = sb.barplot(x=nan_percentage['ColName'], y=nan_percentage['NaN%'])

plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
import ast

for df in (df_train, df_test):

    df.belongs_to_collection = df.belongs_to_collection.apply(lambda x: x if pd.isna(x) else ast.literal_eval(x))

    df.genres = df.genres.apply(lambda x: x if pd.isna(x) else ast.literal_eval(x))

    df.production_companies = df.production_companies.apply(lambda x: x if pd.isna(x) else ast.literal_eval(x))

    df.production_countries = df.production_countries.apply(lambda x: x if pd.isna(x) else ast.literal_eval(x))

    df.spoken_languages = df.spoken_languages.apply(lambda x: x if pd.isna(x) else ast.literal_eval(x))

    df.keywords = df.keywords.apply(lambda x: x if pd.isna(x) else ast.literal_eval(x))

    df.cast = df.cast.apply(lambda x: x if pd.isna(x) else ast.literal_eval(x))

    df.crew = df.crew.apply(lambda x: x if pd.isna(x) else ast.literal_eval(x))
# df_train.belongs_to_collection.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts()
df_train.genres.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts()
sb.barplot(x=df_train.genres.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts(), y=df_train.revenue)
# df_train.production_companies.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts()
sb.barplot(x=df_train.production_companies.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts(), y=df_train.revenue)
# df_train.production_countries.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts()
sb.barplot(x=df_train.production_countries.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts(), y=df_train.revenue)
# df_train.spoken_languages.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts()
sb.barplot(x=df_train.spoken_languages.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts(), y=df_train.revenue)
# df_train.keywords.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts()
sb.barplot(x=df_train.keywords.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts(), y=df_train.revenue)
# df_train.cast.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts()
sb.barplot(x=df_train.cast.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts(), y=df_train.revenue)
# df_train.crew.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts()
sb.barplot(x=df_train.crew.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0).value_counts(), y=df_train.revenue)
def extract_features(df, col):

    #     Separate features into  individual features

    features = set()

    for f in df[col]:

        if np.any(pd.notna(f)):

            for x in range(len(f)):

                features.add(str(f[x]['name']))

            

    return features

    

def create_features(df, col, features):

    for f in features:

        df[col+'_'+f]=0

        

    for index, f in enumerate(df[col]):

        if np.any(pd.notna(f)):

            for x in range(len(f)):

                if f[x]['name'] in features:

                    df.loc[index, col+'_'+f[x]['name']] = 1
f_train = extract_features(df_train, 'genres')

f_test = extract_features(df_test, 'genres')

len(f_train), len(f_test)
f_train - f_test
for df in (df_train, df_test):

    # Create features from belongs to collection

    # df['collection_name'] = df.belongs_to_collection.apply(lambda x: x[0]['name'] if np.any(pd.notna(x)) else x)

    df['is_series'] = df.belongs_to_collection.apply(lambda x: 1 if np.any(pd.notna(x)) else 0)

    

    # Create features for individual genres

    create_features(df, 'genres', f_train)

    df['total_genres'] = df.genres.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0)

    

    

    # Create features for individual production_companies

    # create_features(df, 'production_companies')

    df['total_production_companies'] = df.production_companies.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0)

    

    # Create features for individual production_countries

    # create_features(df, 'production_countries')

    df['total_production_countries'] = df.production_countries.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0)

    

    # Create features for individual spoken_languages

    # create_features(df, 'spoken_languages')

    df['total_spoken_languages'] = df.spoken_languages.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0)

    

    # Create features for individual keywords

    # create_features(df, 'keywords')

    df['total_keywords'] = df.keywords.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0)

    

    # Create features for individual cast

    # create_features(df, 'cast')

    df['total_cast'] = df.cast.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0)

    

    # Create features for individual crew

    # create_features(df, 'crew')

    df['total_crew'] = df.crew.apply(lambda x: len(x) if np.any(pd.notna(x)) else 0)

    

    # Create feature from date

    add_datepart(df, 'release_date')
fig, ax = plt.subplots(4, 5, figsize=(16, 16))

sb.barplot(df_train.genres_Action, df_train.revenue, ax=ax[0,0])

sb.barplot(df_train.genres_Adventure, df_train.revenue, ax=ax[0,1])

sb.barplot(df_train.genres_Animation, df_train.revenue, ax=ax[0,2])

sb.barplot(df_train.genres_Comedy, df_train.revenue, ax=ax[0,3])

sb.barplot(df_train.genres_Crime, df_train.revenue, ax=ax[0,4])

sb.barplot(df_train.genres_Documentary, df_train.revenue, ax=ax[1,0])

sb.barplot(df_train.genres_Drama, df_train.revenue, ax=ax[1,1])

sb.barplot(df_train.genres_Family, df_train.revenue, ax=ax[1,2])

sb.barplot(df_train.genres_Fantasy, df_train.revenue, ax=ax[1,3])

sb.barplot(df_train.genres_Foreign, df_train.revenue, ax=ax[1,4])

sb.barplot(df_train.genres_History, df_train.revenue, ax=ax[2,0])

sb.barplot(df_train.genres_Horror, df_train.revenue, ax=ax[2,1])

sb.barplot(df_train.genres_Music, df_train.revenue, ax=ax[2,2])

sb.barplot(df_train.genres_Mystery, df_train.revenue, ax=ax[2,3])

sb.barplot(df_train.genres_Romance, df_train.revenue, ax=ax[2,4])

sb.barplot(df_train.genres_Thriller, df_train.revenue, ax=ax[3,0])

sb.barplot(df_train.genres_War, df_train.revenue, ax=ax[3,1])

sb.barplot(df_train.genres_Western, df_train.revenue, ax=ax[3,2])
# df_train.production_companies.apply(lambda x: print(x))
# df_train.production_companies.apply(lambda x: ' ' if np.all(pd.isna(x)) else ','.join((map(lambda y: y['name'], x))).replace(' ', '_'))
prod_comp = ','.join(df_train.production_companies.apply(lambda x: 'NaN' if np.all(pd.isna(x)) else ','.join((map(lambda y: y['name'], x))).replace(' ', '_')))

# Create and generate a word cloud image:

wordcloud = WordCloud(background_color="white").generate(prod_comp)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
filtered_words = [word for word in prod_comp.split(',')]

counted_words = collections.Counter(filtered_words)



words = []

counts = []

for letter, count in counted_words.most_common(20):

    words.append(letter)

    counts.append(count)

    



sb.barplot(x=counts, y=words)
df_train['prod_wb'] = df_train.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Warner Bros.' in list((map(lambda y: y['name'], x))) else 0)

df_train['prod_up'] = df_train.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Universal Pictures' in list((map(lambda y: y['name'], x))) else 0)

df_train['prod_pp'] = df_train.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Paramount Pictures' in list((map(lambda y: y['name'], x))) else 0)

df_train['prod_tcffc'] = df_train.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Twentieth Century Fox Film Corporation' in list((map(lambda y: y['name'], x))) else 0)

df_train['prod_cp'] = df_train.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Columbia Pictures' in list((map(lambda y: y['name'], x))) else 0)

df_train['prod_mgm'] = df_train.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Metro-Goldwyn-Mayer (MGM)' in list((map(lambda y: y['name'], x))) else 0)

df_train['prod_nlc'] = df_train.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'New Line Cinema' in list((map(lambda y: y['name'], x))) else 0)

df_train['prod_tp'] = df_train.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Touchstone Pictures' in list((map(lambda y: y['name'], x))) else 0)

df_train['prod_wdp'] = df_train.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Walt Disney Pictures' in list((map(lambda y: y['name'], x))) else 0)

df_train['prod_cpc'] = df_train.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Columbia Pictures Corporation' in list((map(lambda y: y['name'], x))) else 0)



df_train['prod_wb'].value_counts(), df_train['prod_up'].value_counts(), df_train['prod_pp'].value_counts(), df_train['prod_tcffc'].value_counts(), df_train['prod_cp'].value_counts(), df_train['prod_mgm'].value_counts(), df_train['prod_nlc'].value_counts(), df_train['prod_tp'].value_counts(), df_train['prod_wdp'].value_counts(), df_train['prod_cpc'].value_counts()
df_test['prod_wb'] = df_test.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Warner Bros.' in list((map(lambda y: y['name'], x))) else 0)

df_test['prod_up'] = df_test.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Universal Pictures' in list((map(lambda y: y['name'], x))) else 0)

df_test['prod_pp'] = df_test.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Paramount Pictures' in list((map(lambda y: y['name'], x))) else 0)

df_test['prod_tcffc'] = df_test.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Twentieth Century Fox Film Corporation' in list((map(lambda y: y['name'], x))) else 0)

df_test['prod_cp'] = df_test.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Columbia Pictures' in list((map(lambda y: y['name'], x))) else 0)

df_test['prod_mgm'] = df_test.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Metro-Goldwyn-Mayer (MGM)' in list((map(lambda y: y['name'], x))) else 0)

df_test['prod_nlc'] = df_test.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'New Line Cinema' in list((map(lambda y: y['name'], x))) else 0)

df_test['prod_tp'] = df_test.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Touchstone Pictures' in list((map(lambda y: y['name'], x))) else 0)

df_test['prod_wdp'] = df_test.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Walt Disney Pictures' in list((map(lambda y: y['name'], x))) else 0)

df_test['prod_cpc'] = df_test.production_companies.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'Columbia Pictures Corporation' in list((map(lambda y: y['name'], x))) else 0)
fig, ax = plt.subplots(2, 5, figsize=(16, 8))

sb.barplot(df_train.prod_wb, df_train.revenue, ax=ax[0,0])

sb.barplot(df_train.prod_up, df_train.revenue, ax=ax[0,1])

sb.barplot(df_train.prod_pp, df_train.revenue, ax=ax[0,2])

sb.barplot(df_train.prod_tcffc, df_train.revenue, ax=ax[0,3])

sb.barplot(df_train.prod_cp, df_train.revenue, ax=ax[0,4])

sb.barplot(df_train.prod_mgm, df_train.revenue, ax=ax[1,0])

sb.barplot(df_train.prod_nlc, df_train.revenue, ax=ax[1,1])

sb.barplot(df_train.prod_tp, df_train.revenue, ax=ax[1,2])

sb.barplot(df_train.prod_wdp, df_train.revenue, ax=ax[1,3])

sb.barplot(df_train.prod_cpc, df_train.revenue, ax=ax[1,4])
lang = ','.join(df_train.spoken_languages.apply(lambda x: '' if np.any(pd.isna(x)) else ','.join((map(lambda y: y['name'], x)))))

# Create and generate a word cloud image:

wordcloud = WordCloud(background_color="white").generate(lang)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
filtered_words = [word for word in lang.split(',')]

counted_words = collections.Counter(filtered_words)



words = []

counts = []

for letter, count in counted_words.most_common(20):

    words.append(letter)

    counts.append(count)

    

sb.barplot(x=counts, y=words)
# (df_train.keywords.apply(lambda x: '' if np.all(pd.isna(x)) else ','.join((map(lambda y: y['name'], x)))))
keys = ','.join(df_train.keywords.apply(lambda x: 'NaN' if np.all(pd.isna(x)) else ','.join((map(lambda y: y['name'], x)))))

# Create and generate a word cloud image:

wordcloud = WordCloud(background_color="white").generate(keys)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
filtered_words = [word for word in keys.split(',')]

counted_words = collections.Counter(filtered_words)



words = []

counts = []

for letter, count in counted_words.most_common(10):

    words.append(letter)

    counts.append(count)

    

sb.barplot(x=counts, y=words)
df_train['key_women'] = df_train.keywords.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'woman director' in list((map(lambda y: y['name'], x))) else 0)

df_train['key_independent'] = df_train.keywords.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'independent film' in list((map(lambda y: y['name'], x))) else 0)

df_train['key_credit'] = df_train.keywords.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'duringcreditsstinger' in list((map(lambda y: y['name'], x))) else 0)

df_train['key_murder'] = df_train.keywords.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'murder' in list((map(lambda y: y['name'], x))) else 0)

df_train['key_novel'] = df_train.keywords.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'based on novel' in list((map(lambda y: y['name'], x))) else 0)



df_train.key_women.value_counts(), df_train.key_independent.value_counts(), df_train.key_murder.value_counts(), df_train.key_credit.value_counts(), df_train.key_novel.value_counts()
fig, ax = plt.subplots(1, 5, figsize=(16, 4))

sb.barplot(df_train.key_women, df_train.revenue, ax=ax[0])

sb.barplot(df_train.key_independent, df_train.revenue, ax=ax[1])

sb.barplot(df_train.key_credit, df_train.revenue, ax=ax[2])

sb.barplot(df_train.key_murder, df_train.revenue, ax=ax[3])

sb.barplot(df_train.key_novel, df_train.revenue, ax=ax[4])
df_test['key_women'] = df_test.keywords.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'woman director' in list((map(lambda y: y['name'], x))) else 0)

df_test['key_independent'] = df_test.keywords.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'independent film' in list((map(lambda y: y['name'], x))) else 0)

df_test['key_credit'] = df_test.keywords.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'duringcreditsstinger' in list((map(lambda y: y['name'], x))) else 0)

df_test['key_murder'] = df_test.keywords.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'murder' in list((map(lambda y: y['name'], x))) else 0)

df_test['key_novel'] = df_test.keywords.apply(lambda x: x if np.all(pd.isna(x)) else 1 if 'based on novel' in list((map(lambda y: y['name'], x))) else 0)
df_train.columns = df_train.columns.str.strip().str.lower().str.replace(' ', '_')

df_test.columns = df_test.columns.str.strip().str.lower().str.replace(' ', '_')
len(df_train.columns), len(df_test.columns)
for df in (df_train, df_test):

    print('='*10)

    for column in df:

        if (df[column].apply(lambda x: True if np.any(pd.isna(x)) else False).max()): print(column)
df_train.columns
cat_names = ['original_language', 'status',

             'is_series', 'genres_war', 'genres_thriller', 'genres_science_fiction',

             'genres_mystery', 'genres_foreign', 'genres_tv_movie', 'genres_western',

             'genres_drama', 'genres_crime', 'genres_fantasy', 'genres_romance',

             'genres_adventure', 'genres_family', 'genres_comedy',

             'genres_documentary', 'genres_history', 'genres_music', 'genres_action',

             'genres_animation', 'genres_horror', 'total_genres',

             'total_production_companies', 'total_production_countries',

             'total_spoken_languages', 'total_keywords', 'total_cast', 'total_crew',

             'release_year', 'release_month', 'release_week', 'release_day',

             'release_dayofweek', 'release_dayofyear', 'release_is_month_end',

             'release_is_month_start', 'release_is_quarter_end',

             'release_is_quarter_start', 'release_is_year_end',

             'release_is_year_start', 'release_elapsed', 'prod_wb', 'prod_up',

             'prod_pp', 'prod_tcffc', 'prod_cp', 'prod_mgm', 'prod_nlc', 'prod_tp',

             'prod_wdp', 'prod_cpc', 'key_women', 'key_independent', 'key_credit',

             'key_murder', 'key_novel'

            ]

cont_names = ['log_budget', 'log_popularity', 'log_runtime']

dep_var = 'log_revenue'

procs = [Categorify, FillMissing, Normalize]

for c in cont_names:

    df_train[cont_names] = df_train[cont_names].fillna(0).astype('float32')

    df_test[cont_names] = df_test[cont_names].fillna(0).astype('float32')
db_test = TabularList.from_df(df_test, cat_names=cat_names, cont_names=cont_names, procs=procs)
db_train = (TabularList.from_df(df_train, cat_names=cat_names, cont_names=cont_names, procs=procs)

            .split_by_idx(list(range(700)))

            .label_from_df(cols=dep_var)

            .add_test(db_test, label = 0)

            .databunch())
#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)

# def rmsle(y, y_pred):

#     assert len(y) == len(y_pred)

#     terms_to_sum = [(np.log(y_pred[i] + 1) - np.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

#     return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5



def rmsle(y, y_pred):

#     print(len(y), len(y_pred))

    sum=0.0

    assert len(y) == len(y_pred)

    for x in range(len(y_pred)):

#         print(y[x], y_pred[x])

        if y_pred[x]<0 or y[x]<0: #check for negative values

            continue

        p = np.log(y_pred[x]+1)

        r = np.log(y[x]+1)

        sum = sum + (p - r)**2

    return (sum/len(y_pred))**0.5
learn = tabular_learner(db_train, layers=[200, 100], metrics=rmsle)
learn.lr_find()

learn.recorder.plot()
learn.fit(5)
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
learn.save('TMDB')

learn.load('TMDB')
predictions, _ = learn.get_preds(DatasetType.Test)

predictions = np.exp(predictions) - 1
pred = list()

for each in predictions.data.tolist():

    pred.append(each[0])
pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')
submission_df = pd.DataFrame({'id': df_test['id'], 'revenue': pred})

submission_df.to_csv('submission.csv', index=False)
from IPython.display import HTML

import base64



def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = f'<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    return HTML(html)



create_download_link(submission_df)