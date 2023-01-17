import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.colors

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import seaborn as sns

sns.set_style("whitegrid", {'axes.grid' : False})



! pip install -q country_converter

import plotly.express as px

import country_converter as co



from nltk.stem import WordNetLemmatizer



df = pd.read_csv('../input/ramen-ratings/ramen-ratings.csv')
df.head()
# the desired rate of null values per col

nulls_per_col = df.isna().sum(axis=0) / len(df.index)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 5))

    

nulls_per_col.plot(kind='bar', color='steelblue', x=nulls_per_col.values, y=nulls_per_col.index, ax=ax, 

                       width=1, linewidth=1, align='edge', edgecolor='steelblue', label='Null value rate')

    

    

# centered labels

labels=df.columns

ticks = np.arange(0.5, len(labels))

ax.xaxis.set(ticks=ticks, ticklabels=labels)



# workaround to visualize very small amounts of null values per col

na_ticks = ticks[(nulls_per_col > 0) & (nulls_per_col < 0.05)]

if (len(na_ticks) > 0):

    ax.plot(na_ticks, [0,]*len(na_ticks), 's', c='steelblue', markersize=10, 

            label='Very few missing values')

    



ax.set_ylim((0,1))

ax.legend()

fig.suptitle('Null Value Rate per Column', fontsize=30, y=1.05)

fig.tight_layout()
df.describe(include='all')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,12))

styles_oc = df['Style'].value_counts()

styles_oc['others'] = styles_oc[-4:].sum()

styles_oc = styles_oc.drop(labels=styles_oc.index[-5:-1])

styles_oc.plot(kind='pie', ax=ax, colormap='cividis', rotatelabels = 270)

plt.show()
df['Style'] = df['Style'].fillna(value='Pack')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,12))

# yea.. seems like some errors happened during data collection, that's why we have to drop the '\n's 

df['Top Ten'] = df['Top Ten'].replace(to_replace='\n', value=None)

styles_oc = df['Top Ten'].value_counts()

styles_oc.plot(kind='pie', cmap='twilight', rotatelabels = 270, labeldistance= 1.01, ax=ax)

plt.show()
# convert to ranks

df['Top Ten'] = df['Top Ten'].str.slice(start=6)

df['Top Ten'] = df['Top Ten'].fillna('0')



# convert rank strings to integers

df['Top Ten'] = df['Top Ten'].astype(np.int8)

df
country_raw = df['Country']

# this might take a while

country_converted = country_raw.apply(lambda x: co.convert(x, to='ISO3'))
not_found = country_raw[country_converted=='not found'].unique()

print(f'{not_found} haven\'t been converted automatically')
def convert_man(country):

    '''

    country: a string which should be a country name

    

    replaces 'country' by its ISO3 representation, if

    country wasn't already converted automatically.

    '''

    if country=='UK':

        return 'GBR'

    if country=='Dubai':

        return 'ARE'

    if country=='Holland':

        return 'NLD'

    if country=='Sarawak':

        return 'MY'

    else: 

        return country

        

    

    

raw_fixed = country_raw.apply(convert_man)

converted_fixed = raw_fixed.apply(lambda x: co.convert(x, to='ISO3'))

df['Country'] = converted_fixed
# rev:=reviewers

# co:=countries

# get reviewers per country

rev_co = df.groupby('Country').count()['Review #']

rev_co = [pd.Series(rev_co.values, name='reviewers'), pd.Series(rev_co.index, name='country')]

rev_co_df = pd.concat(rev_co, axis=1)

rev_co_df = rev_co_df.groupby(by='country', axis=0, as_index=False).sum()
fig = px.choropleth(rev_co_df, locations="country",

                    color="reviewers",

                    color_continuous_scale='oranges',

                    title="Reviewers per Country",

                   )

fig.show()
df['Variety'] = df['Variety'].str.lower()

descriptions = ' '.join(df['Variety'])

# add some custom stopwords:

custom_stop_words = ['noodle', 'soup', 'instant', 'flavor', 'flavour'] 

stop_words = list(STOPWORDS) + custom_stop_words



#plot the wordcloud

plt.figure(figsize=(15,10))

wordcloud_ramen = WordCloud(stopwords=stop_words, 

                            max_font_size=80, max_words=160, 

                            width=600, height=400, 

                            colormap='inferno', background_color='white'

                           ).generate(descriptions)

plt.imshow(wordcloud_ramen, interpolation='bilinear')

plt.axis('off')

plt.savefig('wordcloud.png')

plt.show()
wnl = WordNetLemmatizer()

descriptions_splitted = pd.Series(descriptions.split())

descriptions_lemmatized = descriptions_splitted.apply(wnl.lemmatize)



# for simplicity, let's hardcode the initial number of new features as 20

common = descriptions_lemmatized.value_counts()[:20]





# we don't want to have any stopwords as features

custom_stops = ['flavour', 'flavor', 'cup', 'soup', '&']

stopwords = custom_stops + list(STOPWORDS)

stop_labels = common[np.in1d(common.index, stopwords)].index

common = common.drop(labels=stop_labels)

common.index
new_cols = common.index

temp = np.empty([len(df.index), len(new_cols)], dtype=np.int8)



def fill(row):

    '''

    row: a row of the dataframe

    

    stores whether the 'Variety' of the row contains the (lemmatized) keywords.

    '''

    # we have to lemmatize each row, since we lemmatized the names of the new columns

    row_lemmatized = pd.Series(row['Variety'].split()).apply(wnl.lemmatize)

    temp[row.name] = np.in1d(new_cols, row_lemmatized)



    



# rowise:

df.apply(fill, axis=1)



new_cols = pd.DataFrame(temp, columns=common.index)

df_augmented = pd.concat([df, new_cols], axis=1)

df_augmented.head()
# we don't have to factorize all columns

df_cat = df_augmented[['Brand', 'Variety', 'Style', 'Country']]

df_done = df_augmented.drop(columns=df_cat.columns)





# factorize the columns of one df and reunite both dataframes

df_factorized = df_cat.copy().apply(lambda x: pd.factorize(x)[0])

df_cat.columns = df_cat.columns + '_cat'

df_augmented = pd.concat([df_factorized, df_done], axis=1)

# for better interpretability, we should save the encoding.

# This allows us to decode the columns after applying our 

# machine learning models.

df_encoding = pd.concat([df_cat, df_factorized], axis=1)
df_augmented.columns = df_augmented.columns.str.lower()



df_augmented