import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import re
df_reviews = pd.read_json('../input/IMDB_reviews.json', lines=True)
df_details = pd.read_json('../input/IMDB_movie_details.json', lines=True)
print('User reviews shape: ', df_reviews.shape)

print('Movie details shape: ', df_details.shape)
df_reviews.head()
df_details.head()
print('Unique films in reviews dataset:', df_reviews['movie_id'].nunique())
reviews_meta = pd.DataFrame()

reviews_meta['is_spoiler'] = df_reviews['is_spoiler']

reviews_meta['has_word_twist'] = df_reviews['review_text'].apply(lambda text: 1 if 'TWIST' in text.upper() else 0)

reviews_meta['has_word_then'] = df_reviews['review_text'].apply(lambda text: 1 if 'THEN' in text.upper() else 0)

reviews_meta['has_words_twist_then'] = reviews_meta['has_word_twist'] & reviews_meta['has_word_then']

reviews_meta['has_word_spoiler'] = df_reviews['review_text'].apply(lambda text: 1 if 'SPOILER' in text.upper() else 0)
pie1 = reviews_meta['is_spoiler'].value_counts().reset_index().sort_values(by='index')

pie2 = reviews_meta[reviews_meta['has_word_twist'] == 1]['is_spoiler'].value_counts().reset_index().sort_values(by='index')

pie3 = reviews_meta[reviews_meta['has_words_twist_then'] == 1]['is_spoiler'].value_counts().reset_index().sort_values(by='index')

pie4 = reviews_meta[reviews_meta['has_word_spoiler'] == 1]['is_spoiler'].value_counts().reset_index().sort_values(by='index')



with plt.style.context('seaborn-talk'):

    fig = plt.figure(figsize=(16, 16))



    ax1 = fig.add_subplot(2, 2, 1)

    ax2 = fig.add_subplot(2, 2, 2)

    ax3 = fig.add_subplot(2, 2, 3)

    ax4 = fig.add_subplot(2, 2, 4)



    ax1.pie(pie1['is_spoiler'])

    ax1.set_title('All reviews')



    ax2.pie(pie2['is_spoiler'])

    ax2.set_title('Reviews containing the word \'twist\'')



    ax3.pie(pie3['is_spoiler'])

    ax3.set_title('Reviews containing the words \'twist\' and \'then\'')



    ax4.pie(pie4['is_spoiler'])

    ax4.set_title('Reviews containing the word \'spoiler\'')



    plt.suptitle('Spoiler distribution within the reviews', fontsize=20)

    fig.legend(labels=['Without spoilers', 'With spoilers'], loc='center')



    plt.show()
reviews_meta['word_count'] = df_reviews['review_text'].apply(lambda text: len(text.split(' ')))
with plt.style.context('seaborn-talk'):

    plt.figure(figsize=(16, 8))

    sns.distplot(reviews_meta[reviews_meta['is_spoiler'] == False]['word_count'], hist=False, label='Without spoilers')

    sns.distplot(reviews_meta[reviews_meta['is_spoiler'] == True]['word_count'], hist=False, label='Containing spoilers')

    plt.legend()

    plt.xlim([-100, 1100])

    plt.xlabel('Word Count')

    plt.title('Distribution according to review length')

    plt.show()
star_wars = df_reviews[df_reviews['movie_id'] == 'tt0080684']

star_wars.is_spoiler.value_counts()
def get_word_frequencies(dataframe, counter_dict):

    for text in dataframe['review_text']:

        text = text.replace('.', ' ')

        text = re.sub('[^A-Za-z0-9 ]+', '', text)

        for word in text.split(' '):

            if word in counter_dict:

                counter_dict[word] += 1

            else:

                counter_dict[word] = 1



def filter_by_word_lenght(dictonary, min_length):

    filtered_dict = {}

    for key, value in dictonary.items():

        if (len(key) >= min_length):

            filtered_dict[key] = value

    return filtered_dict
counter_no_spoiler = {}

counter_spoiler = {}



get_word_frequencies(star_wars[star_wars['is_spoiler'] == False], counter_no_spoiler)

get_word_frequencies(star_wars[star_wars['is_spoiler'] == True], counter_spoiler)



# Filtering by minimum word length

counter_no_spoiler = filter_by_word_lenght(counter_no_spoiler, 5)

counter_spoiler = filter_by_word_lenght(counter_spoiler, 5)
wc_no_spoilers = WordCloud(width=500, height=1000).generate_from_frequencies(counter_no_spoiler)

wc_spoilers = WordCloud(width=500, height=1000).generate_from_frequencies(counter_spoiler)
with plt.style.context('seaborn-talk'):

    fig = plt.figure(figsize=(16, 16))



    ax1 = fig.add_subplot(1, 2, 1)

    ax2 = fig.add_subplot(1, 2, 2)



    ax1.imshow(wc_no_spoilers, interpolation='bilinear')

    ax1.set_title('Wordcloud - Reviews without spoilers')

    ax1.axis('off')



    ax2.imshow(wc_spoilers, interpolation='bilinear')

    ax2.set_title('Wordcloud - Reviews with spoilers')

    ax2.axis('off')



    plt.suptitle('Star Wars: Episode V - The Empire Strikes Back (1980)', fontsize=20)

    plt.subplots_adjust(wspace=0)

    plt.show()
star_wars_meta = pd.DataFrame()

star_wars_meta['is_spoiler'] = star_wars['is_spoiler']

star_wars_meta['has_word_vader'] = star_wars['review_text'].apply(lambda text: 1 if 'VADER' in text.upper() else 0)

star_wars_meta['has_word_father'] = star_wars['review_text'].apply(lambda text: 1 if 'FATHER' in text.upper() else 0)

star_wars_meta['has_word_vader_father'] = star_wars_meta['has_word_vader'] & star_wars_meta['has_word_father']
pie1 = star_wars_meta['is_spoiler'].value_counts().reset_index().sort_values(by='index')

pie2 = star_wars_meta[star_wars_meta['has_word_vader'] == 1]['is_spoiler'].value_counts().reset_index().sort_values(by='index')

pie3 = star_wars_meta[star_wars_meta['has_word_father'] == 1]['is_spoiler'].value_counts().reset_index().sort_values(by='index')

pie4 = star_wars_meta[star_wars_meta['has_word_vader_father'] == 1]['is_spoiler'].value_counts().reset_index().sort_values(by='index')



with plt.style.context('seaborn-talk'):

    fig = plt.figure(figsize=(16, 16))



    ax1 = fig.add_subplot(2, 2, 1)

    ax2 = fig.add_subplot(2, 2, 2)

    ax3 = fig.add_subplot(2, 2, 3)

    ax4 = fig.add_subplot(2, 2, 4)



    ax1.pie(pie1['is_spoiler'])

    ax1.set_title('All reviews')



    ax2.pie(pie2['is_spoiler'])

    ax2.set_title('Reviews containing the word \'Vader\'')



    ax3.pie(pie3['is_spoiler'])

    ax3.set_title('Reviews containing the word \'father\'')



    ax4.pie(pie4['is_spoiler'])

    ax4.set_title('Reviews containing the words \'Vader\' and \'father\'')



    plt.suptitle('Star Wars: Episode V - The Empire Strikes Back (1980)', fontsize=20)

    fig.legend(labels=['Without spoilers', 'With spoilers'], loc='center')



    plt.show()