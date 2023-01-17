%matplotlib inline

import re

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud, ImageColorGenerator
hm_data = pd.read_csv('../input/happydb/cleaned_hm.csv')

hm_data.head()
hm_data.loc[hm_data["reflection_period"]=='3m'].head()
df_hm = hm_data[hm_data['cleaned_hm'].notnull()]

len_count = df_hm['cleaned_hm'].apply(lambda x: len(x.split()))

len_count.describe()
df_hm[df_hm['cleaned_hm'].apply(lambda x: len(x.split()))==2].head()
length_order = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", \

                "40-44", "45-49", ">=50"]

length_category = len_count.apply(lambda x: length_order[min(10, int(x/5))])

length_counts = pd.DataFrame(length_category.value_counts()).reset_index()

length_counts.columns = ['word numbers', '# of moments']



sns.barplot(x='word numbers', y='# of moments', data=length_counts, order=length_order)
text = ' '.join(df_hm['cleaned_hm'].tolist())

text = text.lower()

wordcloud = WordCloud(background_color="white", height=2700, width=3600).generate(text)

plt.figure( figsize=(14,8) )

plt.imshow(wordcloud.recolor(colormap=plt.get_cmap('Set2')), interpolation='bilinear')

plt.axis("off")
LIMIT_WORDS = ['happy', 'day', 'got', 'went', 'today', 'made', 'one', 'two', 'time', 'last', 'first', 'going', 'getting', 'took', 'found', 'lot', 'really', 'saw', 'see', 'month', 'week', 'day', 'yesterday', 'year', 'ago', 'now', 'still', 'since', 'something', 'great', 'good', 'long', 'thing', 'toi', 'without', 'yesteri', '2s', 'toand', 'ing']



text = ' '.join(df_hm['cleaned_hm'].tolist())

text = text.lower()

for w in LIMIT_WORDS:

    text = text.replace(' ' + w, '')

    text = text.replace(w + ' ', '')

wordcloud = WordCloud(background_color="white", height=2700, width=3600).generate(text)

plt.figure( figsize=(14,8) )

plt.imshow(wordcloud.recolor(colormap=plt.get_cmap('Set2')), interpolation='bilinear')

plt.axis("off")
seasons = ['Spring', 'Summer', 'Fall', 'Winter']



# Check each moment, and increase the count for the mentioned season

season_dic = dict((x,0) for x in seasons)

tokens_hm = df_hm['cleaned_hm'].apply(lambda x: x.split())

for _, value in tokens_hm.iteritems():

    for word in value:

        if word in seasons:

            season_dic[word] += 1

            

season_dic
# Read the happyDB sample file

with open('happydb.txt', 'w') as ofile:

    len = int(df_hm.shape[0] / 8)

    for i in range(0, len - 1):

        ofile.write(df_hm['cleaned_hm'].iloc[i] + '\n')

        

print("Happy moments are retrieved!")
with open('../input/kokosamples/purchase_v1.koko', 'r') as file:

    print(file.read())
import koko

import spacy



koko.run('../input/kokosamples/purchase_v1.koko', doc_parser='spacy')
with open('../input/kokosamples/purchase_v2.koko', 'r') as file:

    print(file.read())
koko.run('../input/kokosamples/purchase_v2.koko', doc_parser='spacy')
# Select all happy moments used for entity extraction that contain 'purchased'.

df_hmee = df_hm[:int(df_hm.shape[0]/8)]

df_purchase = df_hmee[df_hmee['cleaned_hm'].apply(lambda x: x.find('purchased') != -1)]



print("Number of happy moments containing 'purchased': {}".format(df_purchase.shape[0]))
with open('../input/kokosamples/purchase_v3.koko', 'r') as file:

    print(file.read())
embedding_doc = "../input/glove840b/glove.840B.300d.txt"

koko.run('../input/kokosamples/purchase_v3.koko', doc_parser='spacy', embedding_file=embedding_doc)
demo_data = pd.read_csv('../input/happydb/demographic.csv')



demo_data.head()
merge_data = pd.merge(hm_data, demo_data, on='wid')

gender_data = merge_data[['cleaned_hm', 'gender']]



gender_data.head()
gender_data.gender.value_counts().plot(kind='bar')
gender_bin_data = gender_data[(gender_data['gender'] == 'm') | (gender_data['gender'] == 'f')]



print("Happy moments written by male/female: {}".format(gender_bin_data['cleaned_hm'].size))
gender_bin_data = gender_bin_data.assign(gender_bin=(np.where(gender_bin_data['gender']=='m', 1, 0)))



gender_bin_data.head()
hm_size = gender_bin_data['cleaned_hm'].size

num_train_hm = int(0.7 * gender_bin_data['cleaned_hm'].size)



train_hm = gender_bin_data.iloc[0:num_train_hm]

test_hm = gender_bin_data.iloc[num_train_hm:hm_size]

test_hm = test_hm.reset_index(drop=True)



test_hm.head()
def clean_up_texts(hm_data):

    prepro_hm = []

    stops = set(stopwords.words("english"))

    for i in range(0, hm_data['cleaned_hm'].size):

        # Remove non-english words, including punctuations and numbers

        letters = re.sub("[^a-zA-Z]", " ", hm_data.iloc[i]['cleaned_hm'])



        # Convert all words to lower case

        lower_words = letters.lower()



        # Tokenize the sentences

        tokens = lower_words.split()



        # Reconstruct the processed tokens into a string

        prepro_string = " ".join(tokens)



        prepro_hm.append(prepro_string)

        

    return prepro_hm

    

prepro_train = clean_up_texts(train_hm)

prepro_test = clean_up_texts(test_hm)

print("Texts cleaned up! \n")
prepro_train[:10]
vectorizer = CountVectorizer()

features_train_hm = vectorizer.fit_transform(prepro_train)

train_array_hm = features_train_hm.toarray()



print("Dimension of the training data: {}".format(train_array_hm.shape))
vocab = vectorizer.get_feature_names()



vocab[:20]
from sklearn.linear_model import LogisticRegression



logi_model = LogisticRegression()

logi_model.fit(train_array_hm, train_hm['gender_bin'])



logi_model.score(train_array_hm, train_hm['gender_bin'])
feature_names = vocab

coefficients = logi_model.coef_.tolist()[0]

weight_df = pd.DataFrame({'Word': feature_names,

                          'Coeff': coefficients})

weight_df = weight_df.sort_values(['Coeff', 'Word'], ascending=[0, 1])

weight_df.head(n=10)
weight_df.tail(n=10)
features_test_hm = vectorizer.transform(prepro_test)

test_array_hm = features_test_hm.toarray()



print("Dimension of the test data: {}".format(test_array_hm.shape))
predictions = logi_model.predict(test_array_hm)
from sklearn import metrics



print(metrics.accuracy_score(test_hm['gender_bin'], predictions))