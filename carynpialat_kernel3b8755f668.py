#!pip install --no-cache-dir --upgrade comet_ml

#from comet_ml import Experiment

#experiment = Experiment(api_key='', project_name='jhb-ss2-classification', workspace='carynpialat')
!pip install emot
# Packages

import numpy as np

import pandas as pd

import re

import spacy

import nltk

from emot.emo_unicode import UNICODE_EMO, EMOTICONS

from io import StringIO

from html.parser import HTMLParser

import string

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import TweetTokenizer

from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.utils import resample



# Visualisations

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud



# Vectorisers

from sklearn.feature_extraction.text import TfidfVectorizer



# Training

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV



# Classification models

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import MultinomialNB, ComplementNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



# Model evaluation

from sklearn.metrics import confusion_matrix, classification_report, f1_score, make_scorer

from sklearn.metrics import accuracy_score, precision_score, recall_score



# Save models

import pickle
nlp = spacy.load('en_core_web_sm')
nltk.download('vader_lexicon')
# Training data

df = pd.read_csv('../input/climate-change-belief-analysis/train.csv')

df.set_index('tweetid', inplace=True)



#Test data

df_test = pd.read_csv('../input/climate-change-belief-analysis/test.csv')

df_test.set_index('tweetid', inplace=True)
# View the training data

df.head()
# View the training data

df_test.head()
# More details on the rows, columns and content of the datasets

print('Training data' + ('\n'))

print(df.info())

print('\n' + 'Shape of the training data: {}' .format(df.shape))

print('Number of unique tweets in the training data: {}'.format(len(set(df['message']))))

print('Number of missing values in the training data:' + '\n' + '{}' .format(df.isnull().sum()))

print('\n\n' + 'Test data' + '\n')

print(df_test.info())

print('\n' + 'Shape of the test data: {}' .format(df_test.shape))

print('Number of unique tweets in the test data: {}'.format(len(set(df_test['message']))))

print('Number of missing values in the test data:' + '\n' + '{}' .format(df_test.isnull().sum()))
# Remove html tags from messages

from io import StringIO

from html.parser import HTMLParser



class MLStripper(HTMLParser):

    def __init__(self):

        super().__init__()

        self.reset()

        self.strict = False

        self.convert_charrefs= True

        self.text = StringIO()

    def handle_data(self, d):

        self.text.write(d)

    def get_data(self):

        return self.text.getvalue()



def strip_tags(html):

    s = MLStripper()

    s.feed(html)

    return s.get_data()



df['msg_clean'] = df['message'].apply(strip_tags)

df_test['msg_clean'] = df_test['message'].apply(strip_tags)
# Remove urls, new lines and hashtags from training data

re_pattern = [r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*']

df['msg_clean'] = df['msg_clean'].replace(to_replace = re_pattern, 

                                          value = r'website', regex = True)

df['msg_clean'] = df['msg_clean'].replace(to_replace = r'\n', 

                                          value = r'', regex = True)

df['msg_clean'] = df['msg_clean'].replace(to_replace = r'@', 

                                          value = r'twitterhandle ', regex = True)



# Remove urls, new lines and hashtags from test data

df_test['msg_clean'] = df_test['msg_clean'].replace(to_replace = re_pattern, 

                                                    value = r'website', regex = True)

df_test['msg_clean'] = df_test['msg_clean'].replace(to_replace = r'\n', 

                                                    value = r'', regex = True)

df_test['msg_clean'] = df_test['msg_clean'].replace(to_replace = r'@', 

                                                    value = r'twitterhandle ', regex = True)
# Converting the emoticons into text 

!pip install emot --upgrade

from emot.emo_unicode import UNICODE_EMO, EMOTICONS

def convert_emojis(text):

    for emot in UNICODE_EMO:

        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").

                                           replace(":","").split()))

        return text

# Function which will change the emoji to a string

def emoji_remove(string):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"  # 

                           u"\U000024C2-\U0001F251"  # 

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', string)



df['msg_clean'] = df['msg_clean'].apply(convert_emojis)

df['msg_clean'] = df['msg_clean'].apply(emoji_remove)



df_test['msg_clean'] = df_test['msg_clean'].apply(convert_emojis)

df_test['msg_clean'] = df_test['msg_clean'].apply(emoji_remove)
# Tokenise tweets

tokenizer = TweetTokenizer()

df['msg_clean'] = df['msg_clean'].apply(tokenizer.tokenize)



df_test['msg_clean'] = df_test['msg_clean'].apply(tokenizer.tokenize)
# Replace contractions with full form of the words

contractions = {

"ain't": "am not / are not",

"aren't": "are not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he had / he would",

"he'd've": "he would have",

"he'll": "he will",

"he'll've": "he will have",

"he's": "he has / he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how has / how is",

"i'd": "I had / I would",

"i'd've": "I would have",

"i'll": "I will",

"i'll've": "I will have",

"i'm": "I am",

"i've": "I have",

"isn't": "is not",

"it'd": "it had / it would",

"it'd've": "it would have",

"it'll": "it will",

"it'll've": "it will have",

"it's": "it has / it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she had / she would",

"she'd've": "she would have",

"she'll": "she will",

"she'll've": "she will have",

"she's": "she has / she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so has",

"that'd": "that would / that had",

"that'd've": "that would have",

"that's": "that has / that is",

"there'd": "there had / there would",

"there'd've": "there would have",

"there's": "there has / there is",

"they'd": "they had / they would",

"they'd've": "they would have",

"they'll": "they will",

"they'll've": "they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we had / we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what will",

"what'll've": "what will have",

"what're": "what are",

"what's": "what has / what is",

"what've": "what have",

"when's": "when has / when is",

"when've": "when have",

"where'd": "where did",

"where's": "where has / where is",

"where've": "where have",

"who'll": "who will",

"who'll've": "who will have",

"who's": "who has / who is",

"who've": "who have",

"why's": "why has / why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you had / you would",

"you'd've": "you would have",

"you'll": "you will",

"you'll've": "you will have",

"you're": "you are",

"you've": "you have"

}



# Replace contractions with full length words

df['msg_clean'] = df['msg_clean'].apply(lambda x: [word.replace(word, contractions[word.lower()]) if word.lower() in contractions else word for word in x])



# Remove collections words

# Collection words are the words that you used to query your data from Twitter.

# Thus, you can expect that these terms will be found in each tweet. This could skew your word frequency analysis.

collection_words = ['climatechange', 'climate', 'change']

df['msg_clean'] = [[w for w in word if not w in collection_words] for word in df['msg_clean']]



# Remove stop words

stop_words = stopwords.words('english')

df['msg_clean'] = df['msg_clean'].apply(lambda row: [word for word in row if word not in stop_words])



# Perform the same steps on the testing data

df_test['msg_clean'] = df_test['msg_clean'].apply(lambda x: [word.replace(word, contractions[word.lower()]) if word.lower() in contractions else word for word in x])

df_test['msg_clean'] = [[w for w in word if not w in collection_words] for word in df_test['msg_clean']]

df_test['msg_clean'] = df_test['msg_clean'].apply(lambda row: [word for word in row if word not in stop_words])



# Transform the list of tokens into a single string

df['msg_clean'] = df['msg_clean'].apply(lambda x: ' '.join([i for i in x]))

df_test['msg_clean'] = df_test['msg_clean'].apply(lambda x: ' '.join([i for i in x]))
# Remove punctutation and numbers from tweets

def remove_punctuation_numbers(post):

    punc_numbers = string.punctuation + '0123456789'

    return ''.join([l for l in post if l not in punc_numbers])



df['msg_clean'] = df['msg_clean'].apply(remove_punctuation_numbers)

df_test['msg_clean'] = df_test['msg_clean'].apply(remove_punctuation_numbers)
df['msg_clean'] = df['msg_clean'].str.lower()

df_test['msg_clean'] = df_test['msg_clean'].str.lower()
# Lemmatise tweets

lemmatizer = WordNetLemmatizer()

df['msg_clean'] = df['msg_clean'].apply(lemmatizer.lemmatize)

df_test['msg_clean'] = df_test['msg_clean'].apply(lemmatizer.lemmatize)
# Bar plot displaying the count of each sentiment for the dataframe

sns.countplot(df['sentiment'])

plt.title('Number of tweets per sentiment group')

plt.show()
#Wordcloud of tweets for news

wordcloud = WordCloud(background_color='white', width=800, height=400).generate(' '.join(df[df['sentiment'] == 2]

                                          ['msg_clean']))

plt.figure( figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
#Wordcloud of tweets for pro climate change

wordcloud = WordCloud(background_color='white', width=800, height=400).generate(' '.join(df[df['sentiment'] == 1]

                                          ['msg_clean']))

plt.figure( figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
#Wordcloud of tweets for neutral 

wordcloud = WordCloud(background_color='white', width=800, height=400).generate(' '.join(df[df['sentiment'] == 0]

                                          ['msg_clean']))

plt.figure( figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
#Wordcloud of tweets for anti climate change

wordcloud = WordCloud(background_color='white', width=800, height=400).generate(' '.join(df[df['sentiment'] == -1]

                                          ['msg_clean']))

plt.figure( figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
# Creating a variable for the sentiment analyser as it is too long

sid = SentimentIntensityAnalyzer()



# Add SlangSD dictionary to vader lexicon 

# (Wu et al. (2016): http://arxiv.org/abs/1608.05129)



slang = pd.read_csv('../input/slangsdtxt/SlangSD.txt', sep='\t', names=['word', 'score'])

slang_dict = dict(zip(slang['word'], slang['score']))

sid.lexicon.update(slang_dict)



#Sentiment analysis of cleaned tweets

df['pos']  = df['message'].apply(lambda review: sid.polarity_scores(review)).apply(lambda score_dict: score_dict['pos'])

df['neg']  = df['message'].apply(lambda review: sid.polarity_scores(review)).apply(lambda score_dict: score_dict['neg'])

df['neu']  = df['message'].apply(lambda review: sid.polarity_scores(review)).apply(lambda score_dict: score_dict['neu'])

df['compound']  = df['message'].apply(lambda review: sid.polarity_scores(review)).apply(lambda score_dict: score_dict['compound'])
plt.figure( figsize=(12,3) )

sns.kdeplot(df['pos'][df['sentiment'] == 2], shade = True, label = 'News')

sns.kdeplot(df['pos'][df['sentiment'] == 1], shade = True, label = 'Pro')

sns.kdeplot(df['pos'][df['sentiment'] == 0], shade = True, label = 'Neutral')

sns.kdeplot(df['pos'][df['sentiment'] == -1], shade = True, label = 'Con')

plt.xlabel('')

plt.title('Positive sentiment score')

plt.show()



plt.figure( figsize=(12,3) )

sns.kdeplot(df['neu'][df['sentiment'] == 2], shade = True, label = 'News')

sns.kdeplot(df['neu'][df['sentiment'] == 1], shade = True, label = 'Pro')

sns.kdeplot(df['neu'][df['sentiment'] == 0], shade = True, label = 'Neutral')

sns.kdeplot(df['neu'][df['sentiment'] == -1], shade = True, label = 'Con')

plt.xlabel('')

plt.title('Neutral sentiment score')

plt.show()



plt.figure( figsize=(12,3) )

sns.kdeplot(df['neg'][df['sentiment'] == 2], shade = True, label = 'News')

sns.kdeplot(df['neg'][df['sentiment'] == 1], shade = True, label = 'Pro')

sns.kdeplot(df['neg'][df['sentiment'] == 0], shade = True, label = 'Neutral')

sns.kdeplot(df['neg'][df['sentiment'] == -1], shade = True, label = 'Con')

plt.xlabel('')

plt.title('Negative sentiment score')

plt.show()



plt.figure( figsize=(12,3) )

sns.kdeplot(df['compound'][df['sentiment'] == 2], shade = True, label = 'News')

sns.kdeplot(df['compound'][df['sentiment'] == 1], shade = True, label = 'Pro')

sns.kdeplot(df['compound'][df['sentiment'] == 0], shade = True, label = 'Neutral')

sns.kdeplot(df['compound'][df['sentiment'] == -1], shade = True, label = 'Con')

plt.xlabel('')

plt.title('Compound sentiment score')

plt.show()
#add word count

df['word_count'] = df['message'].apply(lambda x: len(x.split()))



#add unique word count

df['unique_words'] = df['message'].apply(lambda x: len(set(x.split())))



#add stopword count

stop_words = list(stopwords.words("english"))

df['number_stopwords'] = df['message'].apply(lambda x: len([i for i in x.lower().split() if i in stop_words]))



#add punctuation count

df['punctuation'] = df['message'].apply(lambda x: len([i for i in str(x) if i in string.punctuation]))



#add url count

df['number_urls'] = df['message'].apply(lambda x: len([i for i in x.lower().split() if 'http' in i or 'https' in i]))



#add mention count

df['mentions'] = df['message'].apply(lambda x: len([i for i in str(x) if i == '@']))



#add hashtag count

df['hashtags'] = df['message'].apply(lambda x: len([i for i in str(x) if i == '#']))
plt.subplot(1,4,1)

sns.boxplot(y='word_count', x='sentiment', data=df)

plt.title('Number of words')

plt.ylabel('Count')

plt.xlabel('')

fig = plt.gcf()

fig.set_size_inches( 23, 5)



plt.subplot(1,4,2)

sns.boxplot(y='unique_words', x='sentiment', data=df)

plt.title('Number of unique words')

plt.ylabel('')

plt.xlabel('')

fig = plt.gcf()

fig.set_size_inches( 23, 5)



plt.subplot(1,4,3)

sns.boxplot(y='number_stopwords', x='sentiment', data=df)

plt.title('Number of stop words')

plt.ylabel('')

plt.xlabel('')

fig = plt.gcf()

fig.set_size_inches( 23, 5)



plt.subplot(1,4,4)

sns.boxplot(y='punctuation', x='sentiment', data=df)

plt.title('Number of punctuation symbols')

plt.ylabel('')

plt.xlabel('')

fig = plt.gcf()

fig.set_size_inches( 23, 4)

plt.show()
plt.subplot(1,3,1)

sns.stripplot(y='number_urls', x='sentiment', data=df, jitter=True)

plt.title('Number of urls')

plt.ylabel('Count')

plt.xlabel('')

fig = plt.gcf()

fig.set_size_inches( 23, 5)



plt.subplot(1,3,2)

sns.stripplot(y='mentions', x='sentiment', data=df, jitter=True)

plt.title('Number of mentions')

plt.ylabel('')

plt.xlabel('')

fig = plt.gcf()

fig.set_size_inches( 23, 5)



plt.subplot(1,3,3)

sns.stripplot(y='hashtags', x='sentiment', data=df, jitter=True)

plt.title('Number of hashtags')

plt.ylabel('')

plt.xlabel('')

fig = plt.gcf()

fig.set_size_inches( 23, 5)



plt.show()
df['length'] = df['message'].apply(len)



plt.figure( figsize=(12,6) )

sns.kdeplot(df['length'][df['sentiment'] == 2], shade = True, label = 'News')

sns.kdeplot(df['length'][df['sentiment'] == 1], shade = True, label = 'Pro')

sns.kdeplot(df['length'][df['sentiment'] == 0], shade = True, label = 'Neutral')

sns.kdeplot(df['length'][df['sentiment'] == -1], shade = True, label = 'Con')

plt.xlabel('Length of words')

plt.title('Distribution of length of tweets')

plt.show()



print('\n' + 'Average length of news tweets:\t\t{}'.format(round(df['length'][df['sentiment'] == 2].mean(), 3)))

print('\n' + 'Average length of pro tweets:\t\t{}'.format(round(df['length'][df['sentiment'] == 1].mean(), 3)))

print('\n' + 'Average length of neutral tweets:\t{}'.format(round(df['length'][df['sentiment'] == 0].mean(), 3)))

print('\n' + 'Average length of con tweets:\t\t{}'.format(round(df['length'][df['sentiment'] == -1].mean(), 3)))
def count_pos(df, sentiment):

    pos_dict = {}

    df_pos = df[df['sentiment'] == sentiment]

    for i in range(len(df_pos)):

        text = nlp(df.iloc[i, 1])

        for j in range(len(text)):

            part_of_speech = text[j].pos_

            if part_of_speech in pos_dict.keys():

                pos_dict[part_of_speech] += 1

            else:

                pos_dict[part_of_speech] = 1

    return pos_dict
grp_2_pos = pd.DataFrame.from_dict(count_pos(df, 2), orient='index', columns=['news'])

grp_1_pos = pd.DataFrame.from_dict(count_pos(df, 1), orient='index', columns=['pro'])

grp_0_pos = pd.DataFrame.from_dict(count_pos(df, 0), orient='index', columns=['neutral'])

grp_neg1_pos = pd.DataFrame.from_dict(count_pos(df, -1), orient='index', columns=['con'])



grp_2_pos['news'] = grp_2_pos['news']/(len(df[df['sentiment'] == 2]))

grp_1_pos['pro'] = grp_1_pos['pro']/(len(df[df['sentiment'] == 1]))

grp_0_pos['neutral'] = grp_0_pos['neutral']/(len(df[df['sentiment'] == 0]))

grp_neg1_pos['con'] = grp_neg1_pos['con']/(len(df[df['sentiment'] == -1]))



df_pos = pd.merge(grp_2_pos, grp_1_pos, how='outer', left_index=True, right_index=True)

df_pos = pd.merge(df_pos, grp_0_pos, how='outer', left_index=True, right_index=True)

df_pos = pd.merge(df_pos, grp_neg1_pos, how='outer', left_index=True, right_index=True)



df_pos['type'] = df_pos.index

df_pos = pd.melt(df_pos, id_vars='type', var_name="sentiment", value_name="count")
sns.factorplot(x='type', y='count', hue='sentiment', data=df_pos, kind='bar')

plt.xlabel('Part of Speech', size=20)

plt.ylabel('Count', size=20)

plt.xticks(size = 17)

plt.yticks(size = 17)

plt.legend(prop={'size':20})

fig = plt.gcf()

fig.set_size_inches(25, 12)

plt.show()
def count_ent(df, sentiment):

    ent_dict = {}

    name_dict = {}

    df_pos = df[df['sentiment'] == sentiment]

    for i in range(len(df_pos)):

        text = nlp(df.iloc[i, 1])

        if text.ents:

            for ent in text.ents:

                if ent.label_ in ent_dict.keys():

                    ent_dict[ent.label_] += 1

                else:

                    ent_dict[ent.label_] = 1

    return ent_dict
grp_2_ent = pd.DataFrame.from_dict(count_ent(df, 2), orient='index', columns=['news'])

grp_1_ent = pd.DataFrame.from_dict(count_ent(df, 1), orient='index', columns=['pro'])

grp_0_ent = pd.DataFrame.from_dict(count_ent(df, 0), orient='index', columns=['neutral'])

grp_neg1_ent = pd.DataFrame.from_dict(count_ent(df, -1), orient='index', columns=['con'])



grp_2_ent['news'] = grp_2_ent['news']/(len(df[df['sentiment'] == 2]))

grp_1_ent['pro'] = grp_1_ent['pro']/(len(df[df['sentiment'] == 1]))

grp_0_ent['neutral'] = grp_0_ent['neutral']/(len(df[df['sentiment'] == 0]))

grp_neg1_ent['con'] = grp_neg1_ent['con']/(len(df[df['sentiment'] == -1]))



df_ent = pd.merge(grp_2_ent, grp_1_ent, how='outer', left_index=True, right_index=True)

df_ent = pd.merge(df_ent, grp_0_ent, how='outer', left_index=True, right_index=True)

df_ent = pd.merge(df_ent, grp_neg1_ent, how='outer', left_index=True, right_index=True)



df_ent['type'] = df_ent.index

df_ent = pd.melt(df_ent, id_vars='type', var_name="sentiment", value_name="count")
sns.factorplot(x='type', y='count', hue='sentiment', data=df_ent, kind='bar')

plt.xlabel('Named Entities', size=20)

plt.ylabel('Average number of words', size=20)

plt.xticks(size = 17)

plt.yticks(size = 17)

plt.legend(prop={'size':20})

fig = plt.gcf()

fig.set_size_inches(30, 12)

plt.show()
vectorizer = TfidfVectorizer(sublinear_tf=True, 

                             smooth_idf = True, 

                             max_df = 0.3, 

                             strip_accents = 'ascii', 

                             ngram_range = (1, 2))
# Splitting the data into variable and response 

X = vectorizer.fit_transform(df['msg_clean'].values)

y = df['sentiment'].values



print('Shape of the vectorised training data: {}'.format(X.shape))
# Splitting the training data into training (80%) and validation (20%) sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Evaluate the F1 score of different classifier models by cross-validation

random_state = 42

kf = KFold(n_splits=10, random_state=random_state, shuffle=True)



clf = [LogisticRegression(max_iter = 4000), 

       LinearSVC(random_state=random_state),  

       ComplementNB()]



scores = []

for i in range(len(clf)):

    scores.append(cross_val_score(clf[i], X_train, y_train, 

                                  scoring=make_scorer(f1_score, average='macro'), 

                                  cv=kf).mean())



result = pd.DataFrame({'Algorithm': ['LR', 'LSVC', 'CNB'], 'F1_macro': scores})

result = result.sort_values('F1_macro', ascending=False)



print(result)
# Compare results of best performing versus worst performing model

## Best performing model: LinearSVC()

clf = LinearSVC(random_state=random_state)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



print('Best model performance' + '\n')

print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))

print('Precision: {}'.format(precision_score(y_test, y_pred, average='macro')))

print('Recall: {}'.format(recall_score(y_test, y_pred, average='macro')))

print('F1: {}'.format(f1_score(y_test, y_pred, average='macro')))

print('\n' + classification_report(y_test, y_pred))
# Compare results of best performing versus worst performing model

## Worst performing model: LinearSVC()

clf = KNeighborsClassifier(3)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



print('Worst model performance' + '\n')

print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))

print('Precision: {}'.format(precision_score(y_test, y_pred, average='macro')))

print('Recall: {}'.format(recall_score(y_test, y_pred, average='macro')))

print('F1: {}'.format(f1_score(y_test, y_pred, average='macro')))

print('\n' + classification_report(y_test, y_pred))
# Specify the range of 'C' parameters for LinearSVC

params = {'C': [0.1, 0.5, 1, 5, 10]}



clf = GridSearchCV(LinearSVC(max_iter=4000, multi_class='ovr'), 

                   param_grid=params, cv=kf, 

                   scoring=make_scorer(f1_score, average='macro')).fit(X_train, y_train)



print('Best score: {}'.format(clf.best_score_))

print('Best parameters: {}'.format(clf.best_params_))
svc = LinearSVC(random_state=random_state, C=clf.best_params_['C'], multi_class='ovr')

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)



svc_tuned = LinearSVC(random_state=random_state)

svc_tuned.fit(X_train, y_train)

y_pred_tuned = svc_tuned.predict(X_test)



print('LinearSVC model performance' + '\n')

print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)) + 

      '  >>>  {}'.format(accuracy_score(y_test, y_pred_tuned)))

print('Precision: {}'.format(precision_score(y_test, y_pred, average='macro')) + 

      '  >>>  {}'.format(precision_score(y_test, y_pred_tuned, average='macro')))

print('Recall: {}'.format(recall_score(y_test, y_pred, average='macro')) + 

      '  >>>  {}'.format(recall_score(y_test, y_pred_tuned, average='macro')))

print('F1: {}'.format(f1_score(y_test, y_pred, average='macro')) + 

      '  >>>  {}'.format(f1_score(y_test, y_pred_tuned, average='macro')))

#print('\n' + classification_report(y_test, y_pred) + 

#      '  >>>  {}'.format(classification_report(y_test, y_pred_tuned)))

print('\n' + classification_report(y_test, y_pred))

print('\n' + classification_report(y_test, y_pred_tuned) + '\n')
#Specify the range of alpha parameters for ComplementNB

params = {'alpha': [0.1, 0.5, 1], 

          'norm': [True, False]}



clf2 = GridSearchCV(ComplementNB(), 

                   param_grid=params, 

                   cv=kf, scoring=make_scorer(f1_score, average='macro')).fit(X_train, y_train)



print('Best score: {}'.format(clf2.best_score_))

print('Best parameters: {}'.format(clf2.best_params_))
cnb = ComplementNB(alpha=clf2.best_params_['alpha'], norm=clf2.best_params_['norm'])

cnb.fit(X_train, y_train)

y_pred = cnb.predict(X_test)



cnb_tuned = ComplementNB()

cnb_tuned.fit(X_train, y_train)

y_pred_tuned = cnb_tuned.predict(X_test)



print('ComplementNB model performance' + '\n')

print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)) + 

      '  >>>  {}'.format(accuracy_score(y_test, y_pred_tuned)))

print('Precision: {}'.format(precision_score(y_test, y_pred, average='macro')) + 

      '  >>>  {}'.format(precision_score(y_test, y_pred_tuned, average='macro')))

print('Recall: {}'.format(recall_score(y_test, y_pred, average='macro')) + 

      '  >>>  {}'.format(recall_score(y_test, y_pred_tuned, average='macro')))

print('F1: {}'.format(f1_score(y_test, y_pred, average='macro')) + 

      '  >>>  {}'.format(f1_score(y_test, y_pred_tuned, average='macro')))

#print('\n' + classification_report(y_test, y_pred) + 

#      '  >>>  {}'.format(classification_report(y_test, y_pred_tuned)))

print('\n' + classification_report(y_test, y_pred))

print('\n' + classification_report(y_test, y_pred_tuned) + '\n')
X_test = vectorizer.transform(df_test['msg_clean'].values)
# Running a prediction on the test data

classifier = LinearSVC(max_iter=4000)

linearsvc = classifier.fit(X, y)

y_pred = classifier.predict(X_test)
# Storing the predictions in a CSV

predictions = pd.DataFrame({"tweetid":df_test.index, "sentiment": y_pred})

predictions.to_csv('submission_28.csv', index=False)

predictions.head()
#params = {'alpha': 0.5, 'clf': 'ComplementNB'}

#metrics = {'kaggle_score': 0.755}



#Log parameters and results

#experiment.log_parameters(params)

#experiment.log_metrics(metrics)
#experiment.end()
#experiment.display()