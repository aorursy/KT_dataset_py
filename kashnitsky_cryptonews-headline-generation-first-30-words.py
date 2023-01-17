# https://pypi.org/project/rouge/

!pip install rouge > /dev/null
from rouge import Rouge 



hypothesis = "Some London Underground stations should be closed, as the city is trying to reduce the impact of a coronavirus outbreak.".lower()



reference = "Up to 40 stations on the London Underground network are to be shut as the city attempts to reduce the effect of the coronavirus outbreak.".lower()



rouge = Rouge()

scores = rouge.get_scores(hypothesis, reference)

scores
import pandas as pd

from tqdm import tqdm

tqdm.pandas()

from pathlib import Path



from matplotlib import pyplot as plt

%config InlineBackend.figure_format = 'retina'
PATH_TO_CRYPTO_NEWS = Path('../input/news-about-major-cryptocurrencies-20132018-40k/')
train_df = pd.read_csv(PATH_TO_CRYPTO_NEWS / 'crypto_news_parsed_2013-2017_train.csv')

valid_df = pd.read_csv(PATH_TO_CRYPTO_NEWS / 'crypto_news_parsed_2018_validation.csv')
train_df.info()
# readling empty strings is a bit different locally and here, but not a big deal 

train_df['text'].fillna(' ', inplace=True)
valid_df.info()
train_df.head(2)
train_df['url'].nunique() == len(train_df)
train_df.loc[:5, 'url']
train_df['title'].apply(lambda s: len(s.split())).describe()
from wordcloud import WordCloud, STOPWORDS



wordcloud = WordCloud(background_color='black', stopwords = STOPWORDS,

                max_words = 200, max_font_size = 100, 

                random_state = 17, width=800, height=400)



plt.figure(figsize=(16, 12))

wordcloud.generate(str(train_df['title']))

plt.imshow(wordcloud);
train_df['text'].apply(lambda s: len(s.split())).describe()
first_sentences_dumb = train_df['text'].apply(lambda s: s.split('.')[0])

first_sentences_dumb.apply(lambda s: len(s.split())).describe()
first_ten_words_dumb = first_sentences_dumb.apply(lambda s: " ".join(s.split()[:10]))

first_ten_words_dumb.value_counts().head(20)
from nltk.tokenize import sent_tokenize
def extract_first_sent(text):

    

    sent_tok = sent_tokenize(text)

    

    return sent_tok[0].strip() if sent_tok else ''
first_sentences = train_df['text'].progress_apply(extract_first_sent)

first_sentences.apply(lambda s: len(s.split())).describe()
first_ten_words = first_sentences.apply(lambda s: " ".join(s.split()[:10]))

first_ten_words.value_counts().head(20)
train_df['year'].value_counts()
valid_df['year'].value_counts()
train_df['author'].nunique()
train_df['author'].value_counts().head()
train_df['source'].nunique()
train_df['source'].value_counts().head()
true_val_titles = valid_df['title'].str.lower()
first_sentences_val = valid_df['text'].progress_apply(extract_first_sent)

first_thirty_words_val = first_sentences_val.loc[valid_df.index].apply(lambda s: " ".join(s.split()[:30]).lower())
%%time

rouge = Rouge()

scores = rouge.get_scores(hyps=first_thirty_words_val, refs=true_val_titles, avg=True, ignore_empty=True)
scores
final_metric = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3

final_metric