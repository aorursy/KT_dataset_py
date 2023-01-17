import numpy as np
import pandas as pd
import gc
import os, glob
import spacy
from wordcloud import WordCloud
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Token, Doc
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from scipy.stats import ttest_ind, sem
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
import seaborn as sns
import re
from functools import reduce
from operator import concat
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
%matplotlib inline
articles = pd.DataFrame(pd.concat([pd.read_csv(f'../input/{f}') for f in glob.glob('../input/A*.csv')]))
comments = pd.DataFrame(pd.concat([pd.read_csv(f'../input/{f}') for f in glob.glob('../input/C*.csv')]))
articles['pubDate'] = pd.to_datetime(articles['pubDate'])
sid = SentimentIntensityAnalyzer()
def sentiment_score(text):
    return sid.polarity_scores(text)['compound']

articles = articles[articles['keywords'] != '[]'].reset_index()
articles['keywords'] = articles['keywords'].replace(r'\[\'|\'\]', '', regex=True).str.split(r'\',\s\'')
articles.dropna(axis=1, inplace=True)
articles.drop(['documentType', 'multimedia', 'source', 'webURL', 'snippet'], axis=1, inplace=True)
all_keywords = pd.Series(reduce(concat, articles['keywords']))
articles = pd.DataFrame(articles.values.repeat(articles['keywords'].str.len(), axis=0), columns=articles.columns)
articles = articles.assign(keywords=all_keywords)
articles['sentiment'] = articles['headline'].apply(sentiment_score).astype(float)
articles = articles[~articles['keywords'].str.match(r'^\d+$')]
# nlp = spacy.load('en_core_web_sm')
# combine_matcher = PhraseMatcher(nlp.vocab)
# name_matcher = Matcher(nlp.vocab)
# terminology_list = ['United States']
# combine_patterns = [nlp.make_doc(text) for text in terminology_list]
# name_re = lambda text: bool(re.compile(r'(\(.+\))|([A-Z]{1})').match(text))
# NAME_EXTRA = nlp.vocab.add_flag(name_re)
# name_pattern1 = [{'POS': 'PROPN'},{'ORTH': ','}, {'POS': 'PROPN'}]
# name_pattern2 = name_pattern1 + [{'POS': 'PROPN', 'LENGTH': 1}]
# name_pattern3 = name_pattern1 + [{'ORTH': '('}, {}, {'ORTH': ')'}]
# combine_matcher.add('CombineTokens', None, *combine_patterns)
# name_matcher.add('TruncateTokens', None, name_pattern1, name_pattern2, name_pattern3)
# sid = SentimentIntensityAnalyzer()
# def special_cases(doc):
#     for match_id, start, end in name_matcher(doc):
#         for token in doc[start:end]:
#             token._.is_first_or_middle_name = True
#             print(token)
#         doc[start:end].merge()
#     for match_id, start, end in combine_matcher(doc):
#         doc[start:end].merge()
#     return doc

# def preprocess_keywords(doc):
#     def clean_token(token):
#         token = re.sub(r'[\[\]]|\'', '', token.lower_.strip())
#         return re.sub(r'<.+>', '', token)
#     keywords = [clean_token(token) for token in doc if token.lower_ not in STOP_WORDS
#                 and not token.is_digit and not token.is_punct \
#                 and len(token) > 1]
#     return ' | '.join(OrderedDict(enumerate(keywords)).values())

# if Token.has_extension('is_first_or_middle_name'):
#     Token.remove_extension('is_first_or_middle_name')
# if Doc.has_extension('sentiment_score'):
#     Doc.remove_extension('sentiment_score')
# if 'special_cases' in nlp.pipe_names:
#     nlp.remove_pipe('special_cases')
# if 'preprocess_keywords' in nlp.pipe_names:
#     nlp.remove_pipe("preprocess_keywords")

# Token.set_extension('is_first_or_middle_name', default=False)
# Doc.set_extension('sentiment_score', getter=sentiment_score)
# nlp.add_pipe(special_cases)
# nlp.add_pipe(preprocess_keywords)
comments.dropna(axis=1, inplace=True)
drop_list = ['approveDate', 'articleWordCount', 'commentID', 'commentSequence', 'commentType']
drop_list.extend(['createDate', 'depth', 'inReplyTo', 'newDesk', 'parentID'])
drop_list.extend(['picURL', 'printPage', 'replyCount', 'sharing', 'status', 'timespeople', 'trusted'])
drop_list.extend(['typeOfMaterial', 'updateDate', 'userID'])
comments.drop(drop_list, axis=1, inplace=True)
comments['commentBody'] = comments['commentBody'].replace(r'<.+>|[^0-9\sA-Za-z]', '', regex=True).str.lower()
formattedDate = articles['pubDate'].dt.strftime('%b %d, %Y')
keyword_df = pd.DataFrame({'pubDate': formattedDate, 'keyword': articles['keywords']})
axes = keyword_df.groupby([keyword_df['pubDate'].str.slice(8), keyword_df['pubDate'].str.slice(0, 3)]) \
    .apply(lambda g: g['keyword'].value_counts()[:3]).iloc[::-1].plot.barh(figsize=(10, 15))
axes.set_xlabel('Keyword Occurrences')
axes.set_ylabel('')
plt.title('Top 3 Most Popular Keywords By Month')
del formattedDate
del keyword_df
del axes
gc.collect()
plt.show()
articles['keywords'].value_counts()[:10].iloc[::-1].plot.bar(figsize=(7, 5))
plt.title('Top 10 Keywords Overall')
plt.show()
wc = WordCloud(max_words=100, height=400, width=500, background_color='white').generate(' '.join(articles['keywords']))
plt.figure(figsize=(10,7))
plt.title("WordCloud visualization of top keywords", fontsize=14)
plt.imshow(wc)
plt.show()
front_page_keywords = articles[articles['printPage'] == 1]['keywords'].value_counts().iloc[:3]
front_page_keywords.plot.pie(figsize=(7, 6))
del front_page_keywords
gc.collect()
plt.title('Top 3 Front Page Keywords')
plt.ylabel('')
plt.show()
top_3_keywords = ['Trump, Donald J', 'United States Politics and Government', 'United States International Relations']
top_3_sentiment = articles[(articles['keywords'].isin(top_3_keywords)) & (articles['headline'] != 'Unknown')].drop_duplicates('headline')
top_3_sentiment['sentiment'].plot.hist(bins=30)
unique = articles[articles['headline'] != 'Unknown'].drop_duplicates('headline')
plt.title('Average headline sentiment score for the top 3 keywords is {}'.format(round(top_3_sentiment['sentiment'].mean(), 3)))
plt.xlabel('Sentiment score')
plt.figure(figsize=(8, 7))
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes = axes.flat
axes[0].set_title('Sections with most extreme positive sentiment')
axes[1].set_title('Sections with most extreme negative sentiment')
unique.drop_duplicates('headline').groupby('sectionName')['sentiment'].mean().nlargest(5).plot.bar(ax=axes[0])
unique.drop_duplicates('headline').groupby('sectionName')['sentiment'].mean().nsmallest(5).plot.bar(ax=axes[1])
plt.show()
print("Top 5 Headlines by Positive Sentiment:", end='\n\n')
for i, row in top_3_sentiment.sort_values('sentiment', ascending=False).iloc[:5].iterrows():
    print(row['headline'], row['sentiment'])
print("Top 5 Headlines by Negative Sentiment:", end='\n\n')
for i, row in top_3_sentiment.sort_values('sentiment').iloc[:5].iterrows():
    print(row['headline'], row['sentiment'])
def article_outlier_report(article_df, subset_name):
    article_df = article_df.drop_duplicates('headline')
    iqr = article_df['sentiment'].quantile(0.75) - article_df['sentiment'].quantile(0.25)
    low = article_df['sentiment'].quantile(0.25) - (1.5 * iqr)
    high = article_df['sentiment'].quantile(0.75) + (1.5 * iqr)
    outliers = article_df[(article_df['sentiment'] < low) | (article_df['sentiment'] > high)]
    norms = article_df[(article_df['sentiment'] >= low) & (article_df['sentiment'] <= high)]
    pct_pos = round(float(len(article_df[article_df['sentiment'] > high])) / len(article_df) * 100, 2)
    pct_neg = round(float(len(article_df[article_df['sentiment'] < low])) / len(article_df) * 100, 2)
    pct_outliers = round(float(outliers.shape[0]) / len(article_df) * 100, 2)
    pct_normal = round(float(norms.shape[0]) / len(article_df) * 100, 2)
    report = ""
    report += "{} Outlier Report:\n".format(subset_name.title())
    report += "Headline count: {}\n".format(article_df.shape[0])
    report += "Total outlier sentiment headlines: {}%\n".format(pct_outliers)
    report += "Total extreme positive sentiment headlines: {}%\n".format(pct_pos)
    report += "Total extreme negative sentiment headlines: {}%\n".format(pct_neg)
    report += "Total normal sentiment headlines: {}%\n".format(pct_normal)
    report += "Average sentiment: {}\n".format(round(article_df['sentiment'].mean(), 2))
    report += "Standard deviation: {}".format(round(article_df['sentiment'].std(), 3))
    return report
fig, axes = plt.subplots(2, 2, figsize=(20, 10))
axes = axes.flat
unique['sentiment'].plot.box(ax=axes[0], label='Sentiment for all headlines')
for i, kw in enumerate(top_3_keywords):
    label = 'Sentiment for {}'.format(kw)
    top_3_sentiment[top_3_sentiment['keywords'] == kw]['sentiment'].plot.box(ax=axes[i + 1], label=label)
plt.suptitle('Keyword sentiment score boxplots')
plt.show()
print(article_outlier_report(articles, "all headlines"))
for kw in top_3_keywords:
    print(article_outlier_report(top_3_sentiment[top_3_sentiment['keywords'] == kw], kw), end='\n\n')
print("P-values comparing sentiments for all keywords to those of top 3 keywords:")
for kw in top_3_keywords:
    print("{}: {}".format(kw, ttest_ind(unique['sentiment'], top_3_sentiment[top_3_sentiment['keywords'] == kw]['sentiment']).pvalue))
in_top_3 = articles['keywords'].isin(top_3_keywords).astype(int)
in_top_3.corr(articles['sentiment'])
grouped_comments = comments.groupby('articleID').count().reset_index()[['articleID', 'commentBody']]
articles = pd.merge(articles, grouped_comments.rename(columns={'commentBody': 'commentCount'}), on='articleID', how='inner').sort_values('commentCount', ascending=False)
articles.drop_duplicates(['headline', 'keywords'], inplace=True)
print("Overall average number of comments: {}".format(articles.drop_duplicates('headline')['commentCount'].mean()), end='\n\n')
for kw in top_3_keywords:
    kw_match = articles[articles['keywords'] == kw].drop_duplicates('headline')
    print("Average number of comments for {}: {}".format(kw, kw_match['commentCount'].mean()))
    print("Comment count p-value: {}".format(ttest_ind(articles.drop_duplicates('headline')['commentCount'], kw_match['commentCount']).pvalue), end='\n\n')
sns.scatterplot(x='sentiment', y='commentCount', data=articles.drop_duplicates('headline'))
plt.title('Number of comments vs. keyword sentiment score')
plt.show()
articles.groupby('keywords')['commentCount'].sum().sort_values(ascending=False).iloc[10::-1].plot.barh(figsize=(8, 5))
plt.xlabel('Number of Comments')
plt.show()
articles.sort_values('commentCount', ascending=False).drop_duplicates('headline')[['headline', 'commentCount']].iloc[:10].reset_index()
comment_sample = comments.sample(50000)
comment_sample = comment_sample.assign(sentiment=comment_sample['commentBody'].apply(sentiment_score))
sample_mean = np.round(comment_sample['sentiment'].mean(), 4)
sample_mean_error = np.round(sem(comment_sample['sentiment']), 4)
print("Overall average comment sentiment: {} +/- {}".format(sample_mean, sample_mean_error, 4))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes = axes.flat
sns.distplot(comment_sample.drop_duplicates('commentBody')['sentiment'], ax=axes[0])
sns.distplot(articles.drop_duplicates('headline')['sentiment'], ax=axes[1])
axes[0].set_title('Overall comment sentiment distribution')
axes[1].set_title('Overall article headline sentiment distribution')
plt.show()
