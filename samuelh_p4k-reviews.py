# %matplotlib inline

#Display all outputs from cells
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#Import Packages

#fs
import os as os

#Data manipulation
import numpy as np
import pandas as pd

#Plotting
import plotnine as p9
import matplotlib.pyplot as plt

#iteration
import itertools as it

#Wordclouds
from wordcloud import WordCloud
import wordcloud

#Combinatorics
import collections

#regex
import re

#Dates
import datetime as dt

#NLTK
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error

#Math
from math import sqrt

#remove non-alpha numeric text from str
def stripNonAlphaNum(text):
    import re
    return re.compile(r'\W+', re.UNICODE).split(text)


#Import data
df_p4k = pd.read_csv(filepath_or_buffer = "../input/p4kreviews.csv",
                     encoding='latin1')

#Remove the row count
df_p4k.drop(columns=df_p4k.columns[0],
            inplace= True)

#Convert genre to categorical
df_p4k['genre'] = df_p4k['genre'].astype('category')
#Open console
#%qtconsole
p9.ggplot(data=df_p4k, mapping=p9.aes(x = "score")) + p9.geom_density()
df_p4k_sum = df_p4k.groupby("genre").mean()
df_p4k_sum
df_p4k_best =  df_p4k[df_p4k['best'] == 1]
p9.ggplot(data=df_p4k_best, mapping=p9.aes(x = "score")) + p9.geom_density()
p9.ggplot(data=df_p4k, mapping=p9.aes(x = "score")) + p9.facet_wrap("~genre") + p9.geom_density()
wc_text = " ".join(df_p4k['review'].head(10).as_matrix().astype('str'))
wc_text = " ".join(stripNonAlphaNum(wc_text))
p4k_wordcloud = WordCloud().generate(wc_text)
wordcloud = WordCloud(max_font_size=40).generate(wc_text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")

for g in df_p4k['genre'].cat.categories:
    wc_text = " ".join(df_p4k[df_p4k['genre'] == g]['review'].head(100).as_matrix().astype('str')).lower()
    wc_text = " ".join(stripNonAlphaNum(wc_text))
    p4k_wordcloud = WordCloud().generate(wc_text)
    wordcloud = WordCloud(max_font_size=40).generate(wc_text)
    plt.figure()
    plt.title("word cloud for " + g)
    plt.imshow(wordcloud, interpolation="bilinear")
#Most common words

#Find the most common words
wc_text = " ".join(df_p4k['review'].head(100).as_matrix().astype('str')).lower()
top_words = stripNonAlphaNum(wc_text)
top_words = collections.Counter(top_words)

#Use this to create regex filter
word_filter = top_words.most_common(100)
word_filter = [x[0] for x in word_filter]
word_filter = "|".join(word_filter)

regex = re.compile(word_filter)

#Filter out the matching words and recreate a huge string
wc_text = filter(lambda x: not regex.match(x), wc_text.split())
wc_text = " ".join(wc_text)

p4k_wordcloud = WordCloud().generate(wc_text)
wordcloud = WordCloud(max_font_size=40).generate(wc_text)
plt.figure()
plt.title("word cloud (with common words removed)")
plt.imshow(wordcloud, interpolation="bilinear")

for g in df_p4k['genre'].cat.categories:
    wc_text = " ".join(df_p4k[df_p4k['genre'] == g]['review'].head(100).as_matrix().astype('str')).lower()
    wc_text = " ".join(stripNonAlphaNum(wc_text))
    wc_text = filter(lambda x: not regex.match(x), wc_text.split())
    wc_text = " ".join(wc_text)
    p4k_wordcloud = WordCloud().generate(wc_text)
    wordcloud = WordCloud(max_font_size=40).generate(wc_text)
    plt.figure()
    plt.title("word cloud for " + g)
    plt.imshow(wordcloud, interpolation="bilinear")
#Compute differences between word clouds

#Just pick a few categories so jupyter doesn't cry about too many plots
for g in it.combinations(df_p4k['genre'].cat.categories[0:3], 2):
    
    if g[0] == g[1]:
        continue
    
    #Create genre 1 word count
    wc_text = " ".join(df_p4k[df_p4k['genre'] == g[0]]['review'].head(100).as_matrix().astype('str')).lower()
    top_words = stripNonAlphaNum(wc_text)
    top_words = filter(lambda x: not regex.match(x), top_words)
    top_words = collections.Counter(top_words)
    
    df_top_words = pd.DataFrame(list(top_words.items()),
                                columns = ['name', 'count1'])
    
    #Create genre 2 word count
    wc_text = " ".join(df_p4k[df_p4k['genre'] == g[1]]['review'].head(100).as_matrix().astype('str')).lower()
    top_words2 = stripNonAlphaNum(wc_text)
    top_words2 = filter(lambda x: not regex.match(x), top_words2)
    top_words2 = collections.Counter(top_words2)
    
    df_top_words2 = pd.DataFrame(list(top_words2.items()),
                                columns = ['name', 'count2'])
    
    #Full join 
    df_compare = df_top_words.merge(df_top_words2,
                                    how = "outer",
                                    on = "name")
    df_compare['count1'] = df_compare['count1'].fillna(0)
    df_compare['count2'] = df_compare['count2'].fillna(0)
    
    df_compare['diff1'] = df_compare['count1'] - df_compare['count2']
    df_compare['diff2'] = df_compare['count2'] - df_compare['count1']
    
    genre1_text = " ".join(df_compare[df_compare['diff1'] > 0]['name'])
    genre2_text = " ".join(df_compare[df_compare['diff2'] > 0]['name'])
    
    WordCloud().generate(genre1_text)
    wordcloud = WordCloud(max_font_size=40).generate(genre1_text)
    plt.figure()
    plt.title("word cloud for words appearing more in \n" + g[0] + " reviews than in " + g[1] + " reviews.")
    plt.imshow(wordcloud, interpolation="bilinear")
    
    WordCloud().generate(genre2_text)
    wordcloud = WordCloud(max_font_size=40).generate(genre2_text)
    plt.figure()
    plt.title("word cloud for words appearing more in \n" + g[1] + " reviews than in " + g[0] + " reviews.")
    plt.imshow(wordcloud, interpolation="bilinear")
df_plot_data = pd.DataFrame([],
                            columns=[''])

for g in it.combinations(df_p4k['genre'].cat.categories, 2):
    
    if g[0] == g[1]:
        continue
    
    #Create genre 1 word count
    wc_text = " ".join(df_p4k[df_p4k['genre'] == g[0]]['review'].head(100).as_matrix().astype('str')).lower()
    top_words = stripNonAlphaNum(wc_text)
    top_words = filter(lambda x: not regex.match(x), top_words)
    top_words = collections.Counter(top_words)
    
    df_top_words = pd.DataFrame(list(top_words.items()),
                                columns = ['name', 'count1'])
    
    #Create genre 2 word count
    wc_text = " ".join(df_p4k[df_p4k['genre'] == g[1]]['review'].head(100).as_matrix().astype('str')).lower()
    top_words2 = stripNonAlphaNum(wc_text)
    top_words2 = filter(lambda x: not regex.match(x), top_words2)
    top_words2 = collections.Counter(top_words2)
    
    df_top_words2 = pd.DataFrame(list(top_words2.items()),
                                columns = ['name', 'count2'])
    
    #Full join 
    df_compare = df_top_words.merge(df_top_words2,
                                    how = "outer",
                                    on = "name")
    df_compare['count1'] = df_compare['count1'].fillna(0)
    df_compare['count2'] = df_compare['count2'].fillna(0)
    
    df_compare['genre1'] = g[0]
    df_compare['genre2'] = g[1]
    
    df_compare['diff1'] = df_compare['count1'] - df_compare['count2']
    df_compare['diff2'] = df_compare['count2'] - df_compare['count1']
    
    df_plot_data = df_plot_data.append(df_compare)
    
#Drop empty col
df_plot_data = df_plot_data.drop('', axis = 1)

#Find top 5 positive differences for each genre
df_plot_data2 = df_plot_data.sort_values(by = ["genre1", "genre2", "diff1"], ascending = [True, True, False]).groupby(by = ["genre1", 'genre2']).head(5)
df_plot_data2['name'] = pd.Categorical(df_plot_data2['name'], categories=df_plot_data2['name'].unique())



# #Plot the top word differences for each genre
# #Faceting in p9 doesn't seem to work with scales free categorical scales
# (p9.ggplot(data=df_plot_data,
#           mapping=p9.aes(x = "name", y = "diff1")) +
#  p9.facet_wrap(facets = "~ genre1", scales='free_y') + 
#  p9.geom_col())

#Instead we can just plot each genre combination
for i in df_plot_data2[['genre1', 'genre2']].drop_duplicates().head(1).itertuples():
    # (df_plot_data['genre1'] == i[1]) and (df_plot_data['genre1'] == i[2])
    tmp_df = df_plot_data2.loc[lambda df: (df['genre1'] == i[1]) & (df['genre2'] == i[2])]
    
    tmp_df['name'] = pd.Categorical(tmp_df['name'], categories = tmp_df['name'].unique())
    tmp_df = tmp_df.sort_values(by = ["genre1", "genre2", "diff1"], ascending = [True, True, False]).groupby(by = ["genre1", 'genre2']).head(5)
    
    # print(tmp_df)
    # 
    print(p9.ggplot(data = tmp_df) +
    p9.geom_col(p9.aes(x = 'name', y = 'diff1')) +
          p9.labs(title = 'The 5 words that have the highest difference in utilisation \n between ' + i[1] + ' and ' + i[2] + ' reviews.',
                  x = "Word",
                  y = "Difference in # times utilised"))

tmp_plot_data = df_plot_data.sort_values(by = ['name', 'diff1'], ascending= [True, False]).groupby('name').head(1).sort_values('diff1', ascending=False).head(20)
tmp_plot_data['name'] = pd.Categorical(tmp_plot_data['name'], categories=tmp_plot_data['name'].unique())

(p9.ggplot(data = tmp_plot_data) +
 p9.geom_col(p9.aes(x = "name",
                 y = "diff1",
                 fill = 'genre1',
                colour = 'genre2'))+
 p9.theme(axis_text_x = p9.element_text(rotation=90)))
# tmp_plot_data = df_plot_data.sort_values('diff1', ascending=False).groupby(['genre1', 'genre2']).head(1).sort_values(['genre1', 'genre2'], ascending=False)
tmp_plot_data = df_plot_data.sort_values('diff1', ascending=False).groupby('name').head(1).groupby(['genre1', 'genre2']).head(1).sort_values(['genre1', 'genre2'], ascending=False)

tmp_plot_data['genre1'] = pd.Categorical(tmp_plot_data['genre1'], categories= tmp_plot_data['genre1'].unique())
tmp_plot_data['genre2'] = pd.Categorical(tmp_plot_data['genre2'], categories= tmp_plot_data['genre1'].unique())
#Careful, if you have NaN category values these will get plotted at the first level of the category.
tmp_plot_data = tmp_plot_data[tmp_plot_data['genre1'].notnull()]
tmp_plot_data = tmp_plot_data[tmp_plot_data['genre2'].notnull()]

#Can mirror it, but I think it's clearer without the mirror
# tmp_plot_data2 = pd.DataFrame.copy(tmp_plot_data)
# # tmp_plot_data2 = tmp_plot_data2[tmp_plot_data2['genre2'] != 'Electronic']
# # tmp_plot_data2 = tmp_plot_data2[tmp_plot_data2['genre1'] != 'Electronic']
# tmp_plot_data2['tmp_genre'] = tmp_plot_data2['genre1']
# tmp_plot_data2['genre1'] = tmp_plot_data2['genre2']
# tmp_plot_data2['genre2'] = tmp_plot_data2['tmp_genre']
# tmp_plot_data2['diff1'] = abs(tmp_plot_data2['diff2'])
# tmp_plot_data2 = tmp_plot_data2.drop("tmp_genre", axis = 1)
# tmp_plot_data2 = tmp_plot_data2[tmp_plot_data2['genre1'].notnull()]
# tmp_plot_data2 = tmp_plot_data2[tmp_plot_data2['genre2'].notnull()]
# 
# 
# tmp_plot_data = tmp_plot_data.append(tmp_plot_data2)

(p9.ggplot(data = tmp_plot_data) + 
p9.geom_text(p9.aes(x = "genre2",
                     y = 'genre1',
                     label = 'name',
                     color = 'diff1'))+
 p9.labs(title = 'largest differences in word frequency between genres 1 & 2 \n note: each word can only appear once'))
df_review_sent_agg = pd.DataFrame()

analyzer = SentimentIntensityAnalyzer()

uninteretable_reviews = int(0)

#Look out there is a missing review
# df_p4k.iloc[13300]

#Look out there is a garbage review
# df_p4k.iloc[17166]

for index, row in df_p4k.iterrows():
    
    #Skip missing rows / where the review is not a string
    if not isinstance(row['review'], str):
        uninteretable_reviews += 1
        continue
    
    
    token = sent_tokenize(row['review'])
    scores = list([["compound",
                  "neg",
                  "neu",
                  "pos",
                   "album",
                   "artist"]])
    
    #Skip reviews where no sentences can be tokenized from the review str
    if len(token) == 0:
        uninteretable_reviews += 1
        continue
    
    for j in token:
        #Create sentiment score
        score = analyzer.polarity_scores(j)
        
        #Return list of dict values sorted by key (same order as above)
        score = sorted(list(score.items()), key = lambda x: x[1])
        score = [y[1] for y in score]
        
        #Add static columns of album/artist
        score = list(score) + [row['album'], row['artist']]
        scores.append(score)
    
    #Create data frame of scores and summarise the mean sentiment
    df_sents = pd.DataFrame(scores, columns = scores.pop(0))
    df_sents = df_sents.groupby(['album', 'artist']).aggregate(np.mean)
    #Need to reset index after grouped operation. This is like having to ungroup() in dplyr syntax
    df_sents = df_sents.reset_index()
    
    #If you don't ungroup previously the grouped columns they will appear in the DF but not in its columns attribute
    df_review_sent_agg = df_review_sent_agg.append(df_sents)
    
#Print the number of missing/unusable reviews
if uninteretable_reviews > 0:
    print(str(uninteretable_reviews) + ' reviews could not be parsed.')
    
#Join back the other fields
df_review_sent_agg = df_review_sent_agg.merge(right = df_p4k, 
                                              how = 'inner',
                                              on = ['album', 'artist'])
#Check the results
df_review_sent_agg
#Does this look like this is going to work well
df_review_sent_agg.sort_values('pos', ascending= False).head(5)

df_review_sent_agg.sort_values('neg', ascending= False).head(5)

df_review_sent_agg.sort_values('neu', ascending= False).head(5)

df_review_sent_agg.sort_values('compound', ascending= False).head(5)

#Probably not. Goddamn hipster reviewers.
#One-hot encode the genre variables

#First you have to enumerate the string var
label_encode = LabelEncoder()
label_encode = label_encode.fit(df_review_sent_agg.genre)

df_review_sent_agg['genre_int'] = label_encode.transform(df_review_sent_agg.genre)

#Then one hot encode the genre category enumeration
onehot_encode = OneHotEncoder(sparse = False)
encoded_genre = onehot_encode.fit_transform(df_review_sent_agg.genre_int.values.reshape(-1, 1))
encoded_genre = pd.DataFrame(encoded_genre)
encoded_genre.columns = label_encode.classes_

#Column bind the encoded variables
df_review_sent_agg = pd.concat([df_review_sent_agg, encoded_genre], axis = 1)

#Convert date to ordinal
df_review_sent_agg['date'] = pd.to_datetime(df_review_sent_agg['date'])
df_review_sent_agg['date_int'] = df_review_sent_agg['date'].apply(lambda x: x.toordinal())

#Define split
df_train, df_test, response_train, response_test = train_test_split(df_review_sent_agg,
                                                                    df_review_sent_agg['score'],
                                                                    test_size=0.25,
                                                                    random_state=0)

#select feature columns
df_train = df_train.iloc[:, np.r_[2:6, 12:22]]
df_test = df_test.iloc[:, np.r_[2:6, 12:22]]


#Fit linear model
lm = LinearRegression()
lm.fit(df_train, response_train)
#Predict the test set scores
test_predictions = lm.predict(df_test)

#Find the RMSE of the predictions
rmse = sqrt(mean_squared_error(response_test, test_predictions))
rmse

#Find the SMAPE of the predictions
smape = abs(test_predictions - response_test) / (abs(test_predictions) - abs(response_test) / 2)
smape = np.sum(smape) / len(test_predictions)
smape

#Create a data frame and plot the error dist
df_lm_eval = pd.DataFrame.from_dict({"actual": response_test,
                       "predicted": test_predictions})

df_lm_eval['error'] = df_lm_eval['predicted'] - df_lm_eval['actual']

(p9.ggplot(data=df_lm_eval) +
 p9.geom_density(mapping = p9.aes(x = 'error')))
