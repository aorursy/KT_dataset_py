import numpy as np
import pandas as pd 
from textblob import TextBlob
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import wordcloud
from collections import Counter
from pprint import pprint
import random
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# import re
pd.options.display.float_format = '{:.2f}'.format
sns.set(style="ticks")
plt.rc('figure', figsize=(8, 5), dpi=100)
plt.rc('axes', facecolor="#ffffff", linewidth=0.4, grid=True, labelpad=8, labelcolor='#616161')
plt.rc('patch', linewidth=0)
plt.rc('xtick.major', width=0.2)
plt.rc('ytick.major', width=0.2)
plt.rc('grid', color='#9E9E9E', linewidth=0.4)
plt.rc('text', color='#282828')
plt.rc('savefig', pad_inches=0.3, dpi=300)

# Hiding warnings for cleaner display
warnings.filterwarnings('ignore')

# Configuring some notebook options
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
# If you want interactive plots, uncomment the next line
# %matplotlib notebook
askReddit_df = pd.read_csv('../input/AskReddit.csv')
aww_df = pd.read_csv('../input/aww.csv')
books_df = pd.read_csv('../input/books.csv')
explainlikeimfive_df = pd.read_csv('../input/explainlikeimfive.csv')
food_df = pd.read_csv('../input/food.csv')
funny_df = pd.read_csv('../input/funny.csv')
getMotivated_df = pd.read_csv('../input/GetMotivated.csv')
gifs_df = pd.read_csv('../input/gifs.csv')
iAmA_df = pd.read_csv('../input/IAmA.csv')
jokes_df = pd.read_csv('../input/Jokes.csv')
lifeProTips_df = pd.read_csv('../input/LifeProTips.csv')
movies_df = pd.read_csv('../input/movies.csv')
pics_df = pd.read_csv('../input/pics.csv')
showerthoughts_df = pd.read_csv('../input/Showerthoughts.csv')
todayilearned_df = pd.read_csv('../input/todayilearned.csv')
videos_df = pd.read_csv('../input/videos.csv')
woahdude_df = pd.read_csv('../input/woahdude.csv')
worldnews_df = pd.read_csv('../input/worldnews.csv')

# We create these two lists for easier interaction with the datasets later
subreddits = [askReddit_df, aww_df, books_df, explainlikeimfive_df, food_df, funny_df,
              getMotivated_df, gifs_df, iAmA_df, jokes_df, lifeProTips_df, movies_df,
              pics_df, showerthoughts_df, todayilearned_df, videos_df, 
              woahdude_df, worldnews_df]

subreddit_names = ['AskReddit', 'aww', 'books', 'explainlikeimfive', 'food', 'funny',
                   'GetMotivated', 'gifs', 'IAmA', 'Jokes', 'LifeProTips', 'movies',
                   'pics', 'Showerthoughts', 'todayilearned', 'videos', 'woahdude', 'worldnews']
askReddit_df.head()
askReddit_df.info()
# We loop through all our subreddit datasets and remove some columns
# from each of them
for df in subreddits:
    df.drop(['link_flair_text', 'thumbnail', 'subreddit_id', 'link_flair_css_class', 
                       'author_flair_css_class', 'name', 'url', 'distinguished'],
                      axis=1, inplace=True)
for df, name in zip(subreddits, subreddit_names):
    # get the number of null values in each column of the dataset
    null_sum = df.isna().sum()
    # keep only the columns that have missing values
    null_sum = null_sum[null_sum > 0]
    print(name, 'dataset')
    for k,v in zip(null_sum.index, null_sum.values):
        print(k, ': ', v)
    print('-------------')
for df in subreddits:
    df['selftext'].fillna(value="", inplace=True)
for df in subreddits:
    df['title_length'] = df['title'].apply(lambda x: len(x))
def num_capitalized_word(s):
    c = 0
    for w in s.split():
        if w.isupper():
            c += 1
    return c

for df in subreddits:
    df['num_capitalized'] = df['title'].apply(num_capitalized_word)
contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
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
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
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
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
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
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
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
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

# if a contraction has more than one possible expanded forms, we replace it 
# with a list of these possible forms
tmp = {}
for k,v in contractions.items():
    if "/" in v:
        tmp[k] = [x.strip() for x in v.split(sep="/")]
    else:
        tmp[k] = v
contractions = tmp
tokenizer = RegexpTokenizer(r"[\w']+")
subreddit_words = []
for df, name in zip(subreddits, subreddit_names):
    all_titles = ' '.join([x.lower() for x in df['title']])
    for k,v in contractions.items():
        if isinstance(v, list):
            v = random.choice(v)
        all_titles = all_titles.replace(k.lower(), v.lower())
    words = list(tokenizer.tokenize(all_titles))
    words = [x for x in words if x not in stopwords.words('english')]
    subreddit_words.append(words)
    print('Most common words in ' + name, '*****************', sep='\n')
    pprint(Counter(words).most_common(35), compact=True)
    print()
# flattening the list
subreddit_words_f = [x for y in subreddit_words for x in y]
print('Most common words in all subreddits', '*****************', sep='\n')
pprint(Counter(subreddit_words_f).most_common(35), compact=True)
# a function to get custom colors for the word cloud
def col_func(word, font_size, position, orientation, font_path, random_state):
    colors = ['#b58900', '#cb4b16', '#dc322f', '#d33682', '#6c71c4', 
              '#268bd2', '#2aa198', '#859900']
    return random.choice(colors)

fd = {
    'fontsize': '32',
    'fontweight' : 'normal',
    'verticalalignment': 'baseline',
    'horizontalalignment': 'center',
}

for df, name, words in zip(subreddits, subreddit_names, subreddit_words):
    wc = wordcloud.WordCloud(width=1000, height=500, collocations=False, 
                             background_color="#fdf6e3", color_func=col_func, 
                             max_words=200,random_state=np.random.randint(1,8)
                            ).generate_from_frequencies(dict(Counter(words)))
    fig, ax = plt.subplots(figsize=(20,10))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(name, pad=24, fontdict=fd)
selftext_subreddits = [askReddit_df, explainlikeimfive_df, iAmA_df, jokes_df]
selftext_subreddit_names = ['AskReddit', 'explainlikeimfive', 'IAmA', 'Jokes']
selftext_subreddit_words = []

for df, name in zip(selftext_subreddits, selftext_subreddit_names):
    selftexts = ' '.join([x.lower() for x in df['selftext']])
    for k,v in contractions.items():
        if isinstance(v, list):
            v = random.choice(v)
        selftexts = selftexts.replace(k.lower(), v.lower())
    words = list(tokenizer.tokenize(selftexts))
    words = [x for x in words if x not in stopwords.words('english')]
    selftext_subreddit_words.append(words)
    print(name, '*****************', sep='\n')
    pprint(Counter(words).most_common(25), compact=True)
    print()
for df, name, words in zip(selftext_subreddits, selftext_subreddit_names, selftext_subreddit_words):
    wc = wordcloud.WordCloud(width=1000, height=500, collocations=False, 
                             background_color="#002b36", color_func=col_func, 
                             max_words=200, random_state=np.random.randint(1,8)
                            ).generate_from_frequencies(dict(Counter(words)))
    plt.figure(figsize=(20,10))
    plt.imshow(wc, interpolation='bilinear')
    _ = plt.axis("off")
    _ = plt.title(name, fontdict=fd, pad=24)
from nltk.util import ngrams
subreddit_2ngrams = []
for df, name in zip(subreddits, subreddit_names):
    ng = [ngrams(tokenizer.tokenize(tw.lower()), 
                 n=2) for tw in df['title']]
    # flattening the list
    ng = [x for y in ng for x in y]
    subreddit_2ngrams.append(ng)
    print(name, '*****************', sep='\n')
    pprint(Counter(ng).most_common(25), compact=False)
    print()
# flattening the list
subreddit_2ngrams_f = [x for y in subreddit_2ngrams for x in y]
# removing 2-grams that contain stop words
tmp = []
for n in subreddit_2ngrams_f:
    f = 0
    for w in n:
        if w in stopwords.words('english'):
            f = 1
    if f == 0:
        tmp.append(n)
subreddit_2ngrams_f = tmp
pprint(Counter(subreddit_2ngrams_f).most_common(50), compact=False)
for df, name, ngrams_2 in zip(subreddits, subreddit_names, subreddit_2ngrams):
    wc = wordcloud.WordCloud(width=2000, height=1000, 
                             collocations=False, background_color="black", 
                             colormap="Set3", max_words=66,
                             normalize_plurals=False,
                             regexp=r".+", 
                             random_state=7).generate_from_frequencies(dict(Counter([x + ' ' + y for x,y in ngrams_2])))
    plt.figure(figsize=(20,15))
    plt.imshow(wc, interpolation='bilinear')
    _ = plt.axis("off")
    _ = plt.title(name, fontdict=fd, pad=24)
subreddit_3ngrams = []
for df, name in zip(subreddits, subreddit_names):
    ng = [ngrams(tokenizer.tokenize(tw.lower()), 
                 n=3) for tw in df['title']]
    # flattening the list
    ng = [x for y in ng for x in y]
    subreddit_3ngrams.append(ng)
    print(name, '*****************', sep='\n')
    pprint(Counter(ng).most_common(25), compact=False)
    print()
subreddit_4ngrams = []
for df, name in zip(subreddits, subreddit_names):
    ng = [ngrams(tokenizer.tokenize(tw.lower()), 
                 n=4) for tw in df['title']]
    # flattening the list
    ng = [x for y in ng for x in y]
    subreddit_4ngrams.append(ng)
    print(name, '*****************', sep='\n')
    pprint(Counter(ng).most_common(25), compact=False)
    print()
fig, axes = plt.subplots(6, 3, figsize=(20,30), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.5, wspace=0.4)
# fig.text(0.5, 0.04, 'Title Length', ha='center', fontdict={'size':'18'})
for ax, df, name in zip(axes.flat, subreddits, subreddit_names):
    sns.distplot(df["title_length"], kde=False, hist_kws={'alpha': 1}, color="#0747A6", ax=ax)
    ax.set_title(name, fontdict={'size': 16}, pad=14)
    ax.set(xlabel="Title length", ylabel="Number of posts")
fig, axes = plt.subplots(6, 3, figsize=(20,30))
fig.subplots_adjust(hspace=0.5, wspace=0.4)
# fig.text(0.5, 0.04, 'Number of Comments', ha='center', fontdict={'size':'18'})
# fig.text(0.04, 0.5, 'Number of Posts', va='center', rotation='vertical', fontdict={'size':'18'})
for ax, df, name in zip(axes.flat, subreddits, subreddit_names):
    sns.distplot(df["num_comments"], kde=False, hist_kws={'alpha': 1}, color="#0747A6", ax=ax)
    ax.set_title(name, fontdict={'size': 16}, pad=14)
    ax.set(xlabel="Number of comments", ylabel="Number of posts")
medians = []
for df in subreddits:
    medians.append(df['num_comments'].median())

plt.rc('axes', labelpad=16)
fig, ax = plt.subplots(figsize=(14,8))
d = pd.DataFrame({'subreddit': subreddit_names, 'num_comments_median': medians})
sns.barplot(x="subreddit", y="num_comments_median", data=d, palette=sns.cubehelix_palette(n_colors=24, reverse=True), ax=ax);
ax.set(xlabel="Subreddit", ylabel="Median");
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
plt.rc('axes', labelpad=8)
fig, axes = plt.subplots(6, 3, figsize=(20,30))
fig.subplots_adjust(hspace=0.5, wspace=0.4)
for ax, df, name in zip(axes.flat, subreddits, subreddit_names):
    sns.distplot(df["ups"], kde=False, hist_kws={'alpha': 0.5}, color="#0747A6", ax=ax)
    sns.distplot(df["downs"], kde=False, hist_kws={'alpha': 0.5}, color="#FF5630", ax=ax)
    ax.set_title(name, fontdict={'size': 16}, pad=14)
    ax.set(xlabel="Number of upvotes/downvotes", ylabel="Number of posts")
comments = pd.DataFrame(columns=['subreddit', 'score'])
n = []
l = []
for df, name in zip(subreddits, subreddit_names):
    cl = list(df['score'])
    l.extend(cl)
    n.extend([name] * len(cl))
comments['subreddit'] = pd.Series(n)
comments['score'] = pd.Series(l)
fig, ax = plt.subplots(figsize=(20, 20))
sns.violinplot(x='score', y='subreddit', data=comments, scale='width', inner='box', ax=ax);
fig, axes = plt.subplots(8, 2, figsize=(20,60))
fig.subplots_adjust(hspace=0.7, wspace=0.3)
for ax, df, name in zip(axes.flat, subreddits, subreddit_names):
    sns.heatmap(df[['score', 'ups', 'downs', 'num_comments', 'title_length', 'num_capitalized']].corr(), annot=True, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)
    ax.set_title(name, fontdict={'size':'18'}, pad=14)
    ax.set(xlabel="", ylabel="")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)