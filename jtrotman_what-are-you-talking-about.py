# All users currently ranked here or higher are shown

SHOW_TOP_RANKS = 100



# Words to show as text/search links, above the word cloud

SHOW_TOP_WORDS = 10



# Top words per user to save in CSV file

SAVE_TEXT_WORDS = 200



# Colormap names:

# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

COLORS = {

    4: 'Wistia',  # GM

    3: 'autumn',  # Master

    2: 'spring',  # Expert

}

TIER_NAMES = ['Novice', 'Contributor', 'Expert', 'Master', 'Grand Master']

TIER_COLORS = ['green', 'blue', 'purple', 'orange', 'gold', 'black']
%matplotlib inline

import gc, os, re, sys, time

import pandas as pd, numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

from IPython.display import HTML, display

import plotly.express as px

from wordcloud import WordCloud

from bs4 import BeautifulSoup

import zlib

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")



MK = Path(f'../input/meta-kaggle')

ID = 'Id'

KEY = 'ForumId'



FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

EMOJI = ''.join([chr(c) for c in range(0x1f600, 0x1f641)])

EMOTICON = r"[:;8X]['`\-^]?[\)\(/dp]"



TOKEN_PATTERNS = [

    f"(?:[{EMOJI}])",

    f"(?:{EMOTICON})",

    r"(?:\bt-sne\b)",

    r"(?:\b\w[\w']+\w+\b)",

    r"(?:\b\w\.[\w\.]+\b)", # e.g.  i.e.

    r"(?:\b\w\w+\b)"

]

TOKEN_PATTERN = "|".join(TOKEN_PATTERNS)



# Copied from:

# https://github.com/GaelVaroquaux/my_topics/blob/master/topics_extraction.py

PROTECTED_WORDS = ['pandas', 'itertools', 'physics', 'keras']



def no_plural_stemmer(word):

    """ A stemmer that tries to apply only on plural. The goal is to keep

        the readability of the words.

    """

    word = word.lower()

    if word.endswith('s') and not (word in PROTECTED_WORDS

                                   or word.endswith('sis')):

        stemmed_word = stemmer.stem(word)

        if len(stemmed_word) == len(word) - 1:

            word = stemmed_word

    return word



# Not perfect but better than nothing

class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):

        analyzer = super(TfidfVectorizer, self).build_analyzer()

        return lambda doc: (no_plural_stemmer(w) for w in analyzer(doc))



# this is to ensure :-p :-D and XD are upper case

def fix_emoticons(s):

    return re.sub(f'({EMOTICON})', lambda m: m.group(1).upper(), s)



def simple_slug(txt):

    return re.sub('[^a-zA-Z0-9\-_]+', '-', txt.lower())



def search_url(q):

    return f'https://www.kaggle.com/search?q={q}'



def html_to_text(r):

    return BeautifulSoup(r, 'html').text



def compress(s):

    return zlib.compress(s.lower().encode('utf-8'))
def add_discussion_tier(dst, col):

    df = pd.read_csv(MK / 'UserAchievements.csv')

    df = df.query('AchievementType=="Discussion"').set_index('UserId')

    dst['DiscussionTier'] = dst[col].map(df.Tier)

    dst['DiscussionRanking'] = dst[col].map(df.CurrentRanking) 



users = pd.read_csv(MK / 'Users.csv', index_col=ID)

forums = pd.read_csv(MK / 'Forums.csv', index_col=ID)

topics = pd.read_csv(MK / 'ForumTopics.csv', index_col=ID)

posts_df = pd.read_csv(MK / 'ForumMessages.csv', index_col=ID).dropna(subset=['Message'])

add_discussion_tier(posts_df, 'PostUserId')

posts_df.insert(0, 'ForumId', posts_df.ForumTopicId.map(topics.ForumId))

posts_df.insert(0, 'ParentForumId', posts_df.ForumId.map(forums.ParentForumId))

posts_df.shape
posts_df.query('DiscussionTier>=3').shape[0]  # count posts by masters, GMs
# fork to try "Medal>0 and (DiscussionTier>=3 or DiscussionRanking<=@SHOW_TOP_RANKS)"

sub_df = posts_df.query("DiscussionTier>=3 or DiscussionRanking<=@SHOW_TOP_RANKS")

sub_df.shape
txt = sub_df.Message.apply(html_to_text)

txt = txt.str.replace(r'(?:https?://\S+)', ' ')   # strip URLs

txt = txt.str.replace(r'(?:\[/?quote.*?\])', ' ') # strip [quote] 

txt = txt.str.replace(r"(?:'s\b)", ' ')  # strip 's

txt = txt.str.replace(r'([a-fA-F0-9_\-]{12,})', ' ') # long hash-like strings

sub_df = sub_df.assign(Text=txt)

sub_df.shape
KEY_LIST = ['DiscussionRanking', 'PostUserId']

docs = {rank:'\n'.join(df.Text) for rank, df in sub_df.groupby(KEY_LIST)}

len(docs)
# Inherits from TfidfVectorizer

# Full list of settings here:

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

tfv = StemmedTfidfVectorizer(ngram_range=(1, 2),

                             max_df=0.95,

                             token_pattern=TOKEN_PATTERN,

                             dtype=np.float32,

                             stop_words='english')

xall = tfv.fit_transform(docs.values())

# Post processing!

tfv.vocabulary_ = {fix_emoticons(k):v for k,v in tfv.vocabulary_.items()}

words = tfv.get_feature_names()

np.savetxt(f'stop_words.txt', list(sorted(tfv.stop_words_)), '%s')
# Save IDF weights (rarer words have a higher weight)

idf = pd.Series(tfv.idf_, words).sort_values()

idf.to_frame('IDF').to_csv(f'idf_weights.csv', index_label='Words')
def generate_clouds():

    rows = []

    for row, ((rank, uid), df) in enumerate(sub_df.groupby(KEY_LIST)):

        x = xall[row]

        s = pd.Series(index=words, data=x.toarray().ravel())

        s = s.sort_values(ascending=False)

        l = s.head(SAVE_TEXT_WORDS).index.str.replace(' ', '_').tolist()

        

        tier = df.iloc[0].DiscussionTier

        days = df.PostDate.str[:10].nunique()

        chars = df.Message.str.len().sum()

        top = s.head(SHOW_TOP_WORDS).index

        top = [f"<a href='{search_url(w)}'>{w}</a>" for w in top]

        top = ', '.join(top)

        u = users.loc[uid]



        rows.append([rank, u.UserName, (s > 0).sum(), s.max(), s.mean()] + l)



        html = f"<h1 id={u.UserName}>#{rank:.0f} {u.DisplayName}</h1>"

        html += f"<ul>"

        html += f"<li><b>Discussion {TIER_NAMES[int(tier)]}</b>"

        html += f";   <a href='https://www.kaggle.com/{u.UserName}/discussion'>{u.UserName} discussion index</a>"

        html += f";   Registered: <b>{u.RegisterDate}</b>"

        html += f"<li>Posted in {df.ForumTopicId.nunique()} unique topics"

        html += f"<li>{days} unique days; {df.shape[0]/days:.1f} posts per day"

        html += f"<li>{df.shape[0]} messages; {chars} raw characters; {int(chars/df.shape[0])} chars per message"

        html += f"<li>Top {SHOW_TOP_WORDS} words: {top}"

        html += f"</ul>"

        html += f"<h3>Top Forums</h3>"



        display(HTML(html))

        topf = forums.loc[forums.index.intersection(df.ForumId)].Title.value_counts().to_frame("Message Count")

        topf.index.name = "Forum"

        display(topf.head(5))

        

        wc = WordCloud(background_color='black',

                       width=800,

                       height=600,

                       colormap=COLORS[tier],

                       font_path=FONT_PATH,

                       random_state=1 + row,

                       min_font_size=10,

                       max_font_size=200).generate_from_frequencies(s[s>0])

        fig, ax = plt.subplots(figsize=(12, 9))

        ax.imshow(wc, interpolation='bilinear')

        ax.axis('off')

        plt.tight_layout()

        plt.show()

        

    # save stats of all users to one file

    c1 = [ 'Rank', 'UserName', 'count', 'max', 'mean' ]

    c2 = [f'tok{i}' for i in range(SAVE_TEXT_WORDS)]

    df = pd.DataFrame(rows, columns=c1+c2).set_index('Rank')

    df.to_csv(f'user_word_stats.csv')
generate_clouds()
from sklearn.decomposition import TruncatedSVD

NSVD = 80

svd = TruncatedSVD(n_components=NSVD, random_state=42)

xc = svd.fit_transform(xall)

svd.explained_variance_ratio_.cumsum()
from sklearn.manifold import TSNE

tsne = TSNE(perplexity=20, early_exaggeration=1, init='pca', method='exact', learning_rate=5, n_iter=5000)

x2 = tsne.fit_transform(xc)



users_df = pd.DataFrame(docs.keys(), columns=KEY_LIST)

users_df = pd.concat((users_df, pd.DataFrame(x2).add_prefix('tsne')), axis=1)

users_df = users_df.set_index('PostUserId')

users_df = users_df.join(users)

users_df = users_df.sort_values('DiscussionRanking', ascending=False)
fig = px.scatter(

    users_df.assign(Tier=np.asarray(TIER_NAMES)[users_df.PerformanceTier],

                    Size=(10 / np.log1p(users_df.DiscussionRanking)).round(3),

                    Year=users_df.RegisterDate.str[-4:].astype(int)),

    title='Kaggle Writers 2D Semantic Space',

    x='tsne0',

    y='tsne1',

    #symbol='Year',

    size='Size',

    hover_name='DisplayName',

    hover_data=['UserName', 'RegisterDate', 'DiscussionRanking'],

    color='Tier',

    color_discrete_map=dict(zip(TIER_NAMES, TIER_COLORS)))

fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')),

                  selector=dict(mode='markers'))

fig.update_layout(height=750, showlegend=False)
_ = """

Rerun for recently finished competitions:



    2020-07-15 | Slug:trec-covid-information-retrieval

    2020-07-27 | Slug:prostate-cancer-grade-assessment

    2020-07-28 | Slug:alaska2-image-steganalysis

    2020-07-31 | Slug:hashcode-photo-slideshow

    2020-08-21 | Slug:landmark-retrieval-2020

    2020-08-21 | Slug:siim-isic-melanoma-classification

    2020-08-24 | Slug:global-wheat-detection





"""