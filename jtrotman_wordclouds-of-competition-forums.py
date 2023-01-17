import gc, os, re, sys, time

import pandas as pd, numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

from IPython.display import HTML, display

from wordcloud import WordCloud

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.manifold import TSNE

import plotly.express as px

from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")



MK = Path(f'../input/meta-kaggle')

ID = 'Id'

FORUM_ID = 'ForumId'

HOST = 'https://www.kaggle.com'



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



def simple_slug(txt):

    return re.sub('[^a-zA-Z0-9\-_]+', '-', txt.lower())



def html_to_text(r):

    return BeautifulSoup(r, 'html').text



def search_url(q):

    return f'https://www.kaggle.com/search?q={q}'
NROWS = None # For testing on subset



# Competitions

comps = pd.read_csv(MK / 'Competitions.csv', parse_dates=['DeadlineDate'], index_col=ID)

tags = pd.read_csv(MK / 'Tags.csv', index_col=ID)

ctags = pd.read_csv(MK / 'CompetitionTags.csv')

ctags['Slug'] = ctags.TagId.map(tags.Slug)

comps['Tags'] = ctags.groupby('CompetitionId').Slug.apply(" : ".join)

comps['Tags'] = comps['Tags'].fillna("none")

comps['Year'] = comps.DeadlineDate.dt.year

comps = comps.drop_duplicates(subset=['ForumId'], keep='last') # 575380 and 585319



# Forum Details

forums = pd.read_csv(MK / 'Forums.csv', index_col=ID)

topics = pd.read_csv(MK / 'ForumTopics.csv', index_col=ID)

topics.Title.fillna('', inplace=True)



# Forum Messages

msgs = pd.read_csv(MK / 'ForumMessages.csv', index_col=ID, nrows=NROWS)

msgs = msgs.dropna(subset=['Message'])

msgs.insert(0, 'ForumId', msgs.ForumTopicId.map(topics.ForumId))

msgs.insert(0, 'ParentForumId', msgs.ForumId.map(forums.ParentForumId))

text = ('<html>' + msgs.Message + '</html>').apply(html_to_text)

text = text.str.replace(r'(https?://\S+|\[/?quote.*?\])', ' ') # strip URLs, [quote] 

text = text.str.replace(r'([\W_]{4,})', ' ') # and junk

text = text.str.replace(r'([a-fA-F0-9]{12,})', ' ') # long hash-like strings

# Add topic titles to each post - this over-weights the title words a little,

#  depending on how many messages are in a topic

msgs['Text'] = msgs.ForumTopicId.map(topics.Title) + " " + text

msgs.shape
forums.shape
forums.describe(include='all').T
top_level = forums[forums.ParentForumId.isnull()].copy()

top_level['ForumCount'] = forums.groupby('ParentForumId').size()

top_level
colormaps = [

    # Sequential

    'Purples',

    'Blues',

    'Greens',

    'Oranges',

    'Reds',

    'YlOrBr',

    'YlOrRd',

    'OrRd',

    'PuRd',

    'RdPu',

    'BuPu',

    'GnBu',

    'PuBu',

    'PuBuGn',

    'BuGn',

    'YlGn',

    # Qualitative

    'Paired',

    'Accent',

    'Set1',

    'Set2',

    'Set3',

    'tab10',

    'tab20',

    # Sequential2

    'spring',

    'summer',

    'autumn',

    'winter',

    'cool',

    'Wistia',

    # Miscellaneous

    'gist_rainbow',

    'rainbow'

]
np.random.seed(42) # what else?

np.random.shuffle(colormaps)
def competition_html(forumid):

    df = comps[comps['ForumId'] == forumid]

    if len(df) != 1:

        return ""

    c = df.iloc[0]

    return (

        '<p>'

        f'<i>{c.HostSegmentTitle} Competition</i>:'

        f'   <b><a target=_blank href="{HOST}/c/{c.Slug}">{c.Title}</a></b>'

        f'   "<i>{c.Subtitle}</i>"'

        '<br/>'

        f'<i>TotalTeams</i>: <b>{c.TotalTeams}</b>'

        '<br/>'

        f'<i>DeadlineDate</i>: <b>{c.DeadlineDate.strftime("%c")}</b>'

    )
NTOP = 200

SHOW_TOP_WORDS = 20



# Using a class is better than one big function.

# For example you can fork the Notebook and have a look at the 'tfv' member.

class CloudGenerator:

    def __init__(self, tag, par, max_df=0.95):

        self.par = par

        self.tag = tag

        docs = {}

        # One big document per FORUM_ID.

        # Note: code in 'run' relies on Python 3 feature of storing key/value pairs

        #  in order they were added.

        for fid, df in par.groupby(FORUM_ID):

            docs[fid] = '\n'.join(df.Text)



        self.tfv = StemmedTfidfVectorizer(ngram_range=(1, 1),

                                     max_df=max_df,

                                     dtype=np.float32,

                                     stop_words='english')

        self.xall = self.tfv.fit_transform(docs.values())

        self.words = self.tfv.get_feature_names()

        self.ids = list(docs.keys())

        self.rows = []



    def save(self):

        tag = self.tag

        # save the stop words (determined by the max_df parameter)

        np.savetxt(f'{tag}_stop_words.txt', list(sorted(self.tfv.stop_words_)), '%s')

        # save stats of all forums to one file

        cols = [ FORUM_ID, 'count', 'max', 'mean' ] + [f'tok{i}' for i in range(NTOP)]

        df = pd.DataFrame(self.rows, columns=cols).set_index(FORUM_ID)

        df.insert(0, 'Title', df.index.map(forums.Title))

        df.to_csv(f'{tag}_word_stats.csv')

        

    def run(self):

        for row, (fid, df) in enumerate(self.par.groupby(FORUM_ID)):

            x = self.xall[row]

            s = pd.Series(index=self.words, data=x.toarray().ravel())

            s = s.sort_values(ascending=False)

            

            # save top words for CSV output

            l = s.head(NTOP).index.str.replace(' ', '_').tolist()

            self.rows.append([fid, (s > 0).sum(), s.max(), s.mean()] + l)

        

            title = forums.Title[fid]

            nchars = df.Message.str.len().sum()

            ntopics = df.ForumTopicId.nunique()

            nmsg = df.shape[0]

            top = s.head(SHOW_TOP_WORDS).index

            top = [f"<a href='{search_url(w)}'>{w}</a>" for w in top]

            top = ', '.join(top)

            query = title

        

            html = f"<h1 id='{simple_slug(title)}'>{title}</h1>"

            html += competition_html(fid)

            url = f"{HOST}/search?q={query}+in%3Atopics"



            html += (

                f"<h3>Forum</h3>"

                f"<ul>"

                f"<li>Search Kaggle for <a href='{url}'>{query}</a> in topics"

                f"<li>{ntopics} topics; {nmsg/ntopics:.1f} messages per topic"

                f"<li>{nmsg} messages; {nchars} raw characters; {nchars/nmsg:.0f} chars per message"

                f"<li>{df.PostUserId.nunique()} unique users"

                f"<li>Top {SHOW_TOP_WORDS} words: {top}"

                f"</ul>"

            )

            

            wc = WordCloud(background_color='black',

                           width=800,

                           height=600,

                           colormap=colormaps[row % len(colormaps)],

                           collocations=False,

                           random_state=row,

                           min_font_size=10,

                           max_font_size=200).generate_from_frequencies(s[s>0])

            

            if False:

                # wordcloud library now supports SVG

                #   - but needs latest docker image; and

                #   - renders poorly on this site, with overlapping words

                html += wc.to_svg()

                display(HTML(html))

            else:

                display(HTML(html))

                fig, ax = plt.subplots(figsize=(12, 9))

                ax.imshow(wc, interpolation='bilinear')

                ax.axis('off')

                plt.tight_layout()

                plt.show()
cg = CloudGenerator('Competitions', msgs.query("ParentForumId==8"))

cg.run()

cg.save()
forums.query("ParentForumId==9")
cg2 = CloudGenerator('General', msgs.query("ParentForumId==9")) #, max_df=1.0

cg2.run()

cg2.save()
NSVD = 120

svd = TruncatedSVD(n_components=NSVD, random_state=42)

xc = svd.fit_transform(cg.xall)

np.round(svd.explained_variance_ratio_.cumsum(), 2)
svd_df = pd.DataFrame(xc, index=list(map(int,cg.ids))).add_prefix('svd')

svd_df['Title'] = forums['Title']

svd_df.to_csv("CompetitionForumsSVD.csv", index_label=FORUM_ID)

svd_df.shape
tsne = TSNE(perplexity=20,

            early_exaggeration=1,

            init='pca',

            method='exact',

            learning_rate=5,

            n_iter=5000)

x2 = tsne.fit_transform(xc)

tsne_df = pd.DataFrame(x2, index=list(map(int, cg.ids))).add_prefix('tsne')

tsne_df.shape
CTYPE = 'CType'

comps[CTYPE] = "Default"

comps.loc[comps.Title.str.contains(r"Santa\b"), CTYPE] = 'Santa'

comps.loc[comps.Tags.str.contains("tabular-"), CTYPE] = 'Tabular'

comps.loc[comps.Tags.str.contains("image-"), CTYPE] = 'Image'

comps.loc[comps.Tags.str.contains("basketball"), CTYPE] = 'Basketball'

comps[CTYPE].value_counts()
forums_full = forums.join(tsne_df)

forums_full['TopicCount'] = forums_full.index.map(topics.ForumId.value_counts())

forums_full = forums_full.join(comps.reset_index().set_index(FORUM_ID).drop(["Title"], 1))

forums_full = forums_full.dropna(subset=['Year'] + list(tsne_df.columns))

forums_full.shape
forums_full.to_csv("CompetitionForumsTSNE.csv", index_label=FORUM_ID)
# Forums TSNE

tmp = forums_full.assign(DeadlineDate=forums_full.DeadlineDate.dt.strftime('%c'))

fig = px.scatter(tmp,

                 title='Competition Forums',

                 x='tsne0',

                 y='tsne1',

                 symbol=CTYPE,

                 hover_name='Title',

                 hover_data=[

                     'EvaluationAlgorithmAbbreviation', 'TopicCount',

                     'DeadlineDate', 'TotalTeams', 'Tags'

                 ],

                 color='Year')

fig.update_traces(marker=dict(size=9,

                              line=dict(width=1, color='black')),

                  selector=dict(mode='markers'))

fig.update_layout(height=750, showlegend=False)