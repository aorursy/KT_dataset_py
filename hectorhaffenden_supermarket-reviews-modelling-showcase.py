# Import libraries

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from datetime import datetime



# libraries for visualization

from matplotlib import pyplot as plt

import seaborn as sns

import pyLDAvis

import pyLDAvis.gensim



# For latent dirichlet allocation

import spacy

import gensim

from gensim import corpora



# For modelling and ELI5 analysis

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import plot_confusion_matrix

import eli5



import nltk 

import string

import re

import pandas as pd

import numpy as np



import random

import datetime as dt



from matplotlib import pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib.ticker import MaxNLocator



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import seaborn as sns





from plotly.subplots import make_subplots



import plotly

#import plotly.plotly as py

from plotly import graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline





import PIL

from PIL import Image



from nltk.probability import FreqDist



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import NMF





from sklearn.feature_extraction.text import CountVectorizer



import pyLDAvis.gensim

# Read in data

data = pd.read_csv('/kaggle/input/trust-pilot-reviews/uk_supermarkets_trustpilot_reviews.csv', index_col = 0)

data.head()
# Download some stopwords

nltk.download('wordnet')





def remove_punct(text):

    text  = "".join([char for char in text if char not in string.punctuation])

    text = re.sub('[0-9]+', '', text)

    return text



def tokenization(text):

    text = re.split('\W+', text)

    return text



def remove_stopwords(text, stopword):

    text = [word for word in text if word not in stopword]

    return text



def lemmatizer(text, wn):

    text = [wn.lemmatize(word) for word in text]

    return text







def clean_data(df):

    ## Step 1: Remove duplicate entries

    df.drop_duplicates(inplace=True)

    

    ## Step 2: Remove punctuation

    punctuation = ['.', '?', '!', '$', '£', '\'', ',']

    word_cols = ['title', 'content']

    for col in word_cols:

        for punc in punctuation:

            df[col] = df[col].str.replace(punc, '')

        # And make all lower case

        df[col] = df[col].str.lower()

        df[f'{col}_punct'] = df[col].apply(lambda x: remove_punct(x))

        df[f'{col}_tokenized'] = df[f'{col}_punct'].apply(lambda x: tokenization(x.lower()))

        stopword = nltk.corpus.stopwords.words('english')

        df[f'{col}_nonstop'] = df[f'{col}_tokenized'].apply(lambda x: remove_stopwords(x, stopword))

        wn = nltk.WordNetLemmatizer()

        df[f'{col}_lemmatized'] = df[f'{col}_nonstop'].apply(lambda x: lemmatizer(x, wn))

        df[f'{col}_clean'] = df[f'{col}_lemmatized'].str.join(' ')

        df.drop([f'{col}_punct', f'{col}_tokenized', f'{col}_nonstop', f'{col}_lemmatized'],

                axis = 1, inplace=True)

    

    # Now let's add some date features

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').astype('datetime64[ns]')

    df['year'] = df['date'].dt.year.astype('int16')

    df['month'] = df['date'].dt.month.astype('int16')

    df['week'] = df['date'].dt.week.astype('int16')

    df['day'] = df['date'].dt.day.astype('int16')

    df['wday'] = df['date'].dt.weekday.astype('int16')

    

    # Change words to number of stars

    dict_ratings = {'Excellent': 5, 'Great': 4, 'Average': 3, 'Poor': 2, 'Bad': 1}

    df['num_stars'] = df['rating'].replace(dict_ratings)

    

    return df



def create_fea(df):

    df['title_num_words'] = df['title'].str.split(' ').str.len()

    df['content_num_words'] = df['content'].str.split(' ').str.len()

    df['title_num_char'] = df['title'].str.len()

    df['content_num_char'] = df['content'].str.len()

    return df



def remove_custom_stopwords(df, custom_stop = ['morrisons'],

                            cols = ['title', 'content', 'title_clean', 'content_clean']):

    for col in cols:

        for punc in custom_stop:

            df[col] = df[col].str.replace(punc, '')

    return df

data = clean_data(data)

data.head()
a = [item for sublist in data['content_clean'].str.split().values for item in sublist]

pd.Series(a).value_counts().head(20)
custom_stopwords = data['company'].unique()

data = remove_custom_stopwords(df = data,

                        custom_stop = custom_stopwords,

                        cols = ['title_clean', 'content_clean'])

# Step 4: replace content and title with the clean versions

REPLACE = True

if REPLACE:

    data['title'] = data['title_clean']

    data['content'] = data['content_clean']

    data = data.drop(['title_clean', 'content_clean'], axis = 1)

    

# Step 5: Create quantitative features

data = create_fea(data)

data.shape
data.head(3)
data = data[15 > data['content'].str.split().apply(lambda x: [len(i) for i in x]).map(

        lambda x: np.mean(x))]
top_pct_to_drop = 0.02

data = data.sort_values('content_num_words', ascending=False).iloc[round(data.shape[0] * top_pct_to_drop):,:]
print("Oldest review:", data['date'].min(), ", Newest review:", data['date'].max())
REMOVE_OLD = False # For example, set REMOVE_OLD to '2015-01-01'

if REMOVE_OLD:

    data = data[data['date'] > REMOVE_OLD]
data['num_stars'].value_counts()
def plot_pie_chart(df):



    temp = df.groupby('num_stars')['title'].count().reset_index().sort_values('num_stars')

    labels = temp['num_stars'].values

    cols = ['lightblue', 'red', 'green', 'purple', 'orange']

    plt.pie(temp['title'], radius=2, autopct = '%0.1f%%',

            shadow = True, explode = [0.2,0,0,0,0.2],

            startangle = 0, labels = labels, colors = cols)

    plt.title('Pie chart showing a breakdown of ratings', fontsize=18, y = 1.5)

    plt.show()

    

plot_pie_chart(df = data)
def plot_star_funnel(df):

    temp = df.groupby('num_stars')['title'].count().reset_index().sort_values('num_stars')

    fig = go.Figure(go.Funnelarea(

        text = temp['num_stars'],

        values = temp['title'],

        title = {'position': 'top center', 'text': 'Funnel chart of ratings'}

    ))

    

    fig.update_layout(

        titlefont=dict(

            family="InterFace",

            size=30,

        )

    )

    #fig.show()

    iplot(fig)

    

plot_star_funnel(data)
def stratified_sample_df(df, col, n_samples):

    n = min(n_samples, df[col].value_counts().min())

    df_ = df.groupby(col).apply(lambda x: x.sample(n))

    df_.index = df_.index.droplevel(0)

    return df_



STRATIFY_REVIEWS = False

if STRATIFY_REVIEWS:

    data = stratified_sample_df(data, 'num_stars', 1000)
# Source - https://www.kaggle.com/datafan07/disaster-tweets-nlp-eda-bert-with-transformers

def plot_dist3(df, feature, title):

    # Creating a customized chart. and giving in figsize and everything.

    fig = plt.figure(constrained_layout=True, figsize=(18, 8))

    # Creating a grid of 3 cols and 3 rows.

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)



    # Customizing the histogram grid.

    ax1 = fig.add_subplot(grid[0, :2])

    # Set the title.

    ax1.set_title('Histogram')

    # plot the histogram.

    sns.distplot(df.loc[:, feature],

                 hist=True,

                 kde=True,

                 ax=ax1,

                 color='#e74c3c')

    ax1.set(ylabel='Frequency')

    ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))



    # Customizing the ecdf_plot.

    ax2 = fig.add_subplot(grid[1, :2])

    # Set the title.

    ax2.set_title('Empirical CDF')

    # Plotting the ecdf_Plot.

    sns.distplot(df.loc[:, feature],

                 ax=ax2,

                 kde_kws={'cumulative': True},

                 hist_kws={'cumulative': True},

                 color='#e74c3c')

    ax2.xaxis.set_major_locator(MaxNLocator(nbins=20))

    ax2.set(ylabel='Cumulative Probability')



    # Customizing the Box Plot.

    ax3 = fig.add_subplot(grid[:, 2])

    # Set title.

    ax3.set_title('Box Plot')

    # Plotting the box plot.

    sns.boxplot(x=feature, data=df, orient='v', ax=ax3, color='#e74c3c')

    ax3.yaxis.set_major_locator(MaxNLocator(nbins=25))



    

    

    

    lims = 0, df.loc[:, feature].max()



    ax1.set_xlim(lims)

    ax2.set_xlim(lims)

    

    

    plt.suptitle(f'{title}', fontsize=18)





def plot_word_len_histogram(textno, textye):

    

    """A function for comparing average word length"""

    

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), sharey=True)

    

    left = textno.str.split().apply(lambda x: [len(i) for i in x]).map(

        lambda x: np.mean(x))

    sns.distplot(left, ax=axes[0], color='#e74c3c')

    

    right = textye.str.split().apply(lambda x: [len(i) for i in x]).map(

        lambda x: np.mean(x))

    sns.distplot(right, ax=axes[1], color='#e74c3c')

    

    mx_l = max(left)

    mx_r = max(right)

    

    mi_l = min(left)

    mi_r = min(right)

    lims = min(mi_r, mi_l), max(mx_r, mx_l)

    axes[0].set_xlim(lims)

    axes[1].set_xlim(lims)

    axes[0].set_xlabel('Word Length')

    axes[0].set_ylabel('Frequency')

    axes[0].set_title('Positive reviews')

    axes[1].set_xlabel('Word Length')

    axes[1].set_title('Negative reviews')

    

    fig.suptitle('Mean Word Lengths', fontsize=24, va='baseline')

    fig.tight_layout()

    

    
plot_dist3(data[data['num_stars'] == 5], 'content_num_char',

       'Characters Per "Positive review')

plot_dist3(data[data['num_stars'] == 1], 'content_num_char',

       'Characters Per "Negative review')
plot_word_len_histogram(data[data['num_stars'] == 5]['content'],

                       data[data['num_stars'] == 1]['content'])
plot_dist3(data[data['num_stars'] == 5], 'content_num_words',

       'Words Per "Positive review')
plot_dist3(data[data['num_stars'] == 1], 'content_num_words',

       'Words Per "Negative review')
sns.set(font_scale = 2)

g = sns.FacetGrid(data, col='num_stars', height=4)

g.map(plt.hist,'content_num_char')

plt.subplots_adjust(top=0.8)

g.fig.suptitle(f'Size of review distribution, by number of stars')

plt.show()
# This is not my creation - I have lost the surce but I will try and find and credit, please message me if you know where to find this

def ngrams(df, n, title, mx_df = 0.9, content_or_title = 'content'):

    """

    A Function to plot most common ngrams

    content_or_title - use the content of the review, or the title

    mx_df: Ignore document frequency strictly higher than the given threshold

    """

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes = axes.flatten()

    #plt.rcParams.update({'font.size': 25})

    for rate, j in zip([5, 1], axes):

        new = df[df['num_stars'] == rate][content_or_title].str.split()

        new = new.values.tolist()

        corpus = [word for i in new for word in i]



        def _get_top_ngram(corpus, n=None):

            #getting top ngrams

            vec = CountVectorizer(ngram_range=(n, n),

                                  max_df=mx_df,

                                  stop_words='english').fit(corpus)

            bag_of_words = vec.transform(corpus)

            sum_words = bag_of_words.sum(axis=0)

            words_freq = [(word, sum_words[0, idx])

                          for word, idx in vec.vocabulary_.items()]

            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

            return words_freq[:15]



        top_n_bigrams = _get_top_ngram(df[df['num_stars'] == rate][content_or_title], n)[:15]

        x, y = map(list, zip(*top_n_bigrams))

        sns.barplot(x=y, y=x, palette='plasma', ax=j)



        

        

    title_font = {'fontname':'Arial', 'size':'24', 'color':'black', 'weight':'normal',

              'verticalalignment':'bottom'}

    lab_font = {'fontname':'Arial', 'size':'20'}

    

    

    axes[0].set_title(f'Positive reviews - for {content_or_title} of review', **title_font)

    axes[1].set_title(f'Negative reviews - for {content_or_title} of review', **title_font)

    axes[0].set_xlabel('Count', **lab_font)

    axes[0].set_ylabel('Words', **lab_font)

    axes[1].set_xlabel('Count', **lab_font)

    axes[1].set_ylabel('Words', **lab_font)

    

    axes[0].tick_params(axis='both', which='major', labelsize=15)

    axes[1].tick_params(axis='both', which='major', labelsize=15)

    

    #fig.suptitle(title, fontsize=24, va='baseline')

    plt.tight_layout()
ngrams(df = data, n = 1, title = 'Most Common Unigrams', mx_df = 0.9, content_or_title = 'content')

ngrams(df = data, n = 1, title = 'Most Common Unigrams', mx_df = 0.9, content_or_title = 'title')
ngrams(df = data, n = 2, title = 'Most Common Bigrams', mx_df = 0.9, content_or_title = 'content')

ngrams(df = data, n = 2, title = 'Most Common Bigrams', mx_df = 0.9, content_or_title = 'title')
ngrams(df = data, n = 3, title = 'Most Common Trigrams', mx_df = 0.9, content_or_title = 'content')

ngrams(df = data, n = 3, title = 'Most Common Trigrams', mx_df = 0.9, content_or_title = 'title')
def display_topics(text, no_top_words, topic, components = 10):

    """

    A function for determining the topics present in our corpus with nmf

    """

    no_top_words = no_top_words

    tfidf_vectorizer = TfidfVectorizer(

        max_df=0.90, min_df=25, max_features=5000, use_idf=True)

    tfidf = tfidf_vectorizer.fit_transform(text)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    doc_term_matrix_tfidf = pd.DataFrame(

        tfidf.toarray(), columns=list(tfidf_feature_names))

    nmf = NMF(n_components=components, random_state=0,

              alpha=.1, init='nndsvd', max_iter = 5000).fit(tfidf)

    print(topic)

    for topic_idx, topic in enumerate(nmf.components_):

        print('Topic %d:' % (topic_idx+1))

        print(' '.join([tfidf_feature_names[i]

                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

    

    
print('Topics for TITLE of review')

display_topics(data[data['num_stars'] == 5]['title'], 

               no_top_words = 5,

               topic = 'Positive review topics \n',

               components = 2)

print('\n======================================\n')

print('\n======================================\n')

print('Topics for BODY of review')

display_topics(data[data['num_stars'] == 5]['content'], 

               no_top_words = 5,

               topic = 'Positive review topics \n',

               components = 10)
print('Topics for TITLE of review')

display_topics(data[data['num_stars'] == 1]['title'], 

               no_top_words = 5,

               topic = 'Negative review topics \n',

               components = 2)

print('\n======================================\n')

print('\n======================================\n')

print('Topics for BODY of review')

display_topics(data[data['num_stars'] == 1]['content'], 

               no_top_words = 5,

               topic = 'Negative review topics \n',

               components = 10)
DO_LDA = False

if DO_LDA:

    CONTENT_OR_TITLE = 'content'



    reviews_split = data[CONTENT_OR_TITLE].str.split()

    # creating the term dictionary of our corpus, where every unique term is assigned an index

    dictionary = corpora.Dictionary(reviews_split)

    # convert the list of reviews (reviews_2) into a Document Term Matrix 

    doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_split]

    # Creating the object for LDA model using gensim library

    LDA = gensim.models.LdaMulticore # .ldamodel.LdaModel

    # Build LDA model - we have picked 3 main topics

    # switch to sklearn.decomposition.LatentDirichletAllocation for consistency?

    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=3, random_state=100,

                    chunksize=1000, passes=50, workers=4)

    # print out the topics that our LDA model has learned

    lda_model.print_topics()

    

    pyLDAvis.enable_notebook()

    v = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary, plot_opts = 'ylab')

    v
DO_LDA = False

if DO_LDA:

    CONTENT_OR_TITLE = 'title'



    reviews_split = data[CONTENT_OR_TITLE].str.split()

    # creating the term dictionary of our corpus, where every unique term is assigned an index

    dictionary = corpora.Dictionary(reviews_split)

    # convert the list of reviews (reviews_2) into a Document Term Matrix 

    doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_split]



    # Creating the object for LDA model using gensim library

    LDA = gensim.models.LdaMulticore



    # Build LDA model - we have picked 3 main topics

    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=3, random_state=100,

                    chunksize=1000, passes=50, workers=4)

    # print out the topics that our LDA model has learned

    lda_model.print_topics()



    # Visualize the topics

    pyLDAvis.enable_notebook()

    v = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)

    v
def plot_wordcloud(df, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False, image = 'basket.png',

                  more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}):

    comments_text = str(df.values)

    mask = np.array(PIL.Image.open(image))

    

    stopwords = set(STOPWORDS)

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='white',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    mask = mask)

    wordcloud.generate(comments_text)

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'green', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()
plot_wordcloud(df = data[data['num_stars'] == 5]['content'],

               max_words=200,

               max_font_size=100, 

               title = 'Most common words of POSITIVE reviews',

               title_size=50,

               image = '../input/trust-pilot-reviews/basket.png',

              more_stopwords = {'store', 'order'})
plot_wordcloud(df = data[data['num_stars'] == 1]['content'],

               max_words=200,

               max_font_size=100, 

               title = 'Most common words of NEGATIVE reviews',

               title_size=50,

               image = '../input/trust-pilot-reviews/basket.png',

              more_stopwords = {'store', 'order', 'one'})
def time_series_slider(df, window = 7, add_count = False, add_var = False, add_kurt = False):

    mean_grp = df.groupby('date')['num_stars'].mean().rolling(window).mean().reset_index().iloc[window:,:]

    # Create figure

    fig = make_subplots(specs=[[{"secondary_y": True}]])



    fig.add_trace(

        go.Scatter(x=list(mean_grp['date']), y=list(mean_grp['num_stars']), name="rolling mean"))

    

    if add_count:

        mean_grp = df.groupby('date')['num_stars'].count().rolling(window).mean().reset_index().iloc[window:,:]

        fig.add_trace(go.Scatter(x=list(mean_grp['date']), y=list(mean_grp['num_stars']), name="rolling count"),

                     secondary_y = True)

    

    if add_var:

        mean_grp = df.groupby('date')['num_stars'].mean().rolling(window).std().reset_index().iloc[window:,:]

        fig.add_trace(go.Scatter(x=list(mean_grp['date']), y=list(mean_grp['num_stars']), name="rolling std"),

                     secondary_y = True)

    

    if add_kurt:

        mean_grp = df.groupby('date')['num_stars'].mean().rolling(window).kurt().reset_index().iloc[window:,:]

        fig.add_trace(go.Scatter(x=list(mean_grp['date']), y=list(mean_grp['num_stars']), name="rolling kurtosis"),

                     secondary_y = True)

    

    # Set title

    fig.update_layout(

        title_text="Time series with range slider and selectors"

    )



    # Add range slider

    fig.update_layout(title="Rolling mean of average reviews",

                      xaxis_title="Date",

                      yaxis_title="Value",

        xaxis=dict(

            rangeselector=dict(

                buttons=list([

                    dict(count=1,

                         label="1m",

                         step="month",

                         stepmode="backward"),

                    dict(count=6,

                         label="6m",

                         step="month",

                         stepmode="backward"),

                    dict(count=1,

                         label="YTD",

                         step="year",

                         stepmode="todate"),

                    dict(count=1,

                         label="1y",

                         step="year",

                         stepmode="backward"),

                    dict(step="all")

                ])

            ),

            rangeslider=dict(

                visible=True

            ),

            type="date"

        ),

        yaxis=dict(

           autorange = True,

           fixedrange= False

        )

    )

    fig['layout']['yaxis'].update(title = 'Number of stars', range = [0, 5], autorange = False)

    #fig.show()

    iplot(fig)
time_series_slider(df = data, window = 30, add_count = False, add_var = True, add_kurt = False)
def trend_barplots(df, mean_or_count = 'mean', plots = ['wday', 'day', 'week', 'month', 'year']):

    '''

    Input: 

        df: data

        mean_or_count: Takes values mean, or count, and decides what to plot

        plots: by which grouping do you want the plots

    Output: 

        displays barplots

    '''

    # Prep data

    for col in plots:

        if col == 'wday':

            df[col] = pd.to_datetime(df['date'], format='%Y-%m-%d').astype('datetime64[ns]').dt.weekday

        

        if mean_or_count == 'mean':

            grp_wday = df.groupby([col])['num_stars'].mean()

            lims = (1,5)

            title_besp = f'Average stars by {col}'

        elif mean_or_count == 'count':

            grp_wday = df.groupby([col])['num_stars'].count()

            # Set limit to 5% above the max

            lims = (0,grp_wday.max() + round(grp_wday.max() / 20))

            title_besp = f'Count of reviews by {col}'

        plt.figure(figsize=(18,6))

        sns.barplot(x=grp_wday.index.values,

                    y=grp_wday.values, palette='plasma')

        plt.ylim(lims)

        plt.tick_params(axis='both', which='major', labelsize=12)

        

        plt.title(title_besp, fontsize=20)

        plt.xlabel(f'{col}', fontsize=18)

        plt.ylabel('Number of stars', fontsize=16)

        #plt.tight_layout()

        plt.show()
trend_barplots(df = data,

               mean_or_count = 'count',

               plots = ['wday', 'day', 'week', 'month', 'year'])
trend_barplots(df = data,

               mean_or_count = 'mean',

               plots = ['wday', 'day', 'week', 'month', 'year'])
# Prep data

data_for_reg = data[data['num_stars'] != 3].copy()

data_for_reg.loc[:,'target'] = -9999

data_for_reg.loc[data_for_reg['num_stars'] < 3, 'target'] = 0 # 0 negative

data_for_reg.loc[data_for_reg['num_stars'] > 3, 'target'] = 1 # 1 positive



X_full = data_for_reg['content']

y_full = data_for_reg['target']





vect = TfidfVectorizer()

X = vect.fit_transform(X_full)



y = y_full



X_train, X_valid, y_train, y_valid = train_test_split(

    X, y, test_size=0.20, random_state=23, stratify=y)

X_train.shape, X_valid.shape


model = LogisticRegression()

model.fit(X_train, y_train)

print("Train Set Accuracy: {}".format(metrics.accuracy_score(model.predict(X_train), y_train)))

print("Train Set ROC: {}\n".format(metrics.roc_auc_score(model.predict(X_train), y_train)))



print("Validation Set Accuracy: {}".format(metrics.accuracy_score(model.predict(X_valid), y_valid)))

print("Validation Set ROC: {}".format(metrics.roc_auc_score(model.predict(X_valid), y_valid)))
print(metrics.classification_report(model.predict(X_valid), y_valid))
# Confusion Matrix\

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes = axes.flatten()

sns.set(font_scale=2.0)

for norm, j in zip(['true', None], axes):

    plot_confusion_matrix(model, X_valid, y_valid, normalize = norm, ax = j)

axes[0].set_title(f'Normalised confusion matrix', fontsize = 24)

axes[1].set_title(f'Raw confusion matrix', fontsize = 24)

plt.show()
target_names = [0, 1]

eli5.show_weights(model, vec=vect, top=100,

                  target_names=target_names)
for iteration in range(5):

    samp = random.randint(1,data.shape[0])

    print("Real Label: {}, on {}".format(data['rating'].iloc[samp], data['date'].iloc[samp]))

    display(eli5.show_prediction(model,data["content"].iloc[samp], vec=vect,

                         target_names=target_names))
from lightgbm import LGBMClassifier

model = LGBMClassifier(n_estimators=130,

                            n_jobs=4)



model.fit(X_train, y_train)

print("Train Set Accuracy: {}".format(metrics.accuracy_score(model.predict(X_train), y_train)))

print("Train Set ROC: {}\n".format(metrics.roc_auc_score(model.predict(X_train), y_train)))



print("Validation Set Accuracy: {}".format(metrics.accuracy_score(model.predict(X_valid), y_valid)))

print("Validation Set ROC: {}".format(metrics.roc_auc_score(model.predict(X_valid), y_valid)))
grp_by_company = data.groupby('company')['num_stars']

grp_by_company.describe()
def plot_ts_by_company(df):

    ts_grp_data = df.groupby(['company', 'year'])['num_stars'].mean().reset_index()

    sns.set(style="white")

    a4_dims = (15, 6)

    fig, ax = plt.subplots(figsize=a4_dims)

    sns.set(font_scale=1.5)

    sns.lineplot(data = ts_grp_data, x='year', y='num_stars', hue='company', ax = ax)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax.set_title('Average number of stars, by company, over time')

    ax.set_ylabel('Number of stars')

    ax.set_xlabel('Date')

    plt.show()
plot_ts_by_company(data)
a4_dims = (22.5, 6)

fig, ax = plt.subplots(figsize=a4_dims)

ax = sns.violinplot(x = "company", y = "num_stars", data = data)
a4_dims = (22.5, 6)

fig, ax = plt.subplots(figsize=a4_dims)

ax = sns.boxplot(x = "company", y = "num_stars", data = data)
queue_reviews = data[(data['content'].str.contains('queue')) | (data['title'].str.contains('queue'))] # Goes from 40k to 1.3k reviews

a4_dims = (22.5, 6)

fig, ax = plt.subplots(figsize=a4_dims)

ax = sns.violinplot(x = "company", y = "num_stars", data = queue_reviews)

plt.show()
lims = (1,5)

title_besp = f'Average stars by company from Queue reviews'



plt.figure(figsize=(18,6))

sns.barplot(data = queue_reviews.groupby('company')['num_stars'].mean().reset_index(),

            x = 'company', 

            y = 'num_stars',

            palette='plasma')



plt.ylim(lims)

plt.tick_params(axis='both', which='major', labelsize=12)



plt.title(title_besp, fontsize=20)

plt.xlabel(f'Company', fontsize=18)

plt.ylabel('Number of stars', fontsize=16)

plt.show()
lims = (1,5)

title_besp = f'Average stars by company from Online reviews'

online_reviews = data[(data['content'].str.contains('online')) | (data['title'].str.contains('online'))] # Goes from 40k to 1.3k reviews



plt.figure(figsize=(18,6))

sns.barplot(data = online_reviews.groupby('company')['num_stars'].mean().reset_index(),

            x = 'company', 

            y = 'num_stars',

            palette='plasma')



plt.ylim(lims)

plt.tick_params(axis='both', which='major', labelsize=12)



plt.title(title_besp, fontsize=20)

plt.xlabel(f'Company', fontsize=18)

plt.ylabel('Number of stars', fontsize=16)

plt.show()
pyramid_reviews = data[(data['content'].str.contains('online')) | (data['title'].str.contains('online'))] # Goes from 40k to 1.3k reviews

pyramid_reviews.loc[pyramid_reviews['num_stars'] == 2, 'num_stars'] = 1

pyramid_reviews.loc[pyramid_reviews['num_stars'] == 4, 'num_stars'] = 5

# Groupby company, and count the number of stars

one_stars = pyramid_reviews[['company', 'num_stars']][pyramid_reviews['num_stars'] == 1].groupby('company').count().reset_index()

five_stars = pyramid_reviews[['company', 'num_stars']][pyramid_reviews['num_stars'] == 5].groupby('company').count().reset_index()

one_stars['num_stars'] = -one_stars['num_stars']



plt.figure(figsize=(15,8))

bar_plot = sns.barplot(x='num_stars', y='company', data=one_stars, lw=3)

bar_plot = sns.barplot(x='num_stars', y='company', data=five_stars, lw=3)

bar_plot.set(xlabel="Count of 1 vs 5 star reviews", ylabel="Company",

             title = "Count of 1 (left) and 5 (right) star reviews for reviews containing \"online\" ")

plt.xlim(-1500, 1500)

plt.show()
pyramid_reviews = data[(data['content'].str.contains('variety')) | (data['title'].str.contains('variety'))] # Goes from 40k to 1.3k reviews

pyramid_reviews.loc[pyramid_reviews['num_stars'] == 2, 'num_stars'] = 1

pyramid_reviews.loc[pyramid_reviews['num_stars'] == 4, 'num_stars'] = 5

# Groupby company, and count the number of stars

one_stars = pyramid_reviews[['company', 'num_stars']][pyramid_reviews['num_stars'] == 1].groupby('company').count().reset_index()

five_stars = pyramid_reviews[['company', 'num_stars']][pyramid_reviews['num_stars'] == 5].groupby('company').count().reset_index()

one_stars['num_stars'] = -one_stars['num_stars']



plt.figure(figsize=(15,8))

bar_plot = sns.barplot(x='num_stars', y='company', data=one_stars, lw=3)

bar_plot = sns.barplot(x='num_stars', y='company', data=five_stars, lw=3)

bar_plot.set(xlabel="Count of 1 vs 5 star reviews", ylabel="Company",

             title = "Count of 1 (left) and 5 (right) star reviews for reviews containing \"variety\" ")

lim_val = max(abs(one_stars['num_stars'].min()), five_stars['num_stars'].max()) + 5

plt.xlim(-lim_val, lim_val)

plt.show()