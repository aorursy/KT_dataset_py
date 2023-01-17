from plotly.offline import init_notebook_mode, iplot

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from wordcloud import WordCloud
from textblob import TextBlob 

import plotly.plotly as py
from plotly import tools
import seaborn as sns
import pandas as pd
import string, os, random

init_notebook_mode(connected=True)
punc = string.punctuation
path = '../input/' # 'data/'
file_path = path + 'reuters-news-wire-archive/reuters-newswire-2017.csv'
df = pd.read_csv(file_path)
df.head(10)
df['word_count'] = df['headline_text'].apply(lambda x : len(x.split()))
df['char_count'] = df['headline_text'].apply(lambda x : len(x.replace(" ","")))
df['word_density'] = df['word_count'] / (df['char_count'] + 1)
df['punc_count'] = df['headline_text'].apply(lambda x : len([a for a in x if a in punc]))

df[['word_count', 'char_count', 'word_density', 'punc_count']].head(10)
# function to obtain the sentiment of the headline using textblob package
def get_polarity(text):
    try:
        pol = TextBlob(text).sentiment.polarity
    except:
        pol = 0.0
    return pol

# I have already computed the sentiments and saved the file if file is not present it will compute the sentiment in real time which could be time taking
pre_computed_path = path + 'precomputedpolarity2017headlines/precomputed_polarity.csv'
if os.path.isfile(pre_computed_path):
    df['polarity'] = pd.read_csv(pre_computed_path)['polarity']
else:
    df['polarity'] = df['headline_text'].apply(get_polarity)
    
df[['polarity']].tail(10)
df['month'] = df['publish_time'].apply(lambda x : str(x)[4:6])
df['date'] = df['publish_time'].apply(lambda x : str(x)[6:8])
df['hour'] = df['publish_time'].apply(lambda x : str(x)[8:10])
df['minute'] = df['publish_time'].apply(lambda x : str(x)[10:])

df[['hour', 'month', 'date', 'minute']].tail(10)
xwords = df.word_count
trace1 = go.Histogram(x=xwords, opacity=0.65, name="Word Count", marker=dict(color='rgba(171, 50, 96, 0.6)'))
data = [trace1]
layout = go.Layout(barmode='overlay',
                   title='Word Count of Headlines',
                   xaxis=dict(title='Word Count'),
                   yaxis=dict( title='Numer of Headlines'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
xchars = df.char_count
trace1 = go.Histogram(x=xchars, opacity=0.65, name="Word Count", marker=dict(color='rgba(12, 50, 196, 0.6)'))
data = [trace1]
layout = go.Layout(barmode='overlay',
                   title='Char Count of Headlines',
                   xaxis=dict(title='Char Count'),
                   yaxis=dict( title='Numer of Headlines'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
xwd = df.word_density
trace1 = go.Histogram(x=xwd, opacity=0.65, name="Word Count", marker=dict(color='rgba(0, 0, 0, 0.6)'))
data = [trace1]
layout = go.Layout(barmode='overlay',
                   title='Word Density of Headlines',
                   xaxis=dict(title='Word Density'),
                   yaxis=dict( title='Numer of Headlines'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
xpc = df.punc_count
trace1 = go.Histogram(x=xpc, opacity=0.75, name="Word Count", marker=dict(color='rgba(10, 220, 150, 0.6)'))
data = [trace1]
layout = go.Layout(barmode='overlay',
                   title='Punctuation Count of Headlines',
                   xaxis=dict(title='Punctuation Count'),
                   yaxis=dict( title='Numer of Headlines'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
def create_stack_bar_data(col):
    aggregated = df[col].value_counts()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    return x_values, y_values
x1, y1 = create_stack_bar_data('month')
trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="month", marker=dict(color='rgba(10, 220, 150, 0.6)'))

x2, y2 = create_stack_bar_data('date');
trace2 = go.Bar(x=x2, y=y2, opacity=0.75, name="month-date", marker=dict(color='rgba(0, 20, 50, 0.6)'));

fig = tools.make_subplots(rows=1, cols=2, print_grid=False);
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 1, 2);

fig['layout'].update(height=400, title='Month and Month-Date headline counts in 2017', legend=dict(orientation="h"));
iplot(fig, filename='simple-subplot');
x1, y1 = create_stack_bar_data('hour')
trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="hour", marker=dict(color='rgba(110, 1, 10, 0.3)'))

x2, y2 = create_stack_bar_data('minute');
trace2 = go.Bar(x=x2, y=y2, opacity=0.75, name="week-day", marker=dict(color='rgba(30, 30, 150, 0.3)'));

fig = tools.make_subplots(rows=1, cols=2, print_grid=False);
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 1, 2);

fig['layout'].update(height=400, title='Month and Month-Date headline counts in 2017', legend=dict(orientation="h"));
iplot(fig, filename='simple-subplot');
# time series of sentiment 

aggdf = df.reset_index().groupby(by=['month', 'date']).agg({'polarity':'mean'}).reset_index().rename(columns={'polairty':'sentiment'})
aggdf['month_date'] = "2017-" + aggdf['month'] + "-" + aggdf['date']

trace1 = go.Scatter(x = aggdf.month_date,
                    y = aggdf.polarity,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    )
data = [trace1]
layout = dict(title = 'How sentiment of news varied over time',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
sorteddf = df.sort_values(by='polarity')
posdf = sorteddf.tail(5000)
negdf = sorteddf.head(5000)

posdf[['headline_text']].tail(10)
negdf[['headline_text']].head(10)
pos_text_cln = " ".join(posdf.headline_text)
neg_text_cln = " ".join(negdf.headline_text)

# replacing some most common words present in these texts
noise_words = ['brief', 'say', 'update', 'trump', 'china']
for noise in noise_words:
    pos_text_cln = pos_text_cln.lower().replace(noise," ")
    neg_text_cln = neg_text_cln.lower().replace(noise, " ")

def green_color(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'hsl({:d}, 80%, {:d}%)'.format(random.randint(85, 140), random.randint(60, 80))

def red_color(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'hsl({:d}, 80%, {:d}%)'.format(random.randint(0, 35), random.randint(60, 80))
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15, 8])
wordcloud1 = WordCloud(background_color='white', height=400).generate(pos_text_cln)

ax1.imshow(wordcloud1.recolor(color_func=green_color, random_state=3),interpolation="bilinear")
ax1.axis('off');
ax1.set_title('Positive Headlines');

wordcloud2 = WordCloud(background_color='white', height=400).generate(neg_text_cln)
ax2.imshow(wordcloud2.recolor(color_func=red_color, random_state=3),interpolation="bilinear")
ax2.axis('off');
ax2.set_title('Negative Headlines');
# function to check if a bag's element is present in the text
def get_category(txt, bag):
    category = [x for x in bag if x in txt.lower()]
    if not category:
        category = [""]
    return category[0] 
country_df = pd.read_csv(path + "country-names-codes/countries.csv")
country_bag = list(country_df['COUNTRY'])
country_bag.extend(["united states","america", "u.s."])
df['country_name'] = df['headline_text'].apply(lambda x : get_category(x, country_bag))

# fix US name variations 
df.country_name[df.country_name=="u.s."] = "united states"
df.country_name[df.country_name=="america"] = "united states"

## get country names occurances 

country_agg = pd.DataFrame()
country_agg['COUNTRY'] = df['country_name'].value_counts().index
country_agg['Values'] = df['country_name'].value_counts().values
country_agg = pd.merge(country_agg, country_df)
country_agg.head(10)
data = [ dict(
        type = 'choropleth',
        locations = country_agg['CODE'],
        z = country_agg['Values'],
        text = country_agg['COUNTRY'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'News Count'),
      ) ]

layout = dict(
    title = '',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )

iplot( fig, validate=False, filename='d3-world-map' )
disaster_bag = ['hurricane', 'tornado', 'blizzerd', 'fire', 'earthquake', 'flood', 'cyclone', 'avalanche', 'drought', 'storm', 'lightning', 'tsunami']
df['disaster_category'] = df['headline_text'].apply(lambda x : get_category(x, disaster_bag))
labels = list(df['disaster_category'].value_counts().index)[1:]
values = list(df['disaster_category'].value_counts().values)[1:]
colors = ['lightblue','gray','#eee','#999', '#9f9f']
trace = go.Pie(labels=labels, values=values, hoverinfo='label+percent', 
               textinfo='value', name='headline counts of different disasters',
               marker=dict(colors=colors))
layout = dict(title = 'Distribution of natural disaster headlines',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = [trace], layout = layout)
iplot(fig)
def prepare_text(term):
    hurricane_df = df.headline_text[df.disaster_category==term]
    text = " ".join(hurricane_df)
    noise_words = ['brief', 'say', 'update', 'ing']
    noise_words.append(term)
    for noise in noise_words:
        text = text.lower().replace(noise," ")
    return text

hurricane_text = prepare_text('hurricane')
fire_text = prepare_text('fire')
storm_text = prepare_text('storm')
flood_text = prepare_text('flood')
wordcloud1 = WordCloud(background_color='white').generate(hurricane_text)
wordcloud2 = WordCloud(background_color='white').generate(flood_text)
wordcloud3 = WordCloud(background_color='white').generate(storm_text)
wordcloud4 = WordCloud(background_color='white').generate(fire_text)

fig, axes = plt.subplots(2, 2, figsize=(18, 10))

ax = axes[0, 0]
ax.imshow(wordcloud1)
ax.axis('off');
ax.set_title("Hurricane", fontsize=30);

ax = axes[0, 1]
ax.imshow(wordcloud2)
ax.axis('off');
ax.set_title("Flood", fontsize=30);

ax = axes[1, 0]
ax.imshow(wordcloud3)
ax.axis('off');
ax.set_title("Storm", fontsize=30);

ax = axes[1, 1]
ax.imshow(wordcloud4)
ax.axis('off');
ax.set_title("Fire", fontsize=30);

stopwords = open(path+"country-names-codes/stopwords.txt").read().strip().split("\n")
stopwords = [x.replace("\r","") for x in stopwords]
from collections import Counter 

def clean_text(txt):    
    txt = txt.lower()
    txt = "".join(x for x in txt if x not in punc)
    words = txt.split()
    words = [wrd for wrd in words if wrd not in stopwords]
    words = [wrd for wrd in words if len(wrd) > 1]
    txt = " ".join(words)
    return txt

def ngrams(txt, n):
    txt = txt.split()
    output = []
    for i in range(len(txt)-n+1):
        output.append(" ".join(txt[i:i+n]))
    return output
def get_bigrams_data(txt, tag, col):
    cleaned_text = clean_text(txt)
    all_bigrams = ngrams(cleaned_text, 2)
    topbigrams = Counter(all_bigrams).most_common(25)
    xvals = list(reversed([_[0] for _ in topbigrams]))
    yvals = list(reversed([_[1] for _ in topbigrams]))
    trace = go.Bar(x=yvals, y=xvals, name=tag, marker=dict(color=col), xaxis=dict(linecolor='#fff',), opacity=0.7, orientation='h')
    return trace
trace1 = get_bigrams_data(fire_text, 'fire', '#4286f4')
trace2 = get_bigrams_data(hurricane_text, 'hurricane', '#f44268')
trace3 = get_bigrams_data(flood_text, 'flood', '#e0d75e')
trace4 = get_bigrams_data(storm_text, 'storm', '#3e8441')

fig = tools.make_subplots(rows=1, cols=4, print_grid=False);
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 1, 2);
fig.append_trace(trace3, 1, 3);
fig.append_trace(trace4, 1, 4);

fig['layout'].update(height=800, title='Top Bigrams used in disaster related Headlines', legend=dict(orientation="v"));
iplot(fig, filename='simple-subplot');
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 

def generate_topic_models(category):
    small_df = df[df.disaster_category==category]
    small_df['clean_text'] = small_df['headline_text'].apply(clean_text)

    cvectorizer = CountVectorizer(min_df=4, max_features=4000, ngram_range=(1,2))
    cvz = cvectorizer.fit_transform(small_df['clean_text'])

    lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', max_iter=20, random_state=42)
    X_topics = lda_model.fit_transform(cvz)

    topic_word = lda_model.components_ 
    vocab = cvectorizer.get_feature_names()
    return topic_word, vocab 
n_top_words = 10
topic_word, vocab = generate_topic_models("hurricane")
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print ("Topic " + str(i+1) + ": " + " | ".join(topic_words) + "\n")
n_top_words = 10
topic_word, vocab = generate_topic_models("fire")
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print ("Topic " + str(i+1) + ": " + " | ".join(topic_words) + "\n")
n_top_words = 10
topic_word, vocab = generate_topic_models("flood")
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print ("Topic " + str(i+1) + ": " + " | ".join(topic_words) + "\n")
n_top_words = 10
topic_word, vocab = generate_topic_models("storm")
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print ("Topic " + str(i+1) + ": " + " | ".join(topic_words) + "\n")