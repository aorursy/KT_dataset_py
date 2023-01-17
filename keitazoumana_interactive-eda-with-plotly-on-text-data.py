import numpy as np 

import pandas as pd 

import pandas_profiling as pp

from tqdm import tqdm

from textblob import TextBlob



# Plotly tools

from plotly.offline import init_notebook_mode, iplot

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

import plotly.graph_objs as go

import plotly.plotly as py

from sklearn.feature_extraction.text import CountVectorizer
review_df = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')
# Perform and advanced visualisation using pandas_profiling library

pp.ProfileReport(review_df)
# We are going to print some reviews in order to better understand the kind of preprocessing

# task we will be performing.



some_reviews_index = [7, 9, 16] 



# tqdm is for printing the status bar

for rw in tqdm(some_reviews_index):

    print(review_df['Review Text'].values[rw])

    print(print("="*50))
# 1- Remove the Title, Unnamed columns 

review_df.drop(['Title', 'Unnamed: 0'], axis=1, inplace=True)
# 2- Remove the rows where Review Text is missing

review_df = review_df[~review_df['Review Text'].isnull()] 
# 3- clean Review Text column



# As we can see previously in the six reviews we printed, we will perform many cleaning process

# Remove special characters

# remove all digits

# remove all backslask 

# remove additional spaces

# remove all the stopwords

# put all the into lowercase



#----------------------------

# https://gist.github.com/sebleier/554280

# we are removing the words from the stop words list: 'no', 'nor', 'not'

stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"]
# This function will give the decontracted format of the words.

# https://stackoverflow.com/a/47091490/4084039

import re



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
# ACTION N°1

print("Before decontraction: ")

print(review_df['Review Text'].values[9])

print(print("="*50))





# Let check the review n°9 after decontraction

rew = decontracted(review_df['Review Text'].values[9])

print("After decontraction: ")

print(rew)
preprocessed_review_text = []



for review in tqdm(review_df['Review Text'].values):

    rw = decontracted(review)

    rw = rw.replace('\\r', ' ')

    rw = rw.replace('\\"', ' ')

    rw = rw.replace('\\n', ' ')

    rw = re.sub('[^A-Za-z0-9]+', ' ', rw)

    

    # https://gist.github.com/sebleier/554280

    rw = ' '.join(e for e in rw.split() if e.lower() not in stopwords)

    preprocessed_review_text.append(rw.lower().strip())
# ACTION N°2

# Action of preprocessing

print("Before preprocessing: ")

print(review_df['Review Text'].values[9])



# Let perform the preprocessing

print("\nAfter preprocessing: ")

print(preprocessed_review_text[9])
# 4- Feature for length of review

review_df['review_length'] = review_df['Review Text'].astype(str).apply(len) 



# 5- Create new feature for the word count of the review 

review_df['word_count'] = review_df['Review Text'].apply(lambda x: len(str(x).split())) 



# 6- Calculate sentiment polarity  

review_df['polarity'] = review_df['Review Text'].map(lambda text: TextBlob(text).sentiment.polarity)
# Let check if our new columns have been added to our set of features.

review_df.columns
# 1- Highest sentiment polarity 

print('3 random reviews with the highest positive sentiment polarity: \n')

cl = review_df.loc[review_df.polarity == 1, ['Review Text']].sample(3).values

for c in cl:

    print("review: ",c[0])
# 2- Neutral sentiment polarity

print('3 random reviews with the most neutral sentiment polarity: \n')

cl = review_df.loc[review_df.polarity == 0, ['Review Text']].sample(3).values

for c in cl:

    print("review: ",c[0])
# Most negative sentiment polarity (0)



# As there is no review with the value of polarity to -1, we are going to take 

# those have a polarity less than -0.9.

print('3 random reviews with the most negative sentiment polarity: \n')

cl = review_df.loc[review_df.polarity < -0.90 , ['Review Text']].sample(3).values

for c in cl:

    print("review: ",c[0])
review_df['Review Text'] = preprocessed_review_text
# Let check if the operation worked using the previous polarity analysis. 

# We are goin to take only those for negative polarity

print('3 random reviews with the most negative sentiment polarity: \n')

cl = review_df.loc[review_df.polarity < -0.90 , ['Review Text']].sample(3).values

for c in cl:

    print("review: ",c[0])
review_df['polarity'].iplot(

kind='hist',

    bins=50,

    xTitle='Polarity',

    linecolor='green',

    yTitle='Count',

    title='Sentiment Polarity Distribution'

)
review_df['Rating'].iplot(

kind='hist',

    xTitle='Rating',

    linecolor='green',

    yTitle='Count',

    title='Reviewers rating Distribution'

)
review_df['Age'].iplot(

kind='hist',

    bins=50,

    xTitle='Age',

    linecolor='green',

    yTitle='Count',

    title='Reviewers age Distribution'

)
review_df['review_length'].iplot(

kind='hist',

    bins=50,

    xTitle='Review Length',

    linecolor='green',

    yTitle='Count',

    title='Review Length Distribution'

)
review_df['word_count'].iplot(

kind='hist',

    bins=50,

    xTitle='Word count',

    linecolor='green',

    yTitle='Count',

    title='Review Text Word Count Distribution'

)
review_df.groupby('Class Name').count()['Clothing ID'].sort_values(ascending=False).iplot(kind='bar', 

                                                                                   yTitle='Count', 

                                                                                   linecolor='green', 

                                                                                   opacity=0.8,

                                                                                   title='Bar chart of Class Name', 

                                                                                   xTitle='Class Name')
review_df.groupby('Department Name').count()['Clothing ID'].sort_values(ascending=False).iplot(kind='bar', 

                                                                                   yTitle='Count', 

                                                                                   linecolor='green', 

                                                                                   opacity=0.8,

                                                                                   title='Bar chart of Department Name', 

                                                                                   xTitle='Department Name')
review_df.groupby('Division Name').count()['Clothing ID'].sort_values(ascending=False).iplot(kind='bar', 

                                                                                   yTitle='Count', 

                                                                                   linecolor='green', 

                                                                                   opacity=0.8,

                                                                                   title='Bar chart of Division Name', 

                                                                                   xTitle='Division Name')
# function to get TOP n-grams   

#@corpus: the document to analyse n-gram

#@n_gram_value: 1(unigram), 2(bigram), 3(trigram), etcetera

#@n: top n value to get.



def get_top_n_grams(corpus, n_gram_value, n=None):

    

    vector = CountVectorizer(ngram_range=(n_gram_value, n_gram_value)).fit(corpus)

    

    # We will used bag of words representation

    bow = vector.transform(corpus)

    sum_words = bow.sum(axis=0)

    

    # Determine frequency for the chart

    words_freq = [(word, sum_words[0, idx]) for word, idx in vector.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    

    return words_freq[:n]
# The final function to plot our N-grams

def plot_n_gram(common_words):

    for word, freq in common_words:

        print(word, freq)

        

    df = pd.DataFrame(common_words, columns=['ReviewText' , 'count'])

    df.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(

kind='bar', yTitle='Count', linecolor='black', title='Top 15 bigrams in review')
# Get TOP 15 1-gram

common_words_unigram = get_top_n_grams(review_df['Review Text'], 1, 15)



# Plot them

plot_n_gram(common_words_unigram)
# Get TOP 15 2-gram

common_words_bigram = get_top_n_grams(review_df['Review Text'], 2, 15)



# Plot them

plot_n_gram(common_words_bigram)
# Get TOP 15 3-gram

common_words_trigram = get_top_n_grams(review_df['Review Text'], 3, 15)



# Plot them

plot_n_gram(common_words_trigram)
# Below is the function that will help doing all those analysis  

def show_box_plot_from_feature(feature_name, title):

    y0 = review_df.loc[review_df['Department Name'] == 'Tops'][feature_name]

    y1 = review_df.loc[review_df['Department Name'] == 'Dresses'][feature_name]

    y2 = review_df.loc[review_df['Department Name'] == 'Bottoms'][feature_name]

    y3 = review_df.loc[review_df['Department Name'] == 'Intimate'][feature_name]

    y4 = review_df.loc[review_df['Department Name'] == 'Jackets'][feature_name]

    y5 = review_df.loc[review_df['Department Name'] == 'Trend'][feature_name]

    

    trace0 = go.Box(

    y=y0,

    name = 'Tops',

    )

    trace1 = go.Box(

        y=y1,

        name = 'Dresses',

    )

    trace2 = go.Box(

        y=y2,

        name = 'Bottoms',

    )

    trace3 = go.Box(

        y=y3,

        name = 'Intimate',

    )

    trace4 = go.Box(

        y=y4,

        name = 'Jackets',

    )

    trace5 = go.Box(

        y=y5,

        name = 'Trend',

    )



    data = [trace0, trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(

        title = title

    )

    

    fig = go.Figure(data=data,layout=layout)

    iplot(fig)
show_box_plot_from_feature('review_length', "Review length Boxplot of Department Name")
show_box_plot_from_feature('polarity', "Sentiment polarity Boxplot of Department Name")
show_box_plot_from_feature('Rating', "Rating Boxplot of Department Name")
# The function below will be used to do the analysis    

def distribution_by_recommendations(feature, title):

    axis1 = review_df.loc[review_df['Recommended IND'] == 1, feature]

    axis0 = review_df.loc[review_df['Recommended IND'] == 0, feature]



    trace1 = go.Histogram(

        x=axis0, name='Not recommended',

        opacity=0.75

    )

    trace2 = go.Histogram(

        x=axis1, name = 'Recommended',

        opacity=0.75

    )



    data = [trace1, trace2]

    layout = go.Layout(barmode='overlay', title=title)

    fig = go.Figure(data=data, layout=layout)



    iplot(fig)
distribution_by_recommendations('polarity', 'Distribution of Sentiment polarity of reviews based on Recommendation')
distribution_by_recommendations('Rating', 'Distribution of Sentiment polarity of reviews based on Recommendation')
distribution_by_recommendations('review_length', 'Distribution of review length based on Recommendation')
# below is the function to plot that jointplot. 

def joint_plot_of_features(feature1, feature2):

    trace1 = go.Scatter(

    x=review_df[feature1], y=review_df[feature2], mode='markers', name='points',

    marker=dict(color='rgb(102,0,0)', size=2, opacity=0.4)

    )

    trace2 = go.Histogram2dContour(

        x=review_df[feature1], y=review_df[feature2], name='density', ncontours=20,

        colorscale='Hot', reversescale=True, showscale=False

    )

    trace3 = go.Histogram(

        x=review_df[feature1], name=feature1+' density',

        marker=dict(color='rgb(102,0,0)'),

        yaxis='y2'

    )

    trace4 = go.Histogram(

        y=review_df[feature2], name= feature2+' density', marker=dict(color='rgb(102,0,0)'),

        xaxis='x2'

    )

    data = [trace1, trace2, trace3, trace4]



    layout = go.Layout(

        showlegend=False,

        autosize=False,

        width=600,

        height=550,

        xaxis=dict(

            domain=[0, 0.85],

            showgrid=False,

            zeroline=False

        ),

        yaxis=dict(

            domain=[0, 0.85],

            showgrid=False,

            zeroline=False

        ),

        margin=dict(

            t=50

        ),

        hovermode='closest',

        bargap=0,

        xaxis2=dict(

            domain=[0.85, 1],

            showgrid=False,

            zeroline=False

        ),

        yaxis2=dict(

            domain=[0.85, 1],

            showgrid=False,

            zeroline=False

        )

    )



    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
joint_plot_of_features('Rating', 'polarity')
joint_plot_of_features('Age', 'polarity')