import pandas as pd



import matplotlib

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import nltk



import plotly

import plotly.graph_objs as go



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)

plotly.offline.init_notebook_mode(connected=True)



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt

%matplotlib inline



from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper

from bokeh.layouts import row,column,gridplot

from bokeh.models.widgets import Tabs,Panel

output_notebook()
df = pd.read_csv("../input/train.tsv", sep="\t")
df.head(10)
df.describe()
df.isna().sum()
example = df[(df['PhraseId'] >= 0) & (df['PhraseId'] <= 2)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])
example = df[(df['PhraseId'] >= 517) & (df['PhraseId'] <= 518)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])
example = df[(df['PhraseId'] >= 68) & (df['PhraseId'] <= 69)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])
example = df[(df['PhraseId'] >= 10737) & (df['PhraseId'] <= 10738)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])
example = df[(df['PhraseId'] >= 22) & (df['PhraseId'] <= 24)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])



print()



print(example["Phrase"].values[2], " - Sentiment:", example["Sentiment"].values[2])
example = df[(df['PhraseId'] >= 46) & (df['PhraseId'] <= 47)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])
example = df[(df['PhraseId'] >= 1225) & (df['PhraseId'] <= 1227)]



print(example["Phrase"].values[0], " - Sentiment:", example["Sentiment"].values[0])



print()



print(example["Phrase"].values[1], " - Sentiment:", example["Sentiment"].values[1])



print()



print(example["Phrase"].values[2], " - Sentiment:", example["Sentiment"].values[2])
def tokenize_the_text(phrases):

    

    from nltk.tokenize import word_tokenize

    from nltk.text import Text

    

    tokens = [word for word in phrases]

    tokens = [word.lower() for word in tokens]

    tokens = [word_tokenize(word) for word in tokens]

    

    return tokens



crude_tokens = tokenize_the_text(df.Phrase)

print(crude_tokens[0:10])
def create_a_vocab(tokens):

    

    vocab = set()



    for setence in tokens:

        for word in setence:

            vocab.add(word)



    vocab = list(vocab)



    return vocab

    

vocab = create_a_vocab(crude_tokens)



print("Vocabulary size:", len(vocab), "words")
data = [go.Bar(

            x=['negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive'],

            y=df.Sentiment.value_counts().sort_index(), marker=dict(

                color = df.Sentiment.value_counts().sort_index(),colorscale='Viridis',showscale=True,

                reversescale = False

                ))]



plotly.offline.iplot(data, filename='sentiment distribution')
layout = go.Layout(

    title='Phrases Length Distribution',

    xaxis=dict(

        title='Phrases\' Length in characters',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    ),

    yaxis=dict(

        title='Frequencies',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    )

)



data = [go.Histogram(x=df['Phrase'].apply(lambda x: len(x.split())))]



fig = go.Figure(data=data, layout=layout)



plotly.offline.iplot(fig, filename='sentiment distribution')
def get_word_count_dict(tokens):

    

    words_count_dict = dict()



    for sentence in tokens:

        for word in sentence:

            if word not in words_count_dict:

                words_count_dict[word] = 1

            else:

                words_count_dict[word] += 1

    

    return words_count_dict





words_count_dict = get_word_count_dict(crude_tokens)



import operator



top_uncleaned_words_df = pd.DataFrame()



top_uncleaned_words_df = top_uncleaned_words_df.append(list(sorted(words_count_dict.items(), key=operator.itemgetter(1), reverse=True)))

top_uncleaned_words_df.columns = ['word', 'frequency']

#print(top_uncleaned_words_df.head(10))



limit = 20



trace1 = go.Bar(

            x=top_uncleaned_words_df.head(limit).word,

            y=top_uncleaned_words_df.head(limit).frequency, marker=dict(

                color = top_uncleaned_words_df.head(limit).frequency,colorscale='Cividis',showscale=True,

                reversescale = False

                ))



layout = dict(title= 'Most Frequent words from train set')



fig=dict(data=[trace1], layout=layout)

plotly.offline.iplot(fig)
bigram_dict = dict()

limit = 20



bigrm = list(nltk.bigrams(' '.join(df.Phrase.values).split()))



for elem in bigrm:

    new_elem = ' '.join(elem)

    if new_elem not in bigram_dict:

        bigram_dict[new_elem] = 1

    else:

        bigram_dict[new_elem] += 1  

    

top_uncleaned_bigram_df = pd.DataFrame.from_dict(bigram_dict, orient='index', columns=["frequency"])

top_uncleaned_bigram_df["bigram"] = top_uncleaned_bigram_df.index 

top_uncleaned_bigram_df = top_uncleaned_bigram_df.reset_index(drop=True)

top_uncleaned_bigram_df.sort_values(ascending=False, by="frequency").head(limit)





trace1 = go.Bar(

            y=top_uncleaned_bigram_df.sort_values(ascending=False, by="frequency").head(limit).frequency,

            x=top_uncleaned_bigram_df.sort_values(ascending=False, by="frequency").head(limit).bigram,

    marker=dict(

                color = top_uncleaned_bigram_df.sort_values(ascending=False, by="frequency").head(limit).frequency,colorscale='Cividis',showscale=True,

                reversescale = False

                )

    )



layout = dict(title= 'Top '+str(limit)+' most frequent Bigrams before cleaning')

fig=dict(data=[trace1], layout=layout)

plotly.offline.iplot(fig)



top_uncleaned_bigram_df.sort_values(ascending=False, by="frequency").head(limit)
trigram_dict = dict()

limit = 20



trigrm = list(nltk.trigrams(' '.join(df.Phrase.values).split()))



for elem in trigrm:

    new_elem = ' '.join(elem)

    if new_elem not in trigram_dict:

        trigram_dict[new_elem] = 1

    else:

        trigram_dict[new_elem] += 1





top_uncleaned_trigram_df = pd.DataFrame.from_dict(trigram_dict, orient='index', columns=["frequency"])

top_uncleaned_trigram_df["bigram"] = top_uncleaned_trigram_df.index 

top_uncleaned_trigram_df = top_uncleaned_trigram_df.reset_index(drop=True)

top_uncleaned_trigram_df.sort_values(ascending=False, by="frequency").head(limit)





trace1 = go.Bar(

            y=top_uncleaned_trigram_df.sort_values(ascending=False, by="frequency").head(limit).frequency,

            x=top_uncleaned_trigram_df.sort_values(ascending=False, by="frequency").head(limit).bigram,

    marker=dict(

                color = top_uncleaned_trigram_df.sort_values(ascending=False, by="frequency").head(limit).frequency,colorscale='Cividis',showscale=True,

                reversescale = False

                )

    )



layout = dict(title= 'Top '+str(limit)+' most frequent Trigrams before cleaning')

fig=dict(data=[trace1], layout=layout)

plotly.offline.iplot(fig)



top_uncleaned_trigram_df.sort_values(ascending=False, by="frequency").head(limit)
def custom_initial_clean(df):



    phrases_X = df.Phrase.copy()



    phrases_X = phrases_X.str.replace('\\*', ' ', regex=True)

    phrases_X = phrases_X.str.replace('\\/', ' ', regex=True)

    phrases_X = phrases_X.str.replace('\\\\', ' ', regex=True)

    phrases_X = phrases_X.str.replace('-', ' ', regex=True)

    phrases_X = phrases_X.str.replace(r'/', ' ', regex=True)

    phrases_X = phrases_X.str.replace(r'``', ' ', regex=True)

    phrases_X = phrases_X.str.replace(r'`', ' ', regex=True)

    phrases_X = phrases_X.str.replace(r"''", ' ', regex=True)

    phrases_X = phrases_X.str.replace(r",", ' ', regex=True)

    phrases_X = phrases_X.str.replace(r"\.$", ' ', regex=True)

    phrases_X = phrases_X.str.replace(r":", ' ', regex=True)

    phrases_X = phrases_X.str.replace(r"# ", '#', regex=True)

    phrases_X = phrases_X.str.replace(r";", ' ', regex=True)

    phrases_X = phrases_X.str.replace(r"?", ' ', regex=True)

    phrases_X = phrases_X.str.replace(r"=", ' ', regex=True)

    phrases_X = phrases_X.str.replace("...", ' ', regex=False)

    phrases_X = phrases_X.str.replace("..", ' ', regex=False)



    phrases_X = phrases_X.str.replace(r'LRB', ' ', regex=True)

    phrases_X = phrases_X.str.replace(r'RRB', ' ', regex=True)

    phrases_X = phrases_X.str.replace(r"[C|c]a n't", 'cannot', regex=True)

    phrases_X = phrases_X.str.replace(r"[W|w]o n't", 'will not', regex=True)

    phrases_X = phrases_X.str.replace(r"[W|w]ere n't", 'were not', regex=True)

    phrases_X = phrases_X.str.replace(r"[W|w]as n't", 'was not', regex=True)

    phrases_X = phrases_X.str.replace(r"[W|w]ould n't", 'would not', regex=True)

    phrases_X = phrases_X.str.replace(r"[D|d]oes n't", 'does not', regex=True)

    phrases_X = phrases_X.str.replace(r"[I|i]s n't", 'is not', regex=True)

    phrases_X = phrases_X.str.replace(r"[C|c]ould n't", 'could not', regex=True)

    phrases_X = phrases_X.str.replace(r"[D|d]id n't", 'did not', regex=True)

    phrases_X = phrases_X.str.replace(r"[H|h]as n't", 'has not', regex=True)

    phrases_X = phrases_X.str.replace(r"[H|h]ave n't", 'have not', regex=True)

    phrases_X = phrases_X.str.replace(r"[D|d]o n't", 'do not', regex=True)

    phrases_X = phrases_X.str.replace(r"[A|a]i n't", "not", regex=True)

    phrases_X = phrases_X.str.replace(r"[N|n]eed n't", "need not", regex=True)

    phrases_X = phrases_X.str.replace(r"[A|a]re n't", "are not", regex=True)

    phrases_X = phrases_X.str.replace(r"[S|s]hould n't", "should not", regex=True)

    phrases_X = phrases_X.str.replace(r"[H|h]ad n't", "had not", regex=True)



    phrases_X = phrases_X.str.replace(" 's", " ", regex=False)

    phrases_X = phrases_X.str.replace("'s", "", regex=False)

    phrases_X = phrases_X.str.replace("'ve", "have", regex=False)

    phrases_X = phrases_X.str.replace("'d", "would", regex=False)

    phrases_X = phrases_X.str.replace("'ll", "will", regex=False)

    phrases_X = phrases_X.str.replace("'m", "am", regex=False)

    phrases_X = phrases_X.str.replace("'n", "and", regex=False)

    phrases_X = phrases_X.str.replace("'re", "are", regex=False)

    phrases_X = phrases_X.str.replace("'til", "until", regex=False)

    phrases_X = phrases_X.str.replace(" ' ", " ", regex=False)

    phrases_X = phrases_X.str.replace(" '", " ", regex=False)



    phrases_X = phrases_X.str.replace(r'[ ]{2,}', ' ', regex=True)



    return phrases_X



phrases_X = custom_initial_clean(df)
def tokenize_the_text(phrases):

    

    from nltk.tokenize import word_tokenize

    from nltk.text import Text

    

    tokens = [word for word in phrases]

    tokens = [word.lower() for word in tokens]

    tokens = [word_tokenize(word) for word in tokens]

    

    return tokens
tokens_custom_cleaned = tokenize_the_text(phrases_X)
vocab = create_a_vocab(tokens_custom_cleaned)



print("Vocabulary size after custom cleaning:", len(vocab), "words")
def removing_stopwords(tokens_custom_cleaned):



    from nltk.corpus import stopwords

    stop_words = stopwords.words('english')

    tokens_custom_cleaned_and_without_stopwords = []

    for sentence in tokens_custom_cleaned:

        tokens_custom_cleaned_and_without_stopwords.append([word for word in sentence if word not in stop_words])

        

    return tokens_custom_cleaned_and_without_stopwords



tokens_custom_cleaned_and_without_stopwords = removing_stopwords(tokens_custom_cleaned)
vocab = create_a_vocab(tokens_custom_cleaned_and_without_stopwords)



print("Vocabulary size after custom cleaning and removing stopwords:", len(vocab), "words")
def lemmatizing_the_tokens(tokens_custom_cleaned_and_without_stopwords):



    from nltk.stem.wordnet import WordNetLemmatizer 

    lem = WordNetLemmatizer()



    tokens_custom_cleaned_and_without_stopwords_and_lemmatized = []



    for sentence in tokens_custom_cleaned_and_without_stopwords:

        tokens_custom_cleaned_and_without_stopwords_and_lemmatized.append([lem.lemmatize(word, pos='v') for word in sentence])

        

    return tokens_custom_cleaned_and_without_stopwords_and_lemmatized





tokens_custom_cleaned_and_without_stopwords_and_lemmatized = lemmatizing_the_tokens(tokens_custom_cleaned_and_without_stopwords)
vocab = create_a_vocab(tokens_custom_cleaned_and_without_stopwords_and_lemmatized)



print("Vocabulary size after custom cleaning, removing stopwords and lemmatizing the text:", len(vocab), "words")
from collections import defaultdict



longest_words_dict = defaultdict(list)



for word in vocab:

        longest_words_dict[str(len(word))].append(word)
print("The Biggest number of characters with the longest words in the Train Set is:", 

      max([int(elem) for elem in list(longest_words_dict.keys())]))

print(longest_words_dict[str(max([int(elem) for elem in list(longest_words_dict.keys())]))])



print()



print("The Second biggest number of characters with the longest words in the Train Set is:", 

      max([int(elem) for elem in list(longest_words_dict.keys())]) - 1)

print(longest_words_dict[str(max([int(elem) for elem in list(longest_words_dict.keys())])-1)])



print()



print("The Third biggest number of characters with the longest words in the Train Set is:", 

      max([int(elem) for elem in list(longest_words_dict.keys())]) - 2)

print(longest_words_dict[str(max([int(elem) for elem in list(longest_words_dict.keys())])-2)])
processed_text = [(' '.join(sentence)) for sentence in tokens_custom_cleaned_and_without_stopwords_and_lemmatized]

whole_reviews = ' '.join(processed_text)





# Create and generate a word cloud image:

wordcloud = WordCloud(max_font_size=180, max_words=100, width=800, height=600).generate(whole_reviews)



# Display the generated image:

plt.figure( figsize=(20,10) )

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
#np.shape(processed_text)

processed_df = pd.DataFrame(columns=["Phrase", "Sentiment"])

processed_df.Phrase = processed_text

processed_df.Sentiment = df.Sentiment.values





fig,axes = plt.subplots(3, 2, figsize=(30, 15))



temp = processed_df[processed_df.Sentiment == 0].Phrase

whole_reviews = ' '.join(temp)



wordcloud_1 = WordCloud(max_font_size=180, max_words=100, width=800, height=600).generate(whole_reviews)



ax = axes[0, 0]

ax.imshow(wordcloud_1, interpolation="bilinear")

ax.axis('off')

ax.set_title("Sentiment 0 - negative, Wordcloud", fontsize=30)



temp = processed_df[processed_df.Sentiment == 1].Phrase

whole_reviews = ' '.join(temp)



wordcloud_2 = WordCloud(max_font_size=180, max_words=100, width=800, height=600).generate(whole_reviews)



ax = axes[0, 1]

ax.imshow(wordcloud_2, interpolation="bilinear")

ax.axis('off')

ax.set_title("Sentiment 1 - somewhat negative, Wordcloud", fontsize=30)



temp = processed_df[processed_df.Sentiment == 2].Phrase

whole_reviews = ' '.join(temp)



wordcloud_3 = WordCloud(max_font_size=180, max_words=100, width=800, height=600).generate(whole_reviews)



ax = axes[1, 0]

ax.imshow(wordcloud_3, interpolation="bilinear")

ax.axis('off')

ax.set_title("Sentiment 2 - neutral, Wordcloud", fontsize=30)



temp = processed_df[processed_df.Sentiment == 3].Phrase

whole_reviews = ' '.join(temp)



wordcloud_4 = WordCloud(max_font_size=180, max_words=100, width=800, height=600).generate(whole_reviews)



ax = axes[1, 1]

ax.imshow(wordcloud_4, interpolation="bilinear")

ax.axis('off')

ax.set_title("Sentiment 3 - somewhat positive, Wordcloud", fontsize=30)





temp = processed_df[processed_df.Sentiment == 4].Phrase

whole_reviews = ' '.join(temp)



wordcloud_5 = WordCloud(max_font_size=180, max_words=100, width=800, height=600).generate(whole_reviews)



ax = axes[2, 0]

ax.imshow(wordcloud_5, interpolation="bilinear")

ax.axis('off')

ax.set_title("Sentiment 4 - positive, Wordcloud", fontsize=30)





axes[-1, -1].axis('off')

plt.show()



top_cleaned_words_df = pd.DataFrame()



top_cleaned_words_df = top_cleaned_words_df.append(list(sorted(get_word_count_dict(tokens_custom_cleaned_and_without_stopwords_and_lemmatized).items(), key=operator.itemgetter(1), reverse=True)))

top_cleaned_words_df.columns = ['word', 'frequency']

#print(top_cleaned_words_df.head(10))



limit = 20



trace1 = go.Bar(

            x=top_cleaned_words_df.head(limit).word,

            y=top_cleaned_words_df.head(limit).frequency,

    marker=dict(

                color = top_cleaned_words_df.head(limit).frequency,colorscale='Cividis',showscale=True,

                reversescale = False

                )

    )



layout = dict(title= 'Top '+str(limit)+' most frequent words after cleaning')

fig=dict(data=[trace1], layout=layout)

plotly.offline.iplot(fig)
temp = tokenize_the_text(processed_df[processed_df.Sentiment == 0].Phrase)

top_cleaned_words_df = pd.DataFrame()



top_cleaned_words_df = top_cleaned_words_df.append(list(sorted(get_word_count_dict(temp).items(), key=operator.itemgetter(1), reverse=True)))

top_cleaned_words_df.columns = ['word', 'frequency']



limit = 20



trace1 = go.Bar(

            x=top_cleaned_words_df.head(limit).word,

            y=top_cleaned_words_df.head(limit).frequency,

    marker=dict(

                color = top_cleaned_words_df.head(limit).frequency,colorscale='Cividis',showscale=True,

                reversescale = False

                )

    )



layout = dict(title= 'Top '+str(limit)+' most frequent words after cleaning for sentiment 0')

fig=dict(data=[trace1], layout=layout)

plotly.offline.iplot(fig)
temp = tokenize_the_text(processed_df[processed_df.Sentiment == 1].Phrase)

top_cleaned_words_df = pd.DataFrame()



top_cleaned_words_df = top_cleaned_words_df.append(list(sorted(get_word_count_dict(temp).items(), key=operator.itemgetter(1), reverse=True)))

top_cleaned_words_df.columns = ['word', 'frequency']



limit = 20



trace1 = go.Bar(

            x=top_cleaned_words_df.head(limit).word,

            y=top_cleaned_words_df.head(limit).frequency,

    marker=dict(

                color = top_cleaned_words_df.head(limit).frequency,colorscale='Cividis',showscale=True,

                reversescale = False

                )

    )



layout = dict(title= 'Top '+str(limit)+' most frequent words after cleaning  for sentiment 1')

fig=dict(data=[trace1], layout=layout)

plotly.offline.iplot(fig)
temp = tokenize_the_text(processed_df[processed_df.Sentiment == 2].Phrase)

top_cleaned_words_df = pd.DataFrame()



top_cleaned_words_df = top_cleaned_words_df.append(list(sorted(get_word_count_dict(temp).items(), key=operator.itemgetter(1), reverse=True)))

top_cleaned_words_df.columns = ['word', 'frequency']



limit = 20



trace1 = go.Bar(

            x=top_cleaned_words_df.head(limit).word,

            y=top_cleaned_words_df.head(limit).frequency,

    marker=dict(

                color = top_cleaned_words_df.head(limit).frequency,colorscale='Cividis',showscale=True,

                reversescale = False

                )

    )



layout = dict(title= 'Top '+str(limit)+' most frequent words after cleaning  for sentiment 2')

fig=dict(data=[trace1], layout=layout)

plotly.offline.iplot(fig)
temp = tokenize_the_text(processed_df[processed_df.Sentiment == 3].Phrase)

top_cleaned_words_df = pd.DataFrame()



top_cleaned_words_df = top_cleaned_words_df.append(list(sorted(get_word_count_dict(temp).items(), key=operator.itemgetter(1), reverse=True)))

top_cleaned_words_df.columns = ['word', 'frequency']



limit = 20



trace1 = go.Bar(

            x=top_cleaned_words_df.head(limit).word,

            y=top_cleaned_words_df.head(limit).frequency,

    marker=dict(

                color = top_cleaned_words_df.head(limit).frequency,colorscale='Cividis',showscale=True,

                reversescale = False

                )

    )



layout = dict(title= 'Top '+str(limit)+' most frequent words after cleaning  for sentiment 3')

fig=dict(data=[trace1], layout=layout)

plotly.offline.iplot(fig)
temp = tokenize_the_text(processed_df[processed_df.Sentiment == 4].Phrase)

top_cleaned_words_df = pd.DataFrame()



top_cleaned_words_df = top_cleaned_words_df.append(list(sorted(get_word_count_dict(temp).items(), key=operator.itemgetter(1), reverse=True)))

top_cleaned_words_df.columns = ['word', 'frequency']



limit = 20



trace1 = go.Bar(

            x=top_cleaned_words_df.head(limit).word,

            y=top_cleaned_words_df.head(limit).frequency,

    marker=dict(

                color = top_cleaned_words_df.head(limit).frequency,colorscale='Cividis',showscale=True,

                reversescale = False

                )

    )



layout = dict(title= 'Top '+str(limit)+' most frequent words after cleaning for sentiment 4')

fig=dict(data=[trace1], layout=layout)

plotly.offline.iplot(fig)
bigram_dict = dict()

limit = 20



bigrm = list(nltk.bigrams(whole_reviews.split()))



for elem in bigrm:

    new_elem = ' '.join(elem)

    if new_elem not in bigram_dict:

        bigram_dict[new_elem] = 1

    else:

        bigram_dict[new_elem] += 1

        

    

top_cleaned_bigram_df = pd.DataFrame.from_dict(bigram_dict, orient='index', columns=["frequency"])

top_cleaned_bigram_df["bigram"] = top_cleaned_bigram_df.index 

top_cleaned_bigram_df = top_cleaned_bigram_df.reset_index(drop=True)

top_cleaned_bigram_df.sort_values(ascending=False, by="frequency").head(limit)





trace1 = go.Bar(

            y=top_cleaned_bigram_df.sort_values(ascending=False, by="frequency").head(limit).frequency,

            x=top_cleaned_bigram_df.sort_values(ascending=False, by="frequency").head(limit).bigram,

    marker=dict(

                color = top_cleaned_bigram_df.sort_values(ascending=False, by="frequency").head(limit).frequency,colorscale='Cividis',showscale=True,

                reversescale = False

                )

    )



layout = dict(title= 'Top '+str(limit)+' most frequent Bigrams after cleaning')

fig=dict(data=[trace1], layout=layout)

plotly.offline.iplot(fig)
trigram_dict = dict()





trigrm = list(nltk.trigrams(whole_reviews.split()))



for elem in trigrm:

    if elem not in bigram_dict:

        trigram_dict[elem] = 1

    else:

        trigram_dict[elem] += 1



top_cleaned_trigram_df = pd.DataFrame.from_dict(trigram_dict, orient='index', columns=["frequency"])

top_cleaned_trigram_df["trigram"] = top_cleaned_trigram_df.index 

top_cleaned_trigram_df = top_cleaned_trigram_df.reset_index(drop=True)

top_cleaned_trigram_df.sort_values(ascending=False, by="frequency").head(10)

def get_ne_dict(tokens):

    

    from collections import defaultdict

    ne_dict = defaultdict(list)

    

    for sent in tokens_custom_cleaned_and_without_stopwords_and_lemmatized:

        for entity in nltk.ne_chunk(nltk.pos_tag(sent), binary=False):

            if len(entity) != 2:

                ne_dict[entity.label()].append(entity[0][0])

            

                

    return ne_dict



import operator



ne_dict = get_ne_dict(tokens_custom_cleaned_and_without_stopwords_and_lemmatized)
for elem in ne_dict.items():

    print(elem[0],":", set(elem[1]))
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=5, lowercase=True, ngram_range=(1, 2))





processed_text = [(' '.join(sentence)) for sentence in tokens_custom_cleaned_and_without_stopwords_and_lemmatized]

vz = vectorizer.fit_transform(processed_text)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

tfidf = pd.DataFrame(columns=['tfidf']).from_dict(

                    dict(tfidf), orient='index')

tfidf.columns = ['tfidf']



tfidf.sort_values(by=['tfidf'], ascending=True).head(10)
tfidf.sort_values(by=['tfidf'], ascending=False).head(10)
temp_tokens = tokenize_the_text(processed_df[processed_df.Sentiment == 0].Phrase)



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_sent = TfidfVectorizer(min_df=5, lowercase=True, ngram_range=(1, 2))





processed_text_sent = [(' '.join(sentence)) for sentence in temp_tokens]

vz_sent = vectorizer_sent.fit_transform(processed_text_sent)



tfidf_sent = dict(zip(vectorizer_sent.get_feature_names(), vectorizer_sent.idf_))

tfidf_sent = pd.DataFrame(columns=['tfidf']).from_dict(

                    dict(tfidf_sent), orient='index')

tfidf_sent.columns = ['tfidf']



tfidf_sent.sort_values(by=['tfidf'], ascending=False).head(10)
temp_tokens = tokenize_the_text(processed_df[processed_df.Sentiment == 1].Phrase)



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_sent = TfidfVectorizer(min_df=5, lowercase=True, ngram_range=(1, 2))





processed_text_sent = [(' '.join(sentence)) for sentence in temp_tokens]

vz_sent = vectorizer_sent.fit_transform(processed_text_sent)



tfidf_sent = dict(zip(vectorizer_sent.get_feature_names(), vectorizer_sent.idf_))

tfidf_sent = pd.DataFrame(columns=['tfidf']).from_dict(

                    dict(tfidf_sent), orient='index')

tfidf_sent.columns = ['tfidf']



tfidf_sent.sort_values(by=['tfidf'], ascending=False).head(10)
temp_tokens = tokenize_the_text(processed_df[processed_df.Sentiment == 2].Phrase)



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_sent = TfidfVectorizer(min_df=5, lowercase=True, ngram_range=(1, 2))





processed_text_sent = [(' '.join(sentence)) for sentence in temp_tokens]

vz_sent = vectorizer_sent.fit_transform(processed_text_sent)



tfidf_sent = dict(zip(vectorizer_sent.get_feature_names(), vectorizer_sent.idf_))

tfidf_sent = pd.DataFrame(columns=['tfidf']).from_dict(

                    dict(tfidf_sent), orient='index')

tfidf_sent.columns = ['tfidf']



tfidf_sent.sort_values(by=['tfidf'], ascending=False).head(10)
temp_tokens = tokenize_the_text(processed_df[processed_df.Sentiment == 3].Phrase)



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_sent = TfidfVectorizer(min_df=5, lowercase=True, ngram_range=(1, 2))





processed_text_sent = [(' '.join(sentence)) for sentence in temp_tokens]

vz_sent = vectorizer_sent.fit_transform(processed_text_sent)



tfidf_sent = dict(zip(vectorizer_sent.get_feature_names(), vectorizer_sent.idf_))

tfidf_sent = pd.DataFrame(columns=['tfidf']).from_dict(

                    dict(tfidf_sent), orient='index')

tfidf_sent.columns = ['tfidf']



tfidf_sent.sort_values(by=['tfidf'], ascending=False).head(10)
temp_tokens = tokenize_the_text(processed_df[processed_df.Sentiment == 4].Phrase)



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_sent = TfidfVectorizer(min_df=5, lowercase=True, ngram_range=(1, 2))





processed_text_sent = [(' '.join(sentence)) for sentence in temp_tokens]

vz_sent = vectorizer_sent.fit_transform(processed_text_sent)



tfidf_sent = dict(zip(vectorizer_sent.get_feature_names(), vectorizer_sent.idf_))

tfidf_sent = pd.DataFrame(columns=['tfidf']).from_dict(

                    dict(tfidf_sent), orient='index')

tfidf_sent.columns = ['tfidf']



tfidf_sent.sort_values(by=['tfidf'], ascending=False).head(10)
from sklearn.decomposition import TruncatedSVD



n_comp=30

svd = TruncatedSVD(n_components=n_comp, random_state=42)

svd_tfidf = svd.fit_transform(vz)
from sklearn.manifold import TSNE



tsne_model = TSNE(n_components=2, verbose=1, random_state=42, n_iter=500)

tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
plot_df = pd.DataFrame(columns=["x", "y", "text"])

plot_df.x = tsne_tfidf[:,0]

plot_df.y = tsne_tfidf[:,1]

plot_df.text = processed_text



source = ColumnDataSource(data=dict(x=plot_df['x'], y=plot_df['y'],

                                    text=plot_df['text']))





plot_tfidf = figure(plot_width=700, plot_height=600,

                        title="Tf-IDF text to features representation using tSNE",

    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

    x_axis_type=None, y_axis_type=None, min_border=1)



plot_tfidf.scatter(source = source, x='x', y='y', color='#4286f4')

hover = plot_tfidf.select(dict(type=HoverTool))

hover.tooltips={"text": "@text" }

output_file("Train_set_reviews_visualization_over_the_2-axis_using_t-SNE.html")

show(plot_tfidf)
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.cluster import MiniBatchKMeans





range_n_clusters = [6, 8, 10, 12, 14, 16, 18]



cluster_grid_df = pd.DataFrame(columns=["cluster_size", "silhouette_score"])



for num_clusters in range_n_clusters:

    

    kmeans_model = MiniBatchKMeans(n_clusters=num_clusters,

                               init='k-means++',

                               n_init=1,

                               init_size=1000, batch_size=1000, verbose=0, max_iter=1000, random_state=42)



    kmeans = kmeans_model.fit(vz)

    kmeans_clusters = kmeans.predict(vz)

    silhouette_avg = silhouette_score(vz, kmeans_clusters)

    print("For n_clusters =", num_clusters, "The average silhouette_score is :", silhouette_avg)

    cluster_grid_df = cluster_grid_df.append({'cluster_size': num_clusters, 'silhouette_score': silhouette_avg}, ignore_index=True)
num_clusters = int(cluster_grid_df[np.abs(cluster_grid_df.silhouette_score) == np.max(cluster_grid_df.silhouette_score)].cluster_size)





kmeans_model = MiniBatchKMeans(n_clusters=num_clusters,

                               init='k-means++',

                               n_init=1,

                               init_size=1000, batch_size=1000, verbose=0, max_iter=1000, random_state=42)



kmeans = kmeans_model.fit(vz)

kmeans_clusters = kmeans.predict(vz)

kmeans_distances = kmeans.transform(vz)
num_clusters = int(cluster_grid_df[np.abs(cluster_grid_df.silhouette_score) == np.max(cluster_grid_df.silhouette_score)].cluster_size)



print("Representative terms per cluster center:")

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

for i in range(num_clusters):

    print("Cluster %d:" % i)

    center_cluster_str_words = []

    for ind in order_centroids[i, :10]:

        str = "".join(terms[ind])

        center_cluster_str_words.append(str)

    print(' %s' % "|".join(center_cluster_str_words))

    print()
from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, verbose=1, init='pca', random_state=42, n_iter=500)



tsne_kmeans = tsne_model.fit_transform(kmeans_distances)


colormap = np.array([

    "#000080", "#00BFFF", "#008000", "#808000", "#FF8C00",

    "#FF7F50", "#4ba04e", "#FF6347", "#A9A9A9", "#808080",

    "#ADD8E6", "#FF00FF", "#90EE90", "#C0C0C0", "#FF00FF",

    "#008080", "#4169E1", "#BDB76B", "#F0FFF0", "#F4A460",

    "#4B0082", "#FA8072", "#9ACD32", "#7CFC00", "#DDA0DD",

    "#A52A2A", "#F5F5DC", "#FFEFD5", "#008080", "#000000"

])





plot_df = pd.DataFrame(columns=["x", "y", "text", "cluster"])

plot_df.x = tsne_kmeans[:,0]

plot_df.y = tsne_kmeans[:,1]

plot_df.text = processed_text

plot_df.cluster = kmeans_clusters



source = ColumnDataSource(data=dict(x=plot_df['x'], y=plot_df['y'],

                                    cluster=plot_df['cluster'],

                                    color=colormap[kmeans_clusters],

                                    text=plot_df['text']))





plot_kmeans = figure(plot_width=700, plot_height=600,

                        title="KMeans clustering of the description",

    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

    x_axis_type=None, y_axis_type=None, min_border=1)



plot_kmeans.scatter(source = source, x='x', y='y', color='color')

hover = plot_kmeans.select(dict(type=HoverTool))

hover.tooltips={"text": "@text", "cluster":"@cluster" }

output_file("K-means_reviews_clustering_t-SNE_vizualization.html")

show(plot_kmeans)
from sklearn.feature_extraction.text import CountVectorizer



cvectorizer = CountVectorizer(min_df=5, lowercase=True, ngram_range=(1,2))



processed_text = [(' '.join(sentence)) for sentence in tokens_custom_cleaned_and_without_stopwords_and_lemmatized]

cvz = cvectorizer.fit_transform(processed_text)
from sklearn.decomposition import LatentDirichletAllocation



# Define Search Param

list_of_components= [6,8,10,12]



lda_grid_df = pd.DataFrame(columns=["components", "perplexity", "log_likelihood"])



for components in list_of_components:

    

    lda_model = LatentDirichletAllocation(n_components=components, max_iter=20, learning_method='online', random_state=42)

    X_topics = lda_model.fit_transform(cvz)

    print("number of components: ", components)

    print("perplexity: ", lda_model.perplexity(cvz))

    print("log likelihood score: ", lda_model.score(cvz))

    print()

    

    lda_grid_df = lda_grid_df.append({'components': components, 'perplexity': lda_model.perplexity(cvz), "log_likelihood": lda_model.score(cvz)}, ignore_index=True)
from sklearn.decomposition import LatentDirichletAllocation



n_topics = 12

n_iter = 20 # number of iterations



lda_model = LatentDirichletAllocation(n_components=n_topics,

                                      learning_method='online',

                                      max_iter=n_iter,

                                      random_state=42)



X_topics = lda_model.fit_transform(cvz)
n_top_words = 10

topic_summaries = []



topic_word = lda_model.components_  # get the topic words

vocab = cvectorizer.get_feature_names()



for i, topic_dist in enumerate(topic_word):

    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]

    topic_summaries.append(' '.join(topic_words))

    print('Topic {}: {}'.format(i, ' | '.join(topic_words)))
# reduce dimension to 2 using tsne

from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, verbose=1, init='pca', random_state=42, n_iter=500)

tsne_lda = tsne_model.fit_transform(X_topics)
plot_df = pd.DataFrame(columns=["x", "y", "text", "topic"])

plot_df.x = tsne_lda[:,0]

plot_df.y = tsne_lda[:,1]

plot_df.text = processed_text

plot_df.topic = X_topics.argmax(axis=1)



colormap = np.array([

    "#000080", "#00BFFF", "#008000", "#808000", "#FF8C00",

    "#FF7F50", "#a368a3", "#FF6347", "#A9A9A9", "#808080",

    "#ADD8E6", "#FF00FF", "#90EE90", "#C0C0C0", "#FF00FF",

    "#008080", "#4169E1", "#BDB76B", "#F0FFF0", "#F4A460",

    "#4B0082", "#FA8072", "#9ACD32", "#7CFC00", "#DDA0DD",

    "#A52A2A", "#F5F5DC", "#FFEFD5", "#008080", "#000000"

])



source = ColumnDataSource(data=dict(x=plot_df['x'], y=plot_df['y'],

                                    topic=plot_df['topic'],

                                    color=colormap[plot_df['topic']],

                                    text=plot_df['text']))





plot_lda = figure(plot_width=700, plot_height=600,

                        title="LDA Topic Visualization",

    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

    x_axis_type=None, y_axis_type=None, min_border=1)



plot_lda.scatter(source = source, x='x', y='y', color='color')

hover = plot_lda.select(dict(type=HoverTool))

hover.tooltips={"text": "@text", "topic":"@topic" }

output_file("LDA_and_tSNE_visualization.html")

show(plot_lda)
import gensim



sentences = [' '.join(sent) for sent in tokens_custom_cleaned_and_without_stopwords_and_lemmatized]



# Creating the model and setting values for the various parameters

num_features = 300  # Word vector dimensionality

min_word_count = 10 # Minimum word count

num_workers = 4     # Number of parallel threads

context = 5        # Context window size



# Initializing the train model

from gensim.models import word2vec





print("Training model....")

model = word2vec.Word2Vec(tokens_custom_cleaned_and_without_stopwords_and_lemmatized,\

                          workers=num_workers,\

                          size=num_features,\

                          min_count=min_word_count,\

                          window=context)



# To make the model memory efficient

model.init_sims(replace=True)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



print("Top 4 similarities with positive cosine similarity")

print()



print("top 4 similar words for the word film:", model.wv.most_similar(positive=['film'], topn = 4))

print()



print("top 4 similar words for the word film:",model.wv.most_similar(positive=['great'], topn = 4))

print()
from sklearn.manifold import TSNE



labels = []

w2v_vectors = []



for word in model.wv.vocab:

    w2v_vectors.append(model[word])

    labels.append(word)



tsne_model = TSNE(n_components=2, init='pca', n_iter=500, random_state=42)

tsne_w2v = tsne_model.fit_transform(w2v_vectors)
plotly.offline.init_notebook_mode(connected=True)



data = [go.Scatter(x=tsne_w2v[:,0], y=tsne_w2v[:,1], 

                   mode='markers', text = labels)]



fig = go.Figure(data=data, layout= go.Layout(title="Corpus Representation using Word2Vec and tSNE"))



iplot(fig, filename='2D_trainset_vocabulary_representation_using_w2v_and_tSNE')
tsne_model = TSNE(n_components=3, init='pca', n_iter=500, random_state=42)

tsne_w2v_3d = tsne_model.fit_transform(w2v_vectors)


data = [go.Scatter3d(x=tsne_w2v_3d[:,0], y=tsne_w2v_3d[:,1], z=tsne_w2v_3d[:,2], 

                   mode='markers', text = labels, 

                     marker = dict(size = 4, color = 'rgba(0, 10, 157, .6)'))]



fig = go.Figure(data=data, layout= go.Layout(title="3D Corpus Representation using Word2Vec and tSNE", autosize=False,

    width=900,

    height=900))



iplot(fig, filename='Corpus_representation_using_w2v_and_tSNE')