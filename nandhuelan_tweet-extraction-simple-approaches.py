import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

import plotly.offline as py

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots

import re

from collections import defaultdict,OrderedDict

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from plotly import tools

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import string

import matplotlib.gridspec as gridspec

import seaborn as sns

from nltk import word_tokenize,pos_tag

from nltk import RegexpParser

import json

import cufflinks

from plotly.offline import iplot

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



nlp = spacy.load('en_core_web_sm')

!mkdir -p data



PATH='../input/tweet-sentiment-extraction/'



train=pd.read_csv(PATH+'train.csv')

test=pd.read_csv(PATH+'test.csv')

submission=pd.read_csv(PATH+'sample_submission.csv')



MODEL_PATH = '/kaggle/input/distilbertbaseuncased/'
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values



    return summary
resumetable(train)
resumetable(test)
train=train.dropna()
## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" and token not in STOP_WORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



## Get the bar chart from postive tweets ##

freq_dict = defaultdict(int)

for sent in train[train['sentiment']=='positive']["selected_text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), '#C5197D')



## Get the bar chart from negative tweets ##

freq_dict = defaultdict(int)

for sent in train[train['sentiment']=='negative']["selected_text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), '#C5197D')



## Get the bar chart from neutral questions ##

freq_dict = defaultdict(int)

for sent in train[train['sentiment']=='neutral']["selected_text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(50), '#C5197D')





# Creating two subplots

fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.04,

                          subplot_titles=["Positive", 

                                          "Negative",

                                         "Neutral"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 1, 3)

py.iplot(fig, filename='word-plots')



## Get the bar chart from postive tweets ##

freq_dict = defaultdict(int)

for sent in train[train['sentiment']=='positive']["selected_text"]:

    for word in generate_ngrams(sent,n_gram=2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'rgb(51,255,255)')



## Get the bar chart from negative tweets ##

freq_dict = defaultdict(int)

for sent in train[train['sentiment']=='negative']["selected_text"]:

    for word in generate_ngrams(sent,n_gram=2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'rgb(51,255,255)')



## Get the bar chart from neutral questions ##

freq_dict = defaultdict(int)

for sent in train[train['sentiment']=='neutral']["selected_text"]:

    for word in generate_ngrams(sent,n_gram=2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(50), 'rgb(51,255,255)')





# Creating two subplots

fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.04,

                          subplot_titles=["Positive-bigram", 

                                          "Negative-bigram",

                                         "Neutral-bigram"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 1, 3)



py.iplot(fig, filename='word-plots')

def clean(reg_exp, text):

    text = re.sub(reg_exp, " ", text)



    # replace multiple spaces with one.

    text = re.sub('\s{2,}', ' ', text)



    return text





def remove_urls(text):

    text = clean(r"http\S+", text)

    text = clean(r"www\S+", text)

    text = clean(r"pic.twitter.com\S+", text)



    return text



def basic_clean(text):

    text=remove_urls(text)

    text = clean(r'[\?\.\!]+(?=[\?\.\!])', text) #replace double punctuation with single

    text = clean(r"[^A-Za-z0-9\.\'!\?,\$]", text) #removes unicode characters

    return text

train['text']=train['text'].apply(lambda x: basic_clean(x))
def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOP_WORDS)



    wordcloud = WordCloud(background_color='white',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    mask = mask)

    wordcloud.generate(text)

    

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

    

d = '../input/masks/masks-wordclouds/'
comments_text = str(train.text)

comments_mask = np.array(Image.open(d + 'upvote.png'))

plot_wordcloud(comments_text, comments_mask, max_words=2000, max_font_size=300, 

               title = 'Most common words in all of the tweets', title_size=30)
train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))

test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

train["num_unique_words"] = train["text"].apply(lambda x: len(set(str(x).split())))

test["num_unique_words"] = test["text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train["num_chars"] = train["text"].apply(lambda x: len(str(x)))

test["num_chars"] = test["text"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

train["num_stopwords"] = train["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOP_WORDS]))

test["num_stopwords"] = test["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOP_WORDS]))



## Number of punctuations in the text ##

train["num_punctuations"] =train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test["num_punctuations"] =test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of upper case words in the text ##

train["num_words_upper"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test["num_words_upper"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



## Number of title case words in the text ##

train["num_words_title"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

test["num_words_title"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



## Average length of the words in the text ##

train["mean_word_len"] = train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test["mean_word_len"] = test["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
#Reference:https://www.kaggle.com/parulpandey/basic-preprocessing-and-eda



grid = gridspec.GridSpec(3, 4)

plt.figure(figsize=(16,6*4))



plt.suptitle('Meta features', size=20)

count=0

top_cats=train['sentiment'].value_counts().index

for n, col in enumerate(top_cats):

    colr=['green','black','pink']

    for i, q_t in enumerate(['num_words', 'num_unique_words', 'num_chars','num_stopwords']):

        filter_df=train[train['sentiment']==col]

       

        filter_df[q_t].iplot(

            kind='hist',

            bins=100,

            xTitle='text length',

            linecolor='black',

            color=colr[n],

            yTitle='count',

            title=f'{col} {q_t} Distribution')


def extract(x):

    if len((x.split(' ')))<=6:

        return x

    else:

        result=[]

        pattern = r"""S1: {<PR.*>+<VB.*>+<VB>},

              S2: {<JJ>?<NN.*>?<PR.*>+<VB.*>},

              S3: {<JJ>?<NN.*>}"""



        sentence = word_tokenize(x)

        PChunker = RegexpParser(pattern)

        output= PChunker.parse(pos_tag(sentence))

        

        for subtree in (output.subtrees(filter=lambda t: t.label() == 'S1' or t.label() == 'S2' or t.label() == 'S3')):

          result.append(' '.join([x[0] for x in subtree]))

        

        return ' '.join(result)
train['ex_text']=train['text'].apply(lambda x: extract(x))

test['ex_text']=test['text'].apply(lambda x: extract(x))
def jaccard(strs): 

    str1=strs['selected_text']

    str2=strs['ex_text']

    

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



train['jaccard']=train[['selected_text','ex_text']].apply(lambda x: jaccard(x),axis=1)



print(f"Average jaccard index in training data {train['jaccard'].mean()}")
#Reference: https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0



class TextRank4Keyword():

    """Extract keywords from text"""

    

    def __init__(self):

        self.d = 0.85 # damping coefficient, usually is .85

        self.min_diff = 1e-5 # convergence threshold

        self.steps = 10 # iteration steps

        self.node_weight = None # save keywords and its weight



    

    def set_stopwords(self, stopwords):  

        """Set stop words"""

        for word in STOP_WORDS.union(set(stopwords)):

            lexeme = nlp.vocab[word]

            lexeme.is_stop = True

    

    def sentence_segment(self, doc, candidate_pos, lower):

        """Store those words only in cadidate_pos"""

        sentences = []

        for sent in doc.sents:

            selected_words = []

            for token in sent:

                # Store words only with cadidate POS tag

                if token.pos_ in candidate_pos and token.is_stop is False:

                    if lower is True:

                        selected_words.append(token.text.lower())

                    else:

                        selected_words.append(token.text)

            sentences.append(selected_words)

        return sentences

        

    def get_vocab(self, sentences):

        """Get all tokens"""

        vocab = OrderedDict()

        i = 0

        for sentence in sentences:

            for word in sentence:

                if word not in vocab:

                    vocab[word] = i

                    i += 1

        return vocab

    

    def get_token_pairs(self, window_size, sentences):

        """Build token_pairs from windows in sentences"""

        token_pairs = list()

        for sentence in sentences:

            for i, word in enumerate(sentence):

                for j in range(i+1, i+window_size):

                    if j >= len(sentence):

                        break

                    pair = (word, sentence[j])

                    if pair not in token_pairs:

                        token_pairs.append(pair)

        return token_pairs

        

    def symmetrize(self, a):

        return a + a.T - np.diag(a.diagonal())

    

    def get_matrix(self, vocab, token_pairs):

        """Get normalized matrix"""

        # Build matrix

        vocab_size = len(vocab)

        g = np.zeros((vocab_size, vocab_size), dtype='float')

        for word1, word2 in token_pairs:

            i, j = vocab[word1], vocab[word2]

            g[i][j] = 1

            

        # Get Symmeric matrix

        g = self.symmetrize(g)

        

        # Normalize matrix by column

        norm = np.sum(g, axis=0)

        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm

        

        return g_norm



    

    def get_keywords(self, number=10):

        """Print top number keywords"""

        text_list=[]

        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))

        for i, (key, value) in enumerate(node_weight.items()):

            #print(key + ' - ' + str(value))

            text_list.append(key)

            

            #if i > number:

        return ' '.join(text_list)

        

        

    def analyze(self, text, 

                candidate_pos=['NOUN', 'PROPN'], 

                window_size=4, lower=False, stopwords=list()):

        """Main function to analyze text"""

        

        # Set stop words

        self.set_stopwords(stopwords)

        

        # Pare text by spaCy

        doc = nlp(text)

        

        # Filter sentences

        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words

        

        # Build vocabulary

        vocab = self.get_vocab(sentences)

        

        # Get token_pairs from windows

        token_pairs = self.get_token_pairs(window_size, sentences)

        

        # Get normalized matrix

        g = self.get_matrix(vocab, token_pairs)

        

        # Initionlization for weight(pagerank value)

        pr = np.array([1] * len(vocab))

        

        # Iteration

        previous_pr = 0

        for epoch in range(self.steps):

            pr = (1-self.d) + self.d * np.dot(g, pr)

            if abs(previous_pr - sum(pr))  < self.min_diff:

                break

            else:

                previous_pr = sum(pr)



        # Get weight for each node

        node_weight = dict()

        for word, index in vocab.items():

            node_weight[word] = pr[index]

        

        self.node_weight = node_weight
tr4w = TextRank4Keyword()



def keywordextract(x):

    tr4w.analyze(x, candidate_pos = ['NOUN', 'PROPN','VERB','ADJ'], window_size=4, lower=False)

    return tr4w.get_keywords(10)



train['ex_text']=train['text'].apply(lambda x: keywordextract(x))

test['ex_text']=test['text'].apply(lambda x: keywordextract(x))
def jaccard(strs): 

    str1=strs['selected_text']

    str2=strs['ex_text']

    

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



train['jaccard']=train[['selected_text','ex_text']].apply(lambda x: jaccard(x),axis=1)



print(f"Average jaccard index in training data using textrank {train['jaccard'].mean()}")
train=pd.read_csv(PATH+'train.csv')

test=pd.read_csv(PATH+'test.csv')

submission=pd.read_csv(PATH+'sample_submission.csv')



train=train.dropna()



train_np = np.array(train)

test_np = np.array(test)
def find_all(input_str, search_str):

    l1 = []

    length = len(input_str)

    index = 0

    while index < length:

        i = input_str.find(search_str, index)

        if i == -1:

            return l1

        l1.append(i)

        index = i + 1

    return l1



def do_qa_train(train):



    output = {}

    output['version'] = 'v1.0'

    output['data'] = []

    paragraphs = []

    for line in train:

        context = line[1]



        qas = []

        question = line[-1]

        qid = line[0]

        answers = []

        answer = line[2]

        if type(answer) != str or type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answer_starts = find_all(context, answer)

        for answer_start in answer_starts:

            answers.append({'answer_start': answer_start, 'text': answer.lower()})

            break

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})



        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

        

    return paragraphs



qa_train = do_qa_train(train_np)



with open('data/train.json', 'w') as outfile:

    json.dump(qa_train, outfile)
"""

Prepare testing data in QA-compatible format

"""



output = {}

output['version'] = 'v1.0'

output['data'] = []



def do_qa_test(test):

    paragraphs = []

    for line in test:

        context = line[1]

        qas = []

        question = line[-1]

        qid = line[0]

        if type(context) != str or type(question) != str:

            print(context, type(context))

            print(answer, type(answer))

            print(question, type(question))

            continue

        answers = []

        answers.append({'answer_start': 1000000, 'text': '__None__'})

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})



        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

    return paragraphs



qa_test = do_qa_test(test_np)



with open('data/test.json', 'w') as outfile:

    json.dump(qa_test, outfile)
!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q

!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q
%%time



from simpletransformers.question_answering import QuestionAnsweringModel





# Create the QuestionAnsweringModel

model = QuestionAnsweringModel('distilbert', 

                               MODEL_PATH, 

                               args={'reprocess_input_data': True,

                                     'overwrite_output_dir': True,

                                     'learning_rate': 5e-5,

                                     'num_train_epochs': 3,

                                     'max_seq_length': 192,

                                     'doc_stride': 64,

                                     'fp16': False,

                                    },

                              use_cuda=True)



model.train_model('data/train.json')
%%time



predictions = model.predict(qa_test)

predictions_df = pd.DataFrame.from_dict(predictions)



predictions_df = pd.DataFrame.from_dict(predictions)

submission['selected_text'] = predictions_df['answer']



for i in range(len(submission)):

    id_ = submission['textID'][i]

    if test['sentiment'][i] == 'neutral' or len(test['text'][i].split())<4: # neutral postprocessing

        submission.loc[i, 'selected_text'] = test['text'][i]



submission.to_csv('submission.csv',index=None)

print("File submitted successfully.")
submission.head()