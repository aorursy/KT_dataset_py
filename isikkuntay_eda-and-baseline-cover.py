# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import os

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go

import plotly.express as px

pd.set_option('max_columns', 1000)

from bokeh.models import Panel, Tabs

from bokeh.io import output_notebook, show

from bokeh.plotting import figure

import lightgbm as lgb

import plotly.figure_factory as ff

import gc

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

import json

from keras.preprocessing import text, sequence

from sklearn.feature_extraction.text import CountVectorizer
path = '/kaggle/input/tensorflow2-question-answering/'

train_path = 'simplified-nq-train.jsonl'

test_path = 'simplified-nq-test.jsonl'

sample_submission_path = 'sample_submission.csv'



def read_data(path, sample = True, chunksize = 30000):

    if sample == True:

        df = []

        with open(path, 'rt') as reader:

            for i in range(chunksize):

                df.append(json.loads(reader.readline()))

        df = pd.DataFrame(df)

        print('Our sampled dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))

    else:

        df = pd.read_json(path, orient = 'records', lines = True)

        print('Our dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))

        gc.collect()

    return df



train = read_data(path+train_path, sample = True)

test = read_data(path+test_path, sample = False)

train.head()
sample_submission = pd.read_csv(path + sample_submission_path)

print('Our sample submission have {} rows'.format(sample_submission.shape[0]))

sample_submission.head()
def missing_values(df):

    df = pd.DataFrame(df.isnull().sum()).reset_index()

    df.columns = ['features', 'n_missing_values']

    return df

missing_values(train)
missing_values(test)
question_text_0 = train.loc[30, 'question_text']

question_text_0
document_text_0 = train.loc[30, 'document_text'].split()

" ".join(document_text_0[:800])
long_answer_candidates_0 = train.loc[30, 'long_answer_candidates']

long_answer_candidates_0[0:10]
annotations_0 = train['annotations'][30][0]

annotations_0
print('Our question is : ', question_text_0)

print('Our short answer is : ', " ".join(document_text_0[annotations_0['short_answers'][0]['start_token']:annotations_0['short_answers'][0]['end_token']]))

print('Our long answer is : ', " ".join(document_text_0[annotations_0['long_answer']['start_token']:annotations_0['long_answer']['end_token']]))
yes_no_answer = []

for i in range(len(train)):

    yes_no_answer.append(train['annotations'][i][0]['yes_no_answer'])

yes_no_answer = pd.DataFrame({'yes_no_answer': yes_no_answer})

    

def bar_plot(df, column, title, width, height, n, get_count = True):

    if get_count == True:

        cnt_srs = df[column].value_counts(normalize = True)[:n]

    else:

        cnt_srs = df

        

    trace = go.Bar(

        x = cnt_srs.index,

        y = cnt_srs.values,

        marker = dict(

            color = '#1E90FF',

        ),

    )



    layout = go.Layout(

        title = go.layout.Title(

            text = title,

            x = 0.5

        ),

        font = dict(size = 14),

        width = width,

        height = height,

    )



    data = [trace]

    fig = go.Figure(data = data, layout = layout)

    py.iplot(fig, filename = 'bar_plot')

bar_plot(yes_no_answer, 'yes_no_answer', 'Yes No Answer Distribution', 800, 500, 3)
# this function extract the short answers and fill a dataframe

def extract_target_variable(df, short = True):

    if short:

        short_answer = []

        for i in range(len(df)):

            short = df['annotations'][i][0]['short_answers']

            if short == []:

                yes_no = df['annotations'][i][0]['yes_no_answer']

                if yes_no == 'NO' or yes_no == 'YES':

                    short_answer.append(yes_no)

                else:

                    short_answer.append('EMPTY')

            else:

                short = short[0]

                st = short['start_token']

                et = short['end_token']

                short_answer.append(f'{st}'+':'+f'{et}')

        short_answer = pd.DataFrame({'short_answer': short_answer})

        return short_answer

    else:

        long_answer = []

        for i in range(len(df)):

            long = df['annotations'][i][0]['long_answer']

            if long['start_token'] == -1:

                long_answer.append('EMPTY')

            else:

                st = long['start_token']

                et = long['end_token']

                long_answer.append(f'{st}'+':'+f'{et}')

        long_answer = pd.DataFrame({'long_answer': long_answer})

        return long_answer

        

short_answer = extract_target_variable(train)

short_answer.head()
short_answer['type'] = short_answer['short_answer'].copy()

short_answer.loc[(short_answer['short_answer']!='EMPTY') & (short_answer['short_answer']!='YES') & (short_answer['short_answer']!='NO'), 'type'] =  'TEXT'

bar_plot(short_answer, 'type', 'Short Answer Distribution', 800, 500, 10)
long_answer = extract_target_variable(train, False)

long_answer.head()
long_answer['type'] = long_answer['long_answer'].copy()

long_answer.loc[(long_answer['long_answer']!='EMPTY'), 'type'] =  'TEXT'

bar_plot(long_answer, 'type', 'Long Answer Distribution', 800, 500, 10)
def count_word_frequency(series, top = 0, bot = 20):

    cv = CountVectorizer()   

    cv_fit = cv.fit_transform(series)    

    word_list = cv.get_feature_names(); 

    count_list = cv_fit.toarray().sum(axis=0)

    frequency = pd.DataFrame({'Word': word_list, 'Frequency': count_list})

    frequency.sort_values(['Frequency'], ascending = False, inplace = True)

    frequency['Percentage'] = frequency['Frequency']/frequency['Frequency'].sum()

    frequency.drop('Frequency', inplace = True, axis = 1)

    frequency['Percentage'] = frequency['Percentage'].round(3)

    frequency = frequency.iloc[top:bot]

    frequency.set_index('Word', inplace = True)

    bar_plot(pd.Series(frequency['Percentage']), 'Percentage', 'Question Text Word Frequency Distribution', 800, 500, 20, False)

    return frequency

    

frequency = count_word_frequency(train['question_text'])
frequency = count_word_frequency(train['question_text'], 20, 40)
frequency = count_word_frequency(test['question_text'])
frequency = count_word_frequency(test['question_text'], 20, 40)
frequency = count_word_frequency(test['document_text'])
from collections import defaultdict

from wordcloud import WordCloud, STOPWORDS



long_answer = extract_target_variable(train, False)

short_answer = extract_target_variable(train)



## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    punct = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—"--…""'

    token = [token for token in text.lower().split(" ") if token != "" 

             if token != " " if token not in STOPWORDS if "<" not in token if token not in punct] #modified

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



## Get the bar chart from sincere questions ##

freq_dict = defaultdict(int)

for sent in train['question_text']:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in test['question_text']:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words train", 

                                          "Frequent words test"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')



#plt.figure(figsize=(10,16))

#sns.barplot(x="ngram_count", y="ngram", data=fd_sorted.loc[:50,:], color="b")

#plt.title("Frequent words for Insincere Questions", fontsize=16)

#plt.show()
freq_dict = defaultdict(int)

for sent in train['question_text']:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'orange')





freq_dict = defaultdict(int)

for sent in test['question_text']:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,

                          subplot_titles=["Frequent bigrams train", 

                                          "Frequent bigrams test"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")

py.iplot(fig, filename='word-plots')
freq_dict = defaultdict(int)

cnt = 0

for sent in test['document_text']:

    if cnt < 10:

        #print(cnt)

        #print(sent)

        cnt += 1

    else:

        break

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'orange')



cnt = 0

for sent in test['document_text']:

    if cnt < 30:

        #print(cnt)

        #print(sent)

        cnt += 1

    else:

        break

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,

                          subplot_titles=["Frequent bigrams test", 

                                          "Frequent bigrams test"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")

py.iplot(fig, filename='word-plots')
def build_train_test_long(df, train = True):

    final_long_answer_frame = pd.DataFrame()

    if train == True:

        # get long answer

        long_answer = extract_target_variable(df, False)

        

        # iterate over each row to get the possible answers

        for index, row in df.iterrows():

            start_end_tokens = []

            questions = []

            responds = []

            for i in row['long_answer_candidates']:

                start_token = i['start_token']

                end_token = i['end_token']

                start_end_token = str(i['start_token']) + ':' + str(i['end_token'])

                question = row['question_text']

                respond = " ".join(row['document_text'].split()[start_token : end_token])

                start_end_tokens.append(start_end_token)

                questions.append(question)

                responds.append(respond)



            long_answer_frame = pd.DataFrame({'question': questions, 'respond': responds, 'start_end_token': start_end_tokens})

            long_answer_frame['answer'] = long_answer.iloc[index][0]

            long_answer_frame['target'] = long_answer_frame['start_end_token'] == long_answer_frame['answer']

            long_answer_frame['target'] = long_answer_frame['target'].astype('int16')

            long_answer_frame.drop(['answer'], inplace = True, axis = 1)

            final_long_answer_frame = pd.concat([final_long_answer_frame, long_answer_frame])

        return final_long_answer_frame

    else:

         # iterate over each row to get the possible answers

        for index, row in df.iterrows():

            start_end_tokens = []

            questions = []

            responds = []

            for i in row['long_answer_candidates']:

                start_token = i['start_token']

                end_token = i['end_token']

                start_end_token = str(i['start_token']) + ':' + str(i['end_token'])

                question = row['question_text']

                respond = " ".join(row['document_text'].split()[start_token : end_token])

                start_end_tokens.append(start_end_token)

                questions.append(question)

                responds.append(respond)



            long_answer_frame = pd.DataFrame({'question': questions, 'respond': responds, 'start_end_token': start_end_tokens})

            final_long_answer_frame = pd.concat([final_long_answer_frame, long_answer_frame])

        return final_long_answer_frame

        





def build_train_test_short(df, train = True):

    

    final_short_answer_frame = pd.DataFrame()

    

    if train == True:

        # get short answer

        short_answer = extract_target_variable(df, True)



        # iterate over each row to get the possible answer

        for index, row in df.iterrows():

            start_tokens = []

            end_tokens = []

            start_end_tokens = []

            questions = []

            responds = []

            for i in row['long_answer_candidates']:

                start_token = i['start_token']

                end_token = i['end_token']

                start_end_token = str(i['start_token']) + ':' + str(i['end_token'])

                question = row['question_text']

                respond = " ".join(row['document_text'].split()[int(start_token) : int(end_token)])

                start_tokens.append(start_token)

                end_tokens.append(end_token)

                start_end_tokens.append(start_end_token)

                questions.append(question)

                responds.append(respond)



            short_answer_frame = pd.DataFrame({'question': questions, 'respond': responds, 'start_token': start_tokens, 'end_token': end_tokens, 'start_end_token': start_end_tokens})

            short_answer_frame['answer'] = short_answer.iloc[index][0]

            short_answer_frame['start_token_an'] = short_answer_frame['answer'].apply(lambda x: x.split(':')[0] if ':' in x else 0)

            short_answer_frame['end_token_an'] = short_answer_frame['answer'].apply(lambda x: x.split(':')[1] if ':' in x else 0)

            short_answer_frame['start_token_an'] = short_answer_frame['start_token_an'].astype(int)

            short_answer_frame['end_token_an'] = short_answer_frame['end_token_an'].astype(int)

            short_answer_frame['target'] = 0

            short_answer_frame.loc[(short_answer_frame['start_token_an'] >= short_answer_frame['start_token']) & (short_answer_frame['end_token_an'] <= short_answer_frame['end_token']), 'target'] = 1

            short_answer_frame.drop(['answer', 'start_token', 'end_token', 'start_token_an', 'end_token_an'], inplace = True, axis = 1)

            final_short_answer_frame = pd.concat([final_short_answer_frame, short_answer_frame])

        return final_short_answer_frame

    else:

        # iterate over each row to get the possible answer

        for index, row in df.iterrows():

            start_end_tokens = []

            questions = []

            responds = []

            for i in row['long_answer_candidates']:

                start_token = i['start_token']

                end_token = i['end_token']

                start_end_token = str(i['start_token']) + ':' + str(i['end_token'])

                question = row['question_text']

                respond = " ".join(row['document_text'].split()[int(start_token) : int(end_token)])

                start_end_tokens.append(start_end_token)

                questions.append(question)

                responds.append(respond)



            short_answer_frame = pd.DataFrame({'question': questions, 'respond': responds, 'start_end_token': start_end_tokens})

            final_short_answer_frame = pd.concat([final_short_answer_frame, short_answer_frame])

        return final_short_answer_frame
sh = build_train_test_long(train.head())

sh.head()
sh[sh['target']==1]
long_answer.head()