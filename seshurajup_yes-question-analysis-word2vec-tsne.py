import json



import numpy as np

import pandas as pd

from tqdm import tqdm





def jsonl_to_df(file_path, n_rows=1000, load_annotations=True, truncate=True, offset=200):

    """

    Simple utility function to load the .jsonl files for the 

    TF2.0 QA competition. It creates a dataframe of the dataset.

    

    To use, click "File" > "Add utility script", search the name of this 

    notebook, then run:

    

    >>> from tf_qa_jsonl_to_dataframe import jsonl_to_df

    >>> train = jsonl_to_df("/kaggle/...train.jsonl")

    >>> test = jsonl_to_df("/kaggle/...test.jsonl", load_annotations=False)

    

    Parameters:

        * file_path: The path to your json_file

        * n: The number of rows you are importing

        * truncate: Whether to cut the text before the first answer (long or short)

          and after the last answer (long or short), leaving a space for the offset

        * offset: If offset = k, then keep only keep the interval (answer_start - k, answer_end + k)

        

    Returns:

        A Dataframe containing the following columns:

            * document_text (str): The document split by whitespace, possibly truncated

            * question_text (str): the question posed

            * yes_no_answer (str): Could be "YES", "NO", or "NONE"

            * short_answer_start (int): Start index of token, -1 if does not exist

            * short_answer_start (int): End index of token, -1 if does not exist

            * long_answer_start (int): Start index of token, -1 if does not exist

            * long_answer_start (int): End index of token, -1 if does not exist

        

        And may contain:

            * example_id (str): ID representing the string. Only for test data.

    

    Author: @xhlulu

    Source: https://www.kaggle.com/xhlulu/tf-qa-jsonl-to-dataframe

    """

    json_lines = []

    

    with open(file_path) as f:

        for i, line in enumerate(tqdm(f)):

            if not line:

                break

            if n_rows != -1 and i >= n_rows:

                break

                

            line = json.loads(line)

            last_token = line['long_answer_candidates'][-1]['end_token']



            out_di = {

                'document_text': line['document_text'],

                'question_text': line['question_text']

            }

            

            if 'example_id' in line:

                out_di['example_id'] = line['example_id']

            

            if load_annotations:

                annot = line['annotations'][0]

                

                out_di['yes_no_answer'] = annot['yes_no_answer']

                out_di['long_answer_start'] = annot['long_answer']['start_token']

                out_di['long_answer_end'] = annot['long_answer']['end_token']



                if len(annot['short_answers']) > 0:

                    out_di['short_answer_start'] = annot['short_answers'][0]['start_token']

                    out_di['short_answer_end'] = annot['short_answers'][0]['end_token']

                else:

                    out_di['short_answer_start'] = -1

                    out_di['short_answer_end'] = -1



                if truncate:

                    if out_di['long_answer_start'] == -1:

                        start_threshold = out_di['short_answer_start'] - offset

                    elif out_di['short_answer_start'] == -1:

                        start_threshold = out_di['long_answer_start'] - offset

                    else:

                        start_threshold = min(out_di['long_answer_start'], out_di['short_answer_start']) - offset

                        

                    start_threshold = max(0, start_threshold)

                    end_threshold = max(out_di['long_answer_end'], out_di['short_answer_end']) + offset + 1

                    

                    out_di['document_text'] = " ".join(

                        out_di['document_text'].split(' ')[start_threshold:end_threshold]

                    )



            json_lines.append(out_di)



    df = pd.DataFrame(json_lines).fillna(-1)

    

    return df
if __name__ == '__main__':

    directory = '/kaggle/input/tensorflow2-question-answering/'

    train = jsonl_to_df(directory + 'simplified-nq-train.jsonl', n_rows = 200000)

    test = jsonl_to_df(directory + 'simplified-nq-test.jsonl', n_rows = 1000, load_annotations=False)

    print(train.shape)

    print(test.shape)

    

    print(train.columns)

    print(test.columns)
train.head(5)
test.head(5)
train.count()
train.groupby(by=['yes_no_answer']).count()
train['question_text'].str.len().plot.hist()
train[train['yes_no_answer'] == 'YES']['question_text'].str.len().plot.hist()
train[train['yes_no_answer'] == 'NONE']['question_text'].str.len().plot.hist()
train[train['yes_no_answer'] == 'NO']['question_text'].str.len().plot.hist()
train_yes_questions = train[train['yes_no_answer'] == 'YES']['question_text']

train_no_questions = train[train['yes_no_answer'] == 'NO']['question_text']

train_none_questions = train[train['yes_no_answer'] == 'NONE']['question_text']
def build_corpus(data):

    "Creates a list of lists containing words from each yes/no questions"

    corpus = []

    for sentence in data.iteritems():

            word_list = sentence[1].split(" ")

            corpus.append(word_list)

    return corpus
from gensim.models import word2vec
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

%matplotlib inline



def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(16, 16)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
corpus_yes = build_corpus(train_yes_questions)

model_yes = word2vec.Word2Vec(corpus_yes, size=100, window=30, min_count=20)

tsne_plot(model_yes)
corpus_no = build_corpus(train_no_questions)

model_no = word2vec.Word2Vec(corpus_no, size=100, window=30, min_count=30)

tsne_plot(model_no)
corpus_none = build_corpus(train_none_questions)

model_none = word2vec.Word2Vec(corpus_none, size=100, window=250, min_count=1000)

tsne_plot(model_none)
model_yes.most_similar('is')
model_no.most_similar('is')
model_none.most_similar('is')