from tqdm.autonotebook import tqdm as tqdm

from copy import deepcopy



import matplotlib.pyplot as plt

import pandas as pd

import numpy  as np

import json
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv', header=0)

test_df  = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv',  header=0)



train_df['text'] = train_df.text.apply(lambda x: x.strip() if isinstance(x, str) else '')

test_df['text']  = test_df .text.apply(lambda x: x.strip() if isinstance(x, str) else '')
train_df.head(10)
train_df.loc[train_df.selected_text.isna()]
train_df = train_df.drop(13133)
def plot_pie_charts(*pies, side_by_side=True, figsize=(10, 4)):

    """

    A generic function to plot pie charts in a line. If side_by_side

    is True, said line is horizontal (charts are shown side by side),

    if False, it's vertial (charts are stacked on top of each other).

    """

    _, axes = plt.subplots(nrows=1 if side_by_side else len(pies), ncols=len(pies) if side_by_side else 1, figsize=figsize)

    if not isinstance(axes, np.ndarray):

        axes = np.array([axes])

    for axis, pie in zip(axes, pies):

        title = pie.get('title')

        axis.set_title(title)

        try:

            del pie['title']

        except KeyError:

            pass

        axis.axis('equal')

        axis.pie( ** pie )

    if side_by_side:

        plt.tight_layout()

    plt.show()
train_tweets_by_sentiment, test_tweets_by_sentiment = {'samples': {}, 'full_text': {}, 'partial_text': {}}, {'samples': {}}

for sentiment in ['positive', 'negative', 'neutral']:

    train_tweets_by_sentiment['samples'][sentiment] = len(train_df[train_df.sentiment == sentiment])

    test_tweets_by_sentiment['samples'][sentiment]  = len(test_df[ test_df.sentiment == sentiment ])

    train_tweets_by_sentiment['full_text'][sentiment] = len(train_df[(train_df.sentiment == sentiment) & (train_df.selected_text == train_df.text)])

    train_tweets_by_sentiment['partial_text'][sentiment] = train_tweets_by_sentiment['samples'][sentiment] - train_tweets_by_sentiment['full_text'][sentiment]
plot_pie_charts(

    {'title': 'train tweets by sentiment', 'x': list(train_tweets_by_sentiment['samples'].values()), 'labels': train_tweets_by_sentiment['samples'].keys(), 'explode': [0, 0, 0.05], 'shadow': True, 'autopct': '%1.1f%%', 'startangle': 90},

    {'title': 'test  tweets by sentiment', 'x': list(train_tweets_by_sentiment['samples'].values( )), 'labels': train_tweets_by_sentiment['samples'].keys(), 'explode': [0, 0, 0.05], 'shadow': True, 'autopct': '%1.1f%%', 'startangle': 90}

)
plot_pie_charts(

    {'title': 'positive train tweets selected_text', 'x': [train_tweets_by_sentiment['full_text']['positive'], train_tweets_by_sentiment['partial_text']['positive']], 'labels': ['using all tweet', 'using part of tweet'], 'explode': [0.0, 0.10], 'shadow': True, 'autopct': '%1.1f%%', 'startangle': 30},

    {'title': 'negative train tweets selected_text', 'x': [train_tweets_by_sentiment['full_text']['negative'], train_tweets_by_sentiment['partial_text']['negative']], 'labels': ['using all tweet', 'using part of tweet'], 'explode': [0.0, 0.10], 'shadow': True, 'autopct': '%1.1f%%', 'startangle': 30},

    {'title': 'neutral  train tweets selected_text', 'x': [train_tweets_by_sentiment['full_text']['neutral' ], train_tweets_by_sentiment['partial_text']['neutral' ]], 'labels': ['using all tweet', 'using part of tweet'], 'explode': [0.0, 0.05], 'shadow': True, 'autopct': '%1.1f%%', 'startangle': 30},

    figsize=(15, 4)

)
for sentiment in ['positive', 'negative', 'neutral ']:

    train_df_with_sentiment = train_df.loc[train_df.sentiment == sentiment.strip()]

    selected_text_to_text_ratio = np.mean(np.array([len(r['selected_text']) / len(r['text']) for _, r in train_df_with_sentiment.iterrows()]))

    print(f'For {sentiment} tweets, the selected text covers {round(selected_text_to_text_ratio * 100, 2)}% of the tweet text on average.')
jaccard_of_full_positive_tweets_by_number_of_tokens = {}

jaccard_of_full_negative_tweets_by_number_of_tokens = {}



def jaccard(str1, str2):

    a = set(str1.lower().split())

    b = set(str2.lower().split())

    c = a.intersection(b)

    

    return float(len(c)) / (len(a) + len(b) - len(c))

    



for _, tweet in train_df.iterrows():

    tokens = len(tweet.text.split())

    

    if tweet.sentiment == 'positive':

        if tokens not in jaccard_of_full_positive_tweets_by_number_of_tokens:

            jaccard_of_full_positive_tweets_by_number_of_tokens[tokens] = []



        jaccard_of_full_positive_tweets_by_number_of_tokens[tokens].append(

                                  jaccard(tweet.selected_text, tweet.text))

        

    if tweet.sentiment == 'negative':

        if tokens not in jaccard_of_full_negative_tweets_by_number_of_tokens:

            jaccard_of_full_negative_tweets_by_number_of_tokens[tokens] = []



        jaccard_of_full_negative_tweets_by_number_of_tokens[tokens].append(

                                  jaccard(tweet.selected_text, tweet.text))
average_jaccard_of_full_positive_tweets_by_number_of_tokens = {k: np.mean(v) for k, v in jaccard_of_full_positive_tweets_by_number_of_tokens.items()}

average_jaccard_of_full_negative_tweets_by_number_of_tokens = {k: np.mean(v) for k, v in jaccard_of_full_negative_tweets_by_number_of_tokens.items()}
def plot_opposite_bars(above, below):

    X = np.array(sorted(list({x for x in list(above.keys()) + list(below.keys())})))

    Y1 = np.array([above.get(x, 0.0) for x in X])

    Y2 = np.array([below.get(x, 0.0) for x in X])

    

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    

    ax.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')

    ax.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')



    for x, y in zip(X, Y1):

        ax.text(x, y, '%.2f' % y, ha='center', va='bottom')

    for x, y in zip(X, Y2):

        ax.text(x, -y, '%.2f' % y, ha='center', va='top'  )

        

    plt.xticks(np.arange(np.max(X)))

        

    labels = ['positive', 'negative']

    colors = {'positive': '#9999ff', 'negative': '#ff9999'}

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]

    ax.set_title('Jaccard Distance for Positive and Negative Tweets (use all text)')

    ax.legend(handles, labels)

    plt.show()



    

plot_opposite_bars(

    average_jaccard_of_full_positive_tweets_by_number_of_tokens,

    average_jaccard_of_full_negative_tweets_by_number_of_tokens

)
!mkdir -p data

!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q

!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q
use_cuda = True
def do_qa_train(df):

    output = {}

    output['version'] = 'v1.0'

    output['data'] = []

    paragraphs = []

    

    pbar = tqdm(total=len(df), desc='building training data for simple-transformers')

    

    for _, row in df.iterrows():

        context = row['text']

        answers = []

        qas = []

        qid = row['textID']

        question = row[ 'sentiment' ]

        answer = row['selected_text']

        

        if type(answer) != str or type(context) != str or type(question) != str:

            print(f'skipping {row["textID"]} (invalid format)')

            continue

            

        answers.append({'answer_start': context.find(answer), 'text': answer.lower()})

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

        

        pbar.update(1)

    pbar.close()

        

    return paragraphs



qa_train = do_qa_train(train_df[train_df.sentiment != 'neutral'])  # ignore neutral data points
with open('data/train.json', 'w') as outfile:

    json.dump(qa_train, outfile)
output = {}

output['version'] = 'v1.0'

output['data'] = []



def do_qa_test(df):

    paragraphs = []

    

    pbar = tqdm(total=len(df), desc='building training data for simple-transformers')

    

    for _, row in df.iterrows():

        context = row['text']

        qas = []

        question = row['sentiment']

        qid = row['textID']

        

        if type(context) != str or type(question) != str:

            print(f'skipping {row["textID"]} (invalid format)')

            continue

            

        answers = []

        answers.append({'answer_start': 1000000, 'text': '__None__'})

        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        paragraphs.append({'context': context.lower(), 'qas': qas})

        output['data'].append({'title': 'None', 'paragraphs': paragraphs})

        

        pbar.update(1)

    pbar.close()

    

    return paragraphs



qa_test = do_qa_test(test_df[test_df.sentiment != 'neutral'])  # ignore neutral data points
with open('data/test.json', 'w') as outfile:

    json.dump(qa_test, outfile)
from simpletransformers.question_answering import QuestionAnsweringModel



MODEL_PATH = '/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad/'

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

                              use_cuda=use_cuda)

model.train_model('data/train.json')
predictions = model.predict(qa_test)

predictions_df = pd.DataFrame.from_dict(predictions)

predictions_df
submission_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv', header=0)



def get_span_for_textID(textID):

    row = test_df[test_df.textID == textID]

    if len(row) > 1:

        print(f'id {textID} is duplicated')

    row = row.iloc[0]

    if row.sentiment != 'neutral' and len(row.text.split()) > 2:

        return predictions_df[predictions_df.id == textID].iloc[0].answer

    return row.text.strip()



submission_df['selected_text'] = np.array([get_span_for_textID(textID) for textID in submission_df.textID.values])

submission_df
submission_df.to_csv('submission.csv', index=False)
def check(string, substring): 

    return f' {substring} ' not in f' {string} '



errors = train_df[train_df.apply(lambda row: check(row['text'], row['selected_text']), axis=1)]
errors
train_errors_by_sentiment = {}

for sentiment in ['positive', 'negative', 'neutral']:

    train_errors_by_sentiment[sentiment] = len(errors.loc[errors.sentiment == sentiment])



plot_pie_charts(

    {'title': 'train errors by sentiment', 'x': list(train_errors_by_sentiment.values()), 'labels': list(train_errors_by_sentiment.keys()), 'explode': [0.05, 0.0, 0.0], 'shadow': True, 'autopct': '%1.1f%%', 'startangle': 90}

)