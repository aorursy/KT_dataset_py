def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


Sentence_1 = 'Life well spent is life good'
Sentence_2 = 'Life is an art and it is good so far'
Sentence_3 = 'Life is good'

    
print(jaccard(Sentence_1,Sentence_2))
print(jaccard(Sentence_1,Sentence_3))
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!pip install '/kaggle/input/simple-transformers-pypi/seqeval-0.0.12-py3-none-any.whl' -q
!pip install '/kaggle/input/simple-transformers-pypi/simpletransformers-0.22.1-py3-none-any.whl' -q
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import tensorflow as tf
import json
train = pd.read_csv(r'/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv(r'/kaggle/input/tweet-sentiment-extraction/test.csv')
sample_submission = pd.read_csv(r'/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
print('Train Data Shape:', train.shape)
print('Test Data Shape:', test.shape)
train.describe()
train.isnull().sum()
test.isnull().sum()
train.dropna(inplace = True)
train['sentiment'].value_counts()
plt.figure(figsize = (13,5))
plt.subplot(121)
plt.title('Distribution of sentiments in Train Data')
sns.countplot(train['sentiment'])
plt.subplot(122)
sns.countplot(test['sentiment'])
plt.title('Distribution of sentiments in Test Data')
train.head(1)
def clean_data(data):
    # Removing extra spaces in the beginning of text
    data = data.strip()
    # Lower the Text
    data = data.lower()
    return data
train['text'] = train['text'].apply(lambda x: clean_data(x))
test['text'] = test['text'].apply(lambda x: clean_data(x))
train['selected_text'] = train['selected_text'].apply(lambda x: clean_data(x))
train['num_words_text'] = train['text'].str.split().apply(lambda x: len(x))
test['num_words_text'] = test['text'].str.split().apply(lambda x: len(x))
train['num_words_selected_text'] = train['selected_text'].str.split().apply(lambda x: len(x))
plt.figure(figsize = (15,20))
plt.suptitle('Comparing train and test data based on text word length', fontsize = 22)
plt.subplot(311)
plt.xlabel('Positive Sentiment Text Length', fontsize = 15)
plt.ylabel('Distribution', fontsize = 15)
sns.kdeplot(train[train['sentiment']=='positive']['num_words_text'].values, shade = True, color = 'blue', label = 'train')
sns.kdeplot(test[test['sentiment']=='positive']['num_words_text'].values, shade = True, color = 'red', label = 'train')
plt.subplot(312)
plt.xlabel('Neutral Sentiment Text Length', fontsize = 15)
plt.ylabel('Distribution', fontsize = 15)
sns.kdeplot(train[train['sentiment']=='neutral']['num_words_text'].values, shade = True, color = 'blue', label = 'train')
sns.kdeplot(test[test['sentiment']=='neutral']['num_words_text'].values, shade = True, color = 'red', label = 'train')
plt.subplot(313)
plt.xlabel('Negaitive Sentiment Text Length', fontsize = 15)
plt.ylabel('Distribution', fontsize = 15)
sns.kdeplot(train[train['sentiment']=='negative']['num_words_text'].values, shade = True, color = 'blue', label = 'train')
sns.kdeplot(test[test['sentiment']=='negative']['num_words_text'].values, shade = True, color = 'red', label = 'train')
plt.figure(figsize = (15,20))
plt.suptitle('Comparing train text and selected text word length', fontsize = 22)
plt.subplot(311)
plt.xlabel('Positive Sentiment Text Length', fontsize = 15)
plt.ylabel('Positive Sentiment Selected Text Length', fontsize = 15)
sns.kdeplot(train[train['sentiment']=='positive']['num_words_text'].values, shade = True, color = 'blue', label = 'train')
sns.kdeplot(train[train['sentiment']=='positive']['num_words_selected_text'].values, shade = True, color = 'red', label = 'train')
plt.subplot(312)
plt.xlabel('Neutral Sentiment Text Length', fontsize = 15)
plt.ylabel('Neutral Sentiment Selected Text Length', fontsize = 15)
sns.kdeplot(train[train['sentiment']=='neutral']['num_words_text'].values, shade = True, color = 'blue', label = 'train')
sns.kdeplot(train[train['sentiment']=='neutral']['num_words_selected_text'].values, shade = True, color = 'red', label = 'train')
plt.subplot(313)
plt.xlabel('Negaitive Sentiment Text Length', fontsize = 15)
plt.ylabel('Negative Sentiment Selected Text Length', fontsize = 15)
sns.kdeplot(train[train['sentiment']=='negative']['num_words_text'].values, shade = True, color = 'blue', label = 'train')
sns.kdeplot(train[train['sentiment']=='negative']['num_words_selected_text'].values, shade = True, color = 'red', label = 'train')
def punctuation_count(data):
    x = len([w for w in data if w in string.punctuation])
    return x
train['punct_count_text'] = train['text'].apply(lambda x: punctuation_count(x))
train['punct_count_selected_text'] = train['selected_text'].apply(lambda x: punctuation_count(x))
plt.figure(figsize = (12,6))
plt.suptitle('Comparing train text and selected text punctuation length', fontsize = 22)
plt.xlabel('Punctuation Count', fontsize = 15)
plt.ylabel('Distribution', fontsize = 15)
sns.kdeplot(train['punct_count_text'].values, shade = True, color = 'blue', label = 'train')
sns.kdeplot(train['punct_count_selected_text'].values, shade = True, color = 'red', label = 'train')
positive_tweet = train[train['sentiment']=='positive']
negative_tweet = train[train['sentiment']=='negative']
neutral_tweet = train[train['sentiment']=='neutral']
def get_top_n_words(corpus, ngram_range = (1,1), n = None):
    vec = CountVectorizer(ngram_range = ngram_range, stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis = 0)
    word_freq = [(word, sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
    word_freq = sorted(word_freq, key = lambda x: x[1], reverse = True)
    return word_freq[:n]
pos_unigram = get_top_n_words(positive_tweet['text'], (1,1), 20)
neutral_unigram = get_top_n_words(neutral_tweet['text'], (1,1), 20)
neg_unigram = get_top_n_words(negative_tweet['text'], (1,1), 20)

df1 = pd.DataFrame(pos_unigram, columns = ['word','count'])
df2 = pd.DataFrame(neutral_unigram, columns = ['word','count'])
df3 = pd.DataFrame(neg_unigram, columns = ['word','count'])

plt.tight_layout()
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(12,17))
sns.barplot(x = 'count' , y = 'word', data = df1, orient = 'h',ax = ax1)
ax1.set_title('Most repititve words in positive tweets')
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.grid(False)
sns.barplot(x = 'count' , y = 'word', data = df2, orient = 'h',ax = ax2)
ax2.set_title('Most repititve words in neutral tweets')
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.grid(False)
sns.barplot(x = 'count' , y = 'word', data = df3, orient = 'h',ax = ax3)
ax3.set_title('Most repititve words in negative tweets')
ax3.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.grid(False)
pos_bigram = get_top_n_words(positive_tweet['text'], (2,2), 20)
neutral_bigram = get_top_n_words(neutral_tweet['text'], (2,2), 20)
neg_bigram = get_top_n_words(negative_tweet['text'], (2,2), 20)

df1 = pd.DataFrame(pos_bigram, columns = ['word','count'])
df2 = pd.DataFrame(neutral_bigram, columns = ['word','count'])
df3 = pd.DataFrame(neg_bigram, columns = ['word','count'])

plt.tight_layout()
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(12,17))
sns.barplot(x = 'count' , y = 'word', data = df1, orient = 'h',ax = ax1)
ax1.set_title('Most repititve words in positive tweets')
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.grid(False)
sns.barplot(x = 'count' , y = 'word', data = df2, orient = 'h',ax = ax2)
ax2.set_title('Most repititve words in neutral tweets')
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.grid(False)
sns.barplot(x = 'count' , y = 'word', data = df3, orient = 'h',ax = ax3)
ax3.set_title('Most repititve words in negative tweets')
ax3.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.grid(False)
pos_trigram = get_top_n_words(positive_tweet['text'], (3,3), 20)
neutral_trigram = get_top_n_words(neutral_tweet['text'], (3,3), 20)
neg_trigram = get_top_n_words(negative_tweet['text'], (3,3), 20)

df1 = pd.DataFrame(pos_trigram, columns = ['word','count'])
df2 = pd.DataFrame(neutral_trigram, columns = ['word','count'])
df3 = pd.DataFrame(neg_trigram, columns = ['word','count'])

plt.tight_layout()
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(12,17))
sns.barplot(x = 'count' , y = 'word', data = df1, orient = 'h',ax = ax1)
ax1.set_title('Most repititve words in positive tweets')
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.grid(False)
sns.barplot(x = 'count' , y = 'word', data = df2, orient = 'h',ax = ax2)
ax2.set_title('Most repititve words in neutral tweets')
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.grid(False)
sns.barplot(x = 'count' , y = 'word', data = df3, orient = 'h',ax = ax3)
ax3.set_title('Most repititve words in negative tweets')
ax3.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.grid(False)
pos_unigram = get_top_n_words(positive_tweet['selected_text'], (1,1), 20)
neutral_unigram = get_top_n_words(neutral_tweet['selected_text'], (1,1), 20)
neg_unigram = get_top_n_words(negative_tweet['selected_text'], (1,1), 20)

df1 = pd.DataFrame(pos_unigram, columns = ['word','count'])
df2 = pd.DataFrame(neutral_unigram, columns = ['word','count'])
df3 = pd.DataFrame(neg_unigram, columns = ['word','count'])

plt.tight_layout()
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(12,17))
sns.barplot(x = 'count' , y = 'word', data = df1, orient = 'h',ax = ax1)
ax1.set_title('Most repititve words in positive selected_text')
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.grid(False)
sns.barplot(x = 'count' , y = 'word', data = df2, orient = 'h',ax = ax2)
ax2.set_title('Most repititve words in neutral selected_text')
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.grid(False)
sns.barplot(x = 'count' , y = 'word', data = df3, orient = 'h',ax = ax3)
ax3.set_title('Most repititve words in negative selected_text')
ax3.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.grid(False)
stopwords = set(STOPWORDS)
def word_cloud(data, title=None):
    cloud = WordCloud(background_color = 'black',
                     stopwords = stopwords,
                     max_words = 200,
                     max_font_size = 40,
                     scale = 3).generate(str(data))
    fig = plt.figure(figsize=(15,15))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.25)
        plt.imshow(cloud)
        plt.show()
word_cloud(positive_tweet['text'], 'Most Repeated Words in Positive Text Tweets')
word_cloud(neutral_tweet['text'], 'Most Repeated Words in Neutral Text Tweets')
word_cloud(negative_tweet['text'], 'Most Repeated Words in Negative Text Tweets')
train_data = [
    {
        'context': "This tweet sentiment extraction challenge is great",
        'qas': [
            {
                'id': "00001",
                'question': "positive",
                'answers': [
                    {
                        'text': "is great",
                        'answer_start': 43
                    }
                ]
            }
        ]
    }
    ]
l = []
def start_index(text, selected_text):
    start_index = text.find(selected_text)
    l.append(start_index)
    
for i in range(len(train)):
    start_index(train.iloc[i,1], train.iloc[i,2])
def ques_ans_format_train(train):
    output = []
    for i in range(len(train)):
        context = train.iloc[i,1]
        qas = []
        qid = train.iloc[i,0]
        ques = train.iloc[i,3]
        ans = []
        answer = train.iloc[i,2]
        answer_start = l[i]
        ans.append({'text': answer, 'answer_start': answer_start})
        qas.append({'id': qid, 'question': ques, 'is_impossible': False, 'answers': ans})        
        output.append({'context': context, 'qas': qas})
    return output

train_json_format = ques_ans_format_train(train)
# Save as a JSON file
os.makedirs('data', exist_ok=True)
with open('data/train.json', 'w') as f:
    json.dump(train_json_format, f)
    f.close()
test_data = ([
    {
        'context': "Some context as a demo",
        'qas': [
            {'id': '0', 'question': 'neutral'}
        ]
    }
])
def ques_ans_format_test(test):
    output = []
    for i in range(len(test)):
        context = test.iloc[i,1]
        qas = []
        qid = test.iloc[i,0]
        ques = test.iloc[i,2]
        qas.append({'id': qid, 'question': ques})
        
        output.append({'context': context, 'qas': qas})
    return output

test_json_format = ques_ans_format_test(test)
# Save as a JSON file
os.makedirs('data', exist_ok=True)
with open('data/test.json', 'w') as f:
    json.dump(test_json_format, f)
    f.close()
from simpletransformers.question_answering import QuestionAnsweringModel

use_cuda = True 
model_path = '/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad/'

# Create the QuestionAnsweringModel
model = QuestionAnsweringModel('distilbert', 
                               model_path, 
                               args={'reprocess_input_data': True,
                                     'overwrite_output_dir': True,
                                     'learning_rate': 5e-5,
                                     'num_train_epochs': 4,
                                     'max_seq_length': 128,
                                     'doc_stride': 64,
                                     'fp16': False,
                                    },
                              use_cuda=use_cuda)

model.train_model(r'data/train.json')
pred = model.predict(test_json_format)
df = pd.DataFrame.from_dict(pred)
sample_submission["selected_text"] = df["answer"]
# new_df = sample_submission.merge(test,how="inner",on="textID")
# new_df["selected_text"] = np.where((new_df["sentiment"] == "neutral"),new_df["text"], new_df["selected_text"])
# submission = new_df[["textID", "selected_text"]]
sample_submission.to_csv("submission.csv", index = False)
print("File submitted successfully.")