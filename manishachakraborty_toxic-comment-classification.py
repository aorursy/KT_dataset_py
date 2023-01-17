

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

from nltk import word_tokenize

import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB

from skmultilearn.problem_transform import ClassifierChain

from sklearn.metrics import accuracy_score

from collections import Counter

from wordcloud import WordCloud,STOPWORDS

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

train.head()

test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

test_labels = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')

sample_submission = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')



test.head()

def instance_per_category(data):

    data_categories = data.drop(['id','comment_text'],axis=1)

    count_toxic = Counter(data_categories['toxic'])

    count_severe = Counter(data_categories['severe_toxic'])

    count_obscene = Counter(data_categories['obscene'])

    count_threat = Counter(data_categories['threat'])

    count_insult = Counter(data_categories['insult'])

    count_iden = Counter(data_categories['identity_hate'])

    df = pd.DataFrame.from_dict([count_toxic,count_severe,count_obscene,count_threat,count_insult,count_iden])

    df.index  = ['toxic','severe toxic','obscene','threat','insult','identity hate']

    return df.T



print('total samples: {0}'.format(train.shape[0]))

train_categories = instance_per_category(train)

train_categories
def plot_category(data):

    x = np.arange(len(data.columns))

    labels = list(data.columns)

    width = 0.25

    fig,ax = plt.subplots()

    fig.suptitle('Summary of training dataset')

    ax1= ax.bar(x = x+ width/2,  height = data.iloc[0,:], width = width, label='negetive/ not labeled ')

    ax2 = ax.bar(x = x- width/2,  height = data.iloc[1,:], width = width, label= 'positive')

    ax.set_ylabel('number of comments')

    ax.set_xticks(x)

    ax.set_xticklabels(labels)

    ax.legend()

    plt.show()





plot_category(train_categories)
def multi_label(data):

    sum_ = data.iloc[:,2:].sum(axis=1)

    x = sum_.value_counts()

    fig = plt.figure(figsize = (17,8))

    ax = plt.bar(x.index,x.values )

    for bar in ax:

        bar_h = bar.get_height()

        plt.text(bar.get_x(), bar_h, bar_h)       

    plt.xticks(x.index,['unlabeled','1 category','2 categories','3 categories','4 categories','5 categories'])

    plt.title('number of categories for each comment')

    plt.xlabel('number of categories')

    plt.ylabel('comments')

    plt.show()

    

multi_label(train)
print('total unlabelled data: {0}'.format( sum( (train.iloc[:,2] == 0) & (train.iloc[:,3] == 0) & (train.iloc[:,4] == 0) &(train.iloc[:,5] == 0) & (train.iloc[:,6] == 0) & (train.iloc[:,7] == 0))))

print('number of missing comments: {0}'.format(train.iloc[:,1].isnull().sum()))
def text_cleaning(text: str)-> str:

    text = text.lower()

    text = re.sub(r"n't", " not", text)

    text = re.sub(r"\n" , " ", text)

    text = re.sub(r"\'d'", " ", text)

    text = re.sub(r"\'s'" , " ", text)

    text = re.sub(r"\'ll", " ", text)

    text = re.sub(r"\'m'", " ", text)

    text = re.sub(r"\'re'", " ", text)

    return text

    

train['comment_text'] = train['comment_text'].map( lambda text : text_cleaning(str(text)))

test['comment_text'] = test['comment_text'].map( lambda text : text_cleaning(str(text)))
def word_cloud(text):

    plt.figure(figsize=(20,9))

    texts = text.values

    cloud = WordCloud(stopwords=STOPWORDS, background_color = 'white', collocations = False, width=2000, height= 1000).generate(" ".join(text))

    plt.axis('off')

    plt.title('word cloud')

    plt.imshow(cloud)

word_cloud(train['comment_text'])
def comment_length(data):

    comments = data.iloc[:,1]

    length_comments = [len (comment.split(' ')) for comment in comments]

    max_len = max(length_comments)

    min_len = min(length_comments)

    count = Counter(length_comments)

    print(" frequency of the length of the comments")

    plt.hist(list(count))

    return count

comment_lengths = comment_length(train)
def tokenize(text):

    tokens = word_tokenize(text)

    tokens_alpha = [token for token in tokens if token.isalpha()]

    return tokens_alpha



train['comment_text'] = train['comment_text'].map(lambda text: tokenize(text))

test['comment_text'] = test['comment_text'].map(lambda text: tokenize(text))
train_set,test_set = train_test_split(train,random_state=42,test_size=0.3,shuffle=True)

x_train = train_set['comment_text'].values

y_train= train_set.iloc[:,2:]



x_test = test_set['comment_text'].values

y_test = test_set.iloc[:,2:]
vectorizer = TfidfVectorizer(max_features=1500)

x_train = vectorizer.fit_transform(str(x) for x in x_train)

x_test = vectorizer.transform(str(x) for x in x_test)

sub_x = vectorizer.transform(str(x) for x in test['comment_text'].values)
classifier = ClassifierChain(MultinomialNB(fit_prior=True))

classifier.fit(x_train,y_train)

y_predict = classifier.predict(x_test)

score = accuracy_score(y_test,y_predict)

print('estimated accuracy score: {0}'.format(score))
sample_submission.head()
test_predict = classifier.predict_proba(sub_x)

test_predict= test_predict.todense()
ids = pd.DataFrame(test['id'])

predict = pd.DataFrame(test_predict,columns= ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

submission = pd.concat([ids, predict] ,axis = 1)

submission.to_csv('submission.csv',index=False)