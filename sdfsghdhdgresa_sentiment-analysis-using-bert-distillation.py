import pandas as pd

import numpy as np

tweets = pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv")

tweets.head(0)
tweets.head(3)
del tweets["tweet_id"]

del tweets["airline_sentiment_confidence"]

del tweets["negativereason"]

del tweets["negativereason_confidence"]

del tweets["airline_sentiment_gold"]

del tweets["negativereason_gold"]

del tweets["tweet_coord"]

del tweets["user_timezone"]

del tweets["tweet_location"]

tweets.head(3)
import matplotlib.pyplot as plt

import seaborn as sns



tweets['n_words'] = [len(t.split()) for t in tweets.text]



fig = plt.figure(figsize = (15, 6))

sns.distplot(tweets['n_words'][tweets['airline_sentiment']=='positive'], color='g', label = 'positive')

sns.distplot(tweets['n_words'][tweets['airline_sentiment']=='negative'], color='r', label = 'negative')

sns.distplot(tweets['n_words'][tweets['airline_sentiment']=='neutral'], color='b', label = 'neutral')

plt.legend(loc='best')

plt.xlabel('# of Words', size = 14)

plt.ylabel('Count', size = 14)

plt.title('The Distribution of Number of Words for each Class', fontsize = 14)

plt.show()
import matplotlib.pyplot as plt

sentiment_counts = tweets.airline_sentiment.value_counts()

names = ['negative', 'neutral', 'positive']

values = sentiment_counts.values

plt.figure(figsize=(30, 3))

plt.subplot(131)

plt.bar(names, values)

plt.show()
#check each airline's numbers in each sentiment 

def plot_sub_sentiment(Airline):

    df=tweets[tweets['airline']==Airline]

    count=df['airline_sentiment'].value_counts()

    Index = [1,2,3]

    plt.bar(Index,count)

    plt.xticks(Index,['negative','neutral','positive'])

    plt.ylabel('Mood Count')

    plt.xlabel('Mood')

    plt.title('Count of Moods of '+Airline)

plt.figure(1,figsize=(12, 12))

plt.subplot(231)

plot_sub_sentiment('US Airways')

plt.subplot(232)

plot_sub_sentiment('United')

plt.subplot(233)

plot_sub_sentiment('American')

plt.subplot(234)

plot_sub_sentiment('Southwest')

plt.subplot(235)

plot_sub_sentiment('Delta')

plt.subplot(236)

plot_sub_sentiment('Virgin America')
import nltk

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('wordnet')
import re, nltk

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

wordnet_lemmatizer = WordNetLemmatizer()



def normalizer(tweet):

    text = re.sub(r"http:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})", "", tweet)

    only_letters = re.sub("[^a-zA-Z]", " ",text) 

   #tokens = nltk.word_tokenize(only_letters)[2:] #delete airline name

    tokens = nltk.word_tokenize(only_letters)[:] #include airline name

    lower_case = [l.lower() for l in tokens]

    #filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in lower_case]

    

    return ' '.join(lemmas)

   #return lemmas

def normalizer2(tweet):

    text = re.sub(r"http:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})", "", tweet)

    only_letters = re.sub("[^a-zA-Z]", " ",text) 

    tokens = nltk.word_tokenize(only_letters)[:] #include airline name

    lower_case = [l.lower() for l in tokens]

    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]

    return lemmas

    

def column(matrix, i):

    return [row[i] for row in matrix]
#pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells

tweets['normalized_tweet'] = tweets.text.apply(normalizer)

tweets[['text','normalized_tweet']].head()

tweets['normalized_tweet_tokens'] = tweets.text.apply(normalizer2)

from nltk import ngrams

def ngrams(input_list):

    onegrams = input_list

    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]

    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]

    return bigrams+trigrams

tweets['grams'] = tweets.normalized_tweet_tokens.apply(ngrams)

tweets[['grams']].head()
import collections

def count_words(input):

    cnt = collections.Counter()

    for row in input:

        for word in row:

            cnt[word] += 1

    return cnt
positivewords =tweets[(tweets.airline_sentiment == 'positive')][['grams']].apply(count_words)['grams'].most_common(50)

negativewords =tweets[(tweets.airline_sentiment == 'negative')][['grams']].apply(count_words)['grams'].most_common(50)

neutralwords  =tweets[(tweets.airline_sentiment == 'neutral')][['grams']].apply(count_words)['grams'].most_common(50)
from wordcloud import WordCloud,STOPWORDS

def column(matrix, i):

    return [row[i] for row in matrix]

positiveword=' '.join(column(positivewords, 0))

negativeword=' '.join(column(negativewords, 0))

neutralword=' '.join(column(neutralwords, 0))
wordcloud  = WordCloud(background_color='black').generate(positiveword)

wordcloud2 = WordCloud(background_color='black').generate(negativeword)

plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud)

plt.axis('off')

plt.figure(2,figsize=(12, 12))

plt.imshow(wordcloud2)

plt.axis('off')

plt.show()
def sentiment2target(sentiment):

    return {

        'negative': 0,

        'neutral': 1,

        'positive' : 2

    }[sentiment]

targets = tweets.airline_sentiment.apply(sentiment2target)
from sklearn.model_selection import train_test_split

import numpy as np

#data_train, data_test, y_train, y_test = train_test_split(tweets.normalized_tweet, targets, test_size=0.2, random_state=1)

data_train, data_test, y_train, y_test = train_test_split(tweets.normalized_tweet, tweets.airline_sentiment, test_size=0.2, random_state=1)
traindf = pd.DataFrame(np.array(data_train), columns=["tweet"])

traindf["label"] = np.array(y_train)

devdf = pd.DataFrame(np.array(data_test), columns=["tweet"])

devdf["label"] = np.array(y_test)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.linear_model import LinearRegression
union = FeatureUnion([ ("w_v2", TfidfVectorizer(analyzer = 'char', ngram_range=(1,5)  )),

                      ("w_v3", TfidfVectorizer(analyzer = 'char_wb', ngram_range=(1,5)  )),

                      ("w_v", CountVectorizer( ngram_range=(1,3),stop_words=None )),],

transformer_weights={

            'w_v': 1, 

        'w_v2': 1,   

           'w_v3': 1, 

           },)

X_train = union.fit_transform(data_train)

X_test = union.transform(data_test)

X_train.shape
Accuracy=[]

F1=[]
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import f1_score



model = make_pipeline( (MultinomialNB(alpha=0.03))).fit(X_train, y_train)

predicted = model.predict(X_test) 

score = metrics.accuracy_score(y_test, predicted)*100

Accuracy.append(score)

print("MultinomialNB accuracy:   %0.3f" % score) 



f1=f1_score(y_test, predicted, average='macro')*100

F1.append(f1)



print("MultinomialNB F1-score:   %0.3f" % f1) 

print()

print()

print(classification_report(y_test, predicted))
import seaborn as sn

confusion_matrix = pd.crosstab(y_test, predicted, rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True, fmt='.0f')

plt.show()
#@title Make a predication:

sent = 'i did not like my last flight' #@param {type:"string"}

sent = union.transform([sent])

print(model.predict(sent)[0])
y_test = np.array(y_test)

predicted = np.array(predicted)

data_test = np.array(data_test)

print()
#what did the model misclassifiy

predictedprob = model.predict_proba(X_test) 

for i in range(len(predictedprob)):

    for j in range(len(predictedprob[i])):

        predictedprob[i][j] = '{0:.2f}'.format(predictedprob[i][j])

missclassified=[]

true=[]

prd=[]

prdprob=[]

other_prdprob=[]

for i in range(len(y_test)):

    if y_test[i] != predicted[i]:

        missclassified.append(data_test[i])

        true.append(str(y_test[i]))

        prd.append(str(predicted[i]))

        indx=np.argmax(predictedprob[i])

        prdprob.append(str(predictedprob[i][indx]))

        other_prdprob.append(str(predictedprob[i]))

miss = pd.DataFrame(missclassified,columns=["sentence"])

miss["True"] = true

miss["Predicated"] = prd

miss["Confidance"] = prdprob

miss["All_prob"] = other_prdprob
pd.set_option('display.max_colwidth', -1)

miss.head(100)
data_train, data_test, y_train, y_test = train_test_split(tweets.normalized_tweet, targets, test_size=0.2, random_state=1)
#convert the labels to one hot encoder 

a = y_train

b = np.zeros((a.size, a.max()+1))

b[np.arange(a.size),a] = 1

b
#one-vs-all explicitly

model11 = make_pipeline( (LinearRegression())).fit(X_train, column(b, 0))

predicted11 = model11.predict(X_test) 



model22 = make_pipeline( (LinearRegression())).fit(X_train, column(b, 1))

predicted22 = model22.predict(X_test) 



model33 = make_pipeline( (LinearRegression())).fit(X_train, column(b, 2))

predicted33 = model33.predict(X_test) 



LinearRegression_preds = np.stack((predicted11, predicted22,predicted33), axis=-1)

preds_LinearRegression =[]

for i in LinearRegression_preds:

    preds_LinearRegression.append(np.argmax(i))



score = metrics.accuracy_score(y_test, preds_LinearRegression)*100

Accuracy.append(score)

print("LinearRegression accuracy:   %0.3f" % score) 

f1=f1_score(y_test, preds_LinearRegression, average='macro')*100

F1.append(f1)



print("LinearRegression F1-score:   %0.3f" % f1) 

print()

print()

print(classification_report(y_test, preds_LinearRegression))
import seaborn as sn

confusion_matrix = pd.crosstab(np.array(y_test), np.array(preds_LinearRegression), rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True, fmt='.0f')

plt.show()
Index = [1,2]

sns.set()

plt.subplots(figsize=(20,4),tight_layout=True)

plt.subplot(1,2,1)

plt.bar(Index,Accuracy)



plt.xticks(Index, ["MultinomialNB","LinearRegression"],rotation=0)

plt.ylabel('Accuracy')

plt.xlabel('Model')

plt.title('Accuracies of Models')



for i in range(len(Accuracy)):

    plt.annotate(int(Accuracy[i]), xy=(Index[i],Accuracy[i]))

plt.show()
%%capture cup

!pip install simpletransformers==0.40.2
%%capture cup

!git clone https://github.com/NVIDIA/apex

%cd apex

!pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
data_train, data_test, y_train, y_test = train_test_split(tweets.normalized_tweet, targets, test_size=0.2, random_state=1)

traindf = pd.DataFrame(np.array(data_train), columns=["tweet"])

traindf["label"] = np.array(y_train)

devdf = pd.DataFrame(np.array(data_test), columns=["tweet"])

devdf["label"] = np.array(y_test)

from simpletransformers.classification import ClassificationModel

import pandas as pd

import sklearn

import logging





logging.basicConfig(level=logging.INFO)

transformers_logger = logging.getLogger("transformers")

transformers_logger.setLevel(logging.WARNING)



# Create a ClassificationModel

bertmodel = ClassificationModel('bert', 'bert-base-cased', num_labels=3, use_cuda=True, cuda_device=0, 

                            args={

    'reprocess_input_data': True,

    "learning_rate": 4e-5,

    'overwrite_output_dir': True,

    'num_train_epochs': 3,    "save_eval_checkpoints": False,

    "save_steps": -1,}

    )

print(traindf.head())



# Train the bertmodel

bertmodel.train_model(traindf, eval_df=devdf)

predictions, raw_outputs = bertmodel.predict(np.array(data_test))
from sklearn.metrics import f1_score,classification_report

f1_score(y_true=np.array(y_test), y_pred=predictions, average='micro')

F1.append(f1)

score = metrics.accuracy_score(np.array(y_test), predictions)*100

Accuracy.append(score)
print(classification_report(np.array(y_test),predictions, labels=None))
import seaborn as sn

confusion_matrix = pd.crosstab(np.array(y_test),predictions, rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True, fmt='.0f')

plt.show()
b=[]

for i in y_test:

    if i==2:

        b.append("positive")

    elif i == 0:

        b.append("negative")

    else:

        b.append("neutral")
#what did the model misclassifiy

data_test = np.array(data_test)

predictedprob = raw_outputs

for i in range(len(predictedprob)):

    for j in range(len(predictedprob[i])):

        predictedprob[i][j] = '{0:.2f}'.format(predictedprob[i][j])

missclassified=[]

true=[]

prd=[]

prdprob=[]

other_prdprob=[]

for i in range(len(b)):

    if b[i] != predicted[i]:

        missclassified.append(data_test[i])

        true.append(str(b[i]))

        prd.append(str(predicted[i]))

        indx=np.argmax(predictedprob[i])

        prdprob.append(str(predictedprob[i][indx]))

        other_prdprob.append(str(predictedprob[i]))

miss = pd.DataFrame(missclassified,columns=["sentence"])

miss["True"] = true

miss["Predicated"] = prd

miss["Confidance"] = prdprob

miss["All_prob"] = other_prdprob
miss.head(100)
predictions_train, raw_outputs_train = bertmodel.predict(np.array(data_train))
model1 = make_pipeline( (LinearRegression())).fit(X_train, column(raw_outputs_train, 0))

predicted1 = model1.predict(X_test) 



model2 = make_pipeline( (LinearRegression())).fit(X_train, column(raw_outputs_train, 1))

predicted2 = model2.predict(X_test) 



model3 = make_pipeline( (LinearRegression())).fit(X_train, column(raw_outputs_train, 2))

predicted3 = model3.predict(X_test) 
knowledge_distil_preds = np.stack((predicted1, predicted2,predicted3), axis=-1)

preds_knowlsge_ditil =[]

for i in knowledge_distil_preds:

    preds_knowlsge_ditil.append(np.argmax(i))
score = metrics.accuracy_score(y_test, preds_knowlsge_ditil)*100

Accuracy.append(score)

print("Knowledge distilation accuracy:   %0.3f" % score) 

f1=f1_score(y_test, preds_knowlsge_ditil, average='macro')*100

print("Knowledge distilation F1-score:   %0.3f" % f1) 

F1.append(f1)



import seaborn as sn

confusion_matrix = pd.crosstab(np.array(y_test),np.array(preds_knowlsge_ditil), rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True, fmt='.0f')

plt.show()
Index = [1,2,3,4]

sns.set()

plt.subplots(figsize=(20,4),tight_layout=True)

plt.subplot(1,2,1)

plt.bar(Index,Accuracy)

plt.xticks(Index, ["MultinomialNB","LinearRegression","BERT","DistilBERT"],rotation=0)

plt.ylabel('Accuracy')

plt.xlabel('Model')

plt.title('Accuracies of Models')
time_taken=[]
#count time to predict 1000 sentences:

import time

start_time = time.time()



sent = union.transform(data_test[:1000])

model.predict(sent)



print("MultinomialNB --- %s seconds ---" % (time.time() - start_time))

time_taken.append((time.time() - start_time))
#count time to predict 1000 sentences:

import time

start_time = time.time()



sent = union.transform(data_test[:1000])

predicted11 = model11.predict(sent) 

predicted22 = model22.predict(sent) 

predicted33 = model33.predict(sent) 

LinearRegression_preds = np.stack((predicted11, predicted22,predicted33), axis=-1)

preds_LinearRegression =[]

for i in LinearRegression_preds:

    preds_LinearRegression.append(np.argmax(i))



print("LinearRegression --- %s seconds ---" % (time.time() - start_time))

time_taken.append((time.time() - start_time))
#count time to predict 1000 sentences:

import time

start_time = time.time()



bertmodel.predict(np.array(data_test[:1000]))



print("BERT --- %s seconds ---" % (time.time() - start_time))

time_taken.append((time.time() - start_time))


#count time to predict 1000 sentences:

import time

start_time = time.time()

sent = union.transform(data_test[:1000])



predicted1 = model1.predict(sent) 

predicted2 = model2.predict(sent) 

predicted3 = model3.predict(sent) 

LinearRegression_preds = np.stack((predicted1, predicted2,predicted3), axis=-1)

preds_LinearRegression =[]

for i in LinearRegression_preds:

    preds_LinearRegression.append(np.argmax(i))



print("BERT_distil --- %s seconds ---" % (time.time() - start_time))

time_taken.append((time.time() - start_time))
Index = [1,2,3,4]

sns.set()

plt.subplots(figsize=(20,4),tight_layout=True)

plt.subplot(1,2,1)

plt.bar(Index,time_taken)

plt.xticks(Index, ["MultinomialNB","LinearRegression","BERT","DistilBERT"],rotation=0)

plt.ylabel('Time')

plt.xlabel('Model')

plt.title('Time taken to Predict 1000 sentence')
Index = [1,2,3,4]

sns.set()

plt.subplots(figsize=(20,4),tight_layout=True)

plt.subplot(1,2,1)

plt.bar(Index,Accuracy)

plt.xticks(Index, ["MultinomialNB","LinearRegression","BERT","DistilBERT"],rotation=0)

plt.ylabel('Accuracy')

plt.xlabel('Model')

plt.title('Accuracies of Models')



Index = [1,2,3,4]

sns.set()

plt.subplots(figsize=(20,4),tight_layout=True)

plt.subplot(1,2,1)

plt.bar(Index,F1)

plt.xticks(Index, ["MultinomialNB","LinearRegression","BERT","DistilBERT"],rotation=0)

plt.ylabel('F1-Score')

plt.xlabel('Model')

plt.title('F1-Score of Models')