import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import string

oneSetOfStopWords = set(stopwords.words('english'))

dataset = pd.read_csv('../input/demonetization-tweets.csv', encoding = "ISO-8859-1")
dataset.head()
print ("Following are the columns of the dataset - ")

dataset.columns
import re, sys
def clean_tweet(tweet):
    tweet = re.sub('http\S+\s*', '', tweet)  # remove URLs
    tweet = re.sub('RT|cc', '', tweet)  # remove RT and cc
    tweet = re.sub('#\S+', '', tweet)  # remove hashtags
    tweet = re.sub('@\S+', '', tweet)  # remove mentions
    tweet = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', tweet)  # remove punctuations
    tweet = re.sub('\s+', ' ', tweet)  # remove extra whitespace
    tokens = tweet.split(" ")
    tweet = [ x for x in tokens if len(x) < 25 ]
    tweet = " ".join(tweet)
    return tweet
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
dataset['cleaned_text'] = dataset.text.apply(lambda x: clean_tweet(x))
dataset['sentiment_compound_polarity'] = dataset.cleaned_text.apply(lambda x: analyser.polarity_scores(x)['compound'])
dataset['sentiment_negative'] = dataset.cleaned_text.apply(lambda x: analyser.polarity_scores(x)['neg'])
dataset['sentiment_positive'] = dataset.cleaned_text.apply(lambda x: analyser.polarity_scores(x)['pos'])
dataset['sentiment_type']=''
dataset.loc[dataset.sentiment_compound_polarity>0,'sentiment_type']= 0
dataset.loc[dataset.sentiment_compound_polarity<0,'sentiment_type']= 1
dataset.columns
positive = dataset[dataset['sentiment_type'] == 0]
negative = dataset[dataset['sentiment_type'] == 1]
frames = [positive, negative]
orgDataSet = pd.concat(frames)
orgDataSet = orgDataSet.reset_index()
orgDataSet.head()
print ("Total Number of tweets")
print (orgDataSet.shape[0])
print ("____________________________________")

print ("Number of POSITIVE tweets - ", str(len(orgDataSet[orgDataSet['sentiment_type'] == 0])))
print ("Percentage of POSITIVE tweets")
print (len(orgDataSet[orgDataSet['sentiment_type'] == 0])/ len(orgDataSet) * 100)
print ("____________________________________")

print ("Number of NEGATIVE tweets - ", str(len(orgDataSet[orgDataSet['sentiment_type'] == 1])))
print ("Percentage of NEGATIVE tweets")
print (len(orgDataSet[orgDataSet['sentiment_type'] == 1 ])/ len(orgDataSet) * 100)
print ("____________________________________")
import seaborn as sns
sns.set(style="darkgrid")
sns.countplot(x=orgDataSet['sentiment_type'])
from matplotlib.gridspec import GridSpec
targetCounts = orgDataSet.sentiment_type.value_counts()
targetLabels  =['POSITIVE', 'NEGATIVE']
# Make square figures and axes
plt.figure(1, figsize=(15,15))
the_grid = GridSpec(2, 2)


cmap = plt.get_cmap('Spectral')
colors = [cmap(i) for i in np.linspace(0, 1, 3)]
plt.subplot(the_grid[0, 1], aspect=1, title='TWEET DISTRIBUTION')

source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()
print ("Following are the most common occuring words - ")
totalWords = []
tweets = orgDataSet['cleaned_text'].values
cleanedSentences = ""
for x in range(1,len(orgDataSet['cleaned_text'])):
    cleanedSentence = orgDataSet['cleaned_text'][x]    
    cleanedSentences += cleanedSentence
    requiredWords = nltk.word_tokenize(cleanedSentence)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation and (len(word) < 50):
            totalWords.append(word)
    
wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print(mostcommon)

import re
import operator

hashTagDict = {}
requiredSentences = orgDataSet['text'].values
for sent in requiredSentences:
    requiredHashTagList = re.findall(r'#(\w+)',sent)
    for h in range(len(requiredHashTagList)):
        reqHashTag = requiredHashTagList[h]
        if(reqHashTag in hashTagDict):
            hashTagDict[reqHashTag] += 1
        else:
            hashTagDict[reqHashTag] = 1
hashTagDict_sorted = dict(sorted(hashTagDict.items(),key=operator.itemgetter(1),reverse=True))
requiredKeys = list(hashTagDict_sorted.keys())[:10]
requiredValues = list(hashTagDict_sorted.values())[:10]
d = {'HashTags': requiredKeys , 'Counts': requiredValues}
hashTagData = pd.DataFrame(data=d)
plt.figure(1, figsize=(15,15))
sns.set(style="whitegrid")
sns.barplot(x="HashTags", y="Counts", data=hashTagData)
wc = WordCloud().generate(cleanedSentences)
plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X = orgDataSet['cleaned_text'].values
y = orgDataSet['sentiment_type'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.3)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=5000, stop_words='english')
tfidf_vectorizer.fit(X_train)

X_train_vectorized = tfidf_vectorizer.transform(X_train)
X_test_vectorized = tfidf_vectorizer.transform(X_test)
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

print ("Multinomial Naive Bayes")
mnb_model = MultinomialNB()
print (mnb_model)
y = y_train.astype('int')
t = y_test.astype('int')
mnb_model.fit(X_train_vectorized, y)
test_predictedValues = mnb_model.predict(X_test_vectorized)
train_predictedValues = mnb_model.predict(X_train_vectorized)

trainingAccuracyScore = mnb_model.score(X_train_vectorized, y) * 100
testingAccuracyScore = mnb_model.score(X_test_vectorized, t) * 100
print ("\nTraining Accuracy Score - %.2f " % (trainingAccuracyScore), "%")
print ("Testing Accuracy Score - %.2f " % (testingAccuracyScore), "%")
print ("\n Following is the Classification Report for Multinomial Naive Bayes - ")
print (classification_report(t, test_predictedValues))

# finding the true positive rates and false positive rates using the roc_curve function
test_fpr, test_tpr, test_thresholds = roc_curve(t, test_predictedValues)
test_reqAUCScore = "%.2f " % (auc(test_fpr, test_tpr) * 100)
print ("\nTesting AUC Score - " , test_reqAUCScore, "%")

train_fpr, train_tpr, train_thresholds = roc_curve(y, train_predictedValues)
train_reqAUCScore = "%.2f " % (auc(train_fpr, train_tpr) * 100)
print ("Training  AUC Score - " , train_reqAUCScore, "%")

plt.plot(test_fpr, test_tpr, color='green', label = "Testing ROC Curve")
plt.plot(train_fpr, train_tpr, color='red', label = "Training ROC Curve")
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Multinomial Naive Bayes model')
plt.legend(loc="lower right")
plt.show()

cm = confusion_matrix(t, test_predictedValues)

classNames = ['POSITIVE','NEGATIVE']
labels = [0,1]

plt.figure(figsize=(5,5))
ax= plt.subplot()
df_cm = pd.DataFrame(cm)
sns.heatmap(df_cm, annot=True, ax = ax, cmap="Pastel1", fmt='g', annot_kws={"size": 13}); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix for Multinomial Naive Bayes'); 
ax.xaxis.set_ticklabels(classNames, rotation=0, fontsize="13", va="center", color='black'); 
ax.yaxis.set_ticklabels(classNames, rotation=0, fontsize="13", va="center", color='black');
import sys,re
print ("\n Following is the Classification Report for Multinomial Naive Bayes - ")
cf_report = classification_report(t, test_predictedValues)

def classification_report_to_dataframe(cf_report):
    report_data = []
    lines = cf_report.split('\n')
    for line in lines[2:-3]:
        newline = re.sub(r' +',",",line)
        row_data = newline.split(",")
        if(len(row_data) == 6):
            row = {}
            row['precision'] = float(row_data[2])
            row['recall'] = float(row_data[3])
            row['f1_score'] = float(row_data[4])
            report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe

reqDataFrame = classification_report_to_dataframe(cf_report)

classNames = ['POSITIVE','NEGATIVE']
labels = [0,1]

plt.figure(figsize=(5,5))
ax= plt.subplot()
sns.heatmap(reqDataFrame, annot=True, ax = ax, cmap="Pastel1", cbar=False, linewidths=.5, annot_kws={"size": 13}); #annot=True to annotate cells
ax.yaxis.set_ticklabels(classNames, rotation=0, fontsize="13", va="center", color='black')
ax.xaxis.set_ticklabels(['precision','recall', 'f1-score'], rotation=0, fontsize="13", va="center", color='black')

