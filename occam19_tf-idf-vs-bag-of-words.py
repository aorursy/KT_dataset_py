import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn import metrics
from sklearn.model_selection import train_test_split   
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv',  sep='\t')
sampleSub = pd.read_csv('../input/sampleSubmission.csv')
print(train.shape, "\n", 
      test.shape
     )
print (train.isnull().values.any(), "\n",
      test.isnull().values.any()
      )
train.head()
test.head()
len(train.groupby('SentenceId').nunique())
len(test.groupby('SentenceId').nunique())
#Create df of full sentences
fullSent = train.loc[train.groupby('SentenceId')['PhraseId'].idxmin()]

#Change sentiment to increase readability
fullSent['sentiment_label'] = ''
Sentiment_Label = ['Negative', 'Somewhat Negative', 
                  'Neutral', 'Somewhat Positive', 'Positive']
for sent, label in enumerate(Sentiment_Label):
    fullSent.loc[train.Sentiment == sent, 'sentiment_label'] = label
    
fullSent.head()
#Add non-helpful stopwords to stopword list
Stopwords = list(ENGLISH_STOP_WORDS)
Stopwords.extend(['movie','movies','film','nt','rrb','lrb',
                      'make','work','like','story','time','little'])

#Create tfidf vectorizer object & fit to full sentence training data
tfidf_vectorizor = TfidfVectorizer(min_df=5, 
                             max_df=0.5,
                             analyzer='word',
                             strip_accents='unicode',
                             ngram_range=(1, 3),
                             sublinear_tf=True, 
                             smooth_idf=True,
                             use_idf=True,
                             stop_words=Stopwords)

tfidf_vectorizor.fit(list(fullSent['Phrase']))


#Create bag of word vectorizer for comparison in evaluation section
BoW_vectorizer = CountVectorizer(strip_accents='unicode',
                                 stop_words=Stopwords,
                                 ngram_range=(1,3),
                                 analyzer='word',
                                 min_df=5,
                                 max_df=0.5)

BoW_vectorizer.fit(list(fullSent['Phrase']))
#functions to create graphics below from tf-idf matrices
#adapted from : https://buhrmann.github.io/tfidf-analysis.html
def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=20):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=16):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs, num_class=9):
    fig = plt.figure(figsize=(12, 100), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(num_class, 1, i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=16)
        ax.set_ylabel("Word", labelpad=16, fontsize=16)
        ax.set_title(str(df.label) + ' Sentiment Class', fontsize=25)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        ax.invert_yaxis()
        yticks = ax.set_yticklabels(df.feature)
        
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()

class_Xtr = tfidf_vectorizor.transform(fullSent['Phrase'])
class_y = fullSent['sentiment_label']
class_features = tfidf_vectorizor.get_feature_names()
class_top_dfs = top_feats_by_class(class_Xtr, class_y, class_features)
plot_tfidf_classfeats_h(class_top_dfs, 7)
phrase = np.array(train['Phrase'])
sentiment = np.array(train['Sentiment'])
# build train and test datasets
phrase_train, phrase_test, sentiment_train, sentiment_test = train_test_split(phrase, 
                                                                              sentiment, 
                                                                              test_size=0.2, 
                                                                              random_state=4)

#TF-IDF
train_tfidfmatrix = tfidf_vectorizor.fit_transform(phrase_train)
test_tfidfmatrix = tfidf_vectorizor.transform(phrase_test)

#Vectorizer (Bag of Words Model)
train_simplevector = BoW_vectorizer.transform(phrase_train)
test_simplevector = BoW_vectorizer.transform(phrase_test)
def train_model_predict (classifier, train_features, train_labels,
                      test_features):
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    return predictions
model1 = MultinomialNB() 
NBPredictions = train_model_predict(model1, train_tfidfmatrix, sentiment_train,
                             test_tfidfmatrix)
NBPredictions2 = train_model_predict(model1, train_simplevector, sentiment_train,
                             test_simplevector)
model2 = LogisticRegression(solver = 'liblinear', multi_class = 'ovr')
LogisticRegressionPredictions = train_model_predict(model2, train_tfidfmatrix, sentiment_train,
                             test_tfidfmatrix)
LogisticRegressionPredictions2 = train_model_predict(model2, train_simplevector, sentiment_train,
                             test_simplevector)
def get_metrics(true_labels, predicted_labels, feature):  
    print(feature)
    print('Accuracy:', np.round(metrics.accuracy_score(true_labels, 
                                               predicted_labels), 4))
    print('Precision:', np.round(metrics.precision_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'), 4))
    print('Recall:', np.round(metrics.recall_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'), 4))
    print('F1 Score:', np.round(metrics.f1_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'), 4))
    print('\n')
    
get_metrics(NBPredictions, sentiment_test, 'Naive Bayes & TF-IDF Scores: ')
get_metrics(NBPredictions2, sentiment_test, 'Naive Bayes & Bag of Words Scores: ')
get_metrics(LogisticRegressionPredictions, sentiment_test, 'Logistic Regression & TF-IDF Scores: ')
get_metrics(LogisticRegressionPredictions2, sentiment_test, 'Logistic Regression & Bag of Words Scores: ')
train_tfidf = tfidf_vectorizor.fit_transform(train['Phrase'])
model2.fit(train_tfidf, train['Sentiment'])
test_tfidf = tfidf_vectorizor.transform(test['Phrase'])
predictions = model2.predict(test_tfidf)

test['Sentiment'] = predictions
submission = test[['PhraseId','Sentiment']]
submission.to_csv('submission.csv',index=False)