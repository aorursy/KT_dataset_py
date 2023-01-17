import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

pal = sns.color_palette()

print('# File sizes')
for f in os.listdir('../input'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
# Reading training data
df_train = pd.read_csv('../input/train.csv')
df_train.head()
# Reading test data
df_test = pd.read_csv('../input/test.csv')
df_test.head() 
# checking null values in training & testing data  
df_test[df_test.isnull().any(axis=1)]  
df_train[df_train.isnull().any(axis=1)] 
# Adding the string empty to null values  
df_train = df_train.fillna('empty') 
df_test = df_test.fillna('empty')
print('Total number of question pairs for training: {}'.format(len(df_train)))
print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))
qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
print('Total number of questions in the training data: {}'.format(len(
    np.unique(qids))))
print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

plt.figure(figsize=(12, 5))
plt.hist(qids.value_counts(), bins=50)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
print() 
from sklearn.metrics import log_loss

p = df_train['is_duplicate'].mean() # Our predicted probability
print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))

df_test = pd.read_csv('../input/test.csv')
sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': p}) 
sub.to_csv('naive__mean_submission.csv', index=False)
sub.head()
print('Total number of question pairs for testing: {}'.format(len(df_test)))
# Histogram of character count in train & test questions 

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)
plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')
plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')
plt.title('Normalised histogram of character count in questions', fontsize=15)
plt.legend()
plt.xlabel('Number of characters', fontsize=15)
plt.ylabel('Probability', fontsize=15)

print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
# Histogram of  word count in train & test questions 
dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True, label='train')
plt.hist(dist_test, bins=50, range=[0, 50], color=pal[1], normed=True, alpha=0.5, label='test')
plt.title('Normalised histogram of word count in questions', fontsize=15)
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Probability', fontsize=15)

print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
from wordcloud import WordCloud
cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
from nltk.corpus import stopwords 
from string import punctuation 
import re

stops = set(stopwords.words("english")) 
punctuation = set(punctuation) 
stops_final = stops.union(punctuation)
import re 
import nltk 
def clean(text,remove_stop_words=True,stem_words=False):
    text = str(text)
    text = text.lower() 
    text = re.sub(r"\b([A-Za-z]+)'re\b", '\\1 are', text)
    text = re.sub(r"\b([A-Za-z]+)'s\b", '\\1 is', text) 
    text = re.sub(r"\b([A-Za-z]+)'m\b", '\\1 am', text) 
    text = re.sub(r"\b([A-Za-z]+)'ve\b", '\\1 have', text) 
    text = re.sub(r"\b([A-Za-z]+)'ll\b", '\\1 will', text) 
    text = re.sub(r"\b([A-Za-z]+)'t\b", '\\1 not', text) 
    text = re.sub(r"[\'?,\.]", ' ', text) 
    text = re.sub(r"\s+", ' ', text)  
    text = re.sub(r"quikly", "quickly", text,flags=re.IGNORECASE)
    text = re.sub(r"\busa\b", "America", text,flags=re.IGNORECASE)
    text = re.sub(r"\buk\b", "England", text,flags=re.IGNORECASE) 
    text = re.sub(r"imrovement", "improvement", text,flags=re.IGNORECASE)
    text = re.sub(r"intially", "initially", text,flags=re.IGNORECASE)
    text = re.sub(r"\bdms\b", "direct messages ", text,flags=re.IGNORECASE)  
    text = re.sub(r"demonitization", "demonetization", text,flags=re.IGNORECASE) 
    text = re.sub(r"actived", "active", text,flags=re.IGNORECASE)
    text = re.sub(r"kms", " kilometers ", text,flags=re.IGNORECASE)
    text = re.sub(r"\bcs\b", "computer science", text,flags=re.IGNORECASE) 
    text = re.sub(r"\bupvotes\b", "bup votes", text,flags=re.IGNORECASE)
    text = re.sub(r"\biPhone\b", "phone", text,flags=re.IGNORECASE)
    text = re.sub(r"\0rs ", " rs ", text,flags=re.IGNORECASE) 
    text = re.sub(r"calender", "calendar", text,flags=re.IGNORECASE)
    text = re.sub(r"ios", "operating system", text,flags=re.IGNORECASE)
    text = re.sub(r"gps", "GPS", text,flags=re.IGNORECASE)
    text = re.sub(r"gst", "GST", text,flags=re.IGNORECASE)
    text = re.sub(r"programing", "programming", text,flags=re.IGNORECASE)
    text = re.sub(r"bestfriend", "best friend", text,flags=re.IGNORECASE)
    text = re.sub(r"dna", "DNA", text,flags=re.IGNORECASE)
    text = re.sub(r"III", "3", text,flags=re.IGNORECASE) 
    text = re.sub(r"the US", "America", text,flags=re.IGNORECASE)
    text = re.sub(r"Astrology", "astrology", text,flags=re.IGNORECASE)
    text = re.sub(r"Method", "method", text,flags=re.IGNORECASE)
    text = re.sub(r"Find", "find", text,flags=re.IGNORECASE) 
    text = re.sub(r"banglore", "Banglore", text,flags=re.IGNORECASE) 
    
    #Remove punctuation and stopwords: 
    text = ' '.join([c for c in text.split(' ') if c not in stops_final]) 
   
    if stem_words:
        text = text.split(' ')
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words. 
    return text 
        
# Function to call clean function 
def process_questions(questions,clean_questions):
    for question in questions:
        clean_questions.append(clean(question))
        if len(clean_questions) % 100000 == 0:
           progress = len(clean_questions)/len(questions) *100
           print("Progress is {}% complete".format(round(progress,1)))
# Cleaning Questions on Training Data

df_train_qs1_clean = []
df_train_qs2_clean = []
process_questions(df_train.question1,df_train_qs1_clean) 
process_questions(df_train.question2,df_train_qs2_clean)   

df_train.question1 = df_train_qs1_clean
df_train.question2 = df_train_qs2_clean

df_train.head()
df_test_qs1_clean = []
df_test_qs2_clean = []

# cleaning question1 column in test data
process_questions(df_test.question1,df_test_qs1_clean)  

# cleaning question2 column2 in test data
process_questions(df_test.question2,df_test_qs2_clean)  

# replacing original data 
df_test.question1 = df_test_qs1_clean
df_test.question2 = df_test_qs2_clean 

# new test data looks like
df_test.head()
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

plt.figure(figsize=(15, 5))
train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
plt.hist(train_word_match[df_train['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
plt.hist(train_word_match[df_train['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}
print('Most common words and weights: \n')
print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
print('\nLeast common words and weights: ')
(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])
def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R
plt.figure(figsize=(15, 5))
tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)
plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')
plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over tfidf_word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)
from sklearn.metrics import roc_auc_score
print('Original AUC:', roc_auc_score(df_train['is_duplicate'], train_word_match))
print('   TFIDF AUC:', roc_auc_score(df_train['is_duplicate'], tfidf_train_word_match.fillna(0)))
pos_boostrap_sample = df_train[df_train["is_duplicate"] == 0].sample(n = 500000, replace = True)
df_train_rebalanced = pd.concat((pos_boostrap_sample, df_train)) 
# Recalculating word_share_features  on rebalanced dataset 
tfidf_train_word_match_rebalanced = df_train_rebalanced.apply(tfidf_word_match_share, axis=1, raw=True) 
train_word_match_rebalanced = df_train_rebalanced.apply(word_match_share, axis=1, raw=True) 
#First Lets create training & testing data :
x_train = pd.DataFrame() 
x_test = pd.DataFrame() 
x_train['word_match'] =train_word_match_rebalanced
x_train['tfidf_word_match'] = tfidf_train_word_match_rebalanced 
x_test['word_match'] = df_test.apply(word_match_share,axis=1,raw=True) 
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share,axis=1,raw=True) 

y_train = df_train_rebalanced['is_duplicate'].values
# Finally, we split some of the data off for validation
from sklearn.cross_validation import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242) 
import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb.csv', index=False)
# create dictionary and extract BOW features from questions 
import time 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  

featureExtractionStartTime = time.time() 

maxNumfeatures = 300

#bag of letter sequences (chars) 
BagOfWordsExtractor = CountVectorizer(max_df=0.999 , min_df=50,max_features= maxNumfeatures,analyzer='char',ngram_range=(1,2),binary=True,lowercase=True) 

# Concating training question (1 & 2)
train_qs = pd.Series(df_train_rebalanced['question1'].tolist() + df_train_rebalanced['question2'].tolist()).astype(str)

BagOfWordsExtractor.fit(train_qs) 

train_qs1_BOW = BagOfWordsExtractor.transform(pd.Series(df_train_rebalanced['question1'].tolist())) 
train_qs2_BOW = BagOfWordsExtractor.transform(pd.Series(df_train_rebalanced['question2'].tolist())) 

featureExtractorDurationInMinutes = (time.time() - featureExtractionStartTime)/60.0 

print('feature extraction took {:.2f} minutes'.format(featureExtractorDurationInMinutes))
# Lets look at some of the feature generated by countVectorizer
BagOfWordsExtractor.get_feature_names()[:50]
from sklearn import model_selection
from sklearn import linear_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score



crossValidayionStartTime = time.time() 

numCVSplits = 8 
numSplitsToBreakAfter = 2 

X = -(train_qs1_BOW != train_qs2_BOW).astype(int)  


lables = np.array(pd.Series(df_train_rebalanced['is_duplicate']).astype(int)) 
y = lables 

logisticRegressor = linear_model.LogisticRegression(C= 0.1 ,solver = 'sag') 

logRegAccuracy = [] 
logRegLogLoss = [] 
logRegAUC = [] 

print('---------------------------------------------------------') 

stratifiedCV = model_selection.StratifiedKFold(n_splits = numCVSplits , random_state =2) 
for k , (trainInds,validInds) in enumerate(stratifiedCV.split(X,y)):
    foldTrainingStartTime = time.time() 
    
    X_train_cv = X[trainInds,:] 
    X_valid_cv = X[validInds,:] 
    
    y_train_cv = y[trainInds] 
    y_valid_cv = y[validInds] 
    
    logisticRegressor.fit(X_train_cv,y_train_cv) 
    
    y_train_hat = logisticRegressor.predict_proba(X_train_cv)[:,1]
    y_valid_hat = logisticRegressor.predict_proba(X_valid_cv)[:,1] 
    
    logRegAccuracy.append(accuracy_score(y_valid_cv,(np.array(y_valid_hat > 0.5).astype(int)))) 
    logRegLogLoss.append(log_loss(y_valid_cv,y_valid_hat)) 
    logRegAUC.append(roc_auc_score(y_valid_cv,y_valid_hat)) 
    
    foldTrainingDurationInMinutes = (time.time() - foldTrainingStartTime)/60.0  
    
    print(' fold {:d} took {:.2f} minutes : accuracy = {:.3f} ,log loss = {:.4f} , AUC = {:.3f}'.format(k+1,foldTrainingDurationInMinutes,logRegAccuracy[-1],logRegLogLoss[-1],logRegAUC[-1])) 
    
    if(k+1)>= numSplitsToBreakAfter:
        break
        
crossValidationDurationInMinutes = (time.time() - crossValidayionStartTime)/60.0 

print('-------------------------------------------------------') 

print('cross validation took {:2f} minutes'.format(crossValidationDurationInMinutes)) 
print('mean CV: accuracy = {:.3f},logloss = {:.4f},AUC = {:.3f}'.format(np.array(logRegAccuracy).mean(),np.array(logRegLogLoss).mean(),np.array(logRegAUC).mean())) 
print('------------------------------------------------------')
    
#%% show prediction distribution and "feature importance"
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10,10)

plt.figure(); 
sns.kdeplot(y_valid_hat[y_valid_cv==0], shade=True, color="b", bw=0.01)
sns.kdeplot(y_valid_hat[y_valid_cv==1], shade=True, color="g", bw=0.01)
plt.legend(['non duplicate','duplicate'],fontsize=24)
plt.title('Validation Accuracy = %.3f, Log Loss = %.4f, AUC = %.3f' %(logRegAccuracy[-1],
                                                                      logRegLogLoss[-1],
                                                                      logRegAUC[-1]))
plt.xlabel('Prediction'); plt.ylabel('Probability Density'); plt.xlim(-0.01,1.01)


numFeaturesToShow = 30

sortedCoeffients = np.sort(logisticRegressor.coef_)[0]
featureNames = BagOfWordsExtractor.get_feature_names()
sortedFeatureNames = [featureNames[x] for x in list(np.argsort(logisticRegressor.coef_)[0])]

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10,12)

plt.figure()
plt.suptitle('Feature Importance',fontsize=24)
ax = plt.subplot(1,2,1); plt.title('top non duplicate predictors'); 
plt.xlabel('minus logistic regression coefficient')
ax.barh(range(numFeaturesToShow), -sortedCoeffients[:numFeaturesToShow][::-1], align='center'); 
plt.ylim(-1,numFeaturesToShow); ax.set_yticks(range(numFeaturesToShow)); 
ax.set_yticklabels(sortedFeatureNames[:numFeaturesToShow][::-1],fontsize=20)

ax = plt.subplot(1,2,2); plt.title('top duplicate predictors'); 
plt.xlabel('logistic regression coefficient')
ax.barh(range(numFeaturesToShow), sortedCoeffients[-numFeaturesToShow:], align='center'); 
plt.ylim(-1,numFeaturesToShow); ax.set_yticks(range(numFeaturesToShow)); 
ax.set_yticklabels(sortedFeatureNames[-numFeaturesToShow:],fontsize=20)
#%% train on full training data

trainingStartTime = time.time()

logisticRegressor = linear_model.LogisticRegression(C=0.1, solver='sag', 
                                                    class_weight={1: 0.46, 0: 1.32})
logisticRegressor.fit(X, y)

trainingDurationInMinutes = (time.time()-trainingStartTime)/60.0
print('full training took %.2f minutes' % (trainingDurationInMinutes))
0#%% load test data, extract features and make predictions

testPredictionStartTime = time.time()


testQuestion1_BOW_rep = BagOfWordsExtractor.transform(df_test.loc[:,'question1'])
testQuestion2_BOW_rep = BagOfWordsExtractor.transform(df_test.loc[:,'question2'])

X_test = -(testQuestion1_BOW_rep != testQuestion2_BOW_rep).astype(int)


#fix to avoid memory errors
seperators= [750000,1500000]
testPredictions1 = logisticRegressor.predict_proba(X_test[:seperators[0],:])[:,1]
testPredictions2 = logisticRegressor.predict_proba(X_test[seperators[0]:seperators[1],:])[:,1]
testPredictions3 = logisticRegressor.predict_proba(X_test[seperators[1]:,:])[:,1]
testPredictions = np.hstack((testPredictions1,testPredictions2,testPredictions3))

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9,9)  

plt.figure(); 
plt.subplot(2,1,1); sns.kdeplot(y_valid_hat, shade=True, color="b", bw=0.01); 
plt.ylabel('Probability Density'); plt.xlim(-0.01,1.01)
plt.title('mean valid prediction = ' + str(np.mean(y_valid_hat)))
plt.subplot(2,1,2); sns.kdeplot(testPredictions, shade=True, color="b", bw=0.01);
plt.xlabel('Prediction'); plt.ylabel('Probability Density'); plt.xlim(-0.01,1.01)
plt.title('mean test prediction = ' + str(np.mean(testPredictions)))
#%% create a submission
submissionName = 'bag_of_words'
submission = pd.DataFrame()
submission['test_id'] = df_test['test_id']
submission['is_duplicate'] = testPredictions
submission.to_csv('bag_of_words' + '.csv', index=False)
