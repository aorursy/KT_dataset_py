import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

#Load libraries

import re    # for regular expressions 

import nltk  # for text manipulation 

import string 

import warnings 

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 



pd.set_option("display.max_colwidth", 200) 

warnings.filterwarnings("ignore", category=DeprecationWarning) 



%matplotlib inline



#read train and test datasets.

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")



print("Done reading")
train.head()
length_train = train['text'].str.len() 

length_test = test['text'].str.len() 

plt.hist(length_train, bins=20, label="train_tweets") 

plt.hist(length_test, bins=20, label="test_tweets") 

plt.legend() 

plt.show()



#Function to remove unwanted pattern

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

    return input_txt    



#combining train and test data for preprocessing

combi = train.append(test, ignore_index=True) 

combi.shape
#Data Cleaning

#Removing Twitter Handles (@user)

combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['text'], "@[\w]*") 

# Removing Punctuations, Numbers, and Special Characters

combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ") 

#Removing short words

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#Tokenizing

tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing 

#tokenized_tweet.head()



#Normalize the tokenized tweets

from nltk.stem.porter import * 

stemmer = PorterStemmer() 

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

#stitch these tokens back together

for i in range(len(tokenized_tweet)):

    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])    

combi['tidy_tweet'] = tokenized_tweet
combi['tidy_tweet'].head(10)
#Wordcloud

all_words = ' '.join([text for text in combi['tidy_tweet']]) 

from wordcloud import WordCloud 

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 

plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
#Fake text wordcloud

negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['target'] == 1]]) 

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words) 

plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
#Hashtag analysis

# function to collect hashtags 

def hashtag_extract(x):    

    hashtags = []    # Loop over the words in the tweet    

    for i in x: 

        ht = re.findall(r"#(\w+)", i)  

        hashtags.append(ht)    

    return hashtags



# extracting hashtags from non racist/sexist tweets 

HT_regular = hashtag_extract(combi['tidy_tweet'][combi['target'] == 0]) 

# extracting hashtags from racist/sexist tweets 

HT_negative = hashtag_extract(combi['tidy_tweet'][combi['target'] == 1]) 

# unnesting list 

HT_regular = sum(HT_regular,[]) 

HT_negative = sum(HT_negative,[])
#Fake text hashtag

b = nltk.FreqDist(HT_negative) 

e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())}) 

# selecting top 20 most frequent hashtags 

e = e.nlargest(columns="Count", n = 20)   

plt.figure(figsize=(16,5)) 

ax = sns.barplot(data=e, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count') 

plt.show()
#Word2Vec

#from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer import gensim

import gensim

tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing 

model_w2v = gensim.models.Word2Vec(

            tokenized_tweet,

            size=200, # desired no. of features/independent variables

            window=5, # context window size

            min_count=2,

            sg = 1, # 1 for skip-gram model

            hs = 0,

            negative = 10, # for negative sampling

            workers= 2, # no.of cores

            seed = 34) 



model_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)
model_w2v.wv.most_similar(positive="death")
model_w2v.wv.most_similar(positive="flood")
def word_vector(tokens, size):

    vec = np.zeros(size).reshape((1, size))

    count = 0.

    for word in tokens:

        try:

            vec += model_w2v[word].reshape((1, size))

            count += 1.

        except KeyError: # handling the case where the token is not in vocabulary 

            continue

    if count != 0:

            vec /= count

    return vec



#Preparing word2vec feature set…



wordvec_arrays = np.zeros((len(tokenized_tweet), 200)) 

for i in range(len(tokenized_tweet)):

    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)

    wordvec_df = pd.DataFrame(wordvec_arrays) 

wordvec_df.shape    
#Model 1 Logistic Regression

from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import train_test_split 

from sklearn.metrics import f1_score



# Extracting train and test

ytrain, yvalid = train_test_split(train['target'], random_state=42, test_size=0.3)



# splitting data into training and validation set 

train_w2v = wordvec_df.iloc[:7613,:] 

test_w2v = wordvec_df.iloc[7613:,:] 

xtrain_w2v = train_w2v.iloc[ytrain.index,:] 

xvalid_w2v = train_w2v.iloc[yvalid.index,:]



lreg = LogisticRegression() 

lreg.fit(xtrain_w2v, ytrain) 

prediction = lreg.predict_proba(xvalid_w2v) 

prediction_int = prediction[:,1] >= 0.3 

prediction_int = prediction_int.astype(np.int) 

f1_score(yvalid, prediction_int)
#Submission for logistic regression

test_pred = lreg.predict_proba(test_w2v) 

test_pred_int = test_pred[:,1] >= 0.3 

test_pred_int = test_pred_int.astype(np.int) 

test['target'] = test_pred_int 

submission = test[['id','target']] 

submission.to_csv('fake_lreg_w2v.csv', index=False) # writing data to a CSV file
#SVM

from sklearn import svm

svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_w2v, ytrain) 

prediction = svc.predict_proba(xvalid_w2v) 

prediction_int = prediction[:,1] >= 0.3 

prediction_int = prediction_int.astype(np.int) 

f1_score(yvalid, prediction_int)
#Submission for SVM

test_pred = svc.predict_proba(test_w2v) 

test_pred_int = test_pred[:,1] >= 0.3 

test_pred_int = test_pred_int.astype(np.int) 

test['target'] = test_pred_int 

submission = test[['id','target']] 

submission.to_csv('fake_2svm_w2v.csv', index=False) # writing data to a CSV file
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_w2v, ytrain) 

prediction = rf.predict(xvalid_w2v) 

f1_score(yvalid, prediction)
#Submission for Random Forest

test_pred = rf.predict_proba(test_w2v) 

test_pred_int = test_pred[:,1] >= 0.3 

test_pred_int = test_pred_int.astype(np.int) 

test['target'] = test_pred_int 

submission = test[['id','target']] 

submission.to_csv('fake_3rdmfrt_w2v.csv', index=False) # writing data to a CSV file
#XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_w2v, ytrain) 

prediction = xgb.predict(xvalid_w2v) 

f1_score(yvalid, prediction)
#Submission for XGB

test_pred = xgb.predict_proba(test_w2v) 

test_pred_int = test_pred[:,1] >= 0.3 

test_pred_int = test_pred_int.astype(np.int) 

test['target'] = test_pred_int 

submission = test[['id','target']] 

submission.to_csv('fake_4xgb_w2v.csv', index=False) # writing data to a CSV file
#Fine tuning XGB

import xgboost as xgb

dtrain = xgb.DMatrix(xtrain_w2v, label=ytrain) 

dvalid = xgb.DMatrix(xvalid_w2v, label=yvalid) 

dtest = xgb.DMatrix(test_w2v)



# Parameters that we are going to tune 

params = {

    'objective':'binary:logistic',

    'max_depth':6,

    'min_child_weight': 1,

    'eta':.3,

    'subsample': 1,

    'colsample_bytree': 1

 }



def custom_eval(preds, dtrain):

    labels = dtrain.get_label().astype(np.int)

    preds = (preds >= 0.3).astype(np.int)

    return [('f1_score', f1_score(labels, preds))]
gridsearch_params = [

    (max_depth, min_child_weight)

    for max_depth in range(6,10)

     for min_child_weight in range(5,8)

 ]



max_f1 = 0. # initializing with 0 

best_params = None 

for max_depth, min_child_weight in gridsearch_params:

    print("CV with max_depth={}, min_child_weight={}".format(

                             max_depth,

                             min_child_weight))

     # Update our parameters

    params['max_depth'] = max_depth

    params['min_child_weight'] = min_child_weight



     # Cross-validation

    cv_results = xgb.cv(        params,

        dtrain,        feval= custom_eval,

        num_boost_round=200,

        maximize=True,

        seed=16,

        nfold=5,

        early_stopping_rounds=10

    ) 

    # Finding best F1 Score

    

    mean_f1 = cv_results['test-f1_score-mean'].max()

    

    boost_rounds = cv_results['test-f1_score-mean'].argmax()    

    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))    

    if mean_f1 > max_f1:

        max_f1 = mean_f1

        best_params = (max_depth,min_child_weight) 



print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))

#Updating max_depth and min_child_weight parameters.



params['max_depth'] = 8

params['min_child_weight'] = 6
#Tuning subsample and colsample



gridsearch_params = [

    (subsample, colsample)

    for subsample in [i/10. for i in range(5,10)]

    for colsample in [i/10. for i in range(5,10)] ]



max_f1 = 0. 

best_params = None 

for subsample, colsample in gridsearch_params:

    print("CV with subsample={}, colsample={}".format(

                             subsample,

                             colsample))

     # Update our parameters

    params['colsample'] = colsample

    params['subsample'] = subsample

    cv_results = xgb.cv(

        params,

        dtrain,

        feval= custom_eval,

        num_boost_round=200,

        maximize=True,

        seed=16,

        nfold=5,

        early_stopping_rounds=10

    )

     # Finding best F1 Score

    mean_f1 = cv_results['test-f1_score-mean'].max()

    boost_rounds = cv_results['test-f1_score-mean'].argmax()

    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))

    if mean_f1 > max_f1:

        max_f1 = mean_f1

        best_params = (subsample, colsample) 



print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))
params['subsample'] = .9 

params['colsample_bytree'] = .5
max_f1 = 0. 

best_params = None 

for eta in [.3, .2, .1, .05, .01, .005]:

    print("CV with eta={}".format(eta))

     # Update ETA

    params['eta'] = eta



     # Run CV

    cv_results = xgb.cv(

        params,

        dtrain,

        feval= custom_eval,

        num_boost_round=1000,

        maximize=True,

        seed=16,

        nfold=5,

        early_stopping_rounds=20

    )



     # Finding best F1 Score

    mean_f1 = cv_results['test-f1_score-mean'].max()

    boost_rounds = cv_results['test-f1_score-mean'].argmax()

    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))

    if mean_f1 > max_f1:

        max_f1 = mean_f1

        best_params = eta 

print("Best params: {}, F1 Score: {}".format(best_params, max_f1))
params['eta'] = .05
params
xgb_model = xgb.train(

    params,

    dtrain,

    feval= custom_eval,

    num_boost_round= 1000,

    maximize=True,

    evals=[(dvalid, "Validation")],

    early_stopping_rounds=10

 )
test_pred = xgb_model.predict(dtest) 

test['target'] = (test_pred >= 0.3).astype(np.int) 

submission = test[['id','target']] 

submission.to_csv('fake_xgb_w2v_finetuned.csv', index=False)