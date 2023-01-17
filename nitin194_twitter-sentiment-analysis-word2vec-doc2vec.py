# Let’s load the libraries



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
# Let’s read train and test datasets.



train  = pd.read_csv('../input/train_E6oV3lV.csv') 

test = pd.read_csv('../input/test_tweets_anuFYb8.csv')
train[train['label'] == 0].head(10)




train[train['label'] == 1].head(10)

train.shape, test.shape
train["label"].value_counts()

plt.hist(train.tweet.str.len(), bins=20, label='train')

plt.hist(test.tweet.str.len(), bins=20, label='test')

plt.legend()

plt.show()
combi = train.append(test, ignore_index=True, sort=True)

combi.shape
def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

    return input_txt
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*") 

combi.head(10)
combi.tidy_tweet = combi.tidy_tweet.str.replace("[^a-zA-Z#]", " ")

combi.head(10)
combi.tidy_tweet = combi.tidy_tweet.apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

combi.head(10)
tokenized_tweet = combi.tidy_tweet.apply(lambda x: x.split())

tokenized_tweet.head()
# Now we can normalize the tokenized tweets.



from nltk.stem.porter import * 

stemmer = PorterStemmer() 

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

tokenized_tweet.head()
# Now let’s stitch these tokens back together. It can easily be done using nltk’s MosesDetokenizer function.



for i in range(len(tokenized_tweet)):

    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])    

combi['tidy_tweet'] = tokenized_tweet

combi.head(10)
all_words = ' '.join([text for text in combi['tidy_tweet']]) 



from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 

plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off')

plt.show()
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]]) 



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
# function to collect hashtags 



def hashtag_extract(x):

    hashtags = []    # Loop over the words in the tweet

    for i in x:

        ht = re.findall(r"#(\w+)", i)

        hashtags.append(ht)

    return hashtags
# extracting hashtags from non racist/sexist tweets 



HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0]) 
# extracting hashtags from racist/sexist tweets



HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1]) 
# unnesting list



HT_regular = sum(HT_regular,[]) 

HT_negative = sum(HT_negative,[])
a = nltk.FreqDist(HT_regular)

d = pd.DataFrame(

    {

    'Hashtag': list(a.keys()),

    'Count': list(a.values())

    }

) 
# selecting top 20 most frequent hashtags



d = d.nlargest(columns="Count", n = 20)

plt.figure(figsize=(20,5))

ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

# plt.xticks(rotation=90)

plt.show()
a = nltk.FreqDist(HT_negative)

d = pd.DataFrame(

    {

    'Hashtag': list(a.keys()),

    'Count': list(a.values())

    }

) 
# selecting top 20 most frequent hashtags



d = d.nlargest(columns="Count", n = 20)

plt.figure(figsize=(20,5))

ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

# plt.xticks(rotation=90)

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 

import gensim
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])

bow.shape
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])

tfidf.shape
%%time



tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing 



model_w2v = gensim.models.Word2Vec(

            tokenized_tweet,

            size=200, # desired no. of features/independent variables

            window=5, # context window size

            min_count=2, # Ignores all words with total frequency lower than 2.                                  

            sg = 1, # 1 for skip-gram model

            hs = 0,

            negative = 10, # for negative sampling

            workers= 32, # no.of cores

            seed = 34

) 



model_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)
model_w2v.wv.most_similar(positive="dinner")
model_w2v.most_similar(positive="trump")
model_w2v['food']
len(model_w2v['food']) #The length of the vector is 200
def word_vector(tokens, size):

    vec = np.zeros(size).reshape((1, size))

    count = 0

    for word in tokens:

        try:

            vec += model_w2v[word].reshape((1, size))

            count += 1.

        except KeyError:  # handling the case where the token is not in vocabulary

            continue

    if count != 0:

        vec /= count

    return vec
wordvec_arrays = np.zeros((len(tokenized_tweet), 200)) 

for i in range(len(tokenized_tweet)):

    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)

wordvec_df = pd.DataFrame(wordvec_arrays)

wordvec_df.shape
from tqdm import tqdm 

tqdm.pandas(desc="progress-bar") 

from gensim.models.doc2vec import LabeledSentence
def add_label(twt):

    output = []

    for i, s in zip(twt.index, twt):

        output.append(LabeledSentence(s, ["tweet_" + str(i)]))

    return output



labeled_tweets = add_label(tokenized_tweet) # label all the tweets
labeled_tweets[:6]
%%time 

model_d2v = gensim.models.Doc2Vec(dm=1, # dm = 1 for ‘distributed memory’ model

                                  dm_mean=1, # dm_mean = 1 for using mean of the context word vectors

                                  vector_size=200, # no. of desired features

                                  window=5, # width of the context window                                  

                                  negative=7, # if > 0 then negative sampling will be used

                                  min_count=5, # Ignores all words with total frequency lower than 5.                                  

                                  workers=32, # no. of cores                                  

                                  alpha=0.1, # learning rate                                  

                                  seed = 23, # for reproducibility

                                 ) 



model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])



model_d2v.train(labeled_tweets, total_examples= len(combi['tidy_tweet']), epochs=15)
docvec_arrays = np.zeros((len(tokenized_tweet), 200)) 

for i in range(len(combi)):

    docvec_arrays[i,:] = model_d2v.docvecs[i].reshape((1,200))    



docvec_df = pd.DataFrame(docvec_arrays) 

docvec_df.shape
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
# Extracting train and test BoW features 

train_bow = bow[:31962,:] 

test_bow = bow[31962:,:] 



# splitting data into training and validation set 

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)



lreg = LogisticRegression(solver='lbfgs') 



# training the model 

lreg.fit(xtrain_bow, ytrain) 

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set 

prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0 

prediction_int = prediction_int.astype(np.int) 

f1_score(yvalid, prediction_int) # calculating f1 score for the validation set
test_pred = lreg.predict_proba(test_bow)

test_pred_int = test_pred[:,1] >= 0.3

test_pred_int = test_pred_int.astype(np.int)

test['label'] = test_pred_int

submission = test[['id','label']]

submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file
train_tfidf = tfidf[:31962,:]

test_tfidf = tfidf[31962:,:] 



xtrain_tfidf = train_tfidf[ytrain.index]

xvalid_tfidf = train_tfidf[yvalid.index]



lreg.fit(xtrain_tfidf, ytrain) 



prediction = lreg.predict_proba(xvalid_tfidf)



prediction_int = prediction[:,1] >= 0.3

prediction_int = prediction_int.astype(np.int) 



f1_score(yvalid, prediction_int) # calculating f1 score for the validation set
train_w2v = wordvec_df.iloc[:31962,:]

test_w2v = wordvec_df.iloc[31962:,:]



xtrain_w2v = train_w2v.iloc[ytrain.index,:]

xvalid_w2v = train_w2v.iloc[yvalid.index,:]



lreg.fit(xtrain_w2v, ytrain) 



prediction = lreg.predict_proba(xvalid_w2v)



prediction_int = prediction[:,1] >= 0.3

prediction_int = prediction_int.astype(np.int)



f1_score(yvalid, prediction_int)
train_d2v = docvec_df.iloc[:31962,:]

test_d2v = docvec_df.iloc[31962:,:] 



xtrain_d2v = train_d2v.iloc[ytrain.index,:]

xvalid_d2v = train_d2v.iloc[yvalid.index,:]



lreg.fit(xtrain_d2v, ytrain) 



prediction = lreg.predict_proba(xvalid_d2v)



prediction_int = prediction[:,1] >= 0.3

prediction_int = prediction_int.astype(np.int)



f1_score(yvalid, prediction_int)
from sklearn import svm
svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_bow, ytrain) 

prediction = svc.predict_proba(xvalid_bow) 

prediction_int = prediction[:,1] >= 0.3 

prediction_int = prediction_int.astype(np.int) 

f1_score(yvalid, prediction_int)
test_pred = svc.predict_proba(test_bow) 

test_pred_int = test_pred[:,1] >= 0.3 

test_pred_int = test_pred_int.astype(np.int) 

test['label'] = test_pred_int 

submission = test[['id','label']] 

submission.to_csv('sub_svm_bow.csv', index=False)
svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_tfidf, ytrain) 

prediction = svc.predict_proba(xvalid_tfidf) 

prediction_int = prediction[:,1] >= 0.3 

prediction_int = prediction_int.astype(np.int) 

f1_score(yvalid, prediction_int)
svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_w2v, ytrain) 

prediction = svc.predict_proba(xvalid_w2v) 

prediction_int = prediction[:,1] >= 0.3 

prediction_int = prediction_int.astype(np.int) 

f1_score(yvalid, prediction_int)
svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_d2v, ytrain) 

prediction = svc.predict_proba(xvalid_d2v) 

prediction_int = prediction[:,1] >= 0.3 

prediction_int = prediction_int.astype(np.int) 

f1_score(yvalid, prediction_int)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_bow, ytrain) 

prediction = rf.predict(xvalid_bow) 

f1_score(yvalid, prediction) # validation score
test_pred = rf.predict(test_bow)

test['label'] = test_pred

submission = test[['id','label']]

submission.to_csv('sub_rf_bow.csv', index=False)
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_tfidf, ytrain) 

prediction = rf.predict(xvalid_tfidf)

f1_score(yvalid, prediction)
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_w2v, ytrain) 

prediction = rf.predict(xvalid_w2v)

f1_score(yvalid, prediction)
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_d2v, ytrain) 

prediction = rf.predict(xvalid_d2v)

f1_score(yvalid, prediction)
from xgboost import XGBClassifier
xgb_model = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_bow, ytrain)

prediction = xgb_model.predict(xvalid_bow)

f1_score(yvalid, prediction)
test_pred = xgb_model.predict(test_bow)

test['label'] = test_pred

submission = test[['id','label']]

submission.to_csv('sub_xgb_bow.csv', index=False)
xgb = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_tfidf, ytrain) 

prediction = xgb.predict(xvalid_tfidf)

f1_score(yvalid, prediction)
xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_w2v, ytrain) 

prediction = xgb.predict(xvalid_w2v)

f1_score(yvalid, prediction)
xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_d2v, ytrain) 

prediction = xgb.predict(xvalid_d2v)

f1_score(yvalid, prediction)
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

    print("CV with max_depth={}, min_child_weight={}".format(max_depth,min_child_weight))

    

     # Update our parameters

    params['max_depth'] = max_depth

    params['min_child_weight'] = min_child_weight



     # Cross-validation

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

boost_rounds = cv_results['test-f1_score-mean'].idxmax()    

print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))    



if mean_f1 > max_f1:

        max_f1 = mean_f1

        best_params = (max_depth,min_child_weight) 



print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))
params['max_depth'] = 9 

params['min_child_weight'] = 7
gridsearch_params = [

    (subsample, colsample)

    for subsample in [i/10. for i in range(5,10)]

    for colsample in [i/10. for i in range(5,10)]

]



max_f1 = 0. 

best_params = None 



for subsample, colsample in gridsearch_params:

    print("CV with subsample={}, colsample={}".format(subsample,colsample))

    

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

    boost_rounds = cv_results['test-f1_score-mean'].idxmax()

    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))

    

    if mean_f1 > max_f1:

        max_f1 = mean_f1

        best_params = (subsample, colsample) 



print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))
params['subsample'] = 0.9

params['colsample_bytree'] = 0.5
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

    boost_rounds = cv_results['test-f1_score-mean'].idxmax()

    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))

    

    if mean_f1 > max_f1:

        max_f1 = mean_f1

        best_params = eta 

        

print("Best params: {}, F1 Score: {}".format(best_params, max_f1))
params = {

    'colsample': 0.9,

    'colsample_bytree': 0.5,

    'eta': 0.1,

    'max_depth': 9,

    'min_child_weight': 7,

    'objective': 'binary:logistic',

    'subsample': 0.9

}
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

test['label'] = (test_pred >= 0.3).astype(np.int)

submission = test[['id','label']] 

submission.to_csv('sub_xgb_w2v_finetuned.csv', index=False)