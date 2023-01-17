# base imports 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# for visualization of the word embeddings

from sklearn.decomposition import PCA



# for building the embeddings & preprocessing

from gensim.models import Word2Vec

from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, stem_text, strip_numeric



# for modeling

from sklearn import linear_model

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("../input/nlp-getting-started/train.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train.tail(5)
# clean function cleans all the tweets and splits them into lists of words

def clean(tweet):

    return stem_text(remove_stopwords(strip_numeric(strip_punctuation(tweet)))).split(' ')



sentences = [clean(tweet) for tweet in train['text']]

print(sentences[:3])
# second step: train the model

vectors = Word2Vec(sentences, min_count=3)

# save the model binaries

vectors.save('tweet_vectors.bin')
# load the vectors

vectors = Word2Vec.load('tweet_vectors.bin')

# get the summary of the model

print('vectors: '+str(vectors))

# here are some examples of what the vectors look like

print('example of the vector for word "officers": '+str(vectors[clean('officers')]))
# let's grab the vectors for the keywords

keywords = [clean(x) for x in train['keyword'] if isinstance(x,str)]



flat_keywords = [item for sublist in keywords for item in sublist]

print('Number of unique keywords: '+str(len(set(flat_keywords))))
keyvectors = vectors[[k for k in flat_keywords if k in vectors.wv.vocab]]



pca = PCA(n_components=2)

result = pca.fit_transform(keyvectors)
plt.scatter(result[:,0],result[:,1])

for i, word in enumerate([k for k in flat_keywords if k in vectors.wv.vocab]):

	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
def cap_prop(tweet):

    caps = 0

    for char in tweet:

        if char.isupper():

            caps += 1

    return float(caps/len(tweet))



cap_props = [cap_prop(tweet) for tweet in train['text']]

print('Capitalization proportions of first tweet: '+str(cap_props[1]))
# let's look at the difference between the cap_prop for the disaster tweets and 

# the non-disaster tweets

cap_props_d = [cap_props[p] for p in range(len(cap_props)) if train['target'][p]==1]

cap_props_nd = [cap_props[p] for p in range(len(cap_props)) if train['target'][p]==0]



plt.figure(figsize=(8,6))

plt.hist(cap_props_nd,bins=30,alpha=0.5,label="Non-Disaster",color="black")

plt.axvline(np.mean(cap_props_nd), color='black', linestyle='dashed', linewidth=1)

plt.hist(cap_props_d,bins=30,alpha=0.5,label="Disaster",color="red")

plt.axvline(np.mean(cap_props_d), color='red', linestyle='dashed', linewidth=1)

plt.xlabel("Proportion of Capitalized Letters")

plt.ylabel("Number of Tweets")

plt.title("Disaster Tweets Have (a few) More Capitalized Letters")

plt.legend(loc='upper right')

plt.show()
def hash_prop(tweet):

    h = 0

    last_hash = False

    for char in tweet:

        if last_hash and char.isspace():

            last_hash = False

        if char=='#':

            last_hash = True

            h += 1

    return float(h/len(tweet.split(' ')))

hash_props = [hash_prop(tweet) for tweet in train['text']]

print('Hashtag proportions of first tweet: '+str(hash_props[1]))
# we'll look at the same plot for the hashtag proportion

hash_props_d = [hash_props[p] for p in range(len(hash_props)) if train['target'][p]==1]

hash_props_nd = [hash_props[p] for p in range(len(hash_props)) if train['target'][p]==0]



plt.figure(figsize=(8,6))

plt.hist(hash_props_nd,bins=50,alpha=0.5,label="Non-Disaster",color="black")

plt.axvline(np.mean(hash_props_nd), color='black', linestyle='dashed', linewidth=1)

plt.hist(hash_props_d,bins=50,alpha=0.5,label="Disaster",color="red")

plt.axvline(np.mean(hash_props_d), color='red', linestyle='dashed', linewidth=1)

plt.xlabel("Proportion of Hashtags in Tweet")

plt.ylabel("Number of Tweets")

#plt.title("Disaster Tweets Have (a few) More Capitalized Letters")

plt.legend(loc='upper right')

plt.xlim(0.0,1.0)

plt.show()
def avg_vec(tweet):

    tweets = clean(tweet)

    tweets = [t for t in tweets if t in vectors.wv.vocab]

    if len(tweets) >= 1:

        return np.mean(vectors[tweets],axis=0)

    else:

        return []



v = []

for tweet in train['text']: 

    v.append(avg_vec(tweet))

v[0]
def get_val(v,row,i):

    if v[row] == []:

        return 0.0

    else:

        return float(v[row][i])

feats = {'cap_prop': cap_props,

        'hash_prop': hash_props}

feats = pd.DataFrame(feats,columns=['cap_prop','hash_prop'])

for i in range(len(v[0])):

    feats.insert(i,'e_'+str(i),[get_val(v,row,i) for row in range(len(v))],True)

target = train['target']

print(feats.head(5))
feats.to_csv('doc_embeddings.csv',index=False)


regressor = linear_model.LogisticRegression()

regressor.fit(feats,target)

train_preds = regressor.predict(feats)

print(train_preds[:3])






print(classification_report(target,train_preds))
print('We predicted '+str(sum(train_preds))+' disaster tweets of '+str(sum(target)))

print('Here are the first 5: ')

is_pred = train_preds==1

train[is_pred].head(5)
rf = RandomForestClassifier()

rf.fit(feats,target)

rf_preds = rf.predict(feats)

print(classification_report(target,rf_preds))

print('We predicted '+str(sum(rf_preds))+' disaster tweets of '+str(sum(target)))

print('Here are the first 5: ')

is_pred = rf_preds==1

train[is_pred].head(5)
# define function for calculating features

def get_feats(df):

    cap_props = [cap_prop(tweet) for tweet in df['text']]

    hash_props = [hash_prop(tweet) for tweet in df['text']]

    v = [avg_vec(tweet) for tweet in df['text']]

    #for tweet in df['text']: 

    #    v.append(avg_vec(tweet))

    

    feats = {'cap_prop': cap_props,

            'hash_prop': hash_props}

    feats = pd.DataFrame(feats,columns=['cap_prop','hash_prop'])

    

    for i in range(len(v[0])):

        feats.insert(i,'e_'+str(i),[get_val(v,row,i) for row in range(len(v))],True)



    return feats
def get_submission(df,model):

    feats = get_feats(df)

    preds = model.predict(feats)

    submission = {'id': df['id'],

                  'target': preds}

    submission = pd.DataFrame(submission,columns=['id','target'])

    return submission
submission = get_submission(test,regressor)
submission.to_csv('submission.csv',index=False)
submission = get_submission(test,rf)

submission.to_csv('rf_submission.csv',index=False)