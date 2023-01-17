import numpy as np

import pandas as pd 

import spacy

from spacy.matcher import PhraseMatcher

from collections import defaultdict

import gc
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

submission_df = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
disaster_keywords = list(train_df.loc[(train_df.target == 1) & (train_df.keyword.notnull())].keyword.drop_duplicates())
nlp = spacy.load('en')
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

patterns =[nlp(text) for text in disaster_keywords]

matcher.add("disaster", None, *patterns)
tweets = defaultdict(list)



for idx, tweet in train_df.iterrows():

    doc = nlp(tweet.text)

    matches = matcher(doc)

    

    found_items = set([doc[match[1]:match[2]] for match in matches])

    

    for item in found_items:

        tweets[str(item).lower()].append(tweet.target)
counts = {item: len(ratings) for item, ratings in tweets.items()}



item_counts = sorted(counts, key=counts.get, reverse=True)

for item in item_counts:

    print(f"{item:>30}{counts[item]:>5}")
def load_data(split=0.9):

    data = train_df

    

    # Shuffle data

    train_data = data.sample(frac=1, random_state=7)

    

    texts = train_data.text.values

    labels = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in train_data.target.values]

    split = int(len(train_data) * split)

    

    train_labels = [{"cats": labels} for labels in labels[:split]]

    val_labels = [{"cats": labels} for labels in labels[split:]]

    

    return texts[:split], train_labels, texts[split:], val_labels



train_texts, train_labels, val_texts, val_labels = load_data()
nlp = spacy.blank("en")



# "textcat" - TextCategorizer

pipe = nlp.create_pipe("textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"})



nlp.add_pipe(pipe)



pipe.add_label("NEGATIVE")

pipe.add_label("POSITIVE")
from spacy.util import minibatch

import random



def train(model, train_data, optimizer, batch_size=10):

    losses = {}

    random.seed(1)

    random.shuffle(train_data)

    

    batches = minibatch(train_data, size=batch_size)

    for batch in batches:

        texts, labels = zip(*batch)

        model.update(texts, labels, sgd=optimizer, losses=losses)

        

    return losses
spacy.util.fix_random_seed(1)

random.seed(1)



optimizer = nlp.begin_training()

train_data = list(zip(train_texts, train_labels))

losses = train(nlp, train_data, optimizer)

print(losses['textcat'])
def predict(model, texts):  

    docs = [model.tokenizer(text) for text in texts]

     

    textcat = model.get_pipe('textcat')

    scores, _ = textcat.predict(docs)

     

    predicted_class = scores.argmax(axis=1)

    

    return predicted_class
texts = val_texts[34:38]

predictions = predict(nlp, texts)



for p, t in zip(predictions, texts):

    print(f"{pipe.labels[p]}: {t} \n")
def evaluate(model, texts, labels):

    """ Returns the accuracy of a TextCategorizer model. 

    

        Arguments

        ---------

        model: ScaPy model with a TextCategorizer

        texts: Text samples, from load_data function

        labels: True labels, from load_data function

    

    """

    predicted_class = predict(model, texts)

    

    true_class = [int(each['cats']['POSITIVE']) for each in labels]

    

    correct_predictions = predicted_class == true_class

    

    accuracy = correct_predictions.mean()

    

    return accuracy
accuracy = evaluate(nlp, val_texts, val_labels)
n_iters = 5

for i in range(n_iters):

    losses = train(nlp, train_data, optimizer)

    accuracy = evaluate(nlp, val_texts, val_labels)

    print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")
test_sub1 = test_df
predictions_test = predict(nlp, test_sub1.text.values)



for p, t in zip(predictions_test, test_sub1.text.values):

    print(f"{pipe.labels[p]}: {t} \n")
test_sub1['target'] = predictions_test
test_sub1[['id', 'target']].to_csv('submission.csv', index=False)
del test_sub1

gc.collect()
nlp_v2 = spacy.load('en_core_web_lg')
# We just want the vectors so we can turn off other models in the pipeline

with nlp_v2.disable_pipes():

    vectors = np.array([nlp_v2(tweet.text).vector for idx, tweet in train_df.iterrows()])

    

vectors.shape
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(vectors, train_df.target, test_size=0.2, random_state=1)
# Create the LinearSVC model

model_lsvc = LinearSVC(random_state=1, dual=False)

# Fit the model

model_lsvc.fit(X_train, y_train)
print(f'Model test accuracy: {model_lsvc.score(X_test, y_test)*100:.3f}%')
with nlp_v2.disable_pipes():

    vectors_test1 = np.array([nlp_v2(tweet.text).vector for idx, tweet in test_df.iterrows()])

    

linearsvc_preds = model_lsvc.predict(vectors_test1)
test_sub2 = test_df

test_sub2['target'] = linearsvc_preds

test_sub2[['id', 'target']].to_csv('sub_lsvc.csv', index=False)
from xgboost import XGBClassifier



model_xgbc = XGBClassifier(learning_rate=0.05, random_state=1)

model_xgbc.fit(X_train, y_train)
print(f'Model test accuracy: {model_xgbc.score(X_test, y_test)*100:.3f}%')
with nlp_v2.disable_pipes():

    vectors_test2 = np.array([nlp_v2(tweet.text).vector for idx, tweet in test_df.iterrows()])

    

xgboost_preds = model_xgbc.predict(vectors_test2)
test_sub3 = test_df

test_sub3['target'] = xgboost_preds

test_sub3[['id', 'target']].to_csv('sub_xgboost.csv', index=False)
train_cent = train_df

test_cent = test_df
## Center the document vectors

# Calculate the mean for the document vectors, should have shape (300,)

vec_mean = vectors.mean(axis=0)

# Subtract the mean from the vectors

centered = vectors - vec_mean
def cosine_similarity(a, b):

    return np.dot(a, b)/np.sqrt(a.dot(a)*b.dot(b))
def x_text(x):

    tweet_vec = nlp_v2(x).vector

    sims = np.array([cosine_similarity(tweet_vec - vec_mean, vec) for vec in centered])

    most_similar = sims.argmax()

    target = train_cent.iloc[most_similar].target

    return target
test_cent['target'] = test_cent['text'].apply(x_text)
test_cent[['id', 'target']].to_csv('sub_cent.csv', index=False)