import pandas as pd



train = pd.read_csv('../input/tweet-sentiment-analysis/train.csv')

test = pd.read_csv('../input/tweet-sentiment-analysis/test.csv')
train.head()
print(f"There are {len(train)} tweets in the training data.")

positive = sum(train.target)/len(train)

print(f"{positive:.2f}% of the tweets are positive.")

avg_len = sum([len(text) for text in train.text])/len(train)

print(f"The average length of the tweets is {avg_len:.1f} characters.")
from sklearn.feature_extraction.text import CountVectorizer



count_vect = CountVectorizer()

count_vect.fit(train['text'])



bow_train = count_vect.transform(train['text'])

bow_test = count_vect.transform(test['text'])
print(f"There are {len(count_vect.vocabulary_)} words in the vocabulary.")
count_vect = CountVectorizer(max_features=15000)

count_vect.fit(train['text'])



bow_train = count_vect.transform(train['text'])

bow_test = count_vect.transform(test['text'])
from sklearn.linear_model import LogisticRegression



model = LogisticRegression(max_iter=1000).fit(bow_train, train.target)
predictions = model.predict_proba(bow_test)

predictions = [probs[1] for probs in predictions]
submission = pd.read_csv('../input/tweet-sentiment-analysis/sample_submission.csv')

submission['Predicted'] = predictions
submission.to_csv('/kaggle/working/submission.csv', index=False)