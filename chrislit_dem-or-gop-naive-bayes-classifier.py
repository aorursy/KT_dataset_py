# Load some libraries, mostly pandas & sklearn
# but also NLTK's TweetTokenizer since we're working with Twitter data

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.tokenize.casual import TweetTokenizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# initialize tokenizer
tokenizer = TweetTokenizer(reduce_len=True)
all_tweets = pd.read_csv("../input/ExtractedTweets.csv")
# Get an 80/20 train/test split of Twitter handles, stratified on Party
tweeters = all_tweets.iloc[:,:2].drop_duplicates()
handles_train, handles_test = train_test_split(tweeters.Handle, stratify=tweeters.Party, test_size=0.2, random_state=0)

# extract train & test sets from the all_tweets df
train = all_tweets[all_tweets.Handle.isin(handles_train)].reset_index().drop('index', axis=1)
test = all_tweets[all_tweets.Handle.isin(handles_test)].reset_index().drop('index', axis=1)
nb_pipeline = Pipeline([
    ('vectorize', TfidfVectorizer(tokenizer=tokenizer.tokenize)),
    ('classifier', MultinomialNB())
])
nb_pipeline.fit(train.Tweet, train.Party)
preds = nb_pipeline.predict(test.Tweet)
print('Accuracy: {}'.format(str(round(accuracy_score(test.Party, preds), 4))))
accuracy_df = test.drop(['Tweet'], axis=1)
accuracy_df['preds'] = preds
correct = 0
total = 0

print("Congresspersons whose tweets were mostly mis-classified:")
for name in accuracy_df.Handle.unique():
    sub_df = accuracy_df[accuracy_df.Handle == name].reset_index()
    sub_preds = sub_df.preds.value_counts()

    if sub_preds.index[0] == sub_df.Party[0]:
        correct += 1
    total += 1

    if sub_preds[sub_df.Party[0]]/len(sub_df) < 0.5:
        print("{} ({}) classified with accuracy: {}".format(name, sub_df.Party[0],
                                                                      str(round(sub_preds[sub_df.Party[0]]/len(sub_df), 4))))

print()
print("Accuracy of the model on a per-Congressperson-basis was: {}".format(str(round(correct/total, 4))))
