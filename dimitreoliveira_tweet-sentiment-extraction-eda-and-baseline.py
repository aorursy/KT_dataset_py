import random, warnings

import pandas as pd

import seaborn as sns

from nltk import FreqDist

from nltk.corpus import stopwords

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split



SEED = 0

warnings.filterwarnings("ignore")
# Auxiliary functions

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



def evaluate_model(train_set, validation_set):

    train_set['jaccard'] = train_set.apply(lambda x: jaccard(x['selected_text'], x['prediction']), axis=1)

    validation_set['jaccard'] = validation_set.apply(lambda x: jaccard(x['selected_text'], x['prediction']), axis=1)



    print('Train set Jaccard: %.3f' % train_set['jaccard'].mean())

    print('Validation set Jaccard: %.3f' % validation_set['jaccard'].mean())



    print('\nMetric by sentiment')

    for sentiment in train_df['sentiment'].unique():

        print('\nSentiment == %s' % sentiment)

        print('Train set Jaccard: %.3f' % train_set[train_set['sentiment'] == sentiment]['jaccard'].mean())

        print('Validation set Jaccard: %.3f' % validation_set[validation_set['sentiment'] == sentiment]['jaccard'].mean())
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')



print('Train samples: %s' % len(train))

print('Test samples: %s' % len(test))

display(train.head())
sample_0 = 0

sample_1 = 1

sample_2 = 2



print('Example %d' % sample_0)

print('Text: %s' % train['text'].values[sample_0])

print('Sentiment: %s' % train['sentiment'].values[sample_0])

print('Selected text (label): %s' % train['selected_text'].values[sample_0])



print('\nExample %d' % sample_1)

print('Text: %s' % train['text'].values[sample_1])

print('Sentiment: %s' % train['sentiment'].values[sample_1])

print('Selected text (label): %s' % train['selected_text'].values[sample_1])



print('\nExample %d' % sample_2)

print('Text: %s' % train['text'].values[sample_2])

print('Sentiment: %s' % train['sentiment'].values[sample_2])

print('Selected text (label): %s' % train['selected_text'].values[sample_2])
sentiment_order = train['sentiment'].unique()

sns.set(style="darkgrid")

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

sns.countplot(x='sentiment', data=train, palette="Set3", order=sentiment_order, ax=ax1).set_title("Train")

sns.countplot(x='sentiment', data=test, palette="Set3", order=sentiment_order, ax=ax2).set_title("Test")

plt.show()
train['text'].fillna('', inplace=True)

train['selected_text'].fillna('', inplace=True)



train['text_len'] = train['text'].apply(lambda x : len(x))

train['text_wordCnt'] = train['text'].apply(lambda x : len(x.split(' ')))

train['selected_text_len'] = train['selected_text'].apply(lambda x : len(x))

train['selected_text_wordCnt'] = train['selected_text'].apply(lambda x : len(x.split(' ')))

test['text_len'] = test['text'].apply(lambda x : len(x))

test['text_wordCnt'] = test['text'].apply(lambda x : len(x.split(' ')))



f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 10), sharex=True)

sns.distplot(train['text_len'], ax=ax1).set_title("Train")

sns.distplot(test['text_len'], ax=ax2).set_title("Test")

plt.show()



f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 10), sharex=True)

sns.distplot(train['text_wordCnt'], ax=ax1).set_title("Train")

sns.distplot(test['text_wordCnt'], ax=ax2).set_title("Test")

plt.show()
f, ax = plt.subplots(1, 1, figsize=(24, 5), sharex=True)

sns.distplot(train['selected_text_len'], ax=ax).set_title("Train")

plt.show()



f, ax = plt.subplots(1, 1, figsize=(24, 5), sharex=True)

sns.distplot(train['selected_text_wordCnt'], ax=ax).set_title("Train")

plt.show()
eng_stopwords = stopwords.words('english')



train['text'] = train['text'].str.replace('[^a-z ]','')

train['selected_text'] = train['selected_text'].str.replace('[^a-z ]','')

train['text'] = train['text'].apply(lambda x: x.lower())

train['selected_text'] = train['selected_text'].apply(lambda x: x.lower())



freq_dist = FreqDist([word for comment in train['text'] for word in comment.split() if word not in eng_stopwords])

plt.figure(figsize=(20, 6))

plt.title('Word frequency on text').set_fontsize(20)

freq_dist.plot(60, marker='.', markersize=10)

plt.show()



freq_dist = FreqDist([word for comment in train['selected_text'] for word in comment.split() if word not in eng_stopwords])

plt.figure(figsize=(20, 6))

plt.title('Word frequency on selected_text').set_fontsize(20)

freq_dist.plot(60, marker='.', markersize=10)

plt.show()
test['text'] = test['text'].str.replace('[^a-z ]','')

test['text'] = test['text'].apply(lambda x: x.lower())



freq_dist = FreqDist([word for comment in test['text'] for word in comment.split() if word not in eng_stopwords])

plt.figure(figsize=(20, 6))

plt.title('Word frequency on text').set_fontsize(20)

freq_dist.plot(60, marker='.', markersize=10)

plt.show()
sns.jointplot("selected_text_len", "text_len", data=train, kind="reg", color="m", height=12)

plt.show()
sns.jointplot("selected_text_wordCnt", "text_wordCnt", data=train, kind="reg", color="m", height=12)

plt.show()
train_df, validation_df = train_test_split(train, test_size=0.2, random_state=SEED)



print('Train set size: %s' % len(train_df))

print('Validation set size: %s' % len(validation_df))
# Heuristic model

train_df["prediction"] = train_df["text"].apply(lambda x: " ".join(x.strip().split(' ')[-5:]))

validation_df["prediction"] = validation_df["text"].apply(lambda x: " ".join(x.strip().split(' ')[-5:]))

# Metric does not works with empty labels

train_df["prediction"] = train_df["prediction"].apply(lambda x: '.' if x.strip() == '' else x)

validation_df["prediction"] = validation_df["prediction"].apply(lambda x: '.' if x.strip() == '' else x)



evaluate_model(train_df, validation_df)
# Heuristic model

train_df["prediction"] = train_df["text"].apply(lambda x: " ".join(x.strip().split(' ')[5:]))

validation_df["prediction"] = validation_df["text"].apply(lambda x: " ".join(x.strip().split(' ')[5:]))

# Metric does not works with empty labels

train_df["prediction"] = train_df["prediction"].apply(lambda x: '.' if x.strip() == '' else x)

validation_df["prediction"] = validation_df["prediction"].apply(lambda x: '.' if x.strip() == '' else x)



evaluate_model(train_df, validation_df)
# Heuristic model

train_df["prediction"] = train_df["text"]

validation_df["prediction"] = validation_df["text"]

# Metric does not works with empty labels

train_df["prediction"] = train_df["prediction"].apply(lambda x: '.' if x.strip() == '' else x)

validation_df["prediction"] = validation_df["prediction"].apply(lambda x: '.' if x.strip() == '' else x)



evaluate_model(train_df, validation_df)
# Heuristic model

train_df["prediction"] = train_df["text"].apply(lambda x: " ".join(x.split()[-20:]))

validation_df["prediction"] = validation_df["text"].apply(lambda x: " ".join(x.split()[-20:]))

# Metric does not works with empty labels

train_df["prediction"] = train_df["prediction"].apply(lambda x: '.' if x.strip() == '' else x)

validation_df["prediction"] = validation_df["prediction"].apply(lambda x: '.' if x.strip() == '' else x)



evaluate_model(train_df, validation_df)
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

submission["selected_text"] = test["text"]

submission.to_csv("submission.csv", index=False)

display(submission.head(10))