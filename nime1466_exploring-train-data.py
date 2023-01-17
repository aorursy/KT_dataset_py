import re



import pandas as pd
train_data = pd.read_csv('../input/train.csv')
train_data.info()
train_data[0:5]
label_columns = [x for x in train_data.columns if x not in['id', 'comment_text']]
for label in label_columns:

    print(f"Count for {label}: {train_data[label].sum()/95851}")
def remove_punctuation(row_str):

    return re.sub(r"\W", " ", row_str)
train_data = train_data.assign(comment_text=train_data.comment_text.apply(remove_punctuation))
train_data[0:10]
train_data = train_data.assign(comment_len=train_data.comment_text.str.len())
deciles = [x/10.0 for x in range(1, 10)]

train_data.comment_len.describe(percentiles=deciles)
for label in label_columns:

    print("Correlation with comment length for {}: {}".format(label, train_data[label].corr(train_data.comment_len)))
def get_num_words(row_str):

    return len(row_str.split())
train_data = train_data.assign(num_words=train_data.comment_text.apply(get_num_words))
train_data.num_words.describe(percentiles=deciles)
def get_unique_words(row_str):

    return len(set(row_str.lower().split()))
train_data = train_data.assign(unique_words=train_data.comment_text.apply(get_unique_words))
train_data.unique_words.describe(percentiles=deciles)
for label in label_columns:

    print("Correlation with number of words for {}: {}".format(label, train_data[label].corr(train_data.num_words)))
for label in label_columns:

    print("Correlation with unique words for {}: {}".format(label, train_data[label].corr(train_data.unique_words)))
train_data.eval('mean_word_length = comment_len/num_words', inplace=True)
train_data.mean_word_length.describe(percentiles=deciles)
train_data.mean_word_length.quantile(0.99)
for label in label_columns:

    print("Correlation with unique words for {}: {}".format(label, train_data[label].corr(train_data.mean_word_length)))