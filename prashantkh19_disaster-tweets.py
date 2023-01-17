from fastai import *

from fastai.text import *

import pandas as pd
train_df= pd.read_csv('../input/nlp-getting-started/train.csv')

test_df= pd.read_csv('../input/nlp-getting-started/test.csv')

train_df.head()
test_df.head()
# Data cleaning steps

    

# Remove "@user from tweets

def clean_data(df, text_col, new_col='cleaned_text'):

    

    '''It will remove the noise from the text data(@user, characters not able to encode/decode properly)    

    Arguments:

    df : Data Frame

    col : column name of type string

    '''

    tweets_data = df.copy()

    

    #tweets_data[new_col] = tweets_data[text_col].apply(lambda x : re.sub(

    #   f"@[A-Za-z0-9]+", '', x))

    

    #temp line

    tweets_data[new_col] = tweets_data[text_col]

 



    # Keeping only few punctuations 

    tweets_data[new_col] = tweets_data[new_col].apply(lambda x : re.sub(

        '@[A-Za-z0-9]+', '', x))



    #tweets_data[new_col] = tweets_data[new_col].apply(lambda x : re.sub(

        #f'[^{PUNCTUATION_TO_KEEP}A-Za-z0-9]', '', x))



    # Trimming the sentences

    tweets_data[new_col] = tweets_data[new_col].str.strip()

    

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub('\\n',' ',str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r'\W',' ',str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r'https\s+|www.\s+',r'', str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r'http\s+|www.\s+',r'', str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r'\s+[a-zA-Z]\s+',' ',str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r'\^[a-zA-Z]\s+',' ',str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r'\s+',' ',str(x)))

    tweets_data[new_col] = tweets_data[new_col].str.lower()

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\’", "\'", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"won\'t", "will not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"can\'t", "can not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"don\'t", "do not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"dont", "do not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"n\’t", " not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"n\'t", " not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'re", " are", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'s", " is", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\’d", " would", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\d", " would", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'ll", " will", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'t", " not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'ve", " have", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'m", " am", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\n", "", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\r", "", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"[0-9]", "digit", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'", "", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\"", "", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r'[?|!|\'|"|#]',r'', str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r'[.|,|)|(|\|/]',r' ', str(x)))

    

    return tweets_data

train_df = clean_data(train_df, "text", 'cleaned_text')

from nltk.stem import PorterStemmer

stemming = PorterStemmer()

stemming.stem('runs')

temp = train_df['cleaned_text'].apply(lambda sentence : [stemming.stem(x) for x in sentence.split(" ")])



sentences = []

for i in temp:

    sentences.append(" ".join(i))

sentences[0]



train_df['stemmed_text'] = sentences



del sentences, temp
test_df = clean_data(test_df, "text", 'cleaned_text')

from nltk.stem import PorterStemmer

stemming = PorterStemmer()

stemming.stem('runs')

temp = test_df['cleaned_text'].apply(lambda sentence : [stemming.stem(x) for x in sentence.split(" ")])



sentences = []

for i in temp:

    sentences.append(" ".join(i))

sentences[0]



test_df['stemmed_text'] = sentences



del sentences, temp
# Combining all the text data 

tweets_data = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)

tweets_data.head()
path = Path('/kaggle/working/')
data_lm = (TextList.from_df(tweets_data, path, cols = "cleaned_text").split_by_rand_pct(0.1).label_for_lm().databunch(bs=32))
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.lr_find()
learn.recorder.plot(skip_end=15, suggestion = True)
learn.fit_one_cycle(5, 1e-2, moms=(0.8,0.7))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot(skip_end=15, suggestion = True)
learn.fit_one_cycle(5, 1e-4, moms=(0.8,0.7))
learn.save_encoder('fine_tuned_enc')
path
from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(train_df, test_size = 0.3)
data_classifier = (TextDataBunch.from_df('.', X_train, X_val, test_df, text_cols = "cleaned_text", label_cols = "target", vocab = data_lm.vocab))
data_classifier.show_batch()
learn = text_classifier_learner(data_classifier, AWD_LSTM, drop_mult=0.7)

learn.load_encoder('fine_tuned_enc')
learn.lr_find()

learn.recorder.plot(suggestion= True)
learn.fit_one_cycle(5, 5e-04, moms=(0.8,0.7))
learn.freeze_to(-2)

learn.fit_one_cycle(5, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn.freeze_to(-3)

learn.fit_one_cycle(5, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn.unfreeze()

learn.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn.save('final_learner')
learn.recorder.plot_losses()
def get_preds_as_nparray(ds_type) -> np.ndarray:

    """

    the get_preds method does not yield the elements in order by default

    we borrow the code from the RNNLearner to resort the elements into their correct order

    """

    preds = learn.get_preds(ds_type)[0].detach().cpu().numpy()

    sampler = [i for i in data_classifier.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    return preds[reverse_sampler, :]
test_preds = get_preds_as_nparray(DatasetType.Test)

preds = []
for i in test_preds:

    preds.append(np.argmax(i))
sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sub.head(3)
sub['target'] = preds

sub.to_csv('submission.csv', index=False)

sub.head(3)