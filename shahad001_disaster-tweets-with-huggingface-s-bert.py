import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig


# for text cleaning
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
TRAIN_DATA_PATH = "../input/nlp-getting-started/train.csv"
TEST_DATA_PATH = "../input/nlp-getting-started/test.csv"

train_data = pd.read_csv(TRAIN_DATA_PATH)
test_data = pd.read_csv(TEST_DATA_PATH)

print(color.YELLOW + 'summary of dataset:\n' + color.END,
      train_data.describe())


TEXT_FIELD = "text"
LABEL_FIELD = "target"

# drop the null entitites
train_data.dropna(inplace=True, subset=[TEXT_FIELD])
train_data.drop_duplicates(inplace=True,subset=[TEXT_FIELD])

print(color.YELLOW + '\nsummary of dataset after dropping null rows and duplicate texts:\n' + color.END,
      train_data.describe())


train_data['len'] = train_data[TEXT_FIELD].apply(len)
train_data['len'].value_counts().plot.bar()
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}

stop_words = stopwords.words('english')
stop_words = [w for w in stop_words if not w in ['not', 'no', 'nor']]

def clean_text(text):
    # remove the contractions
    for word in text.split():
        if word.lower() in contractions:
            text = text.replace(word, contractions[word.lower()])
    
    tokens = word_tokenize(text) # divide into tokens
    table = str.maketrans('', '', string.punctuation)
    words = [w.lower().translate(table) for w in tokens] # remove the punc'ns and convert to lower case
    words = [w for w in words if w.isalpha()]
#     words = [w for w in words if not w in stop_words]
    size = len(words)
    
    clean_text = ' '.join(word for word in words)    
    return clean_text, size


def create_inputs(tweets, tokenizer):
    input_ids, attention_masks, token_type_ids=[], [], []
    MAX_LEN = -1
    clean_tweets = []
    for tweet in tweets:
        clean_tweet, size = clean_text(tweet)
        clean_tweets.append(clean_tweet)
        MAX_LEN = max(MAX_LEN, size)
    
    
    for tweet in clean_tweets:
        encoded = tokenizer.encode_plus(tweet, add_special_tokens=True, max_length=64,
                                       pad_to_max_length=True)
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        token_type_ids.append(encoded['token_type_ids'])
    
    return np.asarray(input_ids), np.asarray(attention_masks), np.asarray(token_type_ids)
# create the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# get BERT essentials
train_input_ids, train_attention_masks, train_token_type_ids = create_inputs(
    train_data[TEXT_FIELD], tokenizer)
test_input_ids,test_attention_masks,test_token_type_ids = create_inputs(test_data[TEXT_FIELD], tokenizer)
train_len = train_input_ids.shape[0]
val_len = int(0.2 * train_len) # 20% data will be used for validation

val_inp = train_input_ids[:val_len]
train_inp = train_input_ids[val_len : ]

val_out = train_data[LABEL_FIELD][:val_len]
train_out = train_data[LABEL_FIELD][val_len : ]

val_mask = train_attention_masks[:val_len]
train_mask = train_attention_masks[val_len : ]

val_type_ids = train_token_type_ids[:val_len]
train_type_ids = train_token_type_ids[val_len : ]
def convert_to_features(input_ids,attention_masks,token_type_ids,y):
#     This funciton will convert examples to FEATURES
    return {"input_ids": input_ids,
          "attention_mask": attention_masks,
          "token_type_ids": token_type_ids},y
BATCH_SIZE = 128
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_inp,train_mask,train_type_ids,
     train_out)).map(convert_to_features).shuffle(100).batch(BATCH_SIZE).repeat(5)
val_ds = tf.data.Dataset.from_tensor_slices(
    (val_inp, val_mask, val_type_ids, val_out)).map(convert_to_features).batch(BATCH_SIZE)
# Get and configure the BERT model
config = BertConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=0.3, num_labels=2)
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4, epsilon=0.0015, clipnorm=0.01)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
model.summary()
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
history = model.fit(train_ds, epochs=50, validation_data = val_ds, callbacks=[earlyStopping])
loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
def test_create_feature(input_ids,attention_masks,token_type_ids):
    return {"input_ids": input_ids,
          "attention_mask": attention_masks,
          "token_type_ids": token_type_ids}
inp_ids, attn_msks, tkn_ids = create_inputs(
    test_data[TEXT_FIELD], tokenizer)
inp_ds = test_create_feature(inp_ids, attn_msks, tkn_ids)
pre_result = model.predict(inp_ds)
res_dict = {}

# print(pre_result[0][0])

for i in range(len(pre_result[0])):
    res_dict[test_data['id'][i]] = np.argmax(pre_result[0][i])

print(res_dict)
submission_df = pd.DataFrame.from_dict(res_dict, orient="index")
submission_df.to_csv("submission.csv")
