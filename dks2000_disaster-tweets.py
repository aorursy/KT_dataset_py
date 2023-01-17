!pip3 install bert-for-tf2
!pip3 install sentencepiece
!python3 -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk as nlp
from nltk.corpus import stopwords
import string
import tensorflow as tf
import bert

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,LSTM,Dropout
import tensorflow_hub as tfhub
import tensorflow_datasets as tfds
from datetime import datetime
bert_layer = tfhub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=True)
Train = pd.read_csv('train.csv')
Test = pd.read_csv('test.csv')

# Removing Non-Alphabet Characters
def remove_non_alphabet(x):
    return ' '.join([i for i in x.split() if i.isalpha() == True])

# Lowering Words
def lowerwords(text):
	text = re.sub("[^a-zA-Z]"," ",text) # Excluding Numbers
	text = [word.lower() for word in text.split()]
    # joining the list of words with space separator
	return " ".join(text)


# Removing Punctuation
def remove_punctuation(text):
    '''a function for removing punctuation'''
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)


# Removing StopWords
def remove_stopwords(text):
    StopWords = set(stopwords.words('english'))
    output = ' '.join([i for i in text.split() if i not in StopWords])
    return output


def remove_urls(text):
    text = re.sub(r'ttps?://\S+|www\.\S+<.*?>', '', text, flags=re.MULTILINE)
    return text


# Lemmatizer
def Lemmatizing(description):
    description = nlp.word_tokenize(description)
    #description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    
    return description

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
Train['text'] = Train['text'].apply(remove_urls)
Train['text'] = Train['text'].apply(remove_emoji)
Train['text'] = Train['text'].apply(remove_punctuation)
Train['text'] = Train['text'].apply(remove_non_alphabet)
Train['text'] = Train['text'].apply(lowerwords)
Train['text'] = Train['text'].apply(Lemmatizing)
Train['text'] = Train['text'].apply(remove_stopwords)

Test['text'] = Test['text'].apply(remove_urls)
Test['text'] = Test['text'].apply(remove_emoji)
Test['text'] = Test['text'].apply(remove_punctuation)
Test['text'] = Test['text'].apply(remove_non_alphabet)
Test['text'] = Test['text'].apply(lowerwords)
Test['text'] = Test['text'].apply(Lemmatizing)
Test['text'] = Test['text'].apply(remove_stopwords)


X_Train = Train['text']
y_Labels = Train['target']
X_Test = Test['text']
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = tfhub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
def get_masks(tokens,max_seq_length):
    """
    This Function Trims/ Pads a depending on length of token
    """
    if len(tokens)>max_seq_length:
        # Cutting Down the Excess Length
        tokens = tokens[0:max_seq_length]
        return [1]*len(tokens)
    else :
        return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    
    if len(tokens)>max_seq_length:
        # Cutting Down the Excess Length
        tokens = tokens[:max_seq_length]
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments
    
    else:
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):    
    if len(tokens)>max_seq_length:
        tokens = tokens[:max_seq_length]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        return token_ids
    else:
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
        return input_ids
def CreatingData(X_Train,tokenizer,max_seq_length=150):
    
    X_IDs = []
    X_Masks = []
    X_Segments = []

    for i in range(X_Train.shape[0]):
        x = X_Train[i]
        x = tokenizer.tokenize(x)
        x = ["[CLS]"] + x + ["[SEP]"]

        X_IDs.append(get_ids(x, tokenizer, max_seq_length))
        X_Masks.append(get_masks(x,max_seq_length))
        X_Segments.append(get_segments(x, max_seq_length))

    return np.array(X_IDs), np.array(X_Masks), np.array(X_Segments)

X_Train_IDs, X_Train_Masks, X_Train_Segments = CreatingData(X_Train,tokenizer)
X_Test_IDs, X_Test_Masks, X_Test_Segments = CreatingData(X_Test,tokenizer)
print (X_Train_IDs.shape)
print (X_Test_IDs.shape)
def Build_Model(bert_layer=bert_layer,Max_Seq_Length=150):
    IDs = Input(shape=(Max_Seq_Length,), dtype=tf.int32)
    Masks = Input(shape=(Max_Seq_Length,), dtype=tf.int32)
    Segments = Input(shape=(Max_Seq_Length,), dtype=tf.int32)

    Pooled_Output, Sequence_Output = bert_layer([IDs,Masks,Segments])

    x = Sequence_Output[:,0,:]
    x = Dropout(0.2)(x)
    Outputs = Dense(1,activation="sigmoid")(x)

    return Model(inputs=[IDs,Masks,Segments],outputs=Outputs)

Model = Build_Model()
Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','AUC'])
Model.summary()
Model.fit([X_Train_IDs, X_Train_Masks, X_Train_Segments], y_Labels, epochs=25, batch_size=128, validation_split=0.1)
Predictions = np.array(Model.predict([X_Test_IDs, X_Test_Masks, X_Test_Segments]))
Predictions = np.round(Predictions.flatten()).astype(int)

submission = pd.read_csv('sample_submission.csv')
submission['target'] = Predictions
submission.to_csv('./submission.csv', index=False, header=True)
