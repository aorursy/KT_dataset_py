# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import re

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig
max_len = 100

configuration = BertConfig.from_pretrained(f'/kaggle/input/bert-tensorflow/bert-base-uncased-config.json')
# Save the slow pretrained tokenizer
slow_tokenizer = BertTokenizer.from_pretrained('/kaggle/input/bertbaseuncased/vocab.txt')
save_path = "bert-base-uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer('/kaggle/input/bertbaseuncased/vocab.txt', lowercase=True)
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
class SquadTest:
    def __init__(self, text, sentiment):
        self.text = text
        self.sentiment = sentiment
        
    def preprocess(self):        
        text = self.text
        sentiment = self.sentiment

        text = str(text)
        sentiment = str(sentiment)
                  
        # Tokenize context
        tokenized_context = tokenizer.encode(text)       
        
        # Tokenize question
        tokenized_question = tokenizer.encode(sentiment)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return


        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask

def create_test(raw_data):
    squad_examples = []
    for i in range(raw_data.shape[0]):
        text = raw_data["text"][i]
        sentiment = raw_data["sentiment"][i] 
        squad_eg = SquadTest(text, sentiment)
            
        squad_eg.preprocess()
        squad_examples.append(squad_eg)
        
    return squad_examples

def create_inputs_test(squad_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
    }
    for item in squad_examples:
        for key in dataset_dict:
            dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    return x
base_path = '/kaggle/input/bert-tensorflow/bert-base-uncased-tf_model.h5'

def create_model(path = base_path):
    ## BERT encoder
    encoder = TFBertModel.from_pretrained(path, config=configuration)

    ## QA Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]

    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(keras.activations.softmax)(end_logits)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=5e-5)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model
def ans_convert(start_scores, end_scores, input_ids, tokenizer):
     # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = np.argmax(start_scores)
    answer_end = np.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]
            
    #print(str(answer_start), " - ", str(answer_end))
    return answer
test_df = pd.read_csv(f'/kaggle/input/tweet-sentiment-extraction/test.csv')
test_df['text'] = test_df['text'].str.strip()
test_df.head()
test_examples = create_test(test_df)
x_test = create_inputs_test(test_examples)
model = create_model(base_path)
model.load_weights(f'/kaggle/input/bert-model-weights/bert_model_weights.h5')
pred2 = model.predict(x_test, batch_size=16)
ans = [ans_convert(pred2[0][i],pred2[1][i],x_test[0][i],slow_tokenizer) for i in range(len(x_test[0]))] 
test_sub = test_df.copy()
test_sub['selected_text'] = ans
test_sub = test_sub[['textID','selected_text']]
def replace_blank (text):
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    for i in text:  
        if i in punc:  
            text = text.replace(" " +i, i)  
    
    return text
test_sub['selected_text'] = test_sub['selected_text'].apply(replace_blank)
test_sub['selected_text'] = '"' + test_sub['selected_text']+'"'
test_sub.head(50)
test_sub.to_csv('submission.csv',index = False)
