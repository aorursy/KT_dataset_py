# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#https://keras.io/examples/nlp/text_extraction_with_bert/#train-and-evaluate
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig
from kaggle_datasets import KaggleDatasets
max_len = 300
configuration = BertConfig.from_pretrained(f'/kaggle/input/bert-tensorflow/bert-base-uncased-config.json')
# Save the slow pretrained tokenizer
slow_tokenizer = BertTokenizer.from_pretrained(f'/kaggle/input/bertbaseuncased/vocab.txt')
save_path = "bert-base-uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer(f'/kaggle/input/bertbaseuncased/vocab.txt', lowercase=True)
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_df = pd.read_csv(f'/kaggle/input/tweet-sentiment-extraction/train.csv')
test_df = pd.read_csv(f'/kaggle/input/tweet-sentiment-extraction/train.csv')
train_df.head(10)
train_df['text'] = train_df['text'].str.strip()
train_df['selected_text'] = train_df['selected_text'].str.strip()
train_df.shape
class SquadExample:
    def __init__(self, text, sentiment, selected_text, start_char_idx):
        self.text = text
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.start_char_idx = start_char_idx
        self.skip = False
        
    def preprocess(self):
        
        text = self.text
        sentiment = self.sentiment
        selected_text = self.selected_text
        start_char_idx = self.start_char_idx

        text = str(text)
        sentiment = str(sentiment)
        selected_text = str(selected_text)
        
        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(str(selected_text))
        
                
        if end_char_idx > len(text):
            self.skip = True
            return
        
        
        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(text)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1
            
        # Tokenize context
        tokenized_context = tokenizer.encode(text)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)
        
        if len(ans_token_idx) == 0:
            self.skip = True
            return
        
        
        # Find start and end token index for tokens from answer
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]

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
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.context_token_to_char = tokenized_context.offsets

def create_examples(raw_data):
    squad_examples = []
    for i in range(raw_data.shape[0]):
        text = raw_data["text"][i]
        sentiment = raw_data["sentiment"][i]
        selected_text = raw_data["selected_text"][i]
        start_char_idx = str(raw_data["text"][i]).find(str(raw_data["selected_text"][i]))
        squad_eg = SquadExample(text, sentiment, selected_text, start_char_idx)
        squad_eg.preprocess()
        squad_examples.append(squad_eg)
        
    return squad_examples

def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y
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
train_examples = create_examples(train_df)
print(f"{len(train_examples)} training points created.")
x_train, y_train = create_inputs_targets(train_examples)
len(y_train[0])
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

"""
use_tpu = True
if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = create_model(base_path)
else:
    model = create_model(base_path)

model.summary()
"""
model = create_model(base_path)

model.summary()
#save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
#model.save('./model', options=save_locally) # saving in Tensorflow's "saved model" format
model.fit(
    x_train,
    y_train,
    epochs=5, 
    verbose=2,
    batch_size=16
)
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
test_df['text'] = test_df['text'].str.strip()
test_df.head(5)
test_examples = create_test(test_df)
x_test = create_inputs_test(test_examples)
pred2 = model.predict(x_test)
ans = [ans_convert(pred2[0][i],pred2[1][i],x_test[0][i],slow_tokenizer) for i in range(len(x_test[0]))] 
test_sub = test_df.copy()
test_sub['selected_text'] = ans
test_sub = test_sub[['textID','selected_text']]
test_sub.to_csv('submission.csv',index = False)
test_sub.head(5)
model.save_weights('bert_model_weights.h5')
