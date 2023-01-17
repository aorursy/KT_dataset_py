!pip install torchvision
!pip uninstall -y transformers

!pip install transformers
import os

import re

import json

import string

import numpy as np

import torch

import pickle



import tensorflow as tf

from tensorflow import keras as K



from tokenizers import BertWordPieceTokenizer

from transformers import BertTokenizer, TFBertModel
# Detect hardware, return appropriate distribution strategy.

# You can see that it is pretty easy to set up.

try:

    # TPU detection: no parameters necessary if TPU_NAME environment

    # variable is set (always set in Kaggle)

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    print('Running on TPU ', tpu.master())

except ValueError:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print('Number of replicas:', strategy.num_replicas_in_sync)
DATASET_URL = 'https://raw.githubusercontent.com/deepmind/xquad/99910ec0f10151652f6726282ca922dd8eb0207a/xquad.en.json'

MODEL_NAME = 'bert-base-multilingual-cased'

MAX_LEN = 384



# Set language name to save model

LANGUAGE = 'english'



# Depends on whether we are using TPUs or not, increase BATCH_SIZE

BATCH_SIZE = 8 * strategy.num_replicas_in_sync



# Detect environment

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE',''):

    print('Detected Kaggle environment')

    ARTIFACTS_PATH = 'artifacts/'

else:

    ARTIFACTS_PATH = '../artifacts/'

    

if not os.path.exists(ARTIFACTS_PATH):

    os.makedirs(ARTIFACTS_PATH)
# Import tokenizer from HuggingFace

slow_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)



save_path = '%s%s-%s/' % (ARTIFACTS_PATH, LANGUAGE, MODEL_NAME)

if not os.path.exists(save_path):

    os.makedirs(save_path)



slow_tokenizer.save_pretrained(save_path)



# You can already use the Slow Tokenizer, but its implementation in Rust is much faster.

tokenizer = BertWordPieceTokenizer('%s/vocab.txt' % save_path, lowercase=True)
# This code is a modified version from https://keras.io/examples/nlp/text_extraction_with_bert/

class SquadExample:

    def __init__(

        self,

        question,

        context,

        start_char_idx,

        answer_text,

        all_answers,

        tokenizer

    ):

        self.question = question

        self.context = context

        self.start_char_idx = start_char_idx

        self.answer_text = answer_text

        self.all_answers = all_answers

        self.tokenizer = tokenizer

        self.skip = False



    def preprocess(self):

        context = self.context

        question = self.question

        answer_text = self.answer_text

        start_char_idx = self.start_char_idx



        # Fix white spaces

        context = re.sub(r"\s+", ' ', context).strip()

        question = re.sub(r"\s+", ' ', question).strip()

        answer = re.sub(r"\s+", ' ', answer_text).strip()



        # Find end token index of answer in context

        end_char_idx = start_char_idx + len(answer)

        if end_char_idx >= len(context):

            self.skip = True

            return



        # Mark the character indexes in context that are in answer

        is_char_in_ans = [0] * len(context)

        for idx in range(start_char_idx, end_char_idx):

            is_char_in_ans[idx] = 1



        # Encode context (token IDs, mask and token types)

        tokenized_context = tokenizer.encode(context)



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



        # Encode question (token IDs, mask and token types)

        tokenized_question = tokenizer.encode(question)



        # Create inputs

        input_ids = tokenized_context.ids + tokenized_question.ids[1:]

        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(

            tokenized_question.ids[1:]

        )

        attention_mask = [1] * len(input_ids)



        # Pad and create attention masks.

        # Skip if truncation is needed

        padding_length = MAX_LEN - len(input_ids)

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
def create_squad_examples(raw_data, tokenizer):

    squad_examples = []

    for item in raw_data["data"]:

        for para in item["paragraphs"]:

            context = para["context"]

            for qa in para["qas"]:

                question = qa["question"]

                answer_text = qa["answers"][0]["text"]

                all_answers = [_["text"] for _ in qa["answers"]]

                start_char_idx = qa["answers"][0]["answer_start"]

                squad_eg = SquadExample(

                    question,

                    context,

                    start_char_idx,

                    answer_text,

                    all_answers,

                    tokenizer

                )

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
dataset_path = K.utils.get_file('dataset.json', DATASET_URL)
with open(dataset_path) as fp:

    raw_data = json.load(fp)
# Split into train and test sets

raw_train_data = {}

raw_eval_data = {}

raw_train_data['data'], raw_eval_data['data'] = np.split(np.asarray(raw_data['data']), [int(.8*len(raw_data['data']))])
train_squad_examples = create_squad_examples(raw_train_data, tokenizer)

x_train, y_train = create_inputs_targets(train_squad_examples)

print(f"{len(train_squad_examples)} training points created.")



eval_squad_examples = create_squad_examples(raw_eval_data, tokenizer)

x_eval, y_eval = create_inputs_targets(eval_squad_examples)

print(f"{len(eval_squad_examples)} evaluation points created.")
def create_model():

    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_ids', dtype=tf.int32)

    token_type_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='token_type_ids', dtype=tf.int32)

    attention_mask = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype=tf.int32)

    

    encoder = TFBertModel.from_pretrained(MODEL_NAME)

    x = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    

    # Huggingface transformers have multiple outputs, embeddings are the first one,

    # so let's slice out the first position.

    x = x[0]



    # Define two outputs

    start_logits = tf.keras.layers.Dense(1, name='start_logit', use_bias=False)(x)

    start_logits = tf.keras.layers.Flatten()(start_logits)



    end_logits = tf.keras.layers.Dense(1, name='end_logit', use_bias=False)(x)

    end_logits = tf.keras.layers.Flatten()(end_logits)



    # Normalize outputs with softmax

    start_probs = tf.keras.layers.Activation(tf.keras.activations.softmax, name='start_probs')(start_logits)

    end_probs = tf.keras.layers.Activation(tf.keras.activations.softmax, name='end_probs')(end_logits)



    model = tf.keras.Model(

        inputs=[input_ids, token_type_ids, attention_mask],

        outputs=[start_probs, end_probs],

    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    optimizer = tf.keras.optimizers.Adam(lr=2e-5)

    model.compile(optimizer=optimizer, loss=[loss, loss])

    return model
with strategy.scope():

    model = create_model()

    model.summary()
# Source: https://keras.io/examples/nlp/text_extraction_with_bert/

class ExactMatch(tf.keras.callbacks.Callback):

    def __init__(self, x_eval, y_eval):

        self.x_eval = x_eval

        self.y_eval = y_eval



    def on_epoch_end(self, epoch, logs=None):

        pred_start, pred_end = self.model.predict(self.x_eval)

        count = 0

        eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip == False]

        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):

            squad_eg = eval_examples_no_skip[idx]

            offsets = squad_eg.context_token_to_char

            start = np.argmax(start)

            end = np.argmax(end)

            if start >= len(offsets):

                continue

            

            # Get answer from context text

            pred_char_start = offsets[start][0]

            if end < len(offsets):

                pred_char_end = offsets[end][1]

                pred_ans = squad_eg.context[pred_char_start:pred_char_end]

            else:

                pred_ans = squad_eg.context[pred_char_start:]



            # Normalize answers before comparing prediction and true answers

            normalized_pred_ans = self._normalize_text(pred_ans)

            normalized_true_ans = [self._normalize_text(_) for _ in squad_eg.all_answers]

            

            # If the prediction is contained in the true answer, it counts as a hit

            if normalized_pred_ans in normalized_true_ans:

                count += 1



        acc = count / len(self.y_eval[0])

        print(f"\nepoch={epoch+1}, exact match score={acc:.2f}")

    

    def _normalize_text(self, text):

        text = text.lower()



        # Remove punctuations

        exclude = set(string.punctuation)

        text = ''.join(ch for ch in text if ch not in exclude)



        # Remove articles

        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)

        text = re.sub(regex, ' ', text)



        # Remove extra white spaces

        text = re.sub(r"\s+", ' ', text).strip()



        return text
EPOCHS = 8



with strategy.scope():

    exact_match_callback = ExactMatch(x_eval, y_eval)

    model.fit(

        x_train,

        y_train,

        epochs=EPOCHS,

        verbose=1,

        batch_size=BATCH_SIZE,

        callbacks=[exact_match_callback],

    )
import pickle



weigh = model.get_weights()

pklfile = '%s%s-%s.pickle' % (ARTIFACTS_PATH, LANGUAGE, MODEL_NAME)



with open(pklfile, 'wb') as fp:

    pickle.dump(weigh, fp, protocol= pickle.HIGHEST_PROTOCOL)
pklfile = '%s%s-%s.pickle' % (ARTIFACTS_PATH, LANGUAGE, MODEL_NAME)

with open(pklfile, 'rb') as fp:

    data = pickle.load(fp)

    model.set_weights(data)
def get_answer_question(question, context, model, tokenizer):

    # Fix white spaces

    context = re.sub(r"\s+", ' ', context).strip()

    question = re.sub(r"\s+", ' ', question).strip()



    # Encode context (token IDs, mask and token types)

    tokenized_context = tokenizer.encode(context)

    tokenized_question = tokenizer.encode(question)



    # Create inputs

    input_ids = tokenized_context.ids + tokenized_question.ids[1:]

    token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(

        tokenized_question.ids[1:]

    )

    attention_mask = [1] * len(input_ids)



    # Pad and create attention masks.

    padding_length = MAX_LEN - len(input_ids)

    if padding_length > 0:

        input_ids = input_ids + ([0] * padding_length)

        attention_mask = attention_mask + ([0] * padding_length)

        token_type_ids = token_type_ids + ([0] * padding_length)

    elif padding_length < 0:

        raise Exception('Too long!')



    input_ids = np.asarray(input_ids, dtype='int32')

    token_type_ids = np.asarray(token_type_ids, dtype='int32')

    attention_mask = np.asarray(attention_mask, dtype='int32')

        

    encoded_input = [

        np.asarray([input_ids]),

        np.asarray([token_type_ids]),

        np.asarray([attention_mask])

    ]

    

    # Get prediction of answer for the given question and context.

    pred_start, pred_end = model.predict(encoded_input)

    

    start = np.argmax(pred_start[0])

    end = np.argmax(pred_end[0])

    

    offsets = tokenized_context.offsets

    if start >= len(offsets):

        print('Cannot capture answer.')



    pred_char_start = offsets[start][0]

    if end < len(offsets):

        pred_char_end = offsets[end][1]

        pred_ans = context[pred_char_start:pred_char_end]

    else:

        pred_ans = context[pred_char_start:]



    # Remove extra white spaces

    normalized_pred_ans = re.sub(r"\s+", ' ', pred_ans).strip()

    

    return normalized_pred_ans
# Write a context and a question about the context

context = "The Basílica de la Sagrada Familia is a large unfinished Roman Catholic minor basilica in the Eixample district of Barcelona, Spain. Designed by Spanish architect Antoni Gaudí (1852-1926), his work on the building is part of a UNESCO World Heritage Site. On 7 November 2010, Pope Benedict XVI consecrated the church and proclaimed it a minor basilica."

question = 'Who designed the Sagrada Familia?'



# Import textwrap library to display context

import textwrap

wrapper = textwrap.TextWrapper(width=80) 



# Display

print('='*6, ' TEXT ', '='*6)

print(wrapper.fill(context))

print('='*21)



print('='*5, 'QUESTION ', '='*5)

print(question)

print('='*21)



# Infer answer

answer = get_answer_question(question, context, model, tokenizer)

print('='*5, ' ANSWER ', '='*6)

print(answer)