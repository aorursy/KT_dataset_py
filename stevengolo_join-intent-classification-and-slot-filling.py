# Load packages

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf



from pathlib import Path

from transformers import BertTokenizer, TFBertModel

from urllib.request import urlretrieve



from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.metrics import SparseCategoricalAccuracy

from tensorflow.keras.optimizers import Adam
SNIPS_DATA_BASE_URL = (

    "https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/"

    "master/data/snips/"

)

for filename in ["train", "valid", "test", "vocab.intent", "vocab.slot"]:

    path = Path(filename)

    if not path.exists():

        print(f"Downloading {filename}...")

        urlretrieve(SNIPS_DATA_BASE_URL + filename + "?raw=true", path)
lines_train = Path('train').read_text('utf-8').strip().splitlines()
print(f'First line of training set: {lines_train[0]}.')
def parse_line(line):

    utterance_data, intent_label = line.split(" <=> ")

    items = utterance_data.split()

    words = [item.rsplit(':', 1)[0] for item in items]

    word_labels = [item.rsplit(':', 1)[1] for item in items]

    return {

        'intent_label': intent_label,

        'words': " ".join(words),

        'words_label': " ".join(word_labels),

        'length': len(words)

    }
parse_line(lines_train[0])
print(Path('vocab.intent').read_text('utf-8'))
print(Path('vocab.slot').read_text('utf-8'))
parsed = [parse_line(line) for line in lines_train]

df_train = pd.DataFrame([p for p in parsed if p is not None])
# Print some lines of the training set

df_train.head(5)
# Count the number of lines by intent label

df_train.intent_label.value_counts()
# Histogram of sentence lengths

df_train.hist('length', bins=30)
# Get validation and test set

lines_validation = Path('valid').read_text('utf-8').strip().splitlines()

lines_test = Path('test').read_text('utf-8').strip().splitlines()



df_validation = pd.DataFrame([parse_line(line) for line in lines_validation])

df_test = pd.DataFrame([parse_line(line) for line in lines_test])
model_name = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(model_name)
first_sentence = df_train.iloc[0]['words']

print(first_sentence)
tokenizer.tokenize(first_sentence)
# Encode sentence to id

tokenizer.encode(first_sentence)
# Do the inverse operation

tokenizer.decode(tokenizer.encode(first_sentence))
train_sequence_lengths = [len(tokenizer.encode(text))

                          for text in df_train['words']]

plt.hist(train_sequence_lengths, bins=30)

plt.title(f'Max sequence length: {max(train_sequence_lengths)}')

plt.xlabel('Length')

plt.ylabel('Count')

plt.show()
print(f'Vocabulary size: {tokenizer.vocab_size} words.')
# Get the items in BERT

bert_vocab_items = list(tokenizer.vocab.items())
# Print some examples of items

bert_vocab_items[250:260]
def encode_dataset(tokenizer, text_sequences, max_length):

    token_ids = np.zeros(shape=(len(text_sequences), max_length),

                         dtype=np.int32)

    for i, text_sequence in enumerate(text_sequences):

        encoded = tokenizer.encode(text_sequence)

        token_ids[i, 0:len(encoded)] = encoded

    attention_masks = (token_ids != 0).astype(np.int32)

    

    return {'input_ids': token_ids, 'attention_masks': attention_masks}
encoded_train = encode_dataset(tokenizer, df_train['words'], 45)

encoded_validation = encode_dataset(tokenizer, df_validation['words'], 45)

encoded_test = encode_dataset(tokenizer, df_test['words'], 45)
encoded_train['input_ids']
encoded_train['attention_masks']
intent_names = Path('vocab.intent').read_text('utf-8').split()

intent_map = dict((label, idx) for idx, label in enumerate(intent_names))
intent_map
intent_train = df_train['intent_label'].map(intent_map).values

intent_validation = df_validation['intent_label'].map(intent_map).values

intent_test = df_test['intent_label'].map(intent_map).values
base_bert_model = TFBertModel.from_pretrained('bert-base-cased')

base_bert_model.summary()
outputs = base_bert_model(encoded_validation)
print(f'Shape of the first output of the BERT model: {outputs[0].shape}.')
print(f'Shape of the second output of the BERT model: {outputs[1].shape}.')
# Define IntentClassification model

class IntentClassificationModel(tf.keras.Model):

    def __init__(self, intent_num_labels=None,

                 model_name='bert-base-cased',

                 dropout_prob=0.1):

        super().__init__(name='joint_intent_slot')

        # Let's preload the pretrained model BERT in the constructor

        # of our classifier model.

        self.bert = TFBertModel.from_pretrained(model_name)

        self.dropout = Dropout(dropout_prob)

        

        # Define a (Dense) classification layer to compute for each

        # sequence in a batch of samples. The number of output classes

        # is given by the intent_num_labels parameter.

        # Use the default linear activation (no softmax) to compute

        # logits. The softmax normalization will be computed in the

        # loss function instead of the model itself.

        self.intent_classifier = Dense(intent_num_labels)

        

    def call(self, inputs, **kwargs):

        # Use the pretrained model to extract features from our

        # encoded inputs.

        sequence_output, pooled_output = self.bert(inputs, **kwargs)

        

        # The second output of the main BERT layer has shape:

        # (batch_size, output_dim) and gives a "pooled" representation

        # for the full sequence from the hidden state that corresponds

        # to the "[CLS]" token.

        pooled_output = self.dropout(pooled_output, training=kwargs.get('training', False))

        

        # Use the classifier layer to compute the logits from the

        # pooled features.

        intent_logits = self.intent_classifier(pooled_output)

        return intent_logits
# Build the model

intent_model = IntentClassificationModel(intent_num_labels=len(intent_map))



intent_model.compile(optimizer=Adam(learning_rate=3e-5, epsilon=1e-08),

                     loss=SparseCategoricalCrossentropy(from_logits=True),

                     metrics=[SparseCategoricalAccuracy('accuracy')])
# Train the model

history = intent_model.fit(encoded_train, intent_train,

                           epochs=2, batch_size=32,

                           validation_data=(encoded_validation, intent_validation))
def classify(text, tokenizerzer, model, intent_names):

    inputs = tf.constant(tokenizer.encode(text))[None, :] # Batch size = 1

    class_id = model(inputs).numpy().argmax(axis=1)[0]

    return intent_names[class_id]
# Example of classification

classify('Will it snow tomorrow in Paris?',

         tokenizer, intent_model, intent_names)
slot_names = ["[PAD]"]

slot_names += Path('vocab.slot').read_text('utf-8').strip().splitlines()



slot_map = {}

for label in slot_names:

    slot_map[label] = len(slot_map)
def encode_token_labels(text_sequences, slot_names, tokenizer, slot_map, max_length):

    encoded = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)

    for i, (text_sequence, word_labels) in enumerate(

            zip(text_sequences, slot_names)):

        encoded_labels = []

        for word, word_label in zip(text_sequence.split(), word_labels.split()):

            tokens = tokenizer.tokenize(word)

            encoded_labels.append(slot_map[word_label])

            expand_label = word_label.replace("B-", "I-")

            if not expand_label in slot_map:

                expand_label = word_label

            encoded_labels.extend([slot_map[expand_label]] * (len(tokens) - 1))

        encoded[i, 1:len(encoded_labels) + 1] = encoded_labels

    return encoded
slot_train = encode_token_labels(df_train['words'], df_train['words_label'], tokenizer, slot_map, 45)

slot_validation = encode_token_labels(df_validation['words'], df_validation['words_label'], tokenizer, slot_map, 45)

slot_test = encode_token_labels(df_test['words'], df_test['words_label'], tokenizer, slot_map, 45)
# Define JointIntentAndSlotFilling model

class JointIntentAndSlotFillingModel(tf.keras.Model):



    def __init__(self, intent_num_labels=None, slot_num_labels=None,

                 model_name="bert-base-cased", dropout_prob=0.1):

        super().__init__(name="joint_intent_slot")

        self.bert = TFBertModel.from_pretrained(model_name)

        self.dropout = Dropout(dropout_prob)

        self.intent_classifier = Dense(intent_num_labels,

                                       name="intent_classifier")

        self.slot_classifier = Dense(slot_num_labels,

                                     name="slot_classifier")



    def call(self, inputs, **kwargs):

        sequence_output, pooled_output = self.bert(inputs, **kwargs)



        # The first output of the main BERT layer has shape:

        # (batch_size, max_length, output_dim)

        sequence_output = self.dropout(sequence_output,

                                       training=kwargs.get("training", False))

        slot_logits = self.slot_classifier(sequence_output)



        # The second output of the main BERT layer has shape:

        # (batch_size, output_dim)

        # and gives a "pooled" representation for the full sequence from the

        # hidden state that corresponds to the "[CLS]" token.

        pooled_output = self.dropout(pooled_output,

                                     training=kwargs.get("training", False))

        intent_logits = self.intent_classifier(pooled_output)



        return slot_logits, intent_logits
joint_model = JointIntentAndSlotFillingModel(

    intent_num_labels=len(intent_map), slot_num_labels=len(slot_map))



# Define one classification loss for each output:

opt = Adam(learning_rate=3e-5, epsilon=1e-08)

losses = [SparseCategoricalCrossentropy(from_logits=True),

          SparseCategoricalCrossentropy(from_logits=True)]

metrics = [SparseCategoricalAccuracy('accuracy')]

joint_model.compile(optimizer=opt, loss=losses, metrics=metrics)
history = joint_model.fit(

    encoded_train, (slot_train, intent_train),

    validation_data=(encoded_validation, (slot_validation, intent_validation)),

    epochs=2, batch_size=32)
def show_predictions(text, tokenizer, model, intent_names, slot_names):

    inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1

    outputs = model(inputs)

    slot_logits, intent_logits = outputs

    slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]

    intent_id = intent_logits.numpy().argmax(axis=-1)[0]

    print("## Intent:", intent_names[intent_id])

    print("## Slots:")

    for token, slot_id in zip(tokenizer.tokenize(text), slot_ids):

        print(f"{token:>10} : {slot_names[slot_id]}")
# Example of classification

show_predictions('Will it snow tomorrow in Paris?',

                 tokenizer, joint_model, intent_names, slot_names)
def decode_predictions(text, tokenizer, intent_names, slot_names,

                       intent_id, slot_ids):

    info = {"intent": intent_names[intent_id]}

    collected_slots = {}

    active_slot_words = []

    active_slot_name = None

    for word in text.split():

        tokens = tokenizer.tokenize(word)

        current_word_slot_ids = slot_ids[:len(tokens)]

        slot_ids = slot_ids[len(tokens):]

        current_word_slot_name = slot_names[current_word_slot_ids[0]]

        if current_word_slot_name == "O":

            if active_slot_name:

                collected_slots[active_slot_name] = " ".join(active_slot_words)

                active_slot_words = []

                active_slot_name = None

        else:

            # Naive BIO: handling: treat B- and I- the same...

            new_slot_name = current_word_slot_name[2:]

            if active_slot_name is None:

                active_slot_words.append(word)

                active_slot_name = new_slot_name

            elif new_slot_name == active_slot_name:

                active_slot_words.append(word)

            else:

                collected_slots[active_slot_name] = " ".join(active_slot_words)

                active_slot_words = [word]

                active_slot_name = new_slot_name

    if active_slot_name:

        collected_slots[active_slot_name] = " ".join(active_slot_words)

    info["slots"] = collected_slots

    return info
def nlu(text, tokenizer, model, intent_names, slot_names):

    inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1

    outputs = model(inputs)

    slot_logits, intent_logits = outputs

    slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]

    intent_id = intent_logits.numpy().argmax(axis=-1)[0]



    return decode_predictions(text, tokenizer, intent_names, slot_names,

                              intent_id, slot_ids)
nlu('Will it snow tomorrow in Paris?',

                 tokenizer, joint_model, intent_names, slot_names)
nlu('I would like to listen to Wake me up by Avicii.',

    tokenizer, joint_model, intent_names, slot_names)