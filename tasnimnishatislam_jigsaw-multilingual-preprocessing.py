!pip install -q gcsfs
import os, time

import pandas

import tensorflow as tf

import tensorflow_hub as hub

from kaggle_datasets import  KaggleDatasets 

import warnings

warnings.filterwarnings('ignore')





# We'll use a tokenizer for the BERT model from the modelling demo notebook.

!pip install bert-tensorflow

import bert.tokenization



print(tf.version.VERSION)
#select tpu

# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
SEQUENCE_LENGTH = 128



# Note that private datasets cannot be copied - you'll have to share any pretrained models 

# you want to use with other competitors!

GCS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')

BERT_GCS_PATH = KaggleDatasets().get_gcs_path('bert-multi')

BERT_GCS_PATH_SAVEDMODEL = BERT_GCS_PATH + "/bert_multi_from_tfhub"
def multilingual_bert_model(max_seq_length=SEQUENCE_LENGTH, trainable_bert=True):

    """Build and return a multilingual BERT model and tokenizer."""

    input_word_ids = tf.keras.layers.Input(

        shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")

    input_mask = tf.keras.layers.Input(

        shape=(max_seq_length,), dtype=tf.int32, name="input_mask")

    segment_ids = tf.keras.layers.Input(

        shape=(max_seq_length,), dtype=tf.int32, name="all_segment_id")

    

    # Load a SavedModel on TPU from GCS. This model is available online at 

    # https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1. You can use your own 

    # pretrained models, but will need to add them as a Kaggle dataset.

    bert_layer = tf.saved_model.load(BERT_GCS_PATH_SAVEDMODEL)

    # Cast the loaded model to a TFHub KerasLayer.

    bert_layer = hub.KerasLayer(bert_layer, trainable=trainable_bert)



    pooled_output, _ = bert_layer([input_word_ids, input_mask, segment_ids])

    output = tf.keras.layers.Dense(32)(pooled_output)

    output = tf.keras.layers.LeakyReLU()(output)

    output = tf.keras.layers.Dense(1, activation='sigmoid', name='labels')(output)



    return tf.keras.Model(inputs={'input_word_ids': input_word_ids,

                                  'input_mask': input_mask,

                                  'all_segment_id': segment_ids},

                          outputs=output)
# Training data from our first competition,

# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

train_name = GCS_PATH + "/jigsaw-toxic-comment-train-processed-seqlen128.csv"

test_name = GCS_PATH + "/test.csv"

validation_name = GCS_PATH + "/validation.csv"
train = pandas.read_csv(train_name)

train.head()
test = pandas.read_csv(test_name, encoding= 'utf-8')

test.head()
validation = pandas.read_csv(validation_name, encoding = 'utf')

validation.head()
def get_tokenizer(bert_path=BERT_GCS_PATH_SAVEDMODEL):

    """Get the tokenizer for a BERT layer."""

    bert_layer = tf.saved_model.load(bert_path)

    bert_layer = hub.KerasLayer(bert_layer, trainable = False)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

    cased = bert_layer.resolved_object.do_lower_case.numpy()

    tf.gfile = tf.io.gfile  # for bert.tokenization.load_vocab in tokenizer

    tokenizer = bert.tokenization.FullTokenizer(vocab_file, cased)

  

    return tokenizer



tokenizer = get_tokenizer()
example_sentence = train.iloc[37].comment_text[:150]

print(example_sentence)



example_tokens = tokenizer.tokenize(example_sentence)

print(example_tokens[:17])



example_input_ids = tokenizer.convert_tokens_to_ids(example_tokens)

print(example_input_ids[:17])
def process_sentence(sentence, max_seq_length=SEQUENCE_LENGTH, tokenizer=tokenizer):

    """Helper function to prepare data for BERT. Converts sentence input examples

    into the form ['input_word_ids', 'input_mask', 'segment_ids']."""

    # Tokenize, and truncate to max_seq_length if necessary.

    tokens = tokenizer.tokenize(sentence)

    if len(tokens) > max_seq_length - 2:

        tokens = tokens[:(max_seq_length - 2)]



    # Convert the tokens in the sentence to word IDs.

    input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])



    # The mask has 1 for real tokens and 0 for padding tokens. Only real

    # tokens are attended to.

    input_mask = [1] * len(input_ids)



    # Zero-pad up to the sequence length.

    pad_length = max_seq_length - len(input_ids)

    input_ids.extend([0] * pad_length)

    input_mask.extend([0] * pad_length)



    # We only have one input segment.

    segment_ids = [0] * max_seq_length



    return (input_ids, input_mask, segment_ids)



def preprocess_and_save_dataset(unprocessed_filename, text_label='comment_text',

                                seq_length=SEQUENCE_LENGTH, verbose=True):

    """Preprocess a CSV to the expected TF Dataset form for multilingual BERT,

    and save the result."""

    dataframe = pandas.read_csv(unprocessed_filename,

                                index_col='id', encoding = 'utf-8')

    processed_filename = ("unprocessed_filename.rstrip('.csv')" +

                          "-processed-seqlen{}.csv".format(SEQUENCE_LENGTH))



    pos = 0

    start = time.time()



    while pos < len(dataframe):

        processed_df = dataframe[pos:pos + 10000].copy() 

        processed_df['input_word_ids'], processed_df['input_mask'], processed_df['all_segment_id'] = (

            zip(*processed_df[text_label].apply(process_sentence)))



        if pos == 0:

            processed_df.to_csv(processed_filename, index_label='id', mode='w')

        else:

             processed_df.to_csv(processed_filename, index_label='id', mode='a',

                                header=False)



        if verbose:

            print('Processed {} examples in {}'.format(

                pos + 10000, time.time() - start))

        pos += 10000

    return

  

# Process the training dataset.

preprocess_and_save_dataset(train_name, text_label='comment_text',

                                seq_length=SEQUENCE_LENGTH, verbose=True)
def parse_string_list_into_ints(strlist):

    s = tf.strings.strip(strlist)

    s = tf.strings.substr(

        strlist, 1, tf.strings.length(s) - 2)  # Remove parentheses around list

    s = tf.strings.split(s, ',', maxsplit=SEQUENCE_LENGTH)

    s = tf.strings.to_number(s, tf.int32)

    s = tf.reshape(s, [SEQUENCE_LENGTH])  # Force shape here needed for XLA compilation (TPU)

    return s



def format_sentences(data, label='toxic', remove_language=False):

    labels = {'labels': data.pop(label)}

    if remove_language:

        languages = {'language': data.pop('lang')}

    # The remaining three items in the dict parsed from the CSV are lists of integers

    for k,v in data.items():  # "input_word_ids", "input_mask", "all_segment_id"

        data[k] = parse_string_list_into_ints(v)

    return data, labels



def make_sentence_dataset_from_csv(filename, label='toxic', language_to_filter=None):

    # This assumes the column order label, input_word_ids, input_mask, segment_ids

    SELECTED_COLUMNS = [label, "input_word_ids", "input_mask", "all_segment_id"]

    label_default = tf.int32 if label == 'id' else tf.float32

    COLUMN_DEFAULTS  = [label_default, tf.string, tf.string, tf.string]



    if language_to_filter:

        insert_pos = 0 if label != 'id' else 1

        SELECTED_COLUMNS.insert(insert_pos, 'lang')

        COLUMN_DEFAULTS.insert(insert_pos, tf.string)



    preprocessed_sentences_dataset = tf.data.experimental.make_csv_dataset(

        filename, column_defaults=COLUMN_DEFAULTS, select_columns=SELECTED_COLUMNS,

        batch_size=1, num_epochs=1, shuffle=False)  # We'll do repeating and shuffling ourselves

    # make_csv_dataset required a batch size, but we want to batch later

    preprocessed_sentences_dataset = preprocessed_sentences_dataset.unbatch()

    

    if language_to_filter:

        preprocessed_sentences_dataset = preprocessed_sentences_dataset.filter(

            lambda data: tf.math.equal(data['lang'], tf.constant(language_to_filter)))

        #preprocessed_sentences.pop('lang')

    preprocessed_sentences_dataset = preprocessed_sentences_dataset.map(

        lambda data: format_sentences(data, label=label,

                                      remove_language=language_to_filter))



    return preprocessed_sentences_dataset
def make_dataset_pipeline(dataset, repeat_and_shuffle=True):

    """Set up the pipeline for the given dataset.

    

    Caches, repeats, shuffles, and sets the pipeline up to prefetch batches."""

    cached_dataset = dataset.cache()

    if repeat_and_shuffle:

        cached_dataset = cached_dataset.repeat().shuffle(2048)

    cached_dataset = cached_dataset.batch(32 * strategy.num_replicas_in_sync)

    cached_dataset = cached_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return cached_dataset



# Load the preprocessed English dataframe.

preprocessed_en_filename = (

    GCS_PATH + "/jigsaw-toxic-comment-train-processed-seqlen{}.csv".format(

        SEQUENCE_LENGTH))



# Set up the dataset and pipeline.

english_train_dataset = make_dataset_pipeline(

    make_sentence_dataset_from_csv(preprocessed_en_filename))



# Process the new datasets by language.

preprocessed_val_filename = (

    GCS_PATH + "/validation-processed-seqlen{}.csv".format(SEQUENCE_LENGTH))



nonenglish_val_datasets = {}

for language_name, language_label in [('Spanish', 'es'), ('Italian', 'it'),

                                      ('Turkish', 'tr')]:

    nonenglish_val_datasets[language_name] = make_sentence_dataset_from_csv(

        preprocessed_val_filename, language_to_filter=language_label)

    nonenglish_val_datasets[language_name] = make_dataset_pipeline(

        nonenglish_val_datasets[language_name])



nonenglish_val_datasets['Combined'] = tf.data.experimental.sample_from_datasets(

        (nonenglish_val_datasets['Spanish'], nonenglish_val_datasets['Italian'],

         nonenglish_val_datasets['Turkish']))
with strategy.scope():

    multilingual_bert = multilingual_bert_model()



    # Compile the model. Optimize using stochastic gradient descent.

    multilingual_bert.compile(

        loss=tf.keras.losses.BinaryCrossentropy(),

        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),

        metrics=[tf.keras.metrics.AUC()])



multilingual_bert.summary()
# Test the model's performance on non-English comments before training.

for language in nonenglish_val_datasets:

    results = multilingual_bert.evaluate(nonenglish_val_datasets[language],

                                         steps=100, verbose=0)

    print('{} loss, AUC before training:'.format(language), results)



results = multilingual_bert.evaluate(english_train_dataset,

                                     steps=100, verbose=0)

print('\nEnglish loss, AUC before training:', results)



print()

# Train on English Wikipedia comment data.

history = multilingual_bert.fit(

    # Set steps such that the number of examples per epoch is fixed.

    # This makes training on different accelerators more comparable.

    english_train_dataset, steps_per_epoch=4000/strategy.num_replicas_in_sync,

    epochs=100, verbose=1, validation_data=nonenglish_val_datasets['Combined'],

    validation_steps=500)

print()



# Re-evaluate the model's performance on non-English comments after training.

for language in nonenglish_val_datasets:

    results = multilingual_bert.evaluate(nonenglish_val_datasets[language],

                                         steps=100, verbose=0)

    print('{} loss, AUC after training:'.format(language), results)



results = multilingual_bert.evaluate(english_train_dataset,

                                     steps=100, verbose=0)

print('\nEnglish loss, AUC after training:', results)
import numpy as np



TEST_DATASET_SIZE = 63812



print('Making dataset...')

preprocessed_test_filename = (

    GCS_PATH + "/test-processed-seqlen{}.csv".format(SEQUENCE_LENGTH))

test_dataset = make_sentence_dataset_from_csv(preprocessed_test_filename, label='id')

test_dataset = make_dataset_pipeline(test_dataset, repeat_and_shuffle=False) 

    

print('Computing predictions...')

test_sentences_dataset = test_dataset.map(lambda sentence, idnum: sentence)

probabilities = np.squeeze(multilingual_bert.predict(test_sentences_dataset))

print(probabilities)



print('Generating submission file...')

test_ids_dataset = test_dataset.map(lambda sentence, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_dataset.batch(TEST_DATASET_SIZE)))[

    'labels'].numpy().astype('U')  # All in one batch



np.savetxt('submission.csv', np.rec.fromarrays([test_ids, probabilities]),

           fmt=['%s', '%f'], delimiter=',', header='id,toxic', comments='')

!head submission.csv
