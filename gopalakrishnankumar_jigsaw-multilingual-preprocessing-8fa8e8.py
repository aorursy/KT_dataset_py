import os, time

import pandas

import tensorflow as tf

import tensorflow_hub as hub

from kaggle_datasets import KaggleDatasets



# We'll use a tokenizer for the BERT model from the modelling demo notebook.

!pip install bert-tensorflow

import bert.tokenization



print(tf.version.VERSION)
SEQUENCE_LENGTH = 128



DATA_PATH =  "../input/jigsaw-multilingual-toxic-comment-classification"

BERT_PATH = "../input/bert-multi"

BERT_PATH_SAVEDMODEL = os.path.join(BERT_PATH, "bert_multi_from_tfhub")



OUTPUT_PATH = "/kaggle/working"
# Training data from our first competition,

# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

wiki_toxic_comment_data = "jigsaw-toxic-comment-train.csv"

wiki_toxic_comment_data_A = "validation.csv"

wiki_toxic_comment_train = pandas.read_csv(os.path.join(

    DATA_PATH, wiki_toxic_comment_data))

wiki_toxic_comment_train_A = pandas.read_csv(os.path.join(

    DATA_PATH, wiki_toxic_comment_data))

wiki_toxic_comment_train.head()

wiki_toxic_comment_train_A.head()
def get_tokenizer(bert_path=BERT_PATH_SAVEDMODEL):

    """Get the tokenizer for a BERT layer."""

    bert_layer = tf.saved_model.load(bert_path)

    bert_layer = hub.KerasLayer(bert_layer, trainable=False)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

    cased = bert_layer.resolved_object.do_lower_case.numpy()

    tf.gfile = tf.io.gfile  # for bert.tokenization.load_vocab in tokenizer

    tokenizer = bert.tokenization.FullTokenizer(vocab_file, cased)

  

    return tokenizer



tokenizer = get_tokenizer()
example_sentence = wiki_toxic_comment_train.iloc[37].comment_text[:150]

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

    dataframe = pandas.read_csv(os.path.join(DATA_PATH, unprocessed_filename),

                                index_col='id')

    processed_filename = (unprocessed_filename.rstrip('.csv') +

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

preprocess_and_save_dataset(wiki_toxic_comment_data)

preprocess_and_save_dataset(wiki_toxic_comment_data_A)