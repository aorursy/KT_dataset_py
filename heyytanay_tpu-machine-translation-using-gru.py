import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
data = pd.read_csv("../input/europarl-parallel-corpus-19962011/english_german.csv")
data = data.dropna()
data.head()
# Append <START> and <END> to each english sentence
START = 'ssss '
END = ' eeee'

data['English'] = data['English'].apply(lambda x: START+x+END)
data.head()
eng_text = data['English'].tolist()
deu_text = data['German'].tolist()

print(eng_text[5])
print(deu_text[5])
num_words = 10000
class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""
    
    def __init__(self, texts, padding,
                 reverse=False, num_words=None):
        """
        :param texts: List of strings. This is the data-set.
        :param padding: Either 'post' or 'pre' padding.
        :param reverse: Boolean whether to reverse token-lists.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

        # Convert all texts to lists of integer-tokens.
        # Note that the sequences may have different lengths.
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
            # Reverse the token-sequences.
            self.tokens = [list(reversed(x)) for x in self.tokens]
        
            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        # The number of integer-tokens in each sequence.
        self.num_tokens = [len(x) for x in self.tokens]

        # Max number of tokens to use in all sequences.
        # We will pad / truncate all sequences to this length.
        # This is a compromise so we save a lot of memory and
        # only have to truncate maybe 5% of all the sequences.
        self.max_tokens = np.mean(self.num_tokens) \
                          + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)

        # Pad / truncate all token-sequences to the given length.
        # This creates a 2-dim numpy matrix that is easier to use.
        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=self.max_tokens,
                                           padding=padding,
                                           truncating=truncating)

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        
        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text
    
    def text_to_tokens(self, text, reverse=False, padding=False):
        """
        Convert a single text-string to tokens with optional
        reversal and padding.
        """

        # Convert to tokens. Note that we assume there is only
        # a single text-string so we wrap it in a list.
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            # Reverse the tokens.
            tokens = np.flip(tokens, axis=1)

            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        if padding:
            # Pad and truncate sequences to the given length.
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)

        return tokens
%%time
tokenizer_src = TokenizerWrap(texts=deu_text,
                              padding='pre',
                              reverse=True,
                              num_words=num_words
                             )
%%time
tokenizer_des = TokenizerWrap(texts=eng_text,
                              padding='post',
                              reverse=False,
                              num_words=num_words
                             )
token_src = tokenizer_src.tokens_padded
token_des = tokenizer_des.tokens_padded

print(token_src.shape)
print(token_des.shape)
token_start = tokenizer_des.word_index[START.strip()]
token_end = tokenizer_des.word_index[END.strip()]

print(token_start)
print(token_end)
encoder_inp_data = token_src
decoder_inp_data = token_des[:, :-1]
decoder_out_data = token_des[:, 1:]
# Glue all the encoder components together
def connect_encoder():
    net = encoder_input
    
    net = encoder_emb(net)
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    out = encoder_gru3(net)
    
    return out
def connect_decoder(initial_state):    
    # Start the decoder-network with its input-layer.
    net = decoder_input

    # Connect the embedding-layer.
    net = decoder_emb(net)
    
    # Connect all the GRU-layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)
    
    return decoder_output
# Connect all the models
with strategy.scope():
    
    embedding_size = 128
    state_size = 512

    encoder_input = Input(shape=(None,), name='encoder_input')
    encoder_emb = Embedding(input_dim=num_words, output_dim=embedding_size, name='encoder_embedding')

    encoder_gru1 = GRU(state_size, name='enc_gru1', return_sequences=True)
    encoder_gru2 = GRU(state_size, name='enc_gru2', return_sequences=True)
    encoder_gru3 = GRU(state_size, name='enc_gru3', return_sequences=False)
    
    encoder_op = connect_encoder()
    
    # Initial state placeholder takes a "thought vector" produced by the GRUs
    # That's why it needs the inputs with "state_size" (which was used in GRU size)
    decoder_initial_state = Input(shape=(state_size,), name='decoder_init_state')

    # Decoder also needs an input, which is the basic input setence of the destination language
    decoder_input = Input(shape=(None,), name='decoder_input')

    # Have the decoder embedding
    decoder_emb = Embedding(input_dim=num_words, output_dim=embedding_size, name='decoder_embedding')

    # GRU arch similar to Encoder one with small changes
    decoder_gru1 = GRU(state_size, name='dec_gru1', return_sequences=True)
    decoder_gru2 = GRU(state_size, name='dec_gru2', return_sequences=True)
    decoder_gru3 = GRU(state_size, name='dec_gru3', return_sequences=True)

    # Final dense layer for prediction
    decoder_dense = Dense(num_words, activation='softmax', name='decoder_output')
    decoder_op = connect_decoder(encoder_op)
    model_train = Model(inputs=[encoder_input, decoder_input],
                        outputs=[decoder_op])
    model_train.compile(optimizer=RMSprop(lr=1e-3),
                        loss='sparse_categorical_crossentropy')
tf.keras.utils.plot_model(model_train)
path_checkpoint = '21_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)
callbacks = [callback_early_stopping,
             callback_checkpoint]
try:
    model_train.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)
x_data = {
    "encoder_input": encoder_inp_data,
    "decoder_input": decoder_inp_data
}

y_data = {
    "decoder_output": decoder_out_data
}
validation_split = 10000 / len(encoder_inp_data)
print(f"Validation Split: {validation_split:.4f}%")
# Train the model
with strategy.scope():
    model_train.fit(
        x=x_data,
        y=y_data,
        batch_size=384,
        epochs=10,
        validation_split=validation_split,
        callbacks=callbacks
    )
model_train.save("eng_to_deu.hdf5")
