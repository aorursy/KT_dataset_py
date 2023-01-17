# Get the official BERT tokenizer (please feel free to use the tokenization scheme of your choice)
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
    
# Import stuff
import tensorflow as tf
import tensorflow_hub as hub
import tokenization
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# I'm selecting this one
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"

# The model is available as a layer. The intricacies are hidden inside, so it's just a quick line to call it.
bert_layer = hub.KerasLayer(module_url, trainable=True)
# Define tokenizer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# Define a function that'll create the inputs to be passed to your model
# There are 3 inputs: token_ids, mask_ids, segment_ids
# Please Google and read more about the inputs and outpus. That's how I learned about it.
def encode_text(texts, tokenizer, max_len):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
# Actually tokenize now using the function you defined
# Assuming you have 2 pandas dataframes 'df_train' and 'df_test'
train_text_enc = encode_text(df_train['text'].values, tokenizer, max_len=100)
test_text_enc = encode_text(df_test['text'].values, tokenizer, max_len=100)
# Absract out the model creation
def build_bert_model(bert_layer, max_len):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    # Using [CLS] token output from sequence output
    x = sequence_output[:, 0, :]  # use 0th output - belongs to CLS token - means classification
    
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    
    return model
# Build model
m = build_bert_model(bert_layer, max_len)
m.summary()
m.compile(Adam(lr=1e-1), loss='binary_crossentropy', metrics=['accuracy'])
m.fit(train_text_enc, df_train['target'].values, epochs=2, validation_split=0.2, batch_size=16)