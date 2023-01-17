from IPython.display import Image
Image(width= 550, height= 300, filename= '../input/bert-final-data/bert-diagram.png')
%%capture
!pip install /kaggle/input/bert-for-tf2/py-params-0.8.2/py-params-0.8.2/
!pip install /kaggle/input/bert-for-tf2/params-flow-0.7.4/params-flow-0.7.4/
!pip install /kaggle/input/bert-for-tf2/bert-for-tf2-0.13.2/bert-for-tf2-0.13.2/
!pip install sentencepiece

%pylab inline
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from bert.tokenization.bert_tokenization import FullTokenizer
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from bs4 import BeautifulSoup
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from IPython.display import Image
# Define the three functions that prepare the input for bert.
# Load data (see Appendix A)
train_data = np.load("../input/imbddatao/data.npz")
test_data = np.load("../input/imbddatao/test.npz")
train = train_data["a"]
train_labels = train_data["b"]
test = test_data["a"]
test_labels = test_data["b"]

# Text maximum length for Bert Base
max_seq_length = 512
# Load the Bert base layer.
Bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=True)
# Processing Functions
# to distinguish padding from real words ids
def get_masks(tokens, max_seq_length):
    if len(tokens)>max_seq_length:
        #Cutting down the excess length
        tokens = tokens[0:max_seq_length]
        return [1]*len(tokens)
    else:
        return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

# needed to sentencize (we don't use it really).
def get_segments(tokens, max_seq_length):
    if len(tokens)>max_seq_length:
        #Cutting down the excess length
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
# a bow model
def get_ids(tokens, tokenizer, max_seq_length):    
    if len(tokens)>max_seq_length:
        tokens = tokens[:max_seq_length]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        return token_ids
    else:
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
        return input_ids

# create a vocabulary
vocab_file = Bert_layer.resolved_object.vocab_file.asset_path.numpy()
# lower case bert function
do_lower_case = Bert_layer.resolved_object.do_lower_case.numpy()
# tokenizer
tokenizer = FullTokenizer(vocab_file, do_lower_case)

# preprocessing function
def bert_encode(texts):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        stokens = tokenizer.tokenize(text)
        stokens = ["[CLS]"] + stokens + ["[SEP]"]
        
        input_ids = get_ids(stokens, tokenizer, max_seq_length)
        input_masks = get_masks(stokens, max_seq_length)
        input_segments = get_segments(stokens, max_seq_length)
        
        all_tokens.append(input_ids)
        all_masks.append(input_masks)
        all_segments.append(input_segments)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def review_to_words(raw_body):
    body_text = BeautifulSoup(raw_body).get_text() 
    return(body_text)  
print("Esempio input:")
print(bert_encode(train[1]))

train_text = [review_to_words(x) for x in train]
test_text =  [review_to_words(x) for x in test]

train_input = bert_encode(train_text)
test_input = bert_encode(test_text)

test_labels = test_labels.astype(float)
train_labels = train_labels.astype(float)
# Define the model.
METRICS = [ 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.AUC(name='auc'),
]

input_word_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
segment_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

_ , sequence_output = Bert_layer([input_word_ids, input_mask, segment_ids])
clf_output = sequence_output[:, 0, :] # opzione 1
mean = tf.reduce_mean(sequence_output, 1) # opzione 2
out = Dense(512, activation='relu')(clf_output)
# note: here we could insert a dropout layer: we chose not to put it (so drop = 0) because drop exploration (see Appendix B) pointed no signicant performance oscillation when we vary the drop.
final = Dense(1, activation='sigmoid')(out)
model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=final)
# compile and summary
model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=METRICS)
model.summary()
# plot model structure
plot_model(model,show_layer_names  = False,show_shapes = True)
# early stopping
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        min_delta=1e-2,
        patience=2,
        verbose=1)
]

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=5,
    batch_size=12,
    callbacks=callbacks
)

#model.save('model.h5')
# predict
print('\n# Evaluate on test data')
results = model.evaluate(test_input, test_labels, batch_size=12)
print('test loss, test acc, test auc:', results)
# plot performance-related parameters. Once set drop = 0 (see Appendix B), we trained up to 15 epocs to explore the model.
with open('../input/bert-final-data/trainHistoryDict', 'rb') as f:
        long = pickle.load(f)
        f.close()

figure(figsize=(25, 9))
i = 0
epoch = list(range(15))

for _ in ["auc","accuracy","loss"]:
        i = i+1
        plt.subplot(1, 3, i)
        val = long["val_"+_]
        train = long[_]
        plt.title(_ ,fontsize=20)
        plt.plot(epoch,val,train)
        
plt.tight_layout()
Bert_layerL = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1",trainable=True)

vocab_file = Bert_layerL.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = Bert_layerL.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        stokens = tokenizer.tokenize(text)
        stokens = ["[CLS]"] + stokens + ["[SEP]"]
        
        input_ids = get_ids(stokens, tokenizer, max_seq_length)
        input_masks = get_masks(stokens, max_seq_length)
        input_segments = get_segments(stokens, max_seq_length)
        
        all_tokens.append(input_ids)
        all_masks.append(input_masks)
        all_segments.append(input_segments)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

train_input = bert_encode(train_text)
test_input = bert_encode(test_text)

test_labels = test_labels.astype(float)
train_labels = train_labels.astype(float)

input_word_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
segment_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

_ , sequence_output = Bert_layerL([input_word_ids, input_mask, segment_ids]) 
clf_output = sequence_output[:, 0, :] 
mean = tf.reduce_mean(sequence_output, 1) 
out = tf.keras.layers.Dense(512, activation='relu')(clf_output) #We tried mean three times, with similar results. We couldn't perform futher investigation because of computational limits.
final = tf.keras.layers.Dense(1, activation='sigmoid')(out)

    
modelL = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=final)
modelL.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=METRICS) 

callbacksL = [
    EarlyStopping(
        monitor='val_accuracy',
        min_delta=1e-2,
        patience=2,
        verbose=1)
]

train_historyL = modelL.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=4,
    batch_size=4,
    callbacks=callbacksL
)
# predict Large
print('\n# Evaluate on test data')
resultsL = modelL.evaluate(test_input, test_labels, batch_size=4)
print('test loss, test acc, test auc:', resultsL)
for i in ["0","1","2","3","4","5","6","7","8","9"]:
    with open('../input/bert-final-data/histories/trainHistoryDict_0.'+ i, 'rb') as f:
        globals()["history"+i] = pickle.load(f)
        f.close()
        
print("history_0['roc'] = ",max(history0['val_auc']),"\n", "history_1['roc'] = ",max(history1['val_auc']),"\n","history_2['roc'] = ",max(history2['val_auc']),"\n","history_3['roc'] = ",max(history3['val_auc']),"\n",
     "history_4['roc'] = ",max(history4['val_auc']),"\n","history_5['roc'] = ",max(history5['val_auc']),"\n","history_6['roc'] = ",max(history6['val_auc']),"\n","history_7['roc'] = ",max(history7['val_auc']),"\n",
     "history_8['roc'] = ",max(history8['val_auc']),"\n","history_9['roc'] = ",max(history9['val_auc']),"\n")


print("history_0['val_accuracy'] = ",max(history0['val_accuracy']),"\n", "history_1['val_accuracy'] = ",max(history1['val_accuracy']),"\n","history_2['val_accuracy'] = ",max(history2['val_accuracy']),"\n","history_3['val_accuracy'] = ",max(history3['val_accuracy']),"\n",
     "history_4['val_accuracy'] = ",max(history4['val_accuracy']),"\n","history_5['val_accuracy'] = ",max(history5['val_accuracy']),"\n","history_6['val_accuracy'] = ",max(history6['val_accuracy']),"\n","history_7['val_accuracy'] = ",max(history7['val_accuracy']),"\n",
     "history_8['val_accuracy'] = ",max(history8['val_accuracy']),"\n","history_9['val_accuracy'] = ",max(history9['val_accuracy']),"\n")

figure(figsize=(25, 60))
i = 0
drop = 0
epoch = list(range(6))
for history in (history0,history1,history2,history3,history4,history5,history6,history7,history8,history9):
    drop = drop + 1
    for _ in ["auc","accuracy","loss"]:
        i = i+1
        plt.subplot(10, 3, i)
        val = history["val_"+ _]
        train = history[_]
        plt.title(_ + ", drop = 0." + str(drop),fontsize=20)
        plt.plot(epoch,val,train)
        
plt.tight_layout()
Image(width= "100%",filename= r'../input/bert-final-data/LAR2.png')