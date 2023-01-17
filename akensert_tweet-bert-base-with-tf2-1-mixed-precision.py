import numpy as np
import pandas as pd
from math import ceil, floor
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.initializers import TruncatedNormal
from sklearn import model_selection
from transformers import BertConfig, TFBertPreTrainedModel, TFBertMainLayer
from tokenizers import BertWordPieceTokenizer

import logging
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
    
tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options(
    {"auto_mixed_precision": True})
# read csv files
train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
train_df.dropna(inplace=True)

test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
test_df.loc[:, "selected_text"] = test_df.text.values

submission_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

print("train shape =", train_df.shape)
print("test shape  =", test_df.shape)

# set some global variables
PATH = "../input/bert-base-uncased/"
MAX_SEQUENCE_LENGTH = 128
TOKENIZER = BertWordPieceTokenizer(f"{PATH}/vocab.txt", lowercase=True, add_special_tokens=False)

# let's take a look at the data
train_df.head(10)

def preprocess(tweet, selected_text, sentiment):
    """
    Will be used in tf.data.Dataset.from_generator(...)
    
    """
    
    # The original strings have been converted to 
    # byte strings, so we need to decode it
    tweet = tweet.decode('utf-8')
    selected_text = selected_text.decode('utf-8')
    sentiment = sentiment.decode('utf-8')
    
    # Clean up the strings a bit
    tweet = " ".join(str(tweet).split())
    selected_text = " ".join(str(selected_text).split())
    
    # find the intersection between text and selected text
    idx_start, idx_end = None, None
    for index in (i for i, c in enumerate(tweet) if c == selected_text[0]):
        if tweet[index:index+len(selected_text)] == selected_text:
            idx_start = index
            idx_end = index + len(selected_text)
            break
    
    intersection = [0] * len(tweet)
    if idx_start != None and idx_end != None:
        for char_idx in range(idx_start, idx_end):
            intersection[char_idx] = 1
    
    # tokenize with offsets
    enc = TOKENIZER.encode(tweet)
    input_ids_orig, offsets = enc.ids, enc.offsets

    # compute targets
    target_idx = []
    for i, (o1, o2) in enumerate(offsets):
        if sum(intersection[o1: o2]) > 0:
            target_idx.append(i)
    
    target_start = target_idx[0]
    target_end = target_idx[-1]

    # add and pad data (hardcoded for BERT)
    # --> [CLS] sentiment [SEP] input_ids [SEP] [PAD]
    sentiment_map = {
        'positive': 3893,
        'negative': 4997,
        'neutral': 8699,
    }
    
    input_ids = [101] + [sentiment_map[sentiment]] + [102] + input_ids_orig + [102]
    input_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)
    attention_mask = [1] * (len(input_ids_orig) + 4)
    offsets = [(0, 0), (0, 0), (0, 0)] + offsets + [(0, 0)]
    target_start += 3
    target_end += 3

    padding_length = MAX_SEQUENCE_LENGTH - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        input_type_ids = input_type_ids + ([0] * padding_length)
        offsets = offsets + ([(0, 0)] * padding_length)
    elif padding_length < 0:
        input_ids = input_ids[:padding_length-1] + [102]
        attention_mask = attention_mask[:padding_length-1] + [1]
        input_type_ids = input_type_ids[:padding_length-1] + [1]
        offsets = offsets[:padding_length-1] + [(0, 0)]
        if target_start >= MAX_SEQUENCE_LENGTH:
            target_start = MAX_SEQUENCE_LENGTH - 1
        if target_end >= MAX_SEQUENCE_LENGTH:
            target_end = MAX_SEQUENCE_LENGTH - 1
        
    return (
        input_ids, attention_mask, input_type_ids, offsets,
        target_start, target_end, tweet, selected_text, sentiment, 
    )


class TweetSentimentDataset(tf.data.Dataset):
    
    OUTPUT_TYPES = (
        tf.dtypes.int32,  tf.dtypes.int32,   tf.dtypes.int32, 
        tf.dtypes.int32,  tf.dtypes.float32, tf.dtypes.float32,
        tf.dtypes.string, tf.dtypes.string,  tf.dtypes.string,
    )
    
    OUTPUT_SHAPES = (
        (MAX_SEQUENCE_LENGTH,),   (MAX_SEQUENCE_LENGTH,), (MAX_SEQUENCE_LENGTH,), 
        (MAX_SEQUENCE_LENGTH, 2), (),                     (),
        (),                       (),                     (),
    )
    
    # AutoGraph will automatically convert Python code to
    # Tensorflow graph code. You could also wrap 'preprocess' 
    # in tf.py_function(..) for arbitrary python code
    def _generator(tweet, selected_text, sentiment):
        for tw, st, se in zip(tweet, selected_text, sentiment):
            yield preprocess(tw, st, se)
    
    # This dataset object will return a generator
    def __new__(cls, tweet, selected_text, sentiment):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=cls.OUTPUT_TYPES,
            output_shapes=cls.OUTPUT_SHAPES,
            args=(tweet, selected_text, sentiment)
        )
    
    @staticmethod
    def create(dataframe, batch_size, shuffle_buffer_size=-1):
        dataset = TweetSentimentDataset(
            dataframe.text.values, 
            dataframe.selected_text.values, 
            dataframe.sentiment.values
        )

        dataset = dataset.cache()
        if shuffle_buffer_size != -1:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
        
class BertQAModel(TFBertPreTrainedModel):
    
    DROPOUT_RATE = 0.1
    NUM_HIDDEN_STATES = 2
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        self.bert = TFBertMainLayer(config, name="bert")
        self.concat = L.Concatenate()
        self.dropout = L.Dropout(self.DROPOUT_RATE)
        self.qa_outputs = L.Dense(
            config.num_labels, 
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            dtype='float32',
            name="qa_outputs")
        
    @tf.function
    def call(self, inputs, **kwargs):
        # outputs: Tuple[sequence, pooled, hidden_states]
        _, _, hidden_states = self.bert(inputs, **kwargs)
        
        hidden_states = self.concat([
            hidden_states[-i] for i in range(1, self.NUM_HIDDEN_STATES+1)
        ])
        
        hidden_states = self.dropout(hidden_states, training=kwargs.get("training", False))
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        
        return start_logits, end_logits
    
    
def train(model, dataset, loss_fn, optimizer):
    
    @tf.function
    def train_step(model, inputs, y_true, loss_fn, optimizer):
        with tf.GradientTape() as tape:
            y_pred = model(inputs, training=True)
            loss  = loss_fn(y_true[0], y_pred[0])
            loss += loss_fn(y_true[1], y_pred[1])
            scaled_loss = optimizer.get_scaled_loss(loss)
    
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, y_pred

    epoch_loss = 0.
    for batch_num, sample in enumerate(dataset):
        loss, y_pred = train_step(
            model, sample[:3], sample[4:6], loss_fn, optimizer)

        epoch_loss += loss

        print(
            f"training ... batch {batch_num+1:03d} : "
            f"train loss {epoch_loss/(batch_num+1):.3f} ",
            end='\r')
        
        
def predict(model, dataset, loss_fn, optimizer):
    
    @tf.function
    def predict_step(model, inputs):
        return model(inputs)
        
    def to_numpy(*args):
        out = []
        for arg in args:
            if arg.dtype == tf.string:
                arg = [s.decode('utf-8') for s in arg.numpy()]
                out.append(arg)
            else:
                arg = arg.numpy()
                out.append(arg)
        return out
    
    # Initialize accumulators
    offset = tf.zeros([0, MAX_SEQUENCE_LENGTH, 2], dtype=tf.dtypes.int32)
    text = tf.zeros([0,], dtype=tf.dtypes.string)
    selected_text = tf.zeros([0,], dtype=tf.dtypes.string)
    sentiment = tf.zeros([0,], dtype=tf.dtypes.string)
    pred_start = tf.zeros([0, MAX_SEQUENCE_LENGTH], dtype=tf.dtypes.float32)
    pred_end = tf.zeros([0, MAX_SEQUENCE_LENGTH], dtype=tf.dtypes.float32)
    
    for batch_num, sample in enumerate(dataset):
        
        print(f"predicting ... batch {batch_num+1:03d}"+" "*20, end='\r')
        
        y_pred = predict_step(model, sample[:3])
        
        # add batch to accumulators
        pred_start = tf.concat((pred_start, y_pred[0]), axis=0)
        pred_end = tf.concat((pred_end, y_pred[1]), axis=0)
        offset = tf.concat((offset, sample[3]), axis=0)
        text = tf.concat((text, sample[6]), axis=0)
        selected_text = tf.concat((selected_text, sample[7]), axis=0)
        sentiment = tf.concat((sentiment, sample[8]), axis=0)

    # pred_start = tf.nn.softmax(pred_start)
    # pred_end = tf.nn.softmax(pred_end)
    
    pred_start, pred_end, text, selected_text, sentiment, offset = \
        to_numpy(pred_start, pred_end, text, selected_text, sentiment, offset)
    
    return pred_start, pred_end, text, selected_text, sentiment, offset


def decode_prediction(pred_start, pred_end, text, offset, sentiment):
    
    def decode(pred_start, pred_end, text, offset):

        decoded_text = ""
        for i in range(pred_start, pred_end+1):
            decoded_text += text[offset[i][0]:offset[i][1]]
            if (i+1) < len(offset) and offset[i][1] < offset[i+1][0]:
                decoded_text += " "
        return decoded_text
    
    decoded_predictions = []
    for i in range(len(text)):
        if sentiment[i] == "neutral" or len(text[i].split()) < 2:
            decoded_text = text[i]
        else:
            idx_start = np.argmax(pred_start[i])
            idx_end = np.argmax(pred_end[i])
            if idx_start > idx_end:
                idx_end = idx_start 
            decoded_text = str(decode(idx_start, idx_end, text[i], offset[i]))
            if len(decoded_text) == 0:
                decoded_text = text[i]
        decoded_predictions.append(decoded_text)
    
    return decoded_predictions

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

num_folds = 4
num_epochs = 3
batch_size = 32
learning_rate = 3e-5

optimizer = tf.keras.optimizers.Adam(learning_rate)
optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
    optimizer, 'dynamic')

config = BertConfig(output_hidden_states=True, num_labels=2)
BertQAModel.DROPOUT_RATE = 0.2
BertQAModel.NUM_HIDDEN_STATES = 2
model = BertQAModel.from_pretrained(PATH, config=config)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

kfold = model_selection.StratifiedKFold(
    n_splits=num_folds, shuffle=True, random_state=42)

# initialize test predictions
test_preds_start = np.zeros((len(test_df), MAX_SEQUENCE_LENGTH), dtype=np.float32)
test_preds_end = np.zeros((len(test_df), MAX_SEQUENCE_LENGTH), dtype=np.float32)

for fold_num, (train_idx, valid_idx) in enumerate(
        kfold.split(X=train_df.text, y=train_df.sentiment.values)):
    print("\nfold %02d" % (fold_num+1))
        
    train_dataset = TweetSentimentDataset.create(
        train_df.iloc[train_idx], batch_size, shuffle_buffer_size=2048)
    valid_dataset = TweetSentimentDataset.create(
        train_df.iloc[valid_idx], batch_size, shuffle_buffer_size=-1)
    test_dataset = TweetSentimentDataset.create(
        test_df, batch_size, shuffle_buffer_size=-1)
    
    best_score = float('-inf')
    for epoch_num in range(num_epochs):
        print("\nepoch %03d" % (epoch_num+1))
        
        # train for an epoch
        train(model, train_dataset, loss_fn, optimizer)
        
        # predict validation set and compute jaccardian distances
        pred_start, pred_end, text, selected_text, sentiment, offset = \
            predict(model, valid_dataset, loss_fn, optimizer)
        
        selected_text_pred = decode_prediction(
            pred_start, pred_end, text, offset, sentiment)
        jaccards = []
        for i in range(len(selected_text)):
            jaccards.append(
                jaccard(selected_text[i], selected_text_pred[i]))
        
        score = np.mean(jaccards)
        print(f"valid jaccard epoch {epoch_num+1:03d}: {score}"+" "*15)
        
        if score > best_score:
            best_score = score
            # requires you to have 'fold-{fold_num}' folder in PATH:
            # model.save_pretrained(PATH+f'fold-{fold_num}')
            # or
            # model.save_weights(PATH + f'fold-{fold_num}.h5')
            
            # predict test set
            test_pred_start, test_pred_end, test_text, _, test_sentiment, test_offset = \
                predict(model, test_dataset, loss_fn, optimizer)
    
    # add epoch's best test preds to test preds arrays
    test_preds_start += test_pred_start
    test_preds_end += test_pred_end
    
    # reset model, as well as session and graph (to avoid OOM issues?) 
    session = tf.compat.v1.get_default_session()
    graph = tf.compat.v1.get_default_graph()
    del session, graph, model
    model = BertQAModel.from_pretrained(PATH, config=config)
    
# decode test set and add to submission file
selected_text_pred = decode_prediction(
    test_preds_start, test_preds_end, test_text, test_offset, test_sentiment)

submission_df.loc[:, 'selected_text'] = selected_text_pred
submission_df.to_csv("submission.csv", index=False)
