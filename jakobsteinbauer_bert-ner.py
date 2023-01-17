!pip install tensorflow==2.2.0-rc3
import functools
import json
import logging
from pathlib import Path
import sys
import os

from shutil import copyfile
###### For Kaggle: Import necessary function ######
copyfile(src = "../input/additional-scripts/modeling_tf2_kaggle.py", dst = "../working/modeling_tf2_kaggle.py")
copyfile(src = "../input/additional-scripts/tf_metrics_tf2.py", dst = "../working/tf_metrics_tf2.py")
copyfile(src = "../input/additional-scripts/tokenization.py", dst = "../working/tokenization.py")
###################################################

import tensorflow as tf
import numpy as np
from tokenization import FullTokenizer
from modeling_tf2_kaggle import BertConfig, BertModel, get_assignment_map_from_checkpoint

from tf_metrics_tf2 import precision, recall, f1

Path('results').mkdir(exist_ok=True)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        
        
def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data


class NerProcessor(object):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
 
    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_examples=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(True)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, True)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(True)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [True] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(False)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(False)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if print_examples:
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" %
                            " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" %
                            " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features
# DATA_DIR = "/kaggle/input/conll-data/"
# BERT_DIR = "/kaggle/input/bert-pretrained-models/multi_cased_L-12_H-768_A-12/multi_cased_L-12_H-768_A-12/"

# max_seq_length = 128
# processor = NerProcessor()
# dataset = processor.get_dev_examples(DATA_DIR)

# tokenizer = FullTokenizer(os.path.join(BERT_DIR, "vocab.txt"), do_lower_case = False)

# label_list = processor.get_labels()
# num_labels = len(label_list) + 1
# train_examples = processor.get_train_examples(DATA_DIR)
# te1 = [train_examples[0]]
# # label_map = {i: label for i, label in enumerate(label_list, 1)}

# train_features = convert_examples_to_features(
#      train_examples, label_list, max_seq_length, tokenizer)

# print(len(train_features))


def input_fn(DATA_DIR, TOKEN_DIR, train_val_test, params=None, shuffle_and_repeat=False, do_lower_case = False):
    #params = params if params is not None else {}
    assert train_val_test in ['train', 'valid', 'test'], \
    "train_val_test variable must be one of 3 strings in list!"
    
    max_seq_length = params['max_seq_length']

    tokenizer = FullTokenizer(os.path.join(TOKEN_DIR, "vocab.txt"), do_lower_case = do_lower_case)

    processor = NerProcessor()
    label_list = processor.get_labels() 
    
    if train_val_test == 'train':
        examples = processor.get_train_examples(DATA_DIR)
    elif train_val_test == 'valid':
        examples = processor.get_dev_examples(DATA_DIR)
    else:
        examples = processor.get_test_examples(DATA_DIR)
        
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)
        
    all_input_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_ids for f in features]))
    all_input_mask = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_mask for f in features]))
    all_segment_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.segment_ids for f in features]))
    all_valid_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.valid_ids for f in features]))
    all_label_mask = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.label_mask for f in features]))
    all_label_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.label_id for f in features]))
    

    # Dataset using tf.data
    x_data = tf.data.Dataset.zip(
        (all_input_ids, all_input_mask, all_segment_ids, all_valid_ids))
    y_data = tf.data.Dataset.zip((all_label_ids,all_label_mask))
    full_data = tf.data.Dataset.zip((x_data, y_data))
    
    if shuffle_and_repeat:
        full_data = full_data.shuffle(buffer_size=int(len(features) * 0.1),
                                                 seed = params['seed'], 
                                                 reshuffle_each_iteration=True).repeat(params['epochs'])
        
    full_batched_data = full_data.batch(params.get('batch_size', 32)).prefetch(1)
    
    return full_batched_data

def model_fn_builder(bert_config, init_checkpoint, use_one_hot_embeddings=False):
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for Estimator."""

        label_list = NerProcessor().get_labels()
        num_labels = len(label_list) + 1
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        indices = [idx for idx, label in enumerate(label_list) if label.strip() != 'O']
        
        
        (input_ids, input_mask, segment_ids, valid_ids) = features

        batch_size = params.get('batch_size', 32)
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )
        sequence_output = model.get_sequence_output()

        ragged_output = tf.ragged.boolean_mask(sequence_output, tf.cast(valid_ids,tf.bool))
        valid_output = ragged_output.to_tensor(default_value = 0., shape = tf.shape(sequence_output))

        sequence_output = tf.keras.layers.Dropout(rate=params['dropout'])(valid_output, training=is_training) #should be valid_output instead of sequence_output
        logits = tf.keras.layers.Dense(num_labels, activation='softmax', name='output')(sequence_output)

        pred_ids = tf.argmax(logits, axis=2)

        # Initialize BERT weights from init_checkpoint 
        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
            get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
        
        
        NotBert_variables = []
        Bert_variables = []
        tf.compat.v1.logging.info("**** Trainable Variables ****")
        any_newly_init_vars = False
        for var in tvars:
            #init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                any_newly_init_vars = True 
                Bert_variables.append(var)
            else:
                NotBert_variables.append(var)    
#             tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
#                                 init_string)
        if True: # Do verbose logging
            print('Variables re-initialized from Bert Model Config!')

            
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_ids)
        else:
            (label_ids, label_mask) = labels


            weights = tf.cast(label_mask, tf.float32) # Check if this works, since label_mask is of type bool

            log_logits = tf.math.log(logits)
            one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_logits, axis=-1)
            loss = tf.reduce_sum(per_example_loss*weights)/tf.reduce_sum(weights)
                        
            if mode == tf.estimator.ModeKeys.EVAL:
                metrics = {
                    'acc': tf.compat.v1.metrics.accuracy(label_ids, pred_ids, weights),
                    'precision': precision(label_ids, pred_ids, num_labels, [3, 4], weights),
                    'recall': recall(label_ids, pred_ids, num_labels, [3, 4], weights),
                    'f1': f1(label_ids, pred_ids, num_labels, [3, 4], weights),
                }
                for metric_name, op in metrics.items():
                    tf.compat.v1.summary.scalar(metric_name, op[1])
                    
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            elif mode == tf.estimator.ModeKeys.TRAIN:
                
                if params['train_only_NotBert_variables']:
                    print('Trainable Variables not in original Bert Model:')
                    for var in NotBert_variables:
                        print(var)
                    print('Trainable Variables in original Bert Model:')
                    for var in Bert_variables:
                        print(var)
                    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(
                    loss, global_step=tf.compat.v1.train.get_or_create_global_step(), var_list = NotBert_variables)      
                else:
                    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(
                        loss, global_step=tf.compat.v1.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, train_op=train_op)

    return model_fn
if __name__ == '__main__':
    
    # Data dirs
    DATA_DIR = "/kaggle/input/conll-data/"
    BERT_DIR = "/kaggle/input/bert-pretrained-models/multi_cased_L-12_H-768_A-12/multi_cased_L-12_H-768_A-12/"
    json_file = BERT_DIR + "bert_config.json"
    init_checkpoint = BERT_DIR + "bert_model.ckpt"

    config = BertConfig.from_json_file(json_file)

    # Params
    params = {
        'dropout': 0.2,
        'epochs': 6,
        'batch_size': 32,
        'seed': 2, #Arbitrary, fixed seed
        'max_seq_length': 128,
        'learning_rate':0.00005,
        'train_only_NotBert_variables': False
    }

    with Path('results/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)


    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, DATA_DIR, BERT_DIR, "train",
                                   params=params, shuffle_and_repeat=True, do_lower_case = False)
    eval_inpf = functools.partial(input_fn, DATA_DIR, BERT_DIR, "valid",
                                   params=params, shuffle_and_repeat=True, do_lower_case = False)


    model_fn = model_fn_builder(bert_config=config, init_checkpoint=init_checkpoint)

    cfg = tf.estimator.RunConfig(save_checkpoints_steps=110, keep_checkpoint_max=1)
    estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.estimator.experimental.stop_if_no_increase_hook( #tf.contrib.estimator.stop_if_no_increase_hook( # Change made by Jakob Steinbauer
        estimator, 'f1', 500, min_steps=1000, run_every_secs=None, run_every_steps=110)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


    processor = NerProcessor()
    label_list = processor.get_labels()
    label_map = {i: label for i, label in enumerate(label_list,1)}
    label_map[0] = 'O' # Add default label
    label_map[10] = 'O' # Add default label
    label_map[11] = 'O' # Add default label  

    def write_predictions(name):
        Path('results/score').mkdir(parents=True, exist_ok=True)
        with Path('results/score/{}.preds.txt'.format(name)).open('w') as f:
            test_inpf = functools.partial(input_fn, DATA_DIR, BERT_DIR, name,
                                   params=params, shuffle_and_repeat=False, do_lower_case = False)
            if name == 'train':
                orig_words = processor.get_train_examples(DATA_DIR)
            elif name == 'valid':
                orig_words = processor.get_dev_examples(DATA_DIR)
            else:
                orig_words = processor.get_test_examples(DATA_DIR)
            preds_gen = estimator.predict(test_inpf)

            for orig, preds in zip(orig_words, preds_gen):
                words = orig.text_a.split(' ')
                tags  = orig.label
                preds_short = preds[1:len(words)+1]

                for word, tag, tag_pred in zip(words, tags, preds_short):
                    f.write(' '.join([word, tag, label_map[tag_pred]]) + '\n')
                f.write('\n')

    for name in ['train', 'valid', 'test']:
        write_predictions(name)
# import shutil

# shutil.rmtree('/kaggle/working/results/')
