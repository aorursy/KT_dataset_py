import codecs
import copy
import csv
import gc
import os
import pickle
import random
import time
from typing import Dict, List, Sequence, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.python.framework import ops, tensor_util
from tensorflow.python.keras.utils import losses_utils, tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as tf_losses_util
import tensorflow_addons as tfa
from transformers import AutoTokenizer, XLMRobertaTokenizer
from transformers import TFXLMRobertaModel, XLMRobertaConfig
class LossFunctionWrapper(tf.keras.losses.Loss):
    def __init__(self,
                 fn,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name=None,
                 **kwargs):
        super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
            y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(y_pred, y_true)
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = tf.keras.backend.eval(v) if tf_utils.is_tensor_or_variable(v) \
                else v
        base_config = super(LossFunctionWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def distance_based_log_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    margin = 1.0
    p = (1.0 + tf.math.exp(-margin)) / (1.0 + tf.math.exp(y_pred - margin))
    return tf.keras.backend.binary_crossentropy(target=y_true, output=p)
class DBLLogLoss(LossFunctionWrapper):
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO,
                 name='distance_based_log_loss'):
        super(DBLLogLoss, self).__init__(distance_based_log_loss, name=name,
                                         reduction=reduction)
def dice_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    dice_loss_1 = tf.keras.backend.mean(y_true * y_pred, axis=-1)
    dice_loss_2 = tf.keras.backend.mean(y_true * y_true, axis=-1)
    dice_loss_3 = tf.keras.backend.mean(y_pred * y_pred, axis=-1)
    return 1.0 - (2 * dice_loss_1 + 1.0) / (dice_loss_2 + dice_loss_3 + 1.0)
class DiceLoss(LossFunctionWrapper):
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='dice_loss'):
        super(DiceLoss, self).__init__(dice_loss, name=name, reduction=reduction)
class MaskCalculator(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MaskCalculator, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskCalculator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.keras.backend.permute_dimensions(
            x=tf.keras.backend.repeat(
                x=tf.keras.backend.cast(
                    x=tf.keras.backend.greater(
                        x=inputs,
                        y=0
                    ),
                    dtype='float32'
                ),
                n=self.output_dim
            ),
            pattern=(0, 2, 1)
        )

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 1
        shape = list(input_shape)
        shape.append(self.output_dim)
        return tuple(shape)
def regular_encode(texts: List[str], tokenizer: XLMRobertaTokenizer,
                   maxlen: int) -> Tuple[np.ndarray, np.ndarray]:
    enc_di = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    return np.array(enc_di['input_ids']), np.array(enc_di['attention_mask'])
def load_train_set(file_name: str, text_field: str, sentiment_fields: List[str],
                   lang_field: str) -> Dict[str, List[Tuple[str, int]]]:
    assert len(sentiment_fields) > 0, 'List of sentiment fields is empty!'
    header = []
    line_idx = 1
    data_by_lang = dict()
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        data_reader = csv.reader(fp, quotechar='"', delimiter=',')
        for row in data_reader:
            if len(row) > 0:
                err_msg = 'File "{0}": line {1} is wrong!'.format(file_name, line_idx)
                if len(header) == 0:
                    header = copy.copy(row)
                    err_msg2 = err_msg + ' Field "{0}" is not found!'.format(text_field)
                    assert text_field in header, err_msg2
                    for cur_field in sentiment_fields:
                        err_msg2 = err_msg + ' Field "{0}" is not found!'.format(
                            cur_field)
                        assert cur_field in header, err_msg2
                    text_field_index = header.index(text_field)
                    try:
                        lang_field_index = header.index(lang_field)
                    except:
                        lang_field_index = -1
                    indices_of_sentiment_fields = []
                    for cur_field in sentiment_fields:
                        indices_of_sentiment_fields.append(header.index(cur_field))
                else:
                    if len(row) == len(header):
                        text = row[text_field_index].strip()
                        assert len(text) > 0, err_msg + ' Text is empty!'
                        if lang_field_index >= 0:
                            cur_lang = row[lang_field_index].strip()
                            assert len(cur_lang) > 0, err_msg + ' Language is empty!'
                        else:
                            cur_lang = 'en'
                        max_proba = 0.0
                        for cur_field_idx in indices_of_sentiment_fields:
                            try:
                                cur_proba = float(row[cur_field_idx])
                            except:
                                cur_proba = -1.0
                            err_msg2 = err_msg + ' Value {0} is wrong!'.format(
                                row[cur_field_idx]
                            )
                            assert (cur_proba >= 0.0) and (cur_proba <= 1.0), err_msg2
                            if cur_proba > max_proba:
                                max_proba = cur_proba
                        new_label = 1 if max_proba >= 0.5 else 0
                        if cur_lang not in data_by_lang:
                            data_by_lang[cur_lang] = []
                        data_by_lang[cur_lang].append((text, new_label))
            if line_idx % 10000 == 0:
                print('{0} lines of the "{1}" have been processed...'.format(line_idx,
                                                                             file_name))
            line_idx += 1
    if line_idx > 0:
        if (line_idx - 1) % 10000 != 0:
            print('{0} lines of the "{1}" have been processed...'.format(line_idx - 1,
                                                                         file_name))
    return data_by_lang
def load_test_set(file_name: str, id_field: str, text_field: str,
                  lang_field: str) -> Dict[str, List[Tuple[str, int]]]:
    header = []
    line_idx = 1
    data_by_lang = dict()
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        data_reader = csv.reader(fp, quotechar='"', delimiter=',')
        for row in data_reader:
            if len(row) > 0:
                err_msg = 'File "{0}": line {1} is wrong!'.format(file_name, line_idx)
                if len(header) == 0:
                    header = copy.copy(row)
                    err_msg2 = err_msg + ' Field "{0}" is not found!'.format(text_field)
                    assert text_field in header, err_msg2
                    err_msg2 = err_msg + ' Field "{0}" is not found!'.format(id_field)
                    assert id_field in header, err_msg2
                    err_msg2 = err_msg + ' Field "{0}" is not found!'.format(lang_field)
                    assert lang_field in header, err_msg2
                    id_field_index = header.index(id_field)
                    text_field_index = header.index(text_field)
                    lang_field_index = header.index(lang_field)
                else:
                    if len(row) == len(header):
                        try:
                            id_value = int(row[id_field_index])
                        except:
                            id_value = -1
                        err_msg2 = err_msg + ' {0} is wrong ID!'.format(
                            row[id_field_index])
                        assert id_value >= 0, err_msg2
                        text = row[text_field_index].strip()
                        assert len(text) > 0, err_msg + ' Text is empty!'
                        if lang_field_index >= 0:
                            cur_lang = row[lang_field_index].strip()
                            assert len(cur_lang) > 0, err_msg + ' Language is empty!'
                        else:
                            cur_lang = 'en'
                        if cur_lang not in data_by_lang:
                            data_by_lang[cur_lang] = []
                        data_by_lang[cur_lang].append((text, id_value))
            if line_idx % 10000 == 0:
                print('{0} lines of the "{1}" have been processed...'.format(line_idx,
                                                                             file_name))
            line_idx += 1
    if line_idx > 0:
        if (line_idx - 1) % 10000 != 0:
            print('{0} lines of the "{1}" have been processed...'.format(line_idx - 1,
                                                                         file_name))
    return data_by_lang
def build_siamese_dataset(texts: Dict[str, List[Tuple[str, int]]],
                          dataset_size: int, tokenizer: XLMRobertaTokenizer,
                          maxlen: int, batch_size: int,
                          shuffle: bool) -> Tuple[tf.data.Dataset, int]:
    language_pairs = set()
    for language in texts.keys():
        language_pairs.add(('en', language))
        for other_language in texts:
            if other_language == language:
                language_pairs.add((language, other_language))
            else:
                new_pair = (language, other_language)
                new_pair_2 = (other_language, language)
                if (new_pair not in language_pairs) and (new_pair_2 not in language_pairs):
                    language_pairs.add(new_pair)
    language_pairs = sorted(list(language_pairs))
    print('Possible language pairs are: {0}.'.format(language_pairs))
    err_msg = '{0} is too small size of the data set!'.format(dataset_size)
    assert dataset_size >= (len(language_pairs) * 10), err_msg
    n_samples_for_lang_pair = int(np.ceil(dataset_size / float(len(language_pairs))))
    text_pairs_and_labels = []
    for left_lang, right_lang in language_pairs:
        print('{0}-{1}:'.format(left_lang, right_lang))
        left_positive_indices = list(filter(
            lambda idx: texts[left_lang][idx][1] > 0, range(len(texts[left_lang]))
        ))
        left_negative_indices = list(filter(
            lambda idx: texts[left_lang][idx][1] == 0, range(len(texts[left_lang]))
        ))
        right_positive_indices = list(filter(
            lambda idx: texts[right_lang][idx][1] > 0, range(len(texts[right_lang]))
        ))
        right_negative_indices = list(filter(
            lambda idx: texts[right_lang][idx][1] == 0, range(len(texts[right_lang]))
        ))
        used_pairs = set()
        number_of_samples = 0
        for _ in range(n_samples_for_lang_pair // 4):
            left_idx = random.choice(left_positive_indices)
            right_idx = random.choice(right_positive_indices)
            counter = 0
            while ((right_idx == left_idx) or ((left_idx, right_idx) in used_pairs) or
                   ((right_idx, left_idx) in used_pairs)) and (counter < 100):
                right_idx = random.choice(right_positive_indices)
                counter += 1
            if counter < 100:
                used_pairs.add((left_idx, right_idx))
                used_pairs.add((right_idx, left_idx))
                text_pairs_and_labels.append(
                    (
                        texts[left_lang][left_idx][0],
                        texts[right_lang][right_idx][0],
                        1,
                        1,
                        1
                    )
                )
                number_of_samples += 1
        print('  number of "1-1" pairs is {0};'.format(number_of_samples))
        number_of_samples = 0
        for _ in range(n_samples_for_lang_pair // 4, (2 * n_samples_for_lang_pair) // 4):
            left_idx = random.choice(left_negative_indices)
            right_idx = random.choice(right_negative_indices)
            counter = 0
            while ((right_idx == left_idx) or ((left_idx, right_idx) in used_pairs) or
                   ((right_idx, left_idx) in used_pairs)) and (counter < 100):
                right_idx = random.choice(right_negative_indices)
                counter += 1
            if counter < 100:
                used_pairs.add((left_idx, right_idx))
                used_pairs.add((right_idx, left_idx))
                text_pairs_and_labels.append(
                    (
                        texts[left_lang][left_idx][0],
                        texts[right_lang][right_idx][0],
                        1,
                        0,
                        0
                    )
                )
                number_of_samples += 1
        print('  number of "0-0" pairs is {0};'.format(number_of_samples))
        number_of_samples = 0
        for _ in range((2 * n_samples_for_lang_pair) // 4, n_samples_for_lang_pair):
            left_idx = random.choice(left_negative_indices)
            right_idx = random.choice(right_positive_indices)
            counter = 0
            while ((right_idx == left_idx) or ((left_idx, right_idx) in used_pairs) or
                   ((right_idx, left_idx) in used_pairs)) and (counter < 100):
                right_idx = random.choice(right_positive_indices)
                counter += 1
            if counter < 100:
                used_pairs.add((left_idx, right_idx))
                used_pairs.add((right_idx, left_idx))
                if random.random() >= 0.5:
                    text_pairs_and_labels.append(
                        (
                            texts[left_lang][left_idx][0],
                            texts[right_lang][right_idx][0],
                            0,
                            0,
                            1
                        )
                    )
                else:
                    text_pairs_and_labels.append(
                        (
                            texts[right_lang][right_idx][0],
                            texts[left_lang][left_idx][0],
                            0,
                            1,
                            0
                        )
                    )
                number_of_samples += 1
        print('  number of "0-1" or "1-0" pairs is {0}.'.format(number_of_samples))
    random.shuffle(text_pairs_and_labels)
    n_steps = len(text_pairs_and_labels) // batch_size
    print('Samples number of the data set is {0}.'.format(len(text_pairs_and_labels)))
    print('Samples number per each language pair is {0}.'.format(n_samples_for_lang_pair))
    tokens_of_left_texts, mask_of_left_texts = regular_encode(
        texts=[cur[0] for cur in text_pairs_and_labels],
        tokenizer=tokenizer, maxlen=maxlen
    )
    tokens_of_right_texts, mask_of_right_texts = regular_encode(
        texts=[cur[1] for cur in text_pairs_and_labels],
        tokenizer=tokenizer, maxlen=maxlen
    )
    siamese_labels = np.array([cur[2] for cur in text_pairs_and_labels], dtype=np.int32)
    left_labels = np.array([cur[3] for cur in text_pairs_and_labels], dtype=np.int32)
    right_labels = np.array([cur[4] for cur in text_pairs_and_labels], dtype=np.int32)
    print('Number of positive siamese samples is {0} from {1}.'.format(
        int(sum(siamese_labels)), siamese_labels.shape[0]))
    print('Number of positive left samples is {0} from {1}.'.format(
        int(sum(left_labels)), left_labels.shape[0]))
    print('Number of positive right samples is {0} from {1}.'.format(
        int(sum(right_labels)), right_labels.shape[0]))
    if shuffle:
        err_msg = '{0} is too small number of samples for the data set!'.format(
            len(text_pairs_and_labels))
        assert n_steps >= 50, err_msg
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    tokens_of_left_texts, mask_of_left_texts,
                    tokens_of_right_texts, mask_of_right_texts
                ),
                (
                    siamese_labels,
                    left_labels,
                    right_labels
                )
            )
        ).repeat().batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                (
                    tokens_of_left_texts, mask_of_left_texts,
                    tokens_of_right_texts, mask_of_right_texts
                ),
                (
                    siamese_labels,
                    left_labels,
                    right_labels
                )
            )
        ).batch(batch_size)
    del text_pairs_and_labels
    return dataset, n_steps
def build_feature_extractor(transformer_name: str, hidden_state_size: int,
                            max_len: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                     name="base_word_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                           name="base_attention_mask")
    transformer_layer = TFXLMRobertaModel.from_pretrained(
        pretrained_model_name_or_path=transformer_name,
        name='Transformer'
    )
    sequence_output = transformer_layer([word_ids, attention_mask])[0]
    output_mask = MaskCalculator(
        output_dim=hidden_state_size, trainable=False,
        name='OutMaskCalculator'
    )(attention_mask)
    masked_sequence_output = tf.keras.layers.Multiply(
        name='OutMaskMultiplicator'
    )([output_mask, sequence_output])
    masked_sequence_output = tf.keras.layers.Masking(
        name='OutMasking'
    )(masked_sequence_output)
    pooled_output = tf.keras.layers.GlobalAvgPool1D(name='AvePool')(masked_sequence_output)
    text_embedding = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        name='Emdedding'
    )(pooled_output)
    cls_layer = tf.keras.layers.Dense(
        units=1, activation='sigmoid', use_bias=True,
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
        name='ClsOutput'
    )(pooled_output)
    cls_model = tf.keras.Model(
        inputs=[word_ids, attention_mask],
        outputs=cls_layer,
        name='Classifier'
    )
    cls_model.build(input_shape=[(None, max_len), (None, max_len)])
    fe_model = tf.keras.Model(
        inputs=[word_ids, attention_mask],
        outputs=text_embedding,
        name='FeatureExtractor'
    )
    fe_model.build(input_shape=[(None, max_len), (None, max_len)])
    return cls_model, fe_model
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.keras.backend.sum(tf.keras.backend.square(x - y),
                                      axis=1, keepdims=True)
    return tf.keras.backend.sqrt(
        tf.keras.backend.maximum(sum_square, tf.keras.backend.epsilon())
    )
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
def build_siamese_nn(transformer_name: str, hidden_state_size: int, max_len: int,
                     lr: float) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    left_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                          name="left_word_ids")
    left_attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                                name="left_attention_mask")
    right_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                           name="right_word_ids")
    right_attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                                 name="right_attention_mask")
    cls_, fe_ = build_feature_extractor(transformer_name, hidden_state_size, max_len)
    left_text_embedding = fe_([left_word_ids, left_attention_mask])
    right_text_embedding = fe_([right_word_ids, right_attention_mask])
    distance_layer = tf.keras.layers.Lambda(
        function=euclidean_distance,
        output_shape=eucl_dist_output_shape,
        name='L2DistLayer'
    )([left_text_embedding, right_text_embedding])
    left_cls_layer = cls_([left_word_ids, left_attention_mask])
    right_cls_layer = cls_([right_word_ids, right_attention_mask])
    nn = tf.keras.Model(
        inputs=[left_word_ids, left_attention_mask, right_word_ids, right_attention_mask],
        outputs=[distance_layer, left_cls_layer, right_cls_layer],
        name='SiameseXLMR'
    )
    nn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[
            DBLLogLoss(),
            tf.keras.losses.BinaryCrossentropy(),
            tf.keras.losses.BinaryCrossentropy()
        ],
        loss_weights=[
            1.0,
            0.3,
            0.3
        ]
    )
    fe_.summary()
    nn.summary()
    return nn, cls_, fe_
def build_classifier(config: XLMRobertaConfig, hidden_state_size: int, max_len: int,
                     lr: float, language: str) -> tf.keras.Model:
    word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                     name="word_ids_{0}".format(language))
    attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32,
                                           name="attention_mask_{0}".format(language))
    transformer_layer = TFXLMRobertaModel(
        config=config,
        name='Transformer_{0}'.format(language.title())
    )
    sequence_output = transformer_layer([word_ids, attention_mask])[0]
    output_mask = MaskCalculator(
        output_dim=hidden_state_size, trainable=False,
        name='OutMaskCalculator_{0}'.format(language)
    )(attention_mask)
    masked_sequence_output = tf.keras.layers.Multiply(
        name='OutMaskMultiplicator_{0}'.format(language)
    )([output_mask, sequence_output])
    masked_sequence_output = tf.keras.layers.Masking(
        name='OutMasking_{0}'.format(language)
    )(masked_sequence_output)
    pooled_output = tf.keras.layers.GlobalAvgPool1D(
        name='AvePool_{0}'.format(language)
    )(masked_sequence_output)
    cls_layer = tf.keras.layers.Dense(
        units=1, activation='sigmoid', use_bias=True,
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-3),
        name='ClsOutput_{0}'.format(language.title())
    )(pooled_output)
    cls_model = tf.keras.Model(
        inputs=[word_ids, attention_mask],
        outputs=cls_layer,
        name='Classifier_{0}'.format(language.title())
    )
    cls_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=DiceLoss(),
        metrics=[tf.keras.metrics.AUC()]
    )
    cls_model.summary()
    return cls_model
def show_training_process(history: tf.keras.callbacks.History, metric_name: str,
                          figure_id: int=1):
    val_metric_name = 'val_' + metric_name
    err_msg = 'The metric "{0}" is not found! Available metrics are: {1}'.format(
        metric_name, list(history.history.keys()))
    assert val_metric_name in history.history, err_msg
    err_msg = 'The metric "{0}" is not found! Available metrics are: {1}'.format(
        val_metric_name, list(history.history.keys()))
    assert metric_name in history.history, err_msg
    assert len(history.history[metric_name]) == len(history.history['val_' + metric_name])
    plt.figure(figure_id)
    plt.plot(list(range(len(history.history[metric_name]))),
             history.history[metric_name], label='Training {0}'.format(metric_name))
    plt.plot(list(range(len(history.history['val_' + metric_name]))),
             history.history['val_' + metric_name], label='Validation {0}'.format(metric_name))
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title('Training process')
    plt.legend(loc='best')
    plt.show()
def train_siamese_nn(nn: tf.keras.Model, trainset: tf.data.Dataset, steps_per_trainset: int,
                     steps_per_epoch: int, validset: tf.data.Dataset, max_duration: int,
                     model_weights_path: str):
    assert steps_per_trainset >= steps_per_epoch
    n_epochs = int(round(10.0 * steps_per_trainset / float(steps_per_epoch)))
    print('Maximal duration of the Siamese NN training is {0} seconds.'.format(max_duration))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=7, monitor='val_loss', mode='min',
                                         restore_best_weights=False, verbose=True),
        tfa.callbacks.TimeStopping(seconds=max_duration, verbose=True),
        tf.keras.callbacks.ModelCheckpoint(model_weights_path, monitor='val_loss',
                                           mode='min', save_best_only=True,
                                           save_weights_only=True, verbose=True)
    ]
    history = nn.fit(
        trainset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validset,
        epochs=n_epochs, callbacks=callbacks
    )
    show_training_process(history, 'loss')
def show_roc_auc(y_true: np.ndarray, probabilities: np.ndarray, label: str,
                 figure_id: int=1):
    plt.figure(figure_id)
    plt.plot([0, 1], [0, 1], 'k--')
    print('ROC-AUC score for {0} is {1:.9f}'.format(
        label, roc_auc_score(y_true=y_true, y_score=probabilities)
    ))
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=probabilities)
    plt.plot(fpr, tpr, label=label)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
def calculate_features_of_texts(texts: Dict[str, List[Tuple[str, int]]],
                                tokenizer: XLMRobertaTokenizer, maxlen: int,
                                fe: tf.keras.Model, batch_size: int,
                                max_dataset_size: int = 0) -> \
        Dict[str, Tuple[np.ndarray, np.ndarray]]:
    languages = sorted(list(texts.keys()))
    datasets_by_languages = dict()
    if max_dataset_size > 0:
        max_size_per_lang = max_dataset_size // len(languages)
        err_msg = '{0} is too small number of dataset samples!'.format(max_dataset_size)
        assert max_size_per_lang > 0, err_msg
    else:
        max_size_per_lang = 0
    for cur_lang in languages:
        selected_indices = list(range(len(texts[cur_lang])))
        if max_size_per_lang > 0:
            if len(selected_indices) > max_size_per_lang:
                selected_indices = random.sample(
                    population=selected_indices,
                    k=max_size_per_lang
                )
        tokens_of_texts, mask_of_texts = regular_encode(
            texts=[texts[cur_lang][idx][0] for idx in selected_indices],
            tokenizer=tokenizer, maxlen=maxlen
        )
        X = []
        n_batches = int(np.ceil(len(selected_indices) / float(batch_size)))
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(len(selected_indices), batch_start + batch_size)
            res = fe.predict_on_batch(
                [
                    tokens_of_texts[batch_start:batch_end],
                    mask_of_texts[batch_start:batch_end]
                ]
            )
            if not isinstance(res, np.ndarray):
                res = res.numpy()
            X.append(res)
            del res
        X = np.vstack(X)
        y = np.array([texts[cur_lang][idx][1] for idx in selected_indices], dtype=np.int32)
        datasets_by_languages[cur_lang] = (X, y)
        del X, y, selected_indices
    return datasets_by_languages
def predict_with_model(classifier: tf.keras.Model,
                       input_data: Tuple[np.ndarray, np.ndarray],
                       batch_size: int) -> np.ndarray:
    predicted = []
    n_batches = int(np.ceil(input_data[0].shape[0] / float(batch_size)))
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, input_data[0].shape[0])
        res = classifier.predict_on_batch(
            (
                input_data[0][batch_start:batch_end],
                input_data[1][batch_start:batch_end]
            )
        )
        if not isinstance(res, np.ndarray):
            res = res.numpy()
        predicted.append(res.flatten())
        del res
    return np.concatenate(predicted)
def do_submit(texts_for_training: Dict[str, List[Tuple[str, int]]],
              texts_for_submission: Dict[str, List[Tuple[str, int]]],
              language_for_validation: str, maxlen: int,
              base_classifier: tf.keras.Model, tokenizer: XLMRobertaTokenizer,
              max_duration: int, batch_size: int) -> Dict[int, float]:
    assert language_for_validation in texts_for_training
    print('Duration of the training procedure must be less ' \
          'than {0} seconds.'.format(max_duration))
    print('')
    start_time = time.time()
    texts_for_training_ = []
    texts_for_validation_ = []
    for cur_lang in sorted(list(texts_for_training.keys())):
        if cur_lang == language_for_validation:
            texts_for_validation_ += texts_for_training[cur_lang]
        else:
            texts_for_training_ += texts_for_training[cur_lang]
    X_train = regular_encode(
        texts=[cur[0] for cur in texts_for_training_],
        tokenizer=tokenizer, maxlen=maxlen
    )
    y_train = np.array([cur[1] for cur in texts_for_training_], dtype=np.int32)
    trainset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)
    ).repeat().batch(batch_size)
    print('Labeled dataset for training contains {0} samples.'.format(
        len(texts_for_training_)))
    print('Number of positive samples is {0} from {1}.'.format(
        int(sum(y_train)), y_train.shape[0]))
    print('')
    del X_train, y_train
    steps_per_epoch = int(np.ceil(len(texts_for_training_) / float(batch_size)))
    del texts_for_training_
    X_val = regular_encode(
        texts=[cur[0] for cur in texts_for_validation_],
        tokenizer=tokenizer, maxlen=maxlen
    )
    y_val = np.array([cur[1] for cur in texts_for_validation_], dtype=np.int32)
    validset = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)
    ).batch(batch_size)
    print('Dataset for validation contains {0} samples.'.format(y_val.shape[0]))
    print('Number of positive samples is {0} from {1}.'.format(int(sum(y_val)),
                                                               y_val.shape[0]))
    print('')
    steps_per_validset = int(np.ceil(y_val.shape[0] / float(batch_size)))
    if steps_per_epoch <= (3 * steps_per_validset):
        n_epochs = 15
    else:
        n_epochs = (15 * steps_per_epoch) // (3 * steps_per_validset)
        steps_per_epoch = 3 * steps_per_validset
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_auc', mode='max',
                                         restore_best_weights=True, verbose=True),
        tfa.callbacks.TimeStopping(seconds=max_duration, verbose=True)
    ]
    print('n_epochs = {0}, steps_per_epoch = {1}, val_steps = {2}'.format(
        n_epochs, steps_per_epoch, steps_per_validset))
    history = base_classifier.fit(
        trainset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validset,
        epochs=n_epochs, callbacks=callbacks
    )
    show_training_process(history, 'auc', figure_id=1)
    del history, trainset, validset
    print('')
    probabilities = predict_with_model(
        classifier=base_classifier,
        input_data=X_val,
        batch_size=batch_size
    )
    show_roc_auc(y_true=y_val, probabilities=probabilities,
                 label='language "{0}"'.format(language_for_validation),
                 figure_id=2)
    del probabilities, y_val, X_val
    del texts_for_validation_
    gc.collect()
    texts_for_submission_ = []
    identifiers_for_submission = []
    for cur_lang in sorted(list(texts_for_submission.keys())):
        texts_for_submission_ += texts_for_submission[cur_lang]
        identifiers_for_submission += [
            cur[1] for cur in texts_for_submission[cur_lang]
        ]
    X_submit = regular_encode(
        texts=[cur[0] for cur in texts_for_submission_],
        tokenizer=tokenizer, maxlen=maxlen
    )
    print('Dataset for submission contains {0} samples.'.format(
        len(texts_for_submission_)))
    print('')
    del texts_for_submission_
    print('')
    probabilities = predict_with_model(
        classifier=base_classifier,
        input_data=X_submit,
        batch_size=batch_size
    )
    assert probabilities.shape[0] == len(identifiers_for_submission)
    del X_submit
    submissions = dict()
    for sample_idx, cur_id in enumerate(identifiers_for_submission):
        submissions[cur_id] = probabilities[sample_idx]
    del identifiers_for_submission
    gc.collect()
    end_time = time.time()
    print('Duration of the final submission for language '\
          '"{0}" is {1:.3f} seconds'.format(
        language_for_validation, end_time - start_time))
    return submissions
experiment_start_time = time.time()
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    model_name = 'jplu/tf-xlm-roberta-large'
    max_seq_len = 256
    global_batch_size = 8 * strategy.num_replicas_in_sync
else:
    strategy = tf.distribute.get_strategy()
    if strategy.num_replicas_in_sync == 1:
        physical_devices = tf.config.list_physical_devices('GPU') 
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    max_seq_len = 128
    model_name = 'jplu/tf-xlm-roberta-base'
    global_batch_size = 16 * strategy.num_replicas_in_sync
print("REPLICAS: ", strategy.num_replicas_in_sync)
print('Model name: {0}'.format(model_name))
print('Maximal length of sequence is {0}'.format(max_seq_len))
print('Batch size is {0}'.format(global_batch_size))
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
siamese_learning_rate = 3e-6
classifier_learning_rate = 1e-6
dataset_dir = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification'
tmp_roberta_name = '/kaggle/working/base_nn.h5'
xlmroberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
xlmroberta_config = XLMRobertaConfig.from_pretrained(model_name)
print(xlmroberta_config)
sentence_embedding_size = xlmroberta_config.hidden_size
print('Sentence embedding size is {0}'.format(sentence_embedding_size))
assert max_seq_len <= xlmroberta_config.max_position_embeddings
corpus_for_training = load_train_set(
    os.path.join(dataset_dir, "jigsaw-unintended-bias-train.csv"),
    text_field="comment_text", lang_field="lang",
    sentiment_fields=["toxic", "severe_toxicity", "obscene", "identity_attack", "insult",
                      "threat"]
)
assert 'en' in corpus_for_training
corpus_for_validation = load_train_set(
    os.path.join(dataset_dir, "jigsaw-toxic-comment-train.csv"),
    text_field="comment_text", lang_field="lang",
    sentiment_fields=["toxic", "severe_toxic", "obscene", "threat", "insult",
                      "identity_hate"]
)
assert 'en' in corpus_for_validation
multilingual_corpus = load_train_set(
    os.path.join(dataset_dir, "validation.csv"),
    text_field="comment_text", lang_field="lang", sentiment_fields=["toxic", ]
)
assert 'en' not in multilingual_corpus
max_size = 0
print('Multilingual data:')
for language in sorted(list(multilingual_corpus.keys())):
    print('  {0}\t\t{1} samples'.format(language, len(multilingual_corpus[language])))
    assert set(map(lambda cur: cur[1], multilingual_corpus[language])) == {0, 1}
    if len(multilingual_corpus[language]) > max_size:
        max_size = len(multilingual_corpus[language])
multilingual_corpus['en'] = random.sample(
    population=corpus_for_validation['en'],
    k=max_size
)
del corpus_for_validation
texts_for_submission = load_test_set(
    os.path.join(dataset_dir, "test.csv"),
    text_field="content", lang_field="lang", id_field="id"
)
print('Data for submission:')
for language in sorted(list(texts_for_submission.keys())):
    print('  {0}\t\t{1} samples'.format(language, len(texts_for_submission[language])))
dataset_for_training, n_batches_per_data = build_siamese_dataset(
    texts=corpus_for_training, dataset_size=300000,
    tokenizer=xlmroberta_tokenizer, maxlen=max_seq_len,
    batch_size=global_batch_size, shuffle=True
)
dataset_for_validation, n_batches_per_epoch = build_siamese_dataset(
    texts=multilingual_corpus, dataset_size=2000,
    tokenizer=xlmroberta_tokenizer, maxlen=max_seq_len,
    batch_size=global_batch_size, shuffle=False
)
preparing_duration = int(round(time.time() - experiment_start_time))
print("Duration of data loading and preparing to the Siamese NN training is "
      "{0} seconds.".format(preparing_duration))
with strategy.scope():
    siamese_network, neural_classifier, feature_extractor = build_siamese_nn(
        transformer_name=model_name,
        hidden_state_size=sentence_embedding_size,
        max_len=max_seq_len,
        lr=siamese_learning_rate
    )
train_siamese_nn(nn=siamese_network, trainset=dataset_for_training,
                 steps_per_trainset=n_batches_per_data,
                 steps_per_epoch=min(10 * n_batches_per_epoch, n_batches_per_data),
                 validset=dataset_for_validation,
                 max_duration=int(round(3600 * 1.5 - preparing_duration)),
                 model_weights_path=tmp_roberta_name)
siamese_network.load_weights(tmp_roberta_name)
neural_classifier.save_weights(tmp_roberta_name, overwrite=True, save_format='h5')
del corpus_for_training
if dataset_for_training is not None:
    del dataset_for_training
if dataset_for_validation is not None:
    del dataset_for_validation
del neural_classifier, siamese_network
gc.collect()
dataset_for_training = calculate_features_of_texts(
    texts=multilingual_corpus,
    tokenizer=xlmroberta_tokenizer, maxlen=max_seq_len,
    fe=feature_extractor,
    batch_size=global_batch_size,
    max_dataset_size=min(300, global_batch_size * 3)
)
assert len(dataset_for_training) == 4
dataset_for_submission = calculate_features_of_texts(
    texts=texts_for_submission,
    tokenizer=xlmroberta_tokenizer, maxlen=max_seq_len,
    fe=feature_extractor,
    batch_size=global_batch_size,
    max_dataset_size=min(100, global_batch_size)
)
X_embedded = []
y_embedded = []
for cur_lang in dataset_for_training:
    X_embedded.append(dataset_for_training[cur_lang][0])
    y_embedded.append(dataset_for_training[cur_lang][1])
for cur_lang in dataset_for_submission:
    X_embedded.append(dataset_for_submission[cur_lang][0])
    y_embedded.append(
        np.array(
            [-1 for _ in range(dataset_for_submission[cur_lang][0].shape[0])],
            dtype=np.int32
        )
    )
X_embedded = np.vstack(X_embedded)
y_embedded = np.concatenate(y_embedded)
del dataset_for_training, dataset_for_submission, feature_extractor
X_embedded = TSNE(n_components=2, n_jobs=-1).fit_transform(X_embedded)
indices_of_unknown_classes = list(filter(
    lambda sample_idx: y_embedded[sample_idx] < 0,
    range(len(y_embedded))
))
xy = X_embedded[indices_of_unknown_classes]
plt.plot(xy[:, 0], xy[:, 1], 'o', color='b', markersize=2)
indices_of_negative_classes = list(filter(
    lambda sample_idx: y_embedded[sample_idx] == 0,
    range(len(y_embedded))
))
xy = X_embedded[indices_of_negative_classes]
plt.plot(xy[:, 0], xy[:, 1], 'o', color='g', markersize=4)
indices_of_positive_classes = list(filter(
    lambda sample_idx: y_embedded[sample_idx] > 0,
    range(len(y_embedded))
))
xy = X_embedded[indices_of_positive_classes]
plt.plot(xy[:, 0], xy[:, 1], 'o', color='r', markersize=6)
plt.title('Toxic and normal texts')
plt.show()
del indices_of_negative_classes
del indices_of_positive_classes
del indices_of_unknown_classes
del X_embedded, y_embedded
gc.collect()
tf.keras.backend.clear_session()
if tpu:
    tf.tpu.experimental.shutdown_tpu_system(tpu)
    del strategy
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
experiment_duration = int(round(time.time() - experiment_start_time))
print('Duration of siamese XLM-RoBERTa preparing is {0} seconds.'.format(
    experiment_duration))
all_languages_for_training = sorted(list(multilingual_corpus.keys()))
print('Languages in the labeled trainset: ' \
      '{0}'.format(all_languages_for_training))
assert len(all_languages_for_training) == 4
with strategy.scope():
    neural_classifier = build_classifier(
        config=xlmroberta_config,
        hidden_state_size=sentence_embedding_size,
        max_len=max_seq_len,
        lr=classifier_learning_rate,
        language=all_languages_for_training[0]
    )
neural_classifier.load_weights(tmp_roberta_name)
result_of_submission1 = do_submit(
    texts_for_training=multilingual_corpus,
    texts_for_submission=texts_for_submission,
    language_for_validation=all_languages_for_training[0],
    maxlen=max_seq_len, tokenizer=xlmroberta_tokenizer,
    base_classifier=neural_classifier,
    batch_size=global_batch_size,
    max_duration=int(round(2.5 * 3600) - (time.time() - experiment_start_time)) // 4
)
del neural_classifier
gc.collect()
tf.keras.backend.clear_session()
if tpu:
    tf.tpu.experimental.shutdown_tpu_system(tpu)
    del strategy
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
with strategy.scope():
    neural_classifier = build_classifier(
        config=xlmroberta_config,
        hidden_state_size=sentence_embedding_size,
        max_len=max_seq_len,
        lr=classifier_learning_rate,
        language=all_languages_for_training[1]
    )
neural_classifier.load_weights(tmp_roberta_name)
result_of_submission2 = do_submit(
    texts_for_training=multilingual_corpus,
    texts_for_submission=texts_for_submission,
    language_for_validation=all_languages_for_training[1],
    maxlen=max_seq_len, tokenizer=xlmroberta_tokenizer,
    base_classifier=neural_classifier,
    batch_size=global_batch_size,
    max_duration=int(round(2.5 * 3600) - (time.time() - experiment_start_time)) // 3
)
del neural_classifier
gc.collect()
tf.keras.backend.clear_session()
if tpu:
    tf.tpu.experimental.shutdown_tpu_system(tpu)
    del strategy
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
with strategy.scope():
    neural_classifier = build_classifier(
        config=xlmroberta_config,
        hidden_state_size=sentence_embedding_size,
        max_len=max_seq_len,
        lr=classifier_learning_rate,
        language=all_languages_for_training[2]
    )
neural_classifier.load_weights(tmp_roberta_name)
result_of_submission3 = do_submit(
    texts_for_training=multilingual_corpus,
    texts_for_submission=texts_for_submission,
    language_for_validation=all_languages_for_training[2],
    maxlen=max_seq_len, tokenizer=xlmroberta_tokenizer,
    base_classifier=neural_classifier,
    batch_size=global_batch_size,
    max_duration=int(round(2.5 * 3600) - (time.time() - experiment_start_time)) // 2
)
del neural_classifier
gc.collect()
tf.keras.backend.clear_session()
if tpu:
    tf.tpu.experimental.shutdown_tpu_system(tpu)
    del strategy
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
with strategy.scope():
    neural_classifier = build_classifier(
        config=xlmroberta_config,
        hidden_state_size=sentence_embedding_size,
        max_len=max_seq_len,
        lr=classifier_learning_rate,
        language=all_languages_for_training[3]
    )
neural_classifier.load_weights(tmp_roberta_name)
result_of_submission4 = do_submit(
    texts_for_training=multilingual_corpus,
    texts_for_submission=texts_for_submission,
    language_for_validation=all_languages_for_training[3],
    maxlen=max_seq_len, tokenizer=xlmroberta_tokenizer,
    base_classifier=neural_classifier,
    batch_size=global_batch_size,
    max_duration=int(round(2.5 * 3600) - (time.time() - experiment_start_time))
)
identifies = sorted(list(result_of_submission1.keys()))
with codecs.open('submission.csv', mode='w', encoding='utf-8', errors='ignore') as fp:
    fp.write('id,toxic\n')
    for id_val in identifies:
        proba_val = result_of_submission1[id_val]
        proba_val += result_of_submission2[id_val]
        proba_val += result_of_submission3[id_val]
        proba_val += result_of_submission4[id_val]
        proba_val /= 4.0
        fp.write('{0},{1:.9f}\n'.format(id_val, proba_val))
print('Experiment duration is {0:.3f}.'.format(time.time() - experiment_start_time))