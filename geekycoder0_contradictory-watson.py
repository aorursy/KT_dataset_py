import matplotlib.pyplot as plt
train_dataset = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")
lang_abv = train_dataset.groupby(['lang_abv']).size()
lang_abv_keys = lang_abv.keys().tolist()
lang_abv_vals = lang_abv.tolist()
print(lang_abv_keys)
print(lang_abv_vals)
print(len(lang_abv_keys))
print(len(lang_abv_vals))
y_pos = np.arange(len(lang_abv_keys))
plt.bar(y_pos, lang_abv_vals)
plt.xticks(y_pos, lang_abv_keys)
plt.show()
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow import keras

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)

#https://huggingface.co/transformers/glossary.html#attention-mask
class BertClassifier:

    MODEL_FILE_PATH = './model/bert_model.pkl'
    EPOCHS = 1
    MAX_LEN = 300

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        self.build_model()

    def pre_process_hyp_prem_pairs(self, premiseList, hypothesisList):
        premise_ids = []
        hypothesis_ids = []
        for i,premise in enumerate(premiseList):
            premise_tokens = self.tokenizer.tokenize("[CLS]" + premise + "[SEP]")
            hypothesis_tokens = self.tokenizer.tokenize(hypothesisList[i] + "[SEP]")
            premise_ids.append(self.tokenizer.convert_tokens_to_ids(premise_tokens))
            hypothesis_ids.append(self.tokenizer.convert_tokens_to_ids(hypothesis_tokens))

        premise_ids_tensor = tf.ragged.constant(premise_ids)
        hypthesis_ids_tensor = tf.ragged.constant(hypothesis_ids)
        premise_hypothesis_tensor = tf.concat([premise_ids_tensor, hypthesis_ids_tensor], axis=-1)

        input_mask = tf.ones_like(premise_hypothesis_tensor).to_tensor()
        type_s1 = tf.zeros_like(premise_ids_tensor)
        type_s2 = tf.ones_like(hypthesis_ids_tensor)
        input_type_ids = tf.concat([type_s1, type_s2], axis=-1).to_tensor()

        inputs = {
                    'input_word_ids': premise_hypothesis_tensor.to_tensor(),
                    'input_mask': input_mask,
                    'input_type_ids': input_type_ids
                 }
        return inputs

    def build_model(self):
        input_word_ids = tf.keras.Input(shape=(self.MAX_LEN,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(shape=(self.MAX_LEN,), dtype=tf.int32, name="input_mask")
        input_type_ids = tf.keras.Input(shape=(self.MAX_LEN,), dtype=tf.int32, name="input_type_ids")

        embedding = self.model([input_word_ids, input_mask, input_type_ids])[0]
        output = tf.keras.layers.Dense(3, activation='softmax')(embedding[:,0,:])
        self.nn_model = keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)
        self.nn_model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#         self.nn_model.summary()

    def train(self, inputs, labels):
        self.nn_model.fit(inputs, labels, epochs = 2, verbose = 1, batch_size = 64, validation_split = 0.2)
#         keras.models.save_model(self.nn_model, self.MODEL_FILE_PATH)

    def evaluate(self, inputs, labels):
#         prepared_model = keras.models.load_model(self.MODEL_FILE_PATH)
#         if prepared_model:
#             self.nn_model = prepared_model
        test_loss, test_acc = self.nn_model.evaluate(inputs, labels, verbose=2)
        return test_acc

    def predict(self, inputs):
#         prepared_model = keras.models.load_model(self.MODEL_FILE_PATH)
#         if prepared_model:
#             self.nn_model = prepared_model
        predictions = self.nn_model.predict(inputs)
        return predictions
import pandas as pd
import numpy as np
import tensorflow as tf

classifier = BertClassifier()

def load_sentences(file_path):
    df = pd.read_csv(file_path)
    return df["premise"], df["hypothesis"], df["id"], df["lang_abv"], (df["label"] if "label" in df else None)

def prepare_dataset(is_training):
    if is_training:
        premises, hypothesis, ids, lang_abv, labels = load_sentences("../input/contradictory-my-dear-watson/train.csv")
    else:
        premises, hypothesis, ids, lang_abv, labels = load_sentences("../input/contradictory-my-dear-watson/test.csv")
    inputs = classifier.pre_process_hyp_prem_pairs(premises, hypothesis)
    train_labels = np.array(labels) if labels is not None else None
    outcome = {
        'inputs': inputs,
        'labels': train_labels,
        'ids': ids,
        'lang_abv': lang_abv
    }
    return outcome

def train_and_evaluate():
    # prepare_train_dataset()
    outcome = prepare_dataset(True)
    classifier.train(outcome['inputs'], outcome['labels'])

    # eval_records = train_data[total_eval_len:]
    # eval_record_labels = train_labels[total_eval_len:]
    #
    # classifier.train(train_records, train_record_labels)
    # accuracy = classifier.evaluate(eval_records, eval_record_labels)
    # print(accuracy)

def predict_outcomes():
    outcomes = prepare_dataset(False)
    test_inputs = outcomes['inputs']
    print(test_inputs)
    ids = outcomes['ids']
    print("test ids")
    print(len(ids))
    results = classifier.predict(test_inputs)
    print(results)
    predictions = [np.argmax(i) for i in results]
    submission = pd.DataFrame(ids, columns=['id'])
    # print(submission)
    # print(predictions)
    submission['prediction'] = predictions
    submission.to_csv("submission.csv", index=False)

# prepare_dataset(True)
# train_and_evaluate()
# predict_outcomes()
