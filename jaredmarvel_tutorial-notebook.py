# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.environ["WANDB_API_KEY"] = "0" ## to silence warning
from transformers import BertTokenizer, TFBertModel

import matplotlib.pyplot as plt

import tensorflow as tf
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError:

    strategy = tf.distribute.get_strategy() # for CPU and single GPU

    print('Number of replicas:', strategy.num_replicas_in_sync)
train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")
train.head()
train.premise.values[1]
train.hypothesis.values[1]
train.label.values[1]
labels, frequencies = np.unique(train.language.values, return_counts = True)



plt.figure(figsize = (10,10))

plt.pie(frequencies,labels = labels, autopct = '%1.1f%%')

plt.show()
model_name = 'bert-base-multilingual-cased'

tokenizer = BertTokenizer.from_pretrained(model_name)
def encode_sentence(s):

   tokens = list(tokenizer.tokenize(s))

   tokens.append('[SEP]')

   return tokenizer.convert_tokens_to_ids(tokens)
encode_sentence("I love machine learning")
def bert_encode(hypotheses, premises, tokenizer):

    

  num_examples = len(hypotheses)

  

  sentence1 = tf.ragged.constant([

      encode_sentence(s)

      for s in np.array(hypotheses)])

  sentence2 = tf.ragged.constant([

      encode_sentence(s)

       for s in np.array(premises)])



  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]

  input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)



  input_mask = tf.ones_like(input_word_ids).to_tensor()



  type_cls = tf.zeros_like(cls)

  type_s1 = tf.zeros_like(sentence1)

  type_s2 = tf.ones_like(sentence2)

  input_type_ids = tf.concat(

      [type_cls, type_s1, type_s2], axis=-1).to_tensor()



  inputs = {

      'input_word_ids': input_word_ids.to_tensor(),

      'input_mask': input_mask,

      'input_type_ids': input_type_ids}



  return inputs
train_input = bert_encode(train.premise.values, train.hypothesis.values, tokenizer)
max_len = 50



def build_model():

    bert_encoder = TFBertModel.from_pretrained(model_name)

    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")

    

    embedding = bert_encoder([input_word_ids, input_mask, input_type_ids])[0]

    output = tf.keras.layers.Dense(3, activation='softmax')(embedding[:,0,:])

    

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)

    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    

    return model
with strategy.scope():

    model = build_model()

    model.summary()
model.fit(train_input, train.label.values, epochs = 10, verbose = 1, batch_size = 64, validation_split = 0.2)
train_predict = model.predict(train_input)



train_pred_output = np.argmax(train_predict,axis=1)



from sklearn import metrics



metrics.accuracy_score(train.label.values, train_pred_output)
import numpy as np





def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    """

    given a sklearn confusion matrix (cm), make a nice plot



    Arguments

    ---------

    cm:           confusion matrix from sklearn.metrics.confusion_matrix



    target_names: given classification classes such as [0, 1, 2]

                  the class names, for example: ['high', 'medium', 'low']



    title:        the text to display at the top of the matrix



    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm

                  see http://matplotlib.org/examples/color/colormaps_reference.html

                  plt.get_cmap('jet') or plt.cm.Blues

                  

    normalize:    If False, plot the raw numbers

                  If True, plot the proportions



    Usage

    -----

    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by

                                                              # sklearn.metrics.confusion_matrix

                          normalize    = True,                # show proportions

                          target_names = y_labels_vals,       # list of names of the classes

                          title        = best_estimator_name) # title of graph



    Citiation

    ---------

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



    """

    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy

    

    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

            

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
plot_confusion_matrix(cm           = np.array(metrics.confusion_matrix(train.label.values, train_pred_output, labels=[0,1,2])), 

                      normalize    = True,

                      target_names = ['entailment', 'neutral', 'contradiction'],

                      title        = "Confusion Matrix, Normalized")
# Printing the precision and recall, among other metrics

print(metrics.classification_report(train.label.values, train_pred_output, labels=[0,1,2]))
import numpy as np

import matplotlib.pyplot as plt



from sklearn.metrics import plot_confusion_matrix



np.set_printoptions(precision=2)



title = "Normalized confusion matrix"



disp = plot_confusion_matrix(classifier, X_test, y_test,

                                 display_labels=class_names,

                                 cmap=plt.cm.Blues,

                                 normalize='true')

disp.ax_.set_title(title)



print(title)

print(disp.confusion_matrix)



plt.show()
test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")

test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)
test.head()
predictions = [np.argmax(i) for i in model.predict(test_input)]
submission = test.id.copy().to_frame()

submission['prediction'] = predictions
submission.head()
submission.to_csv("submission.csv", index = False)