from sklearn.linear_model import LogisticRegression

from tensorflow import set_random_seed
from numpy import shape
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
set_random_seed(77)
import numpy
from keras.preprocessing.text import text_to_word_sequence
from matplotlib import pyplot as plt
import itertools


def predict_from_console(word2vecModel, model, words_in_sample, list_diagnoses):
    for i in range(100):
        print('\nВВЕДИТЕ ТЕКСТ: ')
        input_text = input().split()
        w2v_model_encoded_text = numpy.zeros((1, word2vecModel.vector_size*words_in_sample))
        j = 0
        for word in input_text:
            if word in word2vecModel.wv.vocab:
                w2v_model_encoded_text[:, j:j + word2vecModel.vector_size] = (word2vecModel[word])
                j += word2vecModel.vector_size

        predicted_class = model.predict_classes(w2v_model_encoded_text)[0]
        print(str(predicted_class) + " " + list_diagnoses[predicted_class])


def plot_training_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()


def precision(label, confusion_matrix):
    if confusion_matrix[label, label] == 0:
        return 0
    else:
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()


def recall(label, confusion_matrix):
    if confusion_matrix[label, label] == 0:
        return 0
    else:
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()


def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows


def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns


def f_measure_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_f = 0
    for label in range(columns):
        sum_of_f += f_measure(label, confusion_matrix)
    return sum_of_f / rows


def f_measure(label, confusion_matrix):
    p = precision(label, confusion_matrix)
    r = recall(label, confusion_matrix)
    if p + r == 0:
        return 0
    else:
        return (2 * p * r) / (p + r)


def words_in_sample(samples):
    """Number of max words in one sample"""
    max_len = 0
    for sample in samples:
        cur_sample = sample.split()
        max_len = len(cur_sample) if len(cur_sample) > max_len else max_len
    return max_len


def get_embedded_samples(samples, word2vecModel, words_in_sample):
    """get word2vec embeddings for given samples and words absent in given word2vec model"""
    new_x = numpy.zeros((samples.shape[0], word2vecModel.vector_size*words_in_sample))
    absent_words = []
    i = 0
    for sample in samples:
        current_sample = text_to_word_sequence(sample)
        newcur_x = numpy.zeros((1, word2vecModel.vector_size*words_in_sample))
        j = 0
        for word in current_sample:
            if word in word2vecModel.wv.vocab:
                newcur_x[:, j:j+word2vecModel.vector_size] = (word2vecModel[word])
                j += word2vecModel.vector_size
            else:
                absent_words.append(word)
        new_x[i] = newcur_x
        i += 1
    return new_x, absent_words


def vocabulary_size(samples):
    """Takes samples from the dataset and return number of unique words"""
    full_text = ""
    for cur_sample in samples:
        full_text = full_text + cur_sample

    words = set(text_to_word_sequence(full_text))
    return len(words)


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [numpy.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


def plot_confusion_matrix(cm, classes, list_diagnoses,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    #plt.yticks(tick_marks, list_diagnoses.values())
    plt.yticks(tick_marks, list_diagnoses)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # if cm[i, j] == 0:
        #    cm[i,j] = ''
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh or cm[i, j] == 0 else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

import os
print(os.listdir("../input"))
# new dataset of 24
#some_data = pandas.read_csv(
#    "C:/Users/admin/Documents/GitHub/Word-embedding/database_930_24.csv",
#    sep=' ; ', encoding='utf-8', engine='python', index_col=False).values


some_data = pd.read_csv("../input/diagnoses-from-text-rus/database_983_24_2.csv", sep=' ; ', encoding='utf-8', engine='python', index_col=False).values 
data = pd.read_csv("../input/diagnoses-from-text-rus/database_983_24_2.csv", sep=' ; ', encoding='utf-8', engine='python', index_col=False)
data.info()
labels = some_data[:, 0]
samples = some_data[:, 1]

set(labels)
#Word2Vec load
def words_in_sample(samples):
    """Number of max words in one sample"""
    max_len = 0
    for sample in samples:
        cur_sample = sample.split()
        max_len = len(cur_sample) if len(cur_sample) > max_len else max_len
    return max_len

words_in_sample = words_in_sample(samples)
from   gensim.models.keyedvectors   import KeyedVectors
#word2vecModel = Word2Vec.load(EMBEDDING_FILE)

word2vecModel = KeyedVectors.load_word2vec_format('../input/word2vec-model/model.bin', binary = False)
new_X, absentWords = get_embedded_samples(samples, word2vecModel, words_in_sample)
print("The dataset's shape: " + str(some_data.shape))
print("The size of the dataset's vocabulary: " + str(vocabulary_size(samples)))
print("Max length of samples: " + str(words_in_sample) + " words")
print("The Word2vec model loaded: " + str(word2vecModel))
print("Shape of embedded samples: " + str(shape(new_X)))
print("Unique words from dataset absent in the word2vec Model: " + str(len(set(absentWords))))
print("Total words from dataset absent in the word2vec Model: " + str(len(absentWords)))
X_train, X_test, y_train, y_test = train_test_split(new_X, labels, test_size=0.3, random_state=77)
# encode data
encoder = LabelEncoder()
encoder.fit(labels)
encoded_labels = encoder.transform(labels)
Y_encoded = np_utils.to_categorical(encoded_labels)
listDiagnoses = set(labels)
#for label in set(encoded_labels):
#    listDiagnoses[label] = encoder.inverse_transform(label)
classifier_1 = LogisticRegression(random_state=0, solver='lbfgs',
                                  multi_class='multinomial')

classifier_1.fit(X_train, y_train)
score = classifier_1.score(X_test, y_test)

pred_val = classifier_1.predict(X_test)

cm = confusion_matrix(y_test, pred_val)
num_of_classes = len(set(encoded_labels))

#print("{0:50} {1:8} {2:10} {3:10}".format('label', 'precision', 'recall', 'F1'))
#for label in range(num_of_classes):
#    print(f"{listDiagnoses[label]:50} {precision(label, cm):9.3f} {recall(label, cm):6.3f} {f_measure(label, cm):6.3f}")
print('============================================================================')
print('Accuracy test:     %f' % (score*100))
print('Recall average:    %f' % (recall_macro_average(cm)*100))
print('Precision average: %f' % (precision_macro_average(cm)*100))
print('F measure average: %f' % (f_measure_macro_average(cm)*100))
plot_confusion_matrix(cm, list(set(encoded_labels)),
                           listDiagnoses,
                           normalize=True,
                           title='LR Cross-entr loss Confusion Matrix')
plt.show()