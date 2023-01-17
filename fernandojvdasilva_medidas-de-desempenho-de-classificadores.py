
import pickle

from sklearn.metrics import classification_report
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
from sklearn.metrics import f1_score

######## Funções Utilitárias para o modelo #################
stopwords_list = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def event_tokenizer(words):    
    #print("POS Tagging...\n")
    pos_tags = pos_tag(words)
     
    #print("Stopwords removal...\n")
    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]
     
    #print("Punctuation removal...\n")
    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]
     
    #print("Lemmatization...\n")
    lemmas = []
    for w in non_punctuation:
        if w[1].startswith('J'):
             pos = wordnet.ADJ
        elif w[1].startswith('V'):
             pos = wordnet.VERB
        elif w[1].startswith('N'):
             pos = wordnet.NOUN
        elif w[1].startswith('R'):
             pos = wordnet.ADV
        else:
             pos = wordnet.NOUN
         
        lemmas.append(lemmatizer.lemmatize(w[0], pos))

    #print("Done Tokenization!\n")
    return lemmas
    
class SVDDimSelect(object):
    def fit(self, X, y=None):      
        print("SVDDimSelect: Fitting to shape/2\n")         
        self.svd_transformer = TruncatedSVD(n_components=round(X.shape[1]/2))
        self.svd_transformer.fit(X)
        
        print("SVDDimSelect: Sorting cummulative variances\n")
        cummulative_variance = 0.0
        k = 0
        for var in sorted(self.svd_transformer.explained_variance_ratio_)[::-1]:
            cummulative_variance += var
            # I empirically decided to choose only the 20% of the components with 
            # most variance.
            if cummulative_variance >= 0.95:
                break
            else:
                k += 1
                
        
        print("SVDDimSelect: number of components = %d\n" % k)
        self.svd_transformer = TruncatedSVD(n_components=k)
        return self.svd_transformer.fit(X)
    
    def transform(self, X, Y=None):
        return self.svd_transformer.transform(X)

########################


# Carrega dataset de teste    
x_test_evt = pickle.load(open('../input/data-preparation/football_events_x_test.pkl', 'rb'))
y_test_evt = pickle.load(open('../input/data-preparation/football_events_y_test.pkl', 'rb'))

# Carregando modelo de Regressão Logística
model_event = pickle.load(open('../input/events-classifier-models/football_event_model_LOG-REGRESSION.pkl', 'rb'))

y_pred = model_event.predict(x_test_evt)

''' Cada coluna da matriz y_pred representa uma das categorias do evento, sendo:
Attempt, Corner, Foul, Yellow card, Second yellow card, Red card, Substitution, 
Free kick won, Offside, Hand ball, Penalty conceded, Key Pass, Failed through ball, Sending off '''

print(classification_report(y_test_evt, y_pred, target_names=['Attempt', 'Corner', 
                                                              'Foul', 'Yellow card', 
                                                              'Second yellow card', 
                                                              'Red card', 'Substitution', 
                                                              'Free kick won', 'Offside', 
                                                              'Hand ball', 'Penalty conceded', 
                                                              'Key Pass', 
                                                              'Failed through ball', 'Sending off']))

print(f1_score(y_test_evt, y_pred, average=None))
 
probas = model_event.predict_proba(x_test_evt)
print(probas)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Cartões vermelhos estão na coluna 5
RED_CARD_INDEX = 5
fpr, tpr, thresholds = roc_curve(y_test_evt[:,RED_CARD_INDEX], probas[:,RED_CARD_INDEX])
roc_auc = auc(fpr, tpr)

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Aleatório')
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % roc_auc)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Falsos Positivos')
plt.ylabel('Positivos Verdadeiros')
plt.title('ROC do Logistic Regression para Cartões Vermelhos')
plt.legend(loc="lower right")
plt.show()


y_test_evt_multi = []
y_pred_multi = []

for y_sample, pred_sample in zip(y_test_evt, y_pred):    
    for i in range(len(y_sample)):
        if y_sample[i] == 1:
            y_test_evt_multi.append(i)
            break
    check = False
    for i in range(len(pred_sample)):         
        if pred_sample[i] == 1:
            y_pred_multi.append(i)
            check = True
            break
    if not check:
        y_pred_multi.append(len(pred_sample))
            
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test_evt_multi, y_pred_multi)
print(cnf_matrix)
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                         print_numbers=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    if print_numbers:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
import numpy as np
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Attempt', 'Corner', 
                                                              'Foul', 'Yellow card', 
                                                              'Second yellow card', 
                                                              'Red card', 'Substitution', 
                                                              'Free kick won', 'Offside', 
                                                              'Hand ball', 'Penalty conceded', 
                                                              'Key Pass', 
                                                              'Failed through ball', 'Sending off'],
                      title='Confusion matrix sem normalização')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Attempt', 'Corner', 
                                                              'Foul', 'Yellow card', 
                                                              'Second yellow card', 
                                                              'Red card', 'Substitution', 
                                                              'Free kick won', 'Offside', 
                                                              'Hand ball', 'Penalty conceded', 
                                                              'Key Pass', 
                                                              'Failed through ball', 'Sending off'],
                      title='Confusion matrix com normalização', normalize=True)

plt.show()
## Código copiado da versão de desenvolvimento do sklearn ##

import numpy as np
from sklearn.utils import check_consistent_length
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.utils.multiclass import type_of_target, unique_labels
from scipy.sparse import csr_matrix

def _check_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same classification task
    This converts multiclass or binary types to a common shape, and raises a
    ValueError for a mix of multilabel and multiclass targets, a mix of
    multilabel formats, for the presence of continuous-valued or multioutput
    targets, or for targets of different lengths.
    Column vectors are squeezed to 1d, while multilabel formats are returned
    as CSR sparse label indicators.
    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multiclass', 'binary'}
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``
    y_true : array or indicator matrix
    y_pred : array or indicator matrix
    """
    check_consistent_length(y_true, y_pred)
    type_true = type_of_target(y_true)
    type_pred = type_of_target(y_pred)

    y_type = set([type_true, type_pred])
    if y_type == set(["binary", "multiclass"]):
        y_type = set(["multiclass"])

    if len(y_type) > 1:
        raise ValueError("Classification metrics can't handle a mix of {0} "
                         "and {1} targets".format(type_true, type_pred))

    # We can't have more than one value on y_type => The set is no more needed
    y_type = y_type.pop()

    # No metrics support "multiclass-multioutput" format
    if (y_type not in ["binary", "multiclass", "multilabel-indicator"]):
        raise ValueError("{0} is not supported".format(y_type))

    if y_type in ["binary", "multiclass"]:
        y_true = column_or_1d(y_true)
        y_pred = column_or_1d(y_pred)
        if y_type == "binary":
            unique_values = np.union1d(y_true, y_pred)
            if len(unique_values) > 2:
                y_type = "multiclass"

    if y_type.startswith('multilabel'):
        y_true = csr_matrix(y_true)
        y_pred = csr_matrix(y_pred)
        y_type = 'multilabel-indicator'

    return y_type, y_true, y_pred

def multilabel_confusion_matrix(y_true, y_pred, sample_weight=None,
                                labels=None, samplewise=False):
    """Compute a confusion matrix for each class or sample
    .. versionadded:: 0.21
    Compute class-wise (default) or sample-wise (samplewise=True) multilabel
    confusion matrix to evaluate the accuracy of a classification, and output
    confusion matrices for each class or sample.
    In multilabel confusion matrix :math:`MCM`, the count of true negatives
    is :math:`MCM_{:,0,0}`, false negatives is :math:`MCM_{:,1,0}`,
    true positives is :math:`MCM_{:,1,1}` and false positives is
    :math:`MCM_{:,0,1}`.
    Multiclass data will be treated as if binarized under a one-vs-rest
    transformation. Returned confusion matrices will be in the order of
    sorted unique labels in the union of (y_true, y_pred).
    Read more in the :ref:`User Guide <multilabel_confusion_matrix>`.
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        of shape (n_samples, n_outputs) or (n_samples,)
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        of shape (n_samples, n_outputs) or (n_samples,)
        Estimated targets as returned by a classifier
    sample_weight : array-like of shape = (n_samples,), optional
        Sample weights
    labels : array-like
        A list of classes or column indices to select some (or to force
        inclusion of classes absent from the data)
    samplewise : bool, default=False
        In the multilabel case, this calculates a confusion matrix per sample
    Returns
    -------
    multi_confusion : array, shape (n_outputs, 2, 2)
        A 2x2 confusion matrix corresponding to each output in the input.
        When calculating class-wise multi_confusion (default), then
        n_outputs = n_labels; when calculating sample-wise multi_confusion
        (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
        the results will be returned in the order specified in ``labels``,
        otherwise the results will be returned in sorted order by default.
    See also
    --------
    confusion_matrix
    Notes
    -----
    The multilabel_confusion_matrix calculates class-wise or sample-wise
    multilabel confusion matrices, and in multiclass tasks, labels are
    binarized under a one-vs-rest way; while confusion_matrix calculates
    one confusion matrix for confusion between every two classes.
    Examples
    --------
    Multilabel-indicator case:
    >>> import numpy as np
    >>> from sklearn.metrics import multilabel_confusion_matrix
    >>> y_true = np.array([[1, 0, 1],
    ...                    [0, 1, 0]])
    >>> y_pred = np.array([[1, 0, 0],
    ...                    [0, 1, 1]])
    >>> multilabel_confusion_matrix(y_true, y_pred)
    array([[[1, 0],
            [0, 1]],
    <BLANKLINE>
           [[1, 0],
            [0, 1]],
    <BLANKLINE>
           [[0, 1],
            [1, 0]]])
    Multiclass case:
    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> multilabel_confusion_matrix(y_true, y_pred,
    ...                             labels=["ant", "bird", "cat"])
    array([[[3, 1],
            [0, 2]],
    <BLANKLINE>
           [[5, 0],
            [1, 0]],
    <BLANKLINE>
           [[2, 1],
            [1, 2]]])
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    check_consistent_length(y_true, y_pred, sample_weight)

    if y_type not in ("binary", "multiclass", "multilabel-indicator"):
        raise ValueError("%s is not supported" % y_type)

    present_labels = unique_labels(y_true, y_pred)
    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels,
                                                 assume_unique=True)])

    if y_true.ndim == 1:
        if samplewise:
            raise ValueError("Samplewise metrics are not available outside of "
                             "multilabel classification.")

        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = np.bincount(tp_bins, weights=tp_bins_weights,
                                 minlength=len(labels))
        else:
            # Pathological case
            true_sum = pred_sum = tp_sum = np.zeros(len(labels))
        if len(y_pred):
            pred_sum = np.bincount(y_pred, weights=sample_weight,
                                   minlength=len(labels))
        if len(y_true):
            true_sum = np.bincount(y_true, weights=sample_weight,
                                   minlength=len(labels))

        # Retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]
        pred_sum = pred_sum[indices]

    else:
        sum_axis = 1 if samplewise else 0

        # All labels are index integers for multilabel.
        # Select labels:
        if not np.array_equal(labels, present_labels):
            if np.max(labels) > np.max(present_labels):
                raise ValueError('All labels must be in [0, n labels) for '
                                 'multilabel targets. '
                                 'Got %d > %d' %
                                 (np.max(labels), np.max(present_labels)))
            if np.min(labels) < 0:
                raise ValueError('All labels must be in [0, n labels) for '
                                 'multilabel targets. '
                                 'Got %d < 0' % np.min(labels))

        if n_labels is not None:
            y_true = y_true[:, labels[:n_labels]]
            y_pred = y_pred[:, labels[:n_labels]]

        # calculate weighted counts
        true_and_pred = y_true.multiply(y_pred)
        tp_sum = count_nonzero(true_and_pred, axis=sum_axis,
                               sample_weight=sample_weight)
        pred_sum = count_nonzero(y_pred, axis=sum_axis,
                                 sample_weight=sample_weight)
        true_sum = count_nonzero(y_true, axis=sum_axis,
                                 sample_weight=sample_weight)

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum

    if sample_weight is not None and samplewise:
        sample_weight = np.array(sample_weight)
        tp = np.array(tp)
        fp = np.array(fp)
        fn = np.array(fn)
        tn = sample_weight * y_true.shape[1] - tp - fp - fn
    elif sample_weight is not None:
        tn = sum(sample_weight) - tp - fp - fn
    elif samplewise:
        tn = y_true.shape[1] - tp - fp - fn
    else:
        tn = y_true.shape[0] - tp - fp - fn

    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)
# from sklearn.metrics import multilabel_confusion_matrix

multilabel_confusion_matrix(y_test_evt, y_pred)
num_parts = 30

test_sets_x = np.array_split(x_test_evt, num_parts)
test_sets_y = np.array_split(y_test_evt, num_parts)
model_event_rf = pickle.load(open('../input/events-classifier-models/football_event_model_RANDOM-FOREST.pkl', 'rb'))

test_results_lr = []
test_results_rf = []

for i in range(num_parts):
    y_pred_part = model_event.predict(test_sets_x[i])
    test_results_lr.append(f1_score(test_sets_y[i], y_pred_part, average='weighted'))

    y_pred_part = model_event_rf.predict(test_sets_x[i])
    test_results_rf.append(f1_score(test_sets_y[i], y_pred_part, average='weighted'))

from pandas import DataFrame

test_results_pd = DataFrame()
test_results_pd['lr'] = test_results_lr
test_results_pd['rf'] = test_results_rf

print(test_results_pd.describe())
print(test_results_pd.var)

from scipy.stats import normaltest

value, p = normaltest(test_results_lr)

if p >= 0.05:
    print("Resultados de Logistic Regression provavelmente seguem uma distribuição normal")
else:
    print("Resultados de Logistic Regression provavelmente NÃO seguem uma distribuição normal")
    
value, p = normaltest(test_results_rf)

if p >= 0.05:
    print("Resultados de Random Forest provavelmente seguem uma distribuição normal")
else:
    print("Resultados de Random Forest provavelmente NÃO seguem uma distribuição normal")

from scipy.stats import ks_2samp

value, pvalue = ks_2samp(test_results_lr, test_results_rf)

if pvalue >= 0.05:
    print("Resultados dos modelos vem da mesma população e NÃO são estatisticamente diferentes")
else:
    print("Resultados dos modelos vem de populações diferentes e SÃO estatisticamente diferentes")

y_pred = model_event.predict(x_test_evt)

''' Cada coluna da matriz y_pred representa uma das categorias do evento, sendo:
Attempt, Corner, Foul, Yellow card, Second yellow card, Red card, Substitution, 
Free kick won, Offside, Hand ball, Penalty conceded, Key Pass, Failed through ball, Sending off '''

print(classification_report(y_test_evt, y_pred, target_names=['Attempt', 'Corner', 
                                                              'Foul', 'Yellow card', 
                                                              'Second yellow card', 
                                                              'Red card', 'Substitution', 
                                                              'Free kick won', 'Offside', 
                                                              'Hand ball', 'Penalty conceded', 
                                                              'Key Pass', 
                                                              'Failed through ball', 'Sending off']))

y_pred = model_event_rf.predict(x_test_evt)

''' Cada coluna da matriz y_pred representa uma das categorias do evento, sendo:
Attempt, Corner, Foul, Yellow card, Second yellow card, Red card, Substitution, 
Free kick won, Offside, Hand ball, Penalty conceded, Key Pass, Failed through ball, Sending off '''

print(classification_report(y_test_evt, y_pred, target_names=['Attempt', 'Corner', 
                                                              'Foul', 'Yellow card', 
                                                              'Second yellow card', 
                                                              'Red card', 'Substitution', 
                                                              'Free kick won', 'Offside', 
                                                              'Hand ball', 'Penalty conceded', 
                                                              'Key Pass', 
                                                              'Failed through ball', 'Sending off']))

