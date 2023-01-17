from nltk.corpus import treebank

from nltk.tag.sequential import ClassifierBasedPOSTagger
import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline

import numpy as np

import mglearn

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import nltk

from nltk.tokenize import word_tokenize

from nltk.stem.snowball import SnowballStemmer

import itertools
# -------------- Main code

train = pd.read_csv('train.csv')

train_sents = treebank.tagged_sents()

tagger = ClassifierBasedPOSTagger(train=train_sents)

stemmer = SnowballStemmer('english')
# Define tag sequences

SEQ_1 = "SEQ_1: {<DT|PP>?<JJ>*}"

SEQ_2 = "SEQ_2: {<NN><DT|PP\$>?<JJ>}"

SEQ_3 = "SEQ_3: {<NP>?<VERB>?<NP|JJ>}"

SEQ_4 = "SEQ_4: {<VB.*><NP|PP|CLAUSE>+$}"



cp1 = nltk.RegexpParser(SEQ_1)

cp2 = nltk.RegexpParser(SEQ_2)

cp3 = nltk.RegexpParser(SEQ_3)

cp4 = nltk.RegexpParser(SEQ_4)



lst_seq = list([cp1, cp2, cp3, cp4])
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

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

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')





def get_number_of_spaces(sentence):

    return sentence.count(' ')





def get_number_of_capitals(sentence):

    n = sum(1 for c in sentence if c.isupper())

    return n





def get_number_of_nouns(taged_tokens):

    n = sum(1 for word, tag in taged_tokens if tag == 'NN' or tag == 'NNS' \

            or tag == 'NNP' or tag == 'NNP')

    return n





def get_number_of_adjectives(taged_tokens):

    n = sum(1 for word, tag in taged_tokens if tag == 'JJ' or tag == 'JJR' or tag == 'JJS')

    return n





def get_count_of_tagged(taged_tokens, tag_in):

    n = sum(1 for word, tag in taged_tokens if tag == tag_in)

    return n





def is_past_tense(taged_tokens):

    n = sum(1 for word, tag in taged_tokens if tag == 'VBD')

    return (n > 0)





def is_modal(taged_tokens):

    n = sum(1 for word, tag in taged_tokens if tag == 'MD')

    return (n > 0)





def vocab_richness(sentence):

    unique = set(sentence.split())

    count_uniques = len(unique)

    return count_uniques





def get_first_words(sentence, count):

    arr_words = sentence.split()

    ret_words = arr_words[:count]

    str_ret = ' '.join(ret_words)

    return str_ret





def get_one_word(sentence, position):

    arr_words = sentence.split()

    if len(arr_words) >= (position + 1):

        ret_word = arr_words[position]

        return ret_word

    else:

        return False





def exists_she(sentense):

    if 'she' in sentense.lower():

        return True

    else:

        return False





def exists_he(sentense):

    if 'he' in sentense.lower():

        return True

    else:

        return False







def first_tag(taged_tokens):

    return str(taged_tokens[0][1])





def second_tag(taged_tokens):

    if len(taged_tokens) > 1:

        return str(taged_tokens[1][1])

    else:

        return False





def third_tag(taged_tokens):

    if len(taged_tokens) > 2:

        return str(taged_tokens[2][1])

    else:

        return False



def get_consonant_letters(sentence):

    consonants = 0

    for word in sentence:

        for letter in word:

            if letter in 'bcdfghjklmnpqrstvwxz':

                consonants += 1



    return consonants





def get_sonant_letters(sentence):

    sonants = 0

    for word in sentence:

        for letter in word:

            if letter in 'aieouy':

                sonants += 1



    return sonants





def lexical_diversity(text):

    return len(set(text)) / len(text)





def get_sequence_tags(taged_tokens, n_sequence):

    countSequence = 0

    cp = lst_seq[n_sequence-1]

    result = cp.parse(taged_tokens)



    for tre in result:

        if isinstance(tre, nltk.tree.Tree):

            if tre.label() ==  cp._stages[0]._chunk_label:

                countSequence += 1



    return (countSequence > 0)
def get_sentence_features(sentens_in):

    stemmed_words = list()

    for w in sentens_in.split():

        stemmed_words.append(stemmer.stem(w))



    sentence = ' '.join(stemmed_words)

    word_tokens = nltk.wordpunct_tokenize(sentence)



    taged_tokens = tagger.tag(word_tokens)



    X_dict = {}



    X_dict['seq_01'] = get_sequence_tags(taged_tokens, 1)

    X_dict['seq_02'] = get_sequence_tags(taged_tokens, 2)

    X_dict['seq_03'] = get_sequence_tags(taged_tokens, 3)

    X_dict['seq_04'] = get_sequence_tags(taged_tokens, 4)



    X_dict['lexical_diversity'] = lexical_diversity(sentence.lower())

    X_dict['get_consonant_letters'] = get_consonant_letters(sentence.lower())

    X_dict['get_sonant_letters'] = get_sonant_letters(sentence.lower())



    X_dict['count_of_spaces'] = get_number_of_spaces(sentence)

    X_dict['count_capitals'] = get_number_of_capitals(sentence)

    X_dict['count_nouns'] = get_number_of_nouns(taged_tokens)

    X_dict['count_adjectives'] = get_number_of_adjectives(taged_tokens)



    X_dict['count_numbers'] = get_count_of_tagged(taged_tokens, 'CD')

    X_dict['count_NNS'] = get_count_of_tagged(taged_tokens, 'NNS')

    X_dict['count_NNP'] = get_count_of_tagged(taged_tokens, 'NNP')

    X_dict['count_NNPS'] = get_count_of_tagged(taged_tokens, 'NNPS')

    X_dict['count_RBS'] = get_count_of_tagged(taged_tokens, 'RBS')

    X_dict['count_RBR'] = get_count_of_tagged(taged_tokens, 'RBR')

    X_dict['count_WP'] = get_count_of_tagged(taged_tokens, 'WP')

    X_dict['count_WP$'] = get_count_of_tagged(taged_tokens, 'WP$')

    X_dict['count_WRB'] = get_count_of_tagged(taged_tokens, 'WRB')

    X_dict['count_PRP'] = get_count_of_tagged(taged_tokens, 'PRP')

    X_dict['count_POS'] = get_count_of_tagged(taged_tokens, 'POS')

    X_dict['count_FW'] = get_count_of_tagged(taged_tokens, 'FW')

    X_dict['count_VB'] = get_count_of_tagged(taged_tokens, 'VB')

    X_dict['count_VBD'] = get_count_of_tagged(taged_tokens, 'VBD')

    X_dict['count_VBG'] = get_count_of_tagged(taged_tokens, 'VBG')

    X_dict['count_VBN'] = get_count_of_tagged(taged_tokens, 'VBN')

    X_dict['count_CC'] = get_count_of_tagged(taged_tokens, 'CC')



    X_dict['count_DT']         = get_count_of_tagged(taged_tokens, 'DT')

    X_dict['count_UH']         = get_count_of_tagged(taged_tokens, 'UH')

    X_dict['count_SYM']        = get_count_of_tagged(taged_tokens, 'SYM')

    X_dict['count_PDT']        = get_count_of_tagged(taged_tokens, 'PDT')

    X_dict['count_LS']         = get_count_of_tagged(taged_tokens, 'LS')



    X_dict['count_3rd person'] = get_count_of_tagged(taged_tokens, 'VBZ')

    X_dict['count_gerund'] = get_count_of_tagged(taged_tokens, 'VBG')



    X_dict['is_past_tense'] = is_past_tense(taged_tokens)

    X_dict['is_modal'] = is_modal(taged_tokens)

    X_dict['vocab_richness'] = vocab_richness(sentence)

    X_dict['first_tag'] = first_tag(taged_tokens)

    X_dict['second_tag'] = second_tag(taged_tokens)

    X_dict['third_tag'] = third_tag(taged_tokens)

    

    X_dict['first_one_word'] = get_one_word(sentence, 0)

    X_dict['second_one_word'] = get_one_word(sentence, 1)

    X_dict['third_one_word'] = get_one_word(sentence, 2)

    X_dict['forth_one_word'] = get_one_word(sentence, 3)

    X_dict['fifth_one_word'] = get_one_word(sentence, 4)

    X_dict['sixth_one_word'] = get_one_word(sentence, 5)

    X_dict['seventh_one_word'] = get_one_word(sentence, 6)

    X_dict['eith_one_word'] = get_one_word(sentence, 7)

    X_dict['ninth_one_word'] = get_one_word(sentence, 8)

    X_dict['tenth_one_word'] = get_one_word(sentence, 9)



    X_dict['first_6_word'] = get_first_words(sentence, 6)

    X_dict['first_5_word'] = get_first_words(sentence, 5)

    X_dict['first_4_word'] = get_first_words(sentence, 4)

    X_dict['first_3_word'] = get_first_words(sentence, 3)

    X_dict['first_2_word'] = get_first_words(sentence, 2)



    X_dict['exists_she'] = exists_she(sentence)

    X_dict['exists_he']  = exists_he(sentence)



    X_dict['first_word_is_the'] = ('the' == get_first_words(sentence.lower(), 1))

    X_dict['first_word_is_she'] = ('she' == get_first_words(sentence.lower(), 1))

    X_dict['first_word_is_he']  = ('he' == get_first_words(sentence.lower(), 1))

    X_dict['first_word_is_it']  = ('it' == get_first_words(sentence.lower(), 1))

    X_dict['first_word_is_this'] = ('this' == get_first_words(sentence.lower(), 1))

    X_dict['first_word_is_you']  = ('you' == get_first_words(sentence.lower(), 1))





    X_dict['Raymond'] = ('raymond' in sentence.lower())

    X_dict['Perdita'] = ('perdita' in sentence.lower())

    X_dict['Idris']   = ('idris' in sentence.lower())

    X_dict['Adrian']  = ('adrian' in sentence.lower())

    X_dict['Chapter'] = ('chapter' in sentence.lower())

    X_dict['sinister'] = ('sinister' in sentence.lower())

    X_dict['weird']    = ('weird' in sentence.lower())

    X_dict['horrible'] = ('horrible' in sentence.lower())



    return X_dict
lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(train.author.values)
xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y,

                                                  stratify=y,

                                                  random_state=32,

                                                  test_size=0.2, shuffle=True)

print(xtrain.shape)

print(xvalid.shape)
from nltk.corpus import stopwords



stop_words = set(stopwords.words('english'))

stop_words.update( ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
def get_train_features(IN_x, IN_y):

    Out_x, Out_y = [], []

    index = 0



    for sentens_edna in IN_x:

        word_tokens1 = [i for i in word_tokenize(sentens_edna) if i not in stop_words]

        sentens_in = ' '.join(word_tokens1)

        X_feat_dict = get_sentence_features(sentens_in)

        Out_x.append(X_feat_dict)

        Out_y.append(IN_y[index])

        index += 1



    return Out_x, Out_y





X_Train, Y_train = get_train_features(xtrain, ytrain)

X_valid, Y_valid = get_train_features(xvalid, yvalid)
clf = grid.best_estimator_.named_steps['bernoullinb']

coef = grid.best_estimator_.named_steps['bernoullinb'].coef_

best_alpha = grid.best_estimator_.named_steps['bernoullinb'].alpha

print("Best cross-validation alpha: {:.2f}".format(best_alpha))
feature_names = np.array(dict_vect.get_feature_names())



mglearn.tools.visualize_coefficients(coef[0], feature_names, n_top_features=25)

mglearn.tools.visualize_coefficients(coef[1], feature_names, n_top_features=25)

mglearn.tools.visualize_coefficients(coef[2], feature_names, n_top_features=25)
predictions = clf.predict_proba(X_valid_ctv)

predicted_lables = clf.predict(X_valid_ctv)



cnf_matrix = confusion_matrix(Y_valid, predicted_lables)



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')



plt.show()
from sklearn.metrics import roc_curve, auc

n_classes = len(class_names)

from sklearn.preprocessing import label_binarize



# Binarize the output

Y_valid = label_binarize(Y_valid, classes=[0, 1, 2])



# Compute ROC curve and ROC area for each class

fpr     = dict()

tpr     = dict()

roc_auc = dict()



plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')





for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(  Y_valid[:,i] , predictions[:, i] )

    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[i], tpr[i], label=class_names[i] + 'ROC curve (area = %0.2f)' % roc_auc[i])



print('EAP ROC curve (area = %0.2f)' % roc_auc[0])

print('HPL ROC curve (area = %0.2f)' % roc_auc[1])

print('MWS ROC curve (area = %0.2f)' % roc_auc[2])



plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc="lower right")

plt.show()
