from typing import Dict, List, Tuple

import os

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from gensim.models import KeyedVectors

from nltk.corpus import stopwords

from string import punctuation as punct

import numpy as np

from nltk.corpus import wordnet as wn

from lxml import etree

from nltk.corpus import wordnet as wn

from nltk.tokenize import RegexpTokenizer

from keras import backend as K





def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



class InputSentences(object):

    def __init__(self, inputpath):

        self.inputpath = inputpath

 

    def __iter__(self):

        if (os.path.isdir(self.inputpath)):

            for filename in os.listdir(self.inputpath):

                with open(os.path.join(self.inputpath, filename), 'r') as handle:

                    for line in handle:

                        # the [2:-1] indexing was so as to remove the leading "b'" and trailing "'" for each line

                        # using yield so the output can be a generator, optimizing the use of memory

                        yield line.strip().split()

                        

        elif (os.path.isfile(self.inputpath)):        

            with open(self.inputpath, 'r') as handle:

                for line in handle:

                    # the [2:-1] indexing was so as to remove the leading "b'" and trailing "'" for each line

                    # using yield so the output can be a generator, optimizing the use of memory

                    yield line.strip()[2:-1].split()







def load_bn2wn_mapping(bn2wn_mapping_path: str, flip_dxn = False) -> Dict[str, str]:

    """

    :param bn2wn_mapping_path; Full path to the BabelNet to WordNet file mapping for filtering the corpora

    :return bn2wn_mapping; the BabelNet to WordNet mapping file encoded as a dictionary with the BabelNet IDs as keys

    """

    first = 0

    second = 1

    if (flip_dxn):

        first = 1

        second = 0

    bn2wn_mapping = dict()

    with open(bn2wn_mapping_path, 'r') as handle:

        for line in handle:

            line = line.strip().split("\t")

            if (line):

                bn2wn_mapping[line[first]] = line[second]



    return bn2wn_mapping



def load_bn2lex_mapping(bn2lex_mapping_path: str, flip_dxn = False) -> Dict[str, str]:

    """

    :param bn2wn_mapping_path; Full path to the BabelNet to WordNet file mapping for filtering the corpora

    :return bn2wn_mapping; the BabelNet to WordNet mapping file encoded as a dictionary with the BabelNet IDs as keys

    """

    first = 0

    second = 1

    if (flip_dxn):

        first = 1

        second = 0

    bn2lex_mapping = dict()

    with open(bn2lex_mapping_path, 'r') as handle:

        for line in handle:

            line = line.strip().split("\t")

            if (line):

                bn2lex_mapping[line[second]] = line[first]



    return bn2lex_mapping



def load_bn2wndomain_mapping(bn2wndomain_path: str, flip_dxn = False) -> Dict[str, str]:

    """

    :param bn2wn_mapping_path; Full path to the BabelNet to WordNet file mapping for filtering the corpora

    :return bn2wn_mapping; the BabelNet to WordNet mapping file encoded as a dictionary with the BabelNet IDs as keys

    """

    first = 0

    second = 1

    if (flip_dxn):

        first = 1

        second = 0

    bn2wndomain_mapping = dict()

    with open(bn2wndomain_path, 'r') as handle:

        for line in handle:

            line = line.strip().split("\t")

            if (line):

                bn2wndomain_mapping[line[second]] = line[first]



    return bn2wndomain_mapping





def load_gold_data(gold_data_path: str) -> Dict[str, str]:

    gold_mapping = dict()

    with open(gold_data_path, 'r') as handle:

        for line in handle:

            line = line.strip().split(" ")

            if (line):

                gold_mapping[line[0]] = line[1]



    return gold_mapping





def load_test_dataset(file_path: str) -> List[str]:

    """

    :param file_path; Path to the input dataset

    :return sentences; List of sentences in input_file

    """

    sentences = []

    #with open(file_path, "r", encoding="utf-8-sig") as file:

    with open(file_path, "r") as file:

        for line in file:

            sentences.append(line.strip())



    return sentences





def load_sentence_instances(inst_file_path: str) -> Dict[str, str]:

    instances = dict()

    k=0

    with open(inst_file_path, 'r') as handle:

        for line in handle:

            line = line.strip().split("\t")

            if (line):

                #instances[k] = [line[2], line[0], line[3], line[1]]

                instances[k] = [line[0], line[1], line[2], line[3]]

                k += 1



    return instances





def make_X(sentences: List[str], vocab: Dict[str, int]) -> np.ndarray:

    """

    :param sentences; List of sentences

    :param unigrams_vocab; Unigram vocabulary

    :param bigrams_vocab; Bigram vocabulary

    :return X; Matrix storing all sentences' feature vector 

    """

    X1 = []

    for sentence in sentences:

        x_temp = []

        for word in sentence.split():

            if word in vocab:

                x_temp.append(vocab[word])

            else:

                x_temp.append(vocab["-OOV-"])



        X1.append(np.array(x_temp))



    X1 = np.array(X1)

    return X1





def wn_mfs(word):

    # print(word)

    # exit()

    synsets = wn.synsets(word)

    sense2freq = {}

    mfs = ""

    mfs_freq = 0

    for s in synsets:

        freq = 0  

        for lemma in s.lemmas():

            freq += lemma.count()



        #wn_synset_id = "wn:" + str(s.offset()).zfill(8) + s.pos()

        if freq > mfs_freq or mfs_freq == 0:

            mfs = "wn:" + str(s.offset()).zfill(8) + s.pos()

            mfs_freq = freq

        

        #sense2freq[wn_synset_id] = freq



    # for s in sense2freq:

    #     print (s, sense2freq[s])

    return mfs
def extract_eval_data(corpora_xml_path: str) -> None:

    """

    :param corpora_path; Full path to the corpora_path to be parsed

    :param bn2wn_mapping_path; Full path to the BabelNet to WordNet file mapping for filtering the corpora

    :param outfile_path; Folder path to write the sentences extracted from the corpora

    :param c_type; Corpora type "precision" or "coverage"

    :return None



    THIS FUNCTION HANDLES ONLY EUROSENSE CORPORA

    """

    sentence_x = ""

    instances = []

    sentence_pos = -1

    context = etree.iterparse(corpora_xml_path, events=('start', 'end'))

    tokenizer = RegexpTokenizer(r'\w+')



    sentence_count = 0

    

    xpath = "sentences2.txt"

    inst_path = "inst_temp_file2.txt"

    

    flag_wf = False

    flag_inst = False



    with open(inst_path, 'w') as inst_file:

        with open(xpath, 'w') as x_file:

            #with open(ypath, 'w') as y_file:

            for event, elem in context:

                if (event == 'start'):

                    if (elem.tag == 'wf'):

                        if (elem.text):

                            text = elem.text



                            if (len(text) > 1):

                                text = text.replace("-","_")



                            if (len(text.split()) > 1):

                                text = text.replace(" ","_")



                            text = tokenizer.tokenize(elem.text)

                            if (text):

                                sentence_pos += 1

                                sentence_x = sentence_x + " " + "".join(text)

                            



                        else :

                            flag_wf = True

                        

                    elif (elem.tag == 'instance'):

                        sentence_pos += 1

                        instances.append([elem.attrib['lemma'], elem.attrib['id'], sentence_count, sentence_pos])

                

                        if (elem.text):

                            text = elem.text

                            if (len(text) > 1):

                                text = text.replace("-","_")



                            if (len(text.split()) > 1):

                                text = text.replace(" ","_")



                            sentence_x = sentence_x + " " + text

                        else :

                            flag_inst = True



                elif (event == 'end'):

                    if (flag_inst):

                        text = elem.text

                        if (len(text) > 1):

                            text = text.replace("-","_")



                        if (len(text.split()) > 1):

                            text = text.replace(" ","_")



                        sentence_x = sentence_x + " " + text

                        flag_inst = False



                    if (flag_wf):

                        text = elem.text

                        if (len(text) > 1):

                            text = text.replace("-","_")



                        if (len(text.split()) > 1):

                                text = text.replace(" ","_")



                        text = tokenizer.tokenize(elem.text)

                        if (text):

                            sentence_pos += 1

                            sentence_x = sentence_x + " " + "".join(text)

                        flag_wf = False



                    if (elem.tag == 'sentence'):

                        if (sentence_x):

                            #output_file.write("{}\n".format(sentence.encode('utf-8')))

                            x_file.write("{}\n".format(sentence_x[1:]))

                            #y_file.write("{}\n".format(sentence_y[1:]))

                            

                            for instance in instances:

                                inst_file.write("{}\t{}\t{}\t{}\n".format(instance[0], instance[1], instance[2], instance[3]))



                            sentence_count += 1

                            sentence_pos = -1

                            print("{:,d} sentences extracted...".format(sentence_count), end="\r")

                            

                            sentence_x = ""

                            #sentence_y = ""

                            instances = []

                        

                #Freeing up memory before parsing the next tag

                elem.clear()

                while elem.getprevious() is not None:

                    del elem.getparent()[0]

    print ("\nEvaluation data extraction completed")
extract_eval_data("../input/evaluation/ALL.data.xml")
from nltk.corpus import wordnet as wn

from tensorflow.keras.models import *

import os

import json



def predict_babelnet() -> None:

    """

    DO NOT MODIFY THE SIGNATURE!

    This is the skeleton of the prediction function.

    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)

    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").

    

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding



    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).

    :param output_path: the path of the output file (where you save your predictions)

    :param resources_path: the path of the resources folder containing your model and stuff you might need.

    :return: None

    """



    pred_file = "pred_bab1.txt"



    model_name = "../input/senseless2/model_wsd_senseless(1).hdf5"

    print("LOADING RESOURCES...")

    model = load_model(model_name)



    #load the saved vocabularies

    with open("../input/senseless2/words_vocab_new.json", 'r') as file:

        x_vocab = file.read()

        x_vocab = json.loads(x_vocab)



    with open("../input/senseless2/sense_vocab_new.json", 'r') as file:

        y_vocab = file.read()

        y_vocab = json.loads(y_vocab)



    id_to_words = {v:k for k, v in y_vocab.items()}



    bn2wn_mapping = load_bn2wn_mapping("../input/babelnetdata/babelnet2wordnet.tsv", True)



    #preparing the input data for prediction

    print("PREPARING EVALUATION DATA FOR PREDICTION...")

#     extract_eval_data("../input/evaluation-data/semeval2015.data.xml")



    sentences = load_test_dataset("../input/nlphw2/sentences2.txt")

    X_ = make_X(sentences, x_vocab)



    # instances.append([elem.attrib['lemma'], elem.attrib['id'], sentence_count, sentence_pos])

    sentences_instances = load_sentence_instances("../input/nlphw2/inst_temp_file2.txt")



    #predicting and writing to file

    print("Predicting (line by line) and writing to file... This may take a little while...")

    k = 0

    inst_index = 0

    x_len = X_.shape

    with open(pred_file, "w") as file:

        for x in X_:

            if x.size != 0:

                x__ = np.expand_dims(x, axis=0)

                y_pred = model.predict(x__)[0]



                # This loop is meant to handle one instance at a time, saved temporarily in the inst_temp_file.txt file,

                # loaded into the sentences_instances variable, till there's no instance left

                while True:

                    assoc_bn_synsets_vocab_pos = []

                    if inst_index not in sentences_instances:

                        break



                    inst = sentences_instances[inst_index]

                    if (int(inst[2]) != k):

                        break

                    else:

                        inst_index += 1

                        inst_pos_in_sent = int(inst[3])

                        inst_id = inst[1]



                        # Getting associated senses to the lemma of the instance selected

                        inst_synsets = wn.synsets(inst[0])

                        for wn_synset in inst_synsets:

                            wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()

                            if wn_synset_id in bn2wn_mapping and bn2wn_mapping[wn_synset_id] in y_vocab:

                                assoc_bn_synsets_vocab_pos.append(y_vocab[bn2wn_mapping[wn_synset_id]])

                        

                        # Finding argmax over all associated synsets, and defaulting to MFS (pre saved to the vocab) where there's none

                        if assoc_bn_synsets_vocab_pos:

                            pred_word = y_pred[0, inst_pos_in_sent]

                            synset_probs = []

                            for pos in assoc_bn_synsets_vocab_pos:

                                synset_probs.append(pred_word[pos])



                            pred_sense = id_to_words[assoc_bn_synsets_vocab_pos[np.argmax(synset_probs)]]

                        else:

                            #MFS word = inst[0]

                            #pred_sense = y_vocab["__MFS__"]

                            pred_sense = bn2wn_mapping[wn_mfs(inst[0])]



                        file.write("{} {}\n".format(inst_id, pred_sense))

            

            k = k+1

            if k % 100 < 1:

                print ("%d/%d lines done... A few more moments and everything will be done! :)" % (k,x_len[0]));



    print("Prediction complete!")

predict_babelnet()
def predict_lexicographer() -> None:

    """

    DO NOT MODIFY THE SIGNATURE!

    This is the skeleton of the prediction function.

    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)

    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").

    

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding



    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).

    :param output_path: the path of the output file (where you save your predictions)

    :param resources_path: the path of the resources folder containing your model and stuff you might need.

    :return: None

    """



    pred_file = "pred_lex1.txt"



    model_name = "../input/senseless2/model_wsd_senseless(1).hdf5"

    print("LOADING RESOURCES...")

    model = load_model(model_name)



    #load the saved vocabularies

    with open("../input/senseless2/words_vocab_new.json", 'r') as file:

        x_vocab = file.read()

        x_vocab = json.loads(x_vocab)



    with open("../input/senseless2/sense_vocab_new.json", 'r') as file:

        y_vocab = file.read()

        y_vocab = json.loads(y_vocab)

        

    id_to_words = {v:k for k, v in y_vocab.items()}



    bn2wn_mapping = load_bn2wn_mapping("../input/babelnetdata/babelnet2wordnet.tsv", True)

    bn2lex_mapping = load_bn2lex_mapping("../input/babelnetdata/babelnet2lexnames.tsv", True)



    #preparing the input data for prediction

    print("PREPARING EVALUATION DATA FOR PREDICTION...")

#     extract_eval_data("../input/evaluation-data/semeval2015.data.xml")



    sentences = load_test_dataset("../input/nlphw2/sentences2.txt")

    X_ = make_X(sentences, x_vocab)



    # instances.append([elem.attrib['lemma'], elem.attrib['id'], sentence_count, sentence_pos])

    sentences_instances = load_sentence_instances("../input/nlphw2/inst_temp_file2.txt")



    #predicting and writing to file

    print("Predicting (line by line) and writing to file... This may take a little while...")

    k = 0

    inst_index = 0

    x_len = X_.shape

    with open(pred_file, "w") as file:

        for x in X_:

            if x.size != 0:

                x__ = np.expand_dims(x, axis=0)

                y_pred = model.predict(x__)[0]



                # This loop is meant to handle one instance at a time, saved temporarily in the inst_temp_file.txt file,

                # loaded into the sentences_instances variable, till there's no instance left

                while True:

                    assoc_bn_synsets_vocab_pos = []

                    if inst_index not in sentences_instances:

                        break



                    inst = sentences_instances[inst_index]

                    if (int(inst[2]) != k):

                        break

                    else:

                        inst_index += 1

                        inst_pos_in_sent = int(inst[3])

                        inst_id = inst[1]



                        # Getting associated senses to the lemma of the instance selected

                        inst_synsets = wn.synsets(inst[0])

                        for wn_synset in inst_synsets:

                            wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()

                            if wn_synset_id in bn2wn_mapping and bn2wn_mapping[wn_synset_id] in y_vocab:

                                assoc_bn_synsets_vocab_pos.append(y_vocab[bn2wn_mapping[wn_synset_id]])

                        

                        # Finding argmax over all associated synsets, and defaulting to MFS (pre saved to the vocab) where there's none

                        if assoc_bn_synsets_vocab_pos:

                            pred_word = y_pred[0, inst_pos_in_sent]

                            synset_probs = []

                            for pos in assoc_bn_synsets_vocab_pos:

                                synset_probs.append(pred_word[pos])



                            pred_sense = bn2lex_mapping[id_to_words[assoc_bn_synsets_vocab_pos[np.argmax(synset_probs)]]]

                        else:

                            #MFS word = inst[0]

                            #pred_sense = y_vocab["__MFS__"]

                            pred_sense = bn2lex_mapping[bn2wn_mapping[wn_mfs(inst[0])]]



                        file.write("{} {}\n".format(inst_id, pred_sense))

                       

            

            k = k+1

            if k % 100 < 1:

                print ("%d/%d lines done... A few more moments and everything will be done! :)" % (k,x_len[0]));



    print("Prediction complete!")

predict_lexicographer()
def predict_wordnet_domains() -> None:

    """

    DO NOT MODIFY THE SIGNATURE!

    This is the skeleton of the prediction function.

    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)

    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").

    

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding



    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).

    :param output_path: the path of the output file (where you save your predictions)

    :param resources_path: the path of the resources folder containing your model and stuff you might need.

    :return: None

    """



    pred_file = "pred_dom1.txt"

    model_name = "../input/senseless2/model_wsd_senseless(1).hdf5"

    print("LOADING RESOURCES...")

    model = load_model(model_name)



    #load the saved vocabularies

    with open("../input/senseless2/words_vocab_new.json", 'r') as file:

        x_vocab = file.read()

        x_vocab = json.loads(x_vocab)



    with open("../input/senseless2/sense_vocab_new.json", 'r') as file:

        y_vocab = file.read()

        y_vocab = json.loads(y_vocab)



    id_to_words = {v:k for k, v in y_vocab.items()}



    bn2wn_mapping = load_bn2wn_mapping("../input/babelnetdata/babelnet2wordnet.tsv", True)

    bn2wndomain_mapping = load_bn2wndomain_mapping("../input/babelnetdata/babelnet2wndomains.tsv", True)



    #preparing the input data for prediction

    print("PREPARING EVALUATION DATA FOR PREDICTION...")

#     extract_eval_data("../input/evaluation-data/semeval2015.data.xml")



    sentences = load_test_dataset("../input/nlphw2/sentences2.txt")

    X_ = make_X(sentences, x_vocab)



    # instances.append([elem.attrib['lemma'], elem.attrib['id'], sentence_count, sentence_pos])

    sentences_instances = load_sentence_instances("../input/nlphw2/inst_temp_file2.txt")



    #predicting and writing to file

    print("Predicting (line by line) and writing to file... This may take a little while...")

    k = 0

    inst_index = 0

    x_len = X_.shape

    with open(pred_file, "w") as file:

        for x in X_:

            if x.size != 0:

                x__ = np.expand_dims(x, axis=0)

                y_pred = model.predict(x__)[0]



                # This loop is meant to handle one instance at a time, saved temporarily in the inst_temp_file.txt file,

                # loaded into the sentences_instances variable, till there's no instance left

                while True:

                    assoc_bn_synsets_vocab_pos = []

                    if inst_index not in sentences_instances:

                        break



                    inst = sentences_instances[inst_index]

                    if (int(inst[2]) != k):

                        break

                    else:

                        inst_index += 1

                        inst_pos_in_sent = int(inst[3])

                        inst_id = inst[1]



                        # Getting associated senses to the lemma of the instance selected

                        inst_synsets = wn.synsets(inst[0])

                        for wn_synset in inst_synsets:

                            wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()

                            if wn_synset_id in bn2wn_mapping and bn2wn_mapping[wn_synset_id] in y_vocab:

                                assoc_bn_synsets_vocab_pos.append(y_vocab[bn2wn_mapping[wn_synset_id]])

                        

                        # Finding argmax over all associated synsets, and defaulting to MFS (pre saved to the vocab) where there's none

                        if assoc_bn_synsets_vocab_pos:

                            pred_word = y_pred[0, inst_pos_in_sent]

                            synset_probs = []

                            for pos in assoc_bn_synsets_vocab_pos:

                                synset_probs.append(pred_word[pos])

                            try:

                                pred_sense = bn2wndomain_mapping[id_to_words[assoc_bn_synsets_vocab_pos[np.argmax(synset_probs)]]]

                            except KeyError:

                                pred_sense = "factotum"

                        else:

                            #MFS word = inst[0]

                            #pred_sense = y_vocab["__MFS__"]

                            try:

                                pred_sense = bn2wndomain_mapping[bn2wn_mapping[wn_mfs(inst[0])]]

                            except KeyError:

                                pred_sense = "factotum"

#                             print(inst_id, pred_sense)



                        file.write("{} {}\n".format(inst_id, pred_sense))

                       

            

            k = k+1

            if k % 100 < 1:

                print ("%d/%d lines done... A few more moments and everything will be done! :)" % (k,x_len[0]));



    print("Prediction complete!")

predict_wordnet_domains()
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import wordnet as wn

from tensorflow.keras.models import *

import os

import json



def score_predict_babelnet(prediction, gold_file, bn2wn_mapping_file):

    

    gold_mapping = load_gold_data(gold_file)

    pred_mapping = load_gold_data(prediction)



    bn2wn_mapping = load_bn2wn_mapping(bn2wn_mapping_file, True)



    count = len(gold_mapping)

    rights = 0

    not_found = 0



    for inst_id, inst_sense_key in gold_mapping.items():

        print("Scoring {:,d}/{:,d} instances...".format(rights+1, count), end="\r")

        if inst_id in pred_mapping:

            wn_synset = wn.lemma_from_key(inst_sense_key).synset()

            wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()

            if (pred_mapping[inst_id] == bn2wn_mapping[wn_synset_id]):

                rights += 1

        else:

            not_found += 1



    print("\nCorrect predictions: {}".format(rights))

    print("F1 score: {}".format(rights/count))

    print("No of instances with no prediction: {}".format(not_found))
def score_predict_wordnetdomains(prediction, gold_file, bn2wndomain_mapping_file,bn2wn_mapping_file):

    

    gold_mapping = load_gold_data(gold_file)

    pred_mapping = load_gold_data(prediction)



    bn2wn_mapping = load_bn2wn_mapping(bn2wn_mapping_file, True)

    bn2wndomain_mapping = load_bn2wndomain_mapping(bn2wndomain_mapping_file, True)



    count = len(gold_mapping)

    rights = 0

    not_found = 0



    for inst_id, inst_sense_key in gold_mapping.items():

        print("Scoring {:,d}/{:,d} instances...".format(rights+1, count), end="\r")

        if inst_id in pred_mapping:

            wn_synset = wn.lemma_from_key(inst_sense_key).synset()

            wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()

            try:

                dom_map = bn2wndomain_mapping[bn2wn_mapping[wn_synset_id]]

            except:

                dom_map = "factotum"

            if (pred_mapping[inst_id] == dom_map):

                rights += 1

            

        else:

            not_found += 1



    print("\nCorrect predictions: {}".format(rights))

    print("F1 score: {}".format(rights/count))

    print("No of instances with no prediction: {}".format(not_found))
def score_predict_lexicographer(prediction, gold_file, bn2lex_mapping_file,bn2wn_mapping_file):

    

    gold_mapping = load_gold_data(gold_file)

    pred_mapping = load_gold_data(prediction)



    bn2wn_mapping = load_bn2wn_mapping(bn2wn_mapping_file, True)

    bn2lex_mapping = load_bn2lex_mapping(bn2lex_mapping_file, True)



    count = len(gold_mapping)

    rights = 0

    not_found = 0



    for inst_id, inst_sense_key in gold_mapping.items():

        print("Scoring {:,d}/{:,d} instances...".format(rights+1, count), end="\r")

        if inst_id in pred_mapping:

            wn_synset = wn.lemma_from_key(inst_sense_key).synset()

            wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()

            if (pred_mapping[inst_id] == bn2lex_mapping[bn2wn_mapping[wn_synset_id]]):

                rights += 1

        else:

            not_found += 1



    print("\nCorrect predictions: {}".format(rights))

    print("F1 score: {}".format(rights/count))

    print("No of instances with no prediction: {}".format(not_found))
score_predict_babelnet("../input/nlphw2/pred_bab1.txt", "../input/evaluation/ALL.gold.key.txt", "../input/babelnetdata/babelnet2wordnet.tsv")
score_predict_lexicographer("../input/nlphw2/pred_lex1.txt", "../input/evaluation/ALL.gold.key.txt","../input/babelnetdata/babelnet2lexnames.tsv", "../input/babelnetdata/babelnet2wordnet.tsv")
score_predict_wordnetdomains("../input/nlphw2/pred_dom1.txt", "../input/evaluation/ALL.gold.key.txt","../input/babelnetdata/babelnet2wndomains.tsv", "../input/babelnetdata/babelnet2wordnet.tsv")
from tensorflow.keras.models import *

model_name = "../input/nlphw/model_wsd2.hdf5"

print("LOADING RESOURCES...")

model = load_model(model_name)

model.save("model_wsd2.hdf5")
import json



with open("../input/nlphw/words_vocab.json", 'r') as file:

    x_vocab = file.read()

    x_vocab = json.loads(x_vocab)

    json.dump(x_vocab, open( "words_vocab_wsd2.json", 'w' ) )

    

with open("../input/nlphw/sense_vocab.json", 'r') as file:

    y_vocab = file.read()

    y_vocab = json.loads(y_vocab)

    json.dump( y_vocab, open( "sense_vocab_wsd2.json", 'w' ) )



from IPython.display import FileLink

FileLink('words_vocab_wsd2.json')