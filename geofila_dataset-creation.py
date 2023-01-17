import pandas as pd
import numpy as np
import re
import os
import json
from copy import deepcopy

from tqdm.notebook import tqdm
import os, shutil
import gc
import pickle
def save_list(l, filename):
    with open(filename, "wb") as fp:   #Pickling
        pickle.dump(l, fp)
        
def load_list(filename):
    with open(filename, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        return b
#set hyperparameters of the system

params = {
    "min_num_of_words": 8, # the minumum number of words that must contain a sentence to be valid
    "min_sent_text": 10, # the minimum number of sentences that must contain a body text of paper to be valid
    "min_sent_abstract": 5, # the minimum number of sentences that an abstract must contain 
    "tokenization_method": "Fast" #or Slow is rather use standford nlp or not
}
!pip install stanfordnlp
import stanfordnlp

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}
def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


if params["tokenization_method"].lower() == "slow":
    stanfordnlp.download('en')
    nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en')
    

def tokenize_slow(text):
    
    ans = []
    text = clean(text).lower()
    if len(text) < 100:
        return []
    doc = nlp(text)
    for i, sentence in enumerate(doc.sentences):
        x = [token.text for token in sentence.tokens[:-1]]
        if len(x) >= params["min_num_of_words"]:
            ans.append(x)
 
    return ans

#tokenize paragraphs and sentences
def tokenize_fast(text):
    if not isinstance(text, str):
        return []
    text = clean(text).lower()
    text = re.sub(r'(\d+)\.(\d+)',r"\1,\2", text)
    text = re.sub(r'(no)\.',r"\1,", text)
    text = re.sub(r'(eg)\.',r"\1,", text)
    text = re.sub(r'(ie)\.',r"\1,", text)
    text = re.sub(r'(fig)\.',r"\1,", text) #Fig. 10 belongs is the same sentence
    paragraphs = [p for p in text.split('\n') if p]
    # and here, sent_tokenize each one of the paragraphs
    text_l = []
    for paragraph in paragraphs:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
        for sent in sentences:
            sent_split = sent.split(" ")
            if len(sent_split) > params["min_num_of_words"]:
                sent = re.sub(r'(\d+)\,(\d+)',r"\1.\2", sent)
                sent = re.sub(r'(no)\,',r"\1.", sent)
                sent = re.sub(r'(eg)\,',r"\1.", sent)
                sent = re.sub(r'(ie)\,',r"\1.", sent)
                sent = re.sub(r'(fig)\,',r"\1.", sent)
                tokens = []
                for word in sent_split:
                    tokens.append(word)
                text_l.append(tokens)
    return text_l
def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)
def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])
    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)
#Bert Sum
from pytorch_pretrained_bert import BertTokenizer
import os
import torch
import gc

import os
from shutil import copyfile

class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('/kaggle/input/biobert-v2', do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def preprocess(self, src, tgt, oracle_ids):

        if (len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        labels = [0] * len(src)
        for l in oracle_ids:
            labels[l] = 1

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args["min_src_ntokens"])]

        src = [src[i][:self.args["max_src_ntokens"]] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:self.args["max_nsents"]]
        labels = labels[:self.args["max_nsents"]]

        if (len(src) < self.args["min_nsents"]):
            return None
        if (len(labels) == 0):
            return None

        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args["max_src_ntokens"]]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt
def format_body(file, colm):
    if not colm in file.keys():
        return ""
    
    body_text = file[colm]
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body


def load_filenames():
    filenames = []
    for root, _, files in os.walk("/kaggle/input/CORD-19-research-challenge/"):
        for file in files:
            if file.split(".")[-1] == "json":
                filenames.append(root + "/" + file)
    
    return filenames

def generate_dataset(args, all_files, use="train"):
    
    if params["tokenization_method"].lower() == "fast":
        tokenize = tokenize_fast
    else:
        tokenize = tokenize_slow
    bert = BertData(args)
    datasets = []
    i = 0
    
    for file in tqdm(all_files):
        file = json.load(open(file, 'rb'))
        if not "body_text" in file.keys():
            continue
        if not "abstract" in file.keys():
            abstract = ""
        else:
            abstract = format_body(file, "abstract")
            
        body_text = format_body(file, 'body_text')   

        if  len(body_text) < 200:   #an ena paper den exei body text apla to petaw den mporw na kanw tpt me auto 
            continue

        source, tgt = tokenize(body_text), tokenize(abstract)
        if (len (source)< params["min_sent_text"]):
            continue
            
        if use.lower() == "train" and (len(abstract) < 200 or len(tgt)< 4):
            #not for training set
            continue

        oracle_ids = greedy_selection(source, tgt, 5)
        b_data = bert.preprocess(source, tgt, oracle_ids)
        if (b_data is None):
            continue 

        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt + [file["paper_id"]], "tgt_txt": tgt_txt, "paper_id": file["paper_id"]}

        datasets.append(b_data_dict)
            
        if len(datasets) == 2001:
            if use.lower() == "train":
                save_file = "/kaggle/working/BioBertSumData/training_data/cnndm.train." + str(i) + ".bert.pt"
            else:
                 save_file = "/kaggle/working/BioBertSumData/testing_data/cnndm.test." + str(i) + ".bert.pt"
                    
            torch.save(datasets, save_file)
            gc.collect()
            datasets = []
            i += 1
    
    if use.lower() == "train":
        save_file = "/kaggle/working/BioBertSumData/training_data/cnndm.train." + str(i) + ".bert.pt"
    else:
        save_file = "/kaggle/working/BioBertSumData/testing_data/cnndm.test." + str(i) + ".bert.pt"
    torch.save(datasets, save_file)
args = {
    "min_num_of_words": 5, # the minumum number of words that must contain a sentence to be valid
    "min_sent_text": 10, # the minimum number of sentences that must contain a body text of paper to be valid
    "min_sent_abstract": 3,
    
    "max_src_ntokens": 200,
    "min_src_ntokens": 5,
    "max_nsents": 100,
    "min_nsents": 3,
    "oracle_mode": "greedy"
}
#create path for the dataset
os.mkdir("/kaggle/working/BioBertSumData/")
os.mkdir("/kaggle/working/BioBertSumData/training_data")
os.mkdir("/kaggle/working/BioBertSumData/testing_data")

#and read files
all_files = load_filenames()
#create testing dataset
#with all the valid papers
#valid is a paper with valid body text
generate_dataset(args, all_files[:1], use = "train")
generate_dataset(args, all_files, use= "train")
