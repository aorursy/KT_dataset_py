!conda install -y gdown
!gdown --id 11kn1CapC16_af6mY5IBBr04lLaHvBIsH #contest3_ner.zip

!unzip contest3_ner_2020.zip
import glob #เขียน patternค้นหาไฟล์
train_filenames = glob.glob('thai_ner_train/*.txt')
!pip install pythainlp
from pythainlp import word_tokenize
import re
train_filenames = glob.glob('thai_ner_train/*.txt')
w_list_final = [] 
tag_list_final = []
for filename in train_filenames:
  with open(filename) as f:
    text = re.split(r'\n\n+', f.read()) #split by sentences
    for line in text:
      line = line.strip()
      data1 = re.split(r'<(pers|org|place)Name .+?>(.+?)</\1Name>', line) #grouping
      data1 = [re.sub(r'<.+?>', '', x) for x in data1] #เขียนทับข้างบน ให้สิ่งที่อยู่ใน <> เป็นว่าง
      w_list = [] 
      tag_list = [] 
      index = 0 #like enumerate
      while index < len(data1):
        if data1[index] in ['pers','org','place']:
          type_ne = data1[index]
          w = word_tokenize(data1[index+1])
          w_list += w #ใส่ tokenized word in [] of w_list
          tag_list += ['B-' + type_ne] + ['I-' + type_ne] * (len(w)-1) #keep tags in tag_list []
          index += 2
        else:
          w = word_tokenize(data1[index])
          w_list += w #ใส่ tokenized word in []
          tag_list += ['O'] * len(w)
          index += 1
      w_list_final.append(w_list) #All become list of list
      tag_list_final.append(tag_list)
import json
with open('train_data.json', 'w', encoding='utf8') as f: #save to json
  json.dump((w_list_final, tag_list_final), f)
with open('train_data.json') as f: #train file
    #w_list, tag_list = json.load(f) #load to python
    train_data = json.load(f)
    X_train = train_data[0]
    Y_train = train_data[1]
dev_filenames = glob.glob('thai_ner_dev_set/*.txt') #dev
w_list_final_dev = [] 
tag_list_final_dev = []
for filename in dev_filenames:
  with open(filename) as f:
    text = re.split(r'\n\n+', f.read()) 
    for line in text:
      line = line.strip()
      data1 = re.split(r'<(pers|org|place)Name .+?>(.+?)</\1Name>', line)
      data1 = [re.sub(r'<.+?>', '', x) for x in data1]
      w_list_dev = [] 
      tag_list_dev = [] 
      index = 0
      while index < len(data1):
        if data1[index] in ['pers','org','place']:
          type_ne = data1[index]
          w = word_tokenize(data1[index+1])
          w_list_dev += w
          tag_list_dev += ['B-' + type_ne] + ['I-' + type_ne] * (len(w)-1)
          index += 2
        else:
          w = word_tokenize(data1[index])
          w_list_dev += w
          tag_list_dev += ['O'] * len(w)
          index += 1
      w_list_final_dev.append(w_list_dev)
      tag_list_final_dev.append(tag_list_dev)
with open('dev_data.json', 'w', encoding='utf8') as f:
  json.dump((w_list_final_dev, tag_list_final_dev), f)
with open('dev_data.json') as f:
    dev_data = json.load(f)
    X_dev = dev_data[0]
    Y_dev = dev_data[1]
!pip install sklearn-crfsuite
import sklearn_crfsuite
import sklearn_crfsuite.metrics
print(len(X_train[0]))
print(len(Y_train[0]))
!gdown --id 1gbarnuFFuSMK64Q5hHyxuTAGJMjEOThJ #name.txt
!gdown --id 17B0h0zeE8sICTM7HCpxbevlFsny_36O- #countries.txt

def is_word(i, tokens):
  return{'is word': tokens[i][0]}
country_list = []
with open('countries.txt', 'r') as f:
  for line in f:
    if line != '\n':
      country_list.append(line.strip())
def is_in_country_list(i, tokens):
    return{'is country': tokens[i][0]}
    
name_list = []
with open('name.txt', 'r') as f:
  for line in f:
    if line != '\n':
      name_list.append(line.strip())
def is_in_name_list(i, tokens):
  return{'is name': tokens[i][0] in name_list}
#from typing import List, Tuple, Union

#import sklearn_crfsuite
from pythainlp.corpus import download, get_corpus_path, thai_stopwords
from pythainlp.tag import pos_tag
from pythainlp.tokenize import word_tokenize
from pythainlp.util import isthai
#from typing import List, Tuple

from pythainlp.corpus import provinces


def is_thai(i, tokens):
  return{'is thai': isthai(tokens[i][0])}
def is_in_province(i, tokens):
  province_list = list(provinces())
  return{'is province': tokens[i] in province_list}

 
# check if it is in thai_stopword list   
   #in lab tokens are list of tuples
def is_stopword(i, tokens):
  stopword_list = thai_stopwords()
  return{'is stopword': tokens[i][0] in stopword_list}
def is_digit(i, tokens):
  return{'is digit': tokens[i][0].isdigit()}
def is_space(i,tokens):
  return{'is space': tokens[i][0].isspace()}
def word_features(i, tokens):
  if i == 0:
    return {'word_current': tokens[i][0],'word_next': tokens[i+1][0]}
  elif i == len(tokens) - 1:
    return {'word_current': tokens[i][0],'word_next': tokens[i-1][0]}
  else:
    return {'word_current': tokens[i][0],'word_next': tokens[i+1][0], 'word_previous': tokens[i-1][0]}
from sklearn_crfsuite.metrics import flat_classification_report
def featurize_one_sentence(tokens, feature_function_list):
    pos_list = pos_tag(tokens)
    tokens = list(zip(tokens,pos_list))
    feature_dict_seq = []
    for i in range(len(tokens)): # ทุก token
        feature_dict = {}
        for feature_fn in feature_function_list: # ทุก feature function
            feature_dict.update(feature_fn(i, tokens))
        feature_dict_seq.append(feature_dict)
    return feature_dict_seq
    
def train_and_evaluate(X_train, Y_train, X_dev, Y_dev, feature_function_list):
    #featurize_one_sentence(X_train[0])
    training_data = [featurize_one_sentence(token_list,feature_function_list) for token_list in X_train]
    crf = sklearn_crfsuite.CRF()
    crf.fit(training_data, Y_train)
    dev = [featurize_one_sentence(token_list,feature_function_list) for token_list in X_dev]
    pred = crf.predict(dev)
    print(sklearn_crfsuite.metrics.flat_accuracy_score(Y_dev, pred))
    print(bio_classification_report(Y_dev,pred))
    return crf
    
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from itertools import chain
def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelEncoder()
    flattened_y_true = list(chain.from_iterable(y_true))
    flattened_y_pred = list(chain.from_iterable(y_pred))
    y_true_combined = lb.fit_transform(flattened_y_true)
    y_pred_combined = lb.transform(flattened_y_pred)

    tagset = set(lb.classes_) - {'O', 'pad', 'I-place'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return metrics.classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

# postag use all features #word_features included
best = train_and_evaluate(X_train, Y_train, X_dev, Y_dev, [word_features, is_word, is_thai, is_space, is_digit, is_in_name_list, is_in_country_list, is_in_province, is_stopword])
# postag use all features without word_features
train_and_evaluate(X_train, Y_train, X_dev, Y_dev, [is_word, is_thai, is_space, is_digit, is_in_name_list, is_in_country_list, is_in_province, is_stopword])
# postag use all features but drop gazetteers
train_and_evaluate(X_train, Y_train, X_dev, Y_dev, [word_features, is_word, is_thai, is_space, is_digit])
# postag and gazetteers only
train_and_evaluate(X_train, Y_train, X_dev, Y_dev, [is_in_name_list, is_in_country_list, is_in_province, is_stopword])
featurize_one_sentence(X_dev[0], [is_word, is_space, is_digit, is_in_name_list, is_in_province, is_in_country_list])