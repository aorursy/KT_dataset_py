from __future__ import unicode_literals
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import ngrams

import re

import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB

from scipy import sparse
def neg_tag(text):
    
    """
    Input is string (e.g. I am not happy.)
    Output is string with neg tags (e.g. I am not NEG_happy.)
    """
    
    transformed = re.sub(r"\b(?:never|nothing|nowhere|noone|none|not|haven't|hasn't|hasnt|hadn't|hadnt|can't|cant|couldn't|couldnt|shouldn't|shouldnt|won't|wont|wouldn't|wouldnt|don't|dont|doesn't|doesnt|didn't|didnt|isnt|isn't|aren't|arent|aint|ain't|hardly|seldom)\b[\w\s]+[^\w\s]", lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)), text, flags=re.IGNORECASE)
    return(transformed)
def preprocessing_baseline(list_of_sentences):
    
    """
    Input is list of raw sentences.
    Output is sentences without punctuations.
    Used for preprocessing method B
    """
    processed_sentences = []
    
    for sent in list_of_sentences:
        
        processed = [word.lower() for word in sent.split()]
        remove_punc = [re.sub('[^a-zA-Z_]+','',t) for t in processed]
        processed_sentences.append(remove_punc)
    
    if len(processed_sentences) == len(list_of_sentences):
        return processed_sentences
    else:
        print('Length of processed is different from input')
def preprocessing_stpwrd_remvl(list_of_sentences):
    
    """
    Input is list of raw sentences.
    Output is list of processed sentences.
    Used for preprocessing methods S1, S2, S3
    """
    
    processed_sentences = []
    
    for sent in list_of_sentences:
        
        stemmed = [stemmer.stem(word).lower() for word in sent.split() if word.lower() not in stpwrds]
        processed = [re.sub('[^a-zA-Z]+','',t) for t in stemmed]
        morethan3 = [t for t in processed if len(t) > 3]
        processed_sentences.append(' '.join(morethan3))
        

    if len(processed_sentences) == len(list_of_sentences):
        return processed_sentences
    else:
        print('Length of processed is different from input')
def preprocessing_neg_tag(list_of_sentences):
    
    """
    Input is list of raw sentences.
    Output is list of neg-tagged and processed sentences.
    Used for preprocessing method N1, N2, N3
    """
    
    processed_sentences = []
    
    for sent in list_of_sentences:
        
        stemmed = [stemmer.stem(word).lower() 
                   if word.lower() not in neg_stw and word.lower() not in stpwrds_wo_neg 
                   else word.lower() 
                   for word in sent.split()]
        
        processed = [t for t in stemmed if t not in stpwrds_wo_neg]
        
        neg_tagged = neg_tag(' '.join(processed))
        
        remove_punc = [re.sub('[^a-zA-Z_]+','',t) for t in neg_tagged.split() if t not in neg_stw]
        
        morethan3 = [t for t in remove_punc if len(t) > 3]
        
        processed_sentences.append(' '.join(morethan3))

    if len(processed_sentences) == len(list_of_sentences):
        return processed_sentences
    else:
        print('Length of processed is different from input')
def get_uni_bi_tri_grams(list_of_reviews):
    
    """
    Input is list of reviews.
    Outputs are:
    1. Unigrams tokens
    2. Unigrams + Bigrams tokens
    3. Unigrams + Bigrams + Trigram tokens
    """
    
    uni_tokens = [s.split() for s in list_of_reviews]
    bi_tokens = [[' '.join([x,y]) for x,y in ngrams(s.split(),2)] for s in list_of_reviews]
    tri_tokens = [[' '.join([x,y,z]) for x,y,z in ngrams(s.split(),3)] for s in list_of_reviews]

    uni_bi_tokens = [x + uni_tokens[i] for i, x in enumerate(bi_tokens)]
    uni_bi_tri_tokens = [x + uni_tokens[i] + bi_tokens[i] for i, x in enumerate(tri_tokens)]

    return uni_tokens, uni_bi_tokens, uni_bi_tri_tokens
# To allow TFIDF Vectorizer to take in tokenized texts directly

def dummy(doc):
    
    """
    Dummy function to allow TfidVectorizer to take in tokenized texts directly
    """
    
    return doc
def build_models(x_train, y_train, x_test, y_test, feature_range):
    
    """
    Standard model building function for B, S, N and NC methods.    
    """
    
    
    model_list = []
    
    for features in range(feature_range[0],feature_range[1],500):
        
        tfid = TfidfVectorizer(max_features=features,analyzer=dummy,preprocessor=dummy)
        
        train_set = tfid.fit_transform(x_train)
        test_set = tfid.transform(x_test)
        
        mnb = MultinomialNB()
        mnb.fit(train_set,y_train)
        
        r = {}
        r['features'] = features
        r['train_acc'], r['test_acc'], r['train_f1'], r['test_f1'], r['tr_cf'] , r['te_cf'], _, _ = get_train_test_score(mnb,
                                                                                                                         train_set, 
                                                                                                                         test_set, 
                                                                                                                         y_train, 
                                                                                                                         y_test)
        model_list.append(r)
    
    return model_list
def get_train_test_score(model, x_train, x_test, y_train, y_test):
    
    """
    Function to get train and test score
    """
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_acc = accuracy_score(y_train,y_train_pred)
    test_acc = accuracy_score(y_test,y_test_pred)
    
    train_f1 = f1_score(y_train,y_train_pred,average='weighted')
    test_f1 = f1_score(y_test,y_test_pred,average='weighted')
    
    train_cf = confusion_matrix(y_train,y_train_pred)
    test_cf = confusion_matrix(y_test,y_test_pred)
    
    train_acc, test_acc, train_f1, test_f1 = [round(x*100,1) for x in [train_acc, test_acc, train_f1, test_f1]]
    
    return train_acc, test_acc, train_f1, test_f1, train_cf, test_cf, y_train_pred, y_test_pred
def print_classification_results(description, train_acc, test_acc, train_f1, test_f1, train_cf, test_cf):
    
    """
    Plot the classification results
    """
    
    print(description)

    
    plt.subplot(1,4,1)
    plt.bar(['Accuracy','F1-Score'], [train_acc,train_f1] ,color=['blue','orange'])
    plt.ylim(55,100)
    if train_acc <= 90:
        plt.annotate(str(train_acc) + '%', (-0.07, train_acc + 1))
        plt.annotate(str(train_f1) + '%', (0.93, train_f1 + 1))

    else:
        plt.annotate(str(train_acc) + '%', (-0.07, train_acc - 3))
        plt.annotate(str(train_f1) + '%', (0.93, train_f1 - 3))
        
    plt.title('Training Score')

    plt.subplot(1,4,2)
    plt.bar(['Accuracy','F1-Score'], [test_acc,test_f1] ,color=['blue','orange'])
    plt.ylim(55,100)
    
    if test_acc <= 90:
        plt.annotate(str(test_acc) + '%', (-0.07, test_acc + 1))
        plt.annotate(str(test_f1) + '%', (0.93, test_f1 + 1))

    else:
        plt.annotate(str(test_acc) + '%', (-0.07, test_acc - 3))
        plt.annotate(str(test_f1) + '%', (0.93, test_f1 - 3))

        
        
    plt.title('Testing Score')

    plt.subplot(1,4,3)
    plot_confusion_matrix(train_cf,classes=['negative','positive'],title='Confusion Matrix (Training)')

    plt.subplot(1,4,4)
    plot_confusion_matrix(test_cf,classes=['negative','positive'],title='Confusion Matrix (Testing)')

    plt.show()
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Obtained from sklearn documentations.
    
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
def plot_pp_best_models(pp_types, pp_desc):
    
    pp_acc, pp_f1 = [], []
    
    pp_labels = list(pp_desc.values())
    
    for pp in pp_types:
        
        best_acc = max(x['test_acc'] for x in models[pp])
        best_f1 = max(x['test_f1'] for x in models[pp])
        
        pp_acc.append(best_acc)
        pp_f1.append(best_f1)
        
    
    
    plt.plot(pp_labels,pp_acc,label='test_acc',marker='o',markersize =10)
    plt.plot(pp_labels,pp_f1,label='test_f1',marker='o',markersize=10, alpha=0.8)
    plt.ylabel('Percentage (%)',fontsize=15)
    plt.xlabel('Preprocessing Method',fontsize=15)

    plt.ylim(min(pp_f1) - 1, max(pp_f1) + 1)
    plt.xticks(fontsize=8)
    
    for index, x in enumerate(pp_types):
        plt.annotate(x,(index - 0.3,min(pp_f1)-0.88),fontsize=20,bbox=dict(boxstyle="round", fc="w"))
    
    plt.grid(True,which='both')
    plt.legend()
    plt.title('Pre-processing Benchmark Results (Multinomial Naive Bayes)\n',fontsize='15')
    plt.show()
# Plot acc and f1 on y, features on x
def plot_acc_f1(pp,feature_range):
        
    tr_acc =[x['train_acc'] for x in models[pp]]
    tr_f1 =[x['train_f1'] for x in models[pp]]
    tr_fea = [x['features'] for x in models[pp]]
    
    te_acc =[x['test_acc'] for x in models[pp]]
    te_f1 =[x['test_f1'] for x in models[pp]]
    te_fea = [x['features'] for x in models[pp]]

    for x in models[pp]:
        if x['test_acc'] == max(te_acc):
            best_model = x
            break
        
    print('===============================================================================================================================\n')
    print('Pre-processing Method\t\t: {}'.format(pp))
    print('Pre-processing Details\t\t: {}'.format(str(pp_desc[pp].replace('\n',' | '))))
    print('Best Model Training Accuracy\t: {} %'.format(best_model['train_acc']))
    print('Best Model Training F1-Score\t: {} %'.format(best_model['train_f1']))
    print('Best Model Testing Accuracy\t: {} %'.format(best_model['test_acc']))
    print('Best Model Testing F1-Score\t: {} %'.format(best_model['test_f1']))
    print('Best Model No. of Features\t: {}'.format(best_model['features']))

    plt.subplot(1,4,1)
    plot_confusion_matrix(best_model['tr_cf'],
                          classes=['negative','positive'],
                          title='Confusion Matrix \n(Best Model, Train Set)\n')

    plt.subplot(1,4,2)
    plot_confusion_matrix(best_model['te_cf'],
                          classes=['negative','positive'],
                          title='Confusion Matrix \n(Best Model, Test Set)\n')
    
    plt.subplot(1,4,3)
    plt.plot(tr_fea,tr_acc,label='train_acc')
    plt.plot(tr_fea,tr_f1,label='train_f1')
    plt.ylabel('Percentage (%)')
    plt.xlim(feature_range[0],feature_range[1])
    plt.xlabel('Number of Features')
    plt.grid(True,which='both')
    plt.legend()
    plt.title('Train Set Models Results\n')
    
    plt.subplot(1,4,4)
    plt.plot(te_fea,te_acc,label='test_acc')
    plt.plot(te_fea,te_f1,label='test_f1')
    plt.ylabel('Percentage (%)')
    plt.xlim(feature_range[0],feature_range[1])
    plt.xlabel('Number of Features')
    plt.grid(True,which='both')
    plt.legend()
    plt.title('Test Set Models Results\n')
    

    

    plt.tight_layout()
    plt.show()
neg_stw = ["never","nothing","nowhere","noone",
           "none","not","haven't","hasn't","hasnt",
           "hadn't","hadnt","can't","cant","couldn't",
           "couldnt","shouldn't","shouldnt","won't",
           "wont","wouldn't","wouldnt","don't","dont",
           "doesn't","doesnt","didn't","didnt","isnt",
           "isn't","aren't","arent","aint",
           "ain't","hardly","seldom"]
stpwrds_wo_neg = [x for x in stopwords.words('english') if x not in neg_stw]
stpwrds = [x for x in stopwords.words('english')]

stemmer = PorterStemmer()
encoding = 'utf-8'

df_train = pd.read_csv('../input/train.csv', encoding=encoding)
df_test = pd.read_csv('../input/test.csv', encoding=encoding)
df_train.head()
df_test.head()
from sklearn.model_selection import train_test_split
# Split the training set into train set and test set

X = df_train.loc[:,'question_text'].values
Y = df_train.loc[:,'target'].values

x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    stratify=Y,
                                                    test_size=0.3,
                                                    random_state=20)
print(len(x_train),len(x_train),len(x_test),len(y_test))
print('Percentage of insincere questions in train set : {0:.2f} %'.format(float(y_train.sum())/len(y_train)*100))
print('Percentage of insincere questions in test set  : {0:.2f} %'.format(float(y_test.sum())/len(y_test)*100))
pp_types = ['BASE','SIM1','SIM2','SIM3','NEG1','NEG2','NEG3']

pp_desc = {  'BASE': 'Baseline',
             'SIM1': '1-gram\nsw, punc removal\nstemming',
             'SIM2': '1,2-grams\nsw, punc removal\nstemming',
             'SIM3': '1,2,3-grams\nsw, punc removal\nstemming',
             'NEG1': '1-gram\nneg tag\nsw, punc removal\nstemming',
             'NEG2': '1,2-grams\nneg tag\nsw, punc removal\nstemming',
             'NEG3': '1,2,3-grams\nneg tag\nsw, punc removal\nstemming',}

models = dict((pp,[]) for pp in pp_types)
x_train_b = preprocessing_baseline(x_train)
x_train_s1, x_train_s2, x_train_s3 = get_uni_bi_tri_grams(preprocessing_stpwrd_remvl(x_train))
x_train_n1, x_train_n2, x_train_n3 = get_uni_bi_tri_grams(preprocessing_neg_tag(x_train))

x_test_b = preprocessing_baseline(x_test)
x_test_s1, x_test_s2, x_test_s3 = get_uni_bi_tri_grams(preprocessing_stpwrd_remvl(x_test))
x_test_n1, x_test_n2, x_test_n3 = get_uni_bi_tri_grams(preprocessing_neg_tag(x_test))
feature_range = (15000,45000)

models['BASE'] = build_models(x_train_b,y_train,x_test_b,y_test,feature_range)
models['SIM1'] = build_models(x_train_s1,y_train,x_test_s1,y_test,feature_range)
models['SIM2'] = build_models(x_train_s2,y_train,x_test_s2,y_test,feature_range)
models['SIM3'] = build_models(x_train_s3,y_train,x_test_s3,y_test,feature_range)
models['NEG1'] = build_models(x_train_n1,y_train,x_test_n1,y_test,feature_range)
models['NEG2'] = build_models(x_train_n2,y_train,x_test_n2,y_test,feature_range)
models['NEG3'] = build_models(x_train_n3,y_train,x_test_n3,y_test,feature_range)
plt.rcParams['figure.figsize'] = [18,4]

for pp in pp_types:
    plot_acc_f1(pp,feature_range)
plt.rcParams['figure.figsize'] = [18,8]

plot_pp_best_models(pp_types,pp_desc)
x_test_qn_text = df_test.loc[:,'question_text'].values
x_test_qid = df_test.loc[:,'qid'].values


x_test_b = preprocessing_baseline(x_test_qn_text)
tfid = TfidfVectorizer(max_features=15000,analyzer=dummy,preprocessor=dummy)

train_set = tfid.fit_transform(x_train_b)
test_set = tfid.transform(x_test_b)

mnb = MultinomialNB()
mnb.fit(train_set,y_train)
predictions = mnb.predict(test_set)
#prepare the submission
submission = pd.DataFrame({"qid": x_test_qid,"prediction":predictions })
submission.to_csv("submission.csv", index=False)
