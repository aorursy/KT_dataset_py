import numpy as np 
import pandas as pd 
from pathlib import Path
import re
import seaborn as sns

#progress bar

from tqdm import tqdm
tqdm.pandas()
path = '/kaggle/input/'

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv(path+'to-learn12/to_learn_R1R2_new_threshold.csv')
ts = pd.read_csv(path+'tsarfati/token_train.tsv',sep="\t", header=None).rename(columns={0:'talkbacks', 1:'label'})
df['talkbacks'].drop_duplicates()#..head()
!pip install tamnun
import codecs
import re
import numpy as np
from tamnun.bert import BertVectorizer, BertClassifier

def token(df_small, method = 'tf', y_format='categorical', vectorizer = None):
    from keras.preprocessing import text, sequence
    from keras.utils.np_utils import to_categorical

    char_level = False
    
    df_small = df_small.iloc[np.random.permutation(len(df_small))]
    cut1 = int(0.8 * len(df_small)) + 1
    df_train, df_valid = df_small[:cut1], df_small[cut1:]

    talkbacks_train =  np.array([str(i) for i in df_train['text'].to_numpy()])
    talkbacks_valid =  np.array([str(i) for i in df_valid['text'].to_numpy()])

    
    if method == 'tf':
        tokenize = text.Tokenizer(num_words=vocabulary_size, 
                              char_level=char_level,
                              filters='')

        tokenize.fit_on_texts(talkbacks_train)  # only fit on train

        x_token_train  = tokenize.texts_to_sequences(talkbacks_train)
        x_token_train  = sequence.pad_sequences(x_token_train , maxlen=max_document_length, padding='post', truncating='post')


        x_token_valid  = tokenize.texts_to_sequences(talkbacks_valid)
        x_token_valid  = sequence.pad_sequences(x_token_valid , maxlen=max_document_length, padding='post', truncating='post')
    
    elif method == 'bert':
        x_token_valid = vectorizer.transform(talkbacks_train)
        x_token_train = vectorizer.transform(talkbacks_valid)

    if y_format == 'categorical':
        y_token_train = to_categorical(df_train['agreed_score'], 2)
        y_token_valid = to_categorical(df_valid['agreed_score'],2)
    elif y_format == 'numeric':
        y_token_train = df_train['agreed_score']
        y_token_valid = df_valid['agreed_score']

    
    
    return(x_token_train, x_token_valid, y_token_train, y_token_valid)
df = pd.read_csv(path+'to-learn12/to_learn_R1R2_new_threshold.csv')
ynet =  pd.read_csv(path+'ynet-df/ynet_covid_new_10_08_2020.csv')

ts = pd.read_csv(path+'tsarfati/token_train.tsv',sep="\t", header=None).rename(columns={0:'talkbacks', 1:'label'})
ynet.head()
vocabulary_size = 5000
max_document_length = 100


def up_to_max_len(s, maxlen=max_document_length):
    if len(s.split(' ')) < maxlen:
        return (s)

    else:
        print('big sentence!')
        sen = ''
        for w in s.split(' ')[0:maxlen]:
            sen += w + ' '
        return (sen)

def preprocess_text(sen, maxlen=max_document_length):
    # Removing html tags
    TAG_RE = re.compile(r'<[^>]+>')
    sentence = TAG_RE.sub('', sen)

    # Remove punctuations and numbers
    sentence = re.sub('!@#$%^&*', '', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    #sentence = re.sub(r'\s+', ' <spaces> ', sentence) #need to choose by the task  

    #lower case
    sentence = sentence.lower()

    #max_len
    sentence = up_to_max_len(sentence, maxlen)

    return sentence

df['text'] = df.talkbacks.apply(preprocess_text)
ts['text'] = ts.talkbacks.apply(preprocess_text,maxlen=100000000000000000)
text_to_learn_lang_model = df['text'].drop_duplicates()+ts['text']

text_to_learn_lang_model = np.asarray(list(text_to_learn_lang_model))
vectorizer = BertVectorizer(do_truncate=True, bert_model='bert-base-multilingual-cased').fit(text_to_learn_lang_model)
def token(df_small, method = 'tf', y_format='categorical', vectorizer = None):
    from keras.preprocessing import text, sequence
    from keras.utils.np_utils import to_categorical

    char_level = False
    
    df_small = df_small.iloc[np.random.permutation(len(df_small))]
    cut1 = int(0.8 * len(df_small)) + 1
    df_train, df_valid = df_small[:cut1], df_small[cut1:]

    talkbacks_train =  np.array([str(i) for i in df_train['text'].to_numpy()])
    talkbacks_valid =  np.array([str(i) for i in df_valid['text'].to_numpy()])

    
    if method == 'tf':
        tokenize = text.Tokenizer(num_words=vocabulary_size, 
                              char_level=char_level,
                              filters='')

        tokenize.fit_on_texts(talkbacks_train)  # only fit on train

        x_token_train  = tokenize.texts_to_sequences(talkbacks_train)
        x_token_train  = sequence.pad_sequences(x_token_train , maxlen=max_document_length, padding='post', truncating='post')


        x_token_valid  = tokenize.texts_to_sequences(talkbacks_valid)
        x_token_valid  = sequence.pad_sequences(x_token_valid , maxlen=max_document_length, padding='post', truncating='post')
    
    elif method == 'bert':
        x_token_train = vectorizer.transform(talkbacks_train)
        x_token_valid = vectorizer.transform(talkbacks_valid)

    if y_format == 'categorical':
        y_token_train = to_categorical(df_train['agreed_score'], 2)
        y_token_valid = to_categorical(df_valid['agreed_score'],2)
    elif y_format == 'numeric':
        y_token_train = df_train['agreed_score']
        y_token_valid = df_valid['agreed_score']

    
    
    return(x_token_train, x_token_valid, y_token_train, y_token_valid)
import matplotlib.pyplot as plt

def cm(x_token_valid, y_token_valid, model, nn=True):
    from sklearn.metrics import confusion_matrix
    if nn == True:
        y_preds = [i.argmax() for i in model.predict(x_token_valid)]
        y_true = [i.argmax() for i in y_token_valid]
    else:
        y_preds = model.predict(x_token_valid)
        y_true = y_token_valid
    sns.heatmap(confusion_matrix(y_true, y_preds), annot=True, fmt="d")
    plt.title('Confusion matrix'+emo[::-1])
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    from sklearn.metrics import f1_score 
    print(f1_score(y_true, y_preds, average='weighted'))
    return(y_preds)

def compare_clf(X_train, X_test, y_train, y_test):
    '''
    compare between all classification models in scikit
    '''
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier


    names = [
    #         "Naive Bayes",
            "Linear SVM",
            "Logistic Regression",
            "Random Forest",
            "C-Support Vector Classification"
        ]

    classifiers = [
    #    MultinomialNB(),
        LinearSVC(),
        LogisticRegression(n_jobs=10),
        RandomForestClassifier(max_depth = 3, n_jobs=10),
        SVC(),
        #Lasso()
        
    ]
    
    compare_matrix = pd.DataFrame()
    for name, classifier in zip(names, classifiers):
      #if name == 'Ada Boost Classifier':
        clf = classifier.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        
        compare_matrix = compare_matrix.append(
            {'clf_name': name,
                'model': clf,
                'accuracy': score,
                 'f1-score': f1_score(y_test, y_pred, average='weighted')
            }, ignore_index=True)

        #print("{} score: {}".format(name, score))

    return (compare_matrix.sort_values(by='accuracy'))

clfs = pd.DataFrame()

for emo in df['emotion'].unique():
    print(emo)
    df_small = df[df['emotion'] == emo].dropna(subset=['agreed_score'])
    
    size = len(df[df['emotion'] == emo].dropna(subset=['agreed_score']).index)
    positive = len(df[(df['emotion'] == emo) & df['agreed_score'] == 1].dropna(subset=['agreed_score']).index)
    ratio = positive/size
    
    x_token_train, x_token_valid, y_token_train, y_token_valid = token(df_small, 
                                                                      method = 'bert', y_format='categorical', vectorizer = vectorizer)
    y_valid = [i.argmax() for i in y_token_valid]
    y_train = [i.argmax() for i in y_token_train]

    
    models = compare_clf(x_token_train, x_token_valid, y_train, y_valid)
    models['method'] = emo
    clfs = clfs.append(models)


clfs.sort_values('accuracy')

df_small
emo='סנטימנט'
df_small = df[df['emotion'] == emo].dropna(subset=['agreed_score'])
x_token_train, x_token_valid, y_train, y_valid = token(df_small,
                                                       method = 'bert', y_format='numeric', vectorizer = vectorizer)



clf = BertClassifier(num_of_classes=2, bert_model_name='bert-base-multilingual-cased', lr=1e-5).fit(
    x_token_train, np.array(y_train))

predicted = clf.predict(x_token_valid)

from sklearn.metrics import accuracy_score, classification_report
print('Accuracy:', accuracy_score(y_train, predicted))
print(classification_report(test_tags, predicted))
