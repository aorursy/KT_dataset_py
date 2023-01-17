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
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
import matplotlib.pyplot as plt
# extracting the number of examples of each class
real_tweets = df_train[df_train['target'] == 1]
not_real_tweets = df_train[df_train['target'] == 0]
real_len = real_tweets.shape[0]
not_real_len = not_real_tweets.shape[0]
print("Cantidad de tweets reales:", real_len)
print("Cantidad de tweets falsos:",not_real_len)
# bar plot of the 2 classes
plt.bar(10,real_len,3, label="Real", color='blue')
plt.bar(15,not_real_len,3, label="Not", color='red')
plt.legend()
plt.ylabel('Cantidad de tweets')
plt.title('Proporcion reales/falsos')
plt.show()
!pip install symspellpy
#Importamos librerias de pre procesamiento de texto
import nltk
import re
import string
from nltk.corpus import stopwords

from tqdm import tqdm
import spacy
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
import json
print("Train Shape:", df_train.shape)
print("Test Shape:", df_test.shape)
#utilizamos la libreria "re" para limpieza de texto
#uso funcion 
#re.sub(pattern, repl, string, count=0, flags=0)
#elimino corchetes, links, <>, puntuaciones, saltos de linea
#y remuevo las palabras que contienen numeros

#remuevo todas las puntuaciones salvo # y @
#no remuevo urls pq los pienso reemplazar mas adelante
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    #text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    punctuations_to_remove = string.punctuation.replace("#","")
    punctuations_to_remove = punctuations_to_remove.replace("@","")
    text = re.sub('[%s]' % re.escape(punctuations_to_remove), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

#Remuevo emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

sample = "!#$%&hola'()*+, -./:;<=>?@[\]^_`{|}~"
print(clean_text(sample))
#Penn Treebank Tokenizer
#The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank.

def myStemmer(text):
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    stemmer = nltk.stem.PorterStemmer()
    stemmed_list = []
    for token in tokens:
        stemmed_list.append(stemmer.stem(token))
    stemmed_text = ""
    separator = ' '
    stemmed_text = separator.join(stemmed_list)
    return stemmed_text

def myLemmatizer(text):
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_list = []
    for token in tokens:
        lemmatized_list.append(lemmatizer.lemmatize(token))
    lemmatized_text = ""
    separator = ' '
    lemmatized_text = separator.join(lemmatized_list)
    return lemmatized_text
    

#The strip() method returns a 
#copy of the string by removing both the leading and the 
#trailing characters (based on the string argument passed).

def lemmatize(sentence):
    nlp = spacy.load('en')
    return (_lemmatize_text(sentence, nlp).strip())
    
def _lemmatize_text(sentence, nlp):
    sent = ""
    doc = nlp(sentence)
    for token in doc:
        if '@' in token.text:
            sent+=" @MENTION"
        elif '#' in token.text:
            sent+= " #HASHTAG"
        else:
            sent+=" "+token.lemma_
    return sent
def simplify(sentence):
    sent = _replace_urls(sentence)
    sent = _simplify_punctuation(sent)
    sent = _normalize_whitespace(sent)
    return sent

def _replace_urls(text):
    url_regex = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    text = re.sub(url_regex, "<URL>", text)
    return text

def _simplify_punctuation(text):
    """
    This function simplifies doubled or more complex punctuation. The exception is '...'.
    """
    corrected = str(text)
    corrected = re.sub(r'([!?,;])\1+', r'\1', corrected)
    corrected = re.sub(r'\.{2,}', r'...', corrected)
    return corrected

def _normalize_whitespace(text):
    """
    This function normalizes whitespaces, removing duplicates.
    """
    corrected = str(text)
    corrected = re.sub(r"//t",r"\t", corrected)
    corrected = re.sub(r"( )\1+",r"\1", corrected)
    corrected = re.sub(r"(\n)\1+",r"\1", corrected)
    corrected = re.sub(r"(\r)\1+",r"\1", corrected)
    corrected = re.sub(r"(\t)\1+",r"\1", corrected)
    return corrected.strip(" ")
def normalize_contractions(text, contractions):
    """
    This function normalizes english contractions.
    """
    new_token_list = []
    token_list = text.split()
    for word_pos in range(len(token_list)):
        word = token_list[word_pos]
        first_upper = False
        if word[0].isupper():
            first_upper = True
        if word.lower() in contractions:
            replacement = contractions[word.lower()]
            if first_upper:
                replacement = replacement[0].upper()+replacement[1:]
            replacement_tokens = replacement.split()
            if len(replacement_tokens)>1:
                new_token_list.append(replacement_tokens[0])
                new_token_list.append(replacement_tokens[1])
            else:
                new_token_list.append(replacement_tokens[0])
        else:
            new_token_list.append(word)
    sentence = " ".join(new_token_list).strip(" ")
    return sentence
contraction_list = json.loads(open('../input/contractions/english_contractions.json', 'r').read())    
sample = "I'm afraid btw..."
print(normalize_contractions(sample, contraction_list))
def init_spellchecker():
    max_edit_distance_dictionary= 3
    prefix_length = 4
    spellchecker = SymSpell(max_edit_distance_dictionary, prefix_length)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    spellchecker.load_dictionary(dictionary_path, term_index=0, count_index=1)
    spellchecker.load_bigram_dictionary(dictionary_path, term_index=0, count_index=2)
    return spellchecker
    
def spell_correction(text, spellchecker):
    """
    This function does very simple spell correction normalization using pyspellchecker module. It works over a tokenized sentence and only the token representations are changed.
    """
    if len(text) < 1:
        return ""
    #Spell checker config
    max_edit_distance_lookup = 2
    suggestion_verbosity = Verbosity.TOP # TOP, CLOSEST, ALL
    #End of Spell checker config
    token_list = text.split()
    for word_pos in range(len(token_list)):
        word = token_list[word_pos]
        if word is None:
            token_list[word_pos] = ""
            continue
        if not '\n' in word and word not in string.punctuation and not is_numeric(word) and not (word.lower() in spellchecker.words.keys()):
            suggestions = spellchecker.lookup(word.lower(), suggestion_verbosity, max_edit_distance_lookup)
            #Checks first uppercase to conserve the case.
            upperfirst = word[0].isupper()
            #Checks for correction suggestions.
            if len(suggestions) > 0:
                correction = suggestions[0].term
                replacement = correction
            #We call our _reduce_exaggerations function if no suggestion is found. Maybe there are repeated chars.
            else:
                replacement = _reduce_exaggerations(word)
            #Takes the case back to the word.
            if upperfirst:
                replacement = replacement[0].upper()+replacement[1:]
            word = replacement
            token_list[word_pos] = word
    return " ".join(token_list).strip()

def _reduce_exaggerations(text):
    """
    Auxiliary function to help with exxagerated words.
    Examples:
        woooooords -> words
        yaaaaaaaaaaaaaaay -> yay
    """
    correction = str(text)
    #TODO work on complexity reduction.
    return re.sub(r'([\w])\1+', r'\1', correction)

def is_numeric(text):
    for char in text:
        if not (char in "0123456789" or char in ",%.$"):
            return False
    return True

def tokenize(words_list):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(words_list)

def remove_stopwords(text, stop_words_list = stopwords.words('english')):
    words = []
    for w in text:
        if w not in stop_words_list:
            words.append(w)
    return words

def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text
def text_preprocessing(text, nlp, contractions, spellchecker):
    text = simplify(text)
    text = normalize_contractions(text, contractions)
    text = spell_correction(text, spellchecker)
    #la limpieza la hago al final dado 
    #que remueve todas las puntuaciones y muchas son importantes
    #para la normalizacion del texto como (')
    text = clean_text(text)
    #text = myLemmatizer(text)
    #text = myStemmer(text)
    text = _lemmatize_text(text, nlp).strip()
    return text
#pre- procesamiento
df_train.head(10)
df_train['text'][5]
nlp = spacy.load('en')
spellchecker = init_spellchecker()
contraction_list = json.loads(open('../input/contractions/english_contractions.json', 'r').read())    
#Train set
df_train['text'] = df_train['text']\
    .apply(lambda x: text_preprocessing(x, nlp, contraction_list, spellchecker))
#Test set
df_test['text'] = df_test['text']\
    .apply(lambda x: text_preprocessing(x, nlp, contraction_list, spellchecker))
df_train['text'][5]
df_train.head(10)
df_test.head(10)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, Normalizer
from sklearn.pipeline import make_pipeline

from matplotlib import pyplot

#Al final el paramtro 'balanced' termino dando mejores resultados que predefinir el peso de las clases
class_weight={1:0.5, 
              0:0.5}

def get_models():
    models = dict()
    #linear_model
    models['lr'] = LogisticRegression(max_iter=1000, class_weight='balanced', C=1e-1)
    models['rc'] = RidgeClassifier()
    models['sdgc'] = SGDClassifier()
    models['knn'] = KNeighborsClassifier()
    models['cart'] = DecisionTreeClassifier()
    #models['svm'] = SVC()
    models['lSvc'] = LinearSVC(max_iter = 1500)
    models['nuSvc'] = NuSVC()
    #models['OCSvm'] = OneClassSVM()
    #models['gnb'] = GaussianNB()
    models['mnb'] = MultinomialNB()
    models['cnb'] = ComplementNB()
    models['per'] = Perceptron()
    return models

def get_stacking(models):
    #define the base models
    level_0 = list()
    for name, model in models.items():
        level_0.append((name,model))
    level_1 = LogisticRegression(max_iter=1000)
    model = StackingClassifier(estimators=level_0, 
                               final_estimator=level_1)
    return model

def evaluate_model(model, train_vectors):
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = model_selection.cross_val_score(model, train_vectors, df_train['target'], cv=5, scoring = 'f1')
    return scores 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#Feature extraction - BOW
count_vectorizer = CountVectorizer()
train_bow = count_vectorizer.fit_transform(df_train['text'])
#print(count_vectorizer.get_feature_names())
test_bow = count_vectorizer.transform(df_test['text'])
#TF IDF
tfidf_vectorizer = TfidfVectorizer()
train_tfidf = tfidf_vectorizer.fit_transform(df_train['text'])
#print(tfidf_vectorizer.get_feature_names())
test_tfidf = tfidf_vectorizer.transform(df_test['text'])
models = get_models()
results, names = list(), list()
print("BOW")
for name, model in models.items():
    scores = evaluate_model(model, train_bow)
    results.append(scores)
    names.append(name)
    print(name, scores)
plt.boxplot(results, labels=names, showmeans=True)
plt.title('Modelos con BOW')
plt.show()
models = get_models()
results_tfidf, names_tfidf = list(), list()
print("TF IDF")
for name, model in models.items():
    scores = evaluate_model(model, train_tfidf)
    results_tfidf.append(scores)
    names_tfidf.append(name)
    #print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    print(name, scores)
plt.boxplot(results_tfidf, labels=names_tfidf, showmeans=True)
plt.title('Modelos con TF-IDF')
plt.show()
def get_best_stacking():
    level_0 = list()
    level_0.append(("lr", LogisticRegression(max_iter = 150, class_weight='balanced', C=1e-1)))
    level_0.append(("cnb", ComplementNB(alpha=1)))
    level_0.append(("mnb", MultinomialNB(alpha=1)))
    level_0.append(("nuSvc", NuSVC()))
    level_1= LogisticRegression()
    model = StackingClassifier(estimators=level_0, final_estimator=level_1)
    return model
best_stacking = get_best_stacking()
#Stacking 3 modelos + BOW
scores = evaluate_model(best_stacking, train_bow) 
print(scores)
#Stacking 3 modelos + TF-IDF
scores_tfidf = evaluate_model(best_stacking, train_tfidf)
print(scores_tfidf)
results.append(scores)
names.append("stacking")
plt.boxplot(results, labels=names, showmeans=True)
plt.title('Modelos con BOW y stacking')
plt.show()
results_tfidf.append(scores_tfidf)
names_tfidf.append("stacking")
plt.boxplot(results_tfidf, labels=names_tfidf, showmeans=True)
plt.title('Modelos con TFIDF y stacking')
plt.show()
from xgboost import XGBClassifier
pipeline_models, pipeline_names= list(), list()

models_pip = dict()
models_pip['svc'] = SVC()
#models_pip['ocs'] = OneClassSVM()
#models_pip['gnb'] = GaussianNB()

models_pip['svc_p'] = make_pipeline(StandardScaler(with_mean=False),SVC())
#models_pip['gnb_p'] = make_pipeline(StandardScaler(with_mean=False), GaussianNB())
#models_pip['ocs_p'] = make_pipeline(StandardScaler(with_mean=False), OneClassSVM())

#sacado de Fork of TP2,
clf = XGBClassifier(objective = 'binary:logistic', 
                    random_state=42, 
                    seed=2, 
                    colsample_bytree=0.5, 
                    subsample=0.7,
                    learning_rate=0.1,
                    n_estimators=300
                    )
print(evaluate_model(clf, train_bow))
clf.fit(train_bow,df_train['target'])
best_stacking.fit(train_bow, df_train['target'])
def submission(submission_file_path, model ,test_vectors):
    sample_submission = pd.read_csv(submission_file_path)
    sample_submission['target'] = model.predict(test_vectors)
    sample_submission.to_csv("submission.csv", index=False)
submission_file_path = "../input/nlp-getting-started/sample_submission.csv"
submission_test_vectors=test_bow
#Eligo el metodo que me dio mayor puntaje, en este caso fue el Naive Bayes con TF_IDF
submission(submission_file_path,best_stacking, submission_test_vectors)

df_sub = pd.read_csv("./submission.csv")
df_test_copy = df_test.copy(deep=True)
df_test_copy.head()
df_test_copy['target'] = df_sub['target']
df_test_copy.head(20)