import numpy as np

import pandas as pd

import os

debug = False

articles = {}

stat = { }

for dirpath, subdirs, files in os.walk('/kaggle/input'):

    for x in files:

        if x.endswith(".json"):

            articles[x] = os.path.join(dirpath, x)        

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
df.columns
virus_ref = ['covid-19', 'coronavirus', 'cov-2', 'sars-cov-2', 'sars-cov', 'hcov', '2019-ncov', 'sars', 'cov', 'mers']
from tqdm import tqdm

import json

import re

def virus_match(text):

    return len(re.findall(rf'({"|".join(virus_ref)})', text, flags=re.IGNORECASE)) > 0



literature = []

nonlit = []

for index, row in tqdm(df.iterrows(), total=df.shape[0]):

    sha = str(row['sha'])

    license = str(row['license'])

    if sha != 'nan':

        sha = sha + '.json';

        try:

            found = False

            with open(articles[sha]) as f:

                data = json.load(f)

                for key in ['abstract', 'body_text']:

                    if found == False and key in data:

                        for content in data[key]:

                            text = content['text']

                            if virus_match(text) == True:                                

                                literature.append({'file': articles[sha], 'body': text, 'label' : 1, 'license' : license}) 

                            elif virus_match(text) == False:

                                nonlit.append({'file': articles[sha], 'body': text, 'label' : 0, 'license' : license})

        except KeyError:

            pass
dfpos = pd.DataFrame().from_dict(literature)

dfneg = pd.DataFrame().from_dict(nonlit)

print(dfpos.shape)

print(dfneg.shape)

del(literature)

del(nonlit)
dfpos.head(10)

dfpos.head(1).file.values[0].split('/')[7].split('.')[0]
from sklearn.model_selection import train_test_split

trainpos,testpos = train_test_split(dfpos,test_size = 0.2,random_state = 0, shuffle = False)

trainneg,testneg = train_test_split(dfneg,test_size = 0.2,random_state = 0, shuffle = False)

train = trainpos.append(trainneg)

test = testpos.append(testneg)

from sklearn.utils import shuffle

train = shuffle(train)

test = shuffle(test)

y_test = test.label

embedding_path = "../input/glove840b300dtxt/glove.840B.300d.txt"

embed_size = 300

max_features = 50000

max_len = 150
list_classes = ['label']

y = train[list_classes].values

train["body"].fillna("no comment")

test["body"].fillna("no comment")

X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size = 0.05)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

max_len = 150

raw_text_train = X_train["body"].str.lower()

raw_text_valid = X_valid["body"].str.lower()

raw_text_test = test["body"].str.lower()



tk = Tokenizer(num_words = max_features, lower = True)

tk.fit_on_texts(raw_text_train)

x_train = tk.texts_to_sequences(raw_text_train)

x_valid = tk.texts_to_sequences(raw_text_valid)

x_test = tk.texts_to_sequences(raw_text_test)



X_train = pad_sequences(x_train, maxlen = max_len)

X_valid = pad_sequences(x_valid, maxlen = max_len)

test = pad_sequences(x_test, maxlen = max_len)
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
word_index = tk.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras.engine import InputSpec, Layer

import logging

from sklearn.metrics import roc_auc_score

from keras.callbacks import Callback

from sklearn.model_selection import StratifiedShuffleSplit

from keras.optimizers import Adam, RMSprop

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from keras.layers import GRU, BatchNormalization

class RocAucEvaluation(Callback):

    def __init__(self, validation_data=(), interval=1):

        super(Callback, self).__init__()



        self.interval = interval

        self.X_val, self.y_val = validation_data



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict(self.X_val, verbose=0)

            score = roc_auc_score(self.y_val, y_pred)

            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))



file_path = "best_model.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                              save_best_only = True, mode = "min")

ra_val = RocAucEvaluation(validation_data=(X_valid, Y_valid), interval = 1)

early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)



def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):

    inp = Input(shape = (max_len,))

    x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = False)(inp)

    x = SpatialDropout1D(dr)(x)



    x = Bidirectional(GRU(units, return_sequences = True))(x)

    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)

    avg_pool = GlobalAveragePooling1D()(x)

    max_pool = GlobalMaxPooling1D()(x)

    x = concatenate([avg_pool, max_pool])

    x = Dense(1, activation = "sigmoid")(x)

    model = Model(inputs = inp, outputs = x)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, Y_train, batch_size = 128, epochs = 1, validation_data = (X_valid, Y_valid), 

                        verbose = 1, callbacks = [ra_val, check_point, early_stop])

    model = load_model(file_path)

    return model
model = build_model(lr = 1e-3, lr_d = 0, units = 128, dr = 0.2)

model.summary()
def find_label(x):

    if float(x) <= 0.5:

        return 0

    else:

        return 1

pred = model.predict(test, batch_size = 1024, verbose = 1)

pred = pd.DataFrame(pred)

pred.columns = ['proba']

pred["label"] = pred.apply(lambda x : find_label(x['proba']),axis = 1)

from sklearn.metrics import confusion_matrix,f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

print(confusion_matrix(y_test,pred["label"]))

print('accuracy : ',accuracy_score(y_test,pred["label"]))

print('precision : ',precision_score(y_test,pred["label"]))

print('recall : ',recall_score(y_test,pred["label"]))

print('f1 score : ',f1_score(y_test,pred["label"]))

print('roc auc score : ',roc_auc_score(y_test,pred['proba']))

del(train,test,trainpos,trainneg,testpos,testneg,X_train, X_valid, Y_train, Y_valid,y_test,pred,word_index,nb_words,embedding_matrix)
from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation





from scipy.spatial.distance import jensenshannon



import joblib



from IPython.display import HTML, display



from ipywidgets import interact, Layout, HBox, VBox, Box

import ipywidgets as widgets

from IPython.display import clear_output



from tqdm import tqdm

from os.path import isfile



import seaborn as sb

import matplotlib.pyplot as plt

plt.style.use("default")
import nltk

from nltk.corpus import stopwords

customize_stop_words = [

    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 

    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI',

    '-PRON-'

]

stopwords = set(stopwords.words('english')).union(set(customize_stop_words))

all_texts = dfpos.body

vectorizer = CountVectorizer(stop_words = stopwords,min_df=2)

data_vectorized = vectorizer.fit_transform(all_texts)
word_count = pd.DataFrame({'word': vectorizer.get_feature_names(), 'count': np.asarray(data_vectorized.sum(axis=0))[0]})



word_count.sort_values('count', ascending=False).set_index('word')[:20].sort_values('count', ascending=True).plot(kind='barh')
filepath = '../output'

joblib.dump(vectorizer, 'vectorizer.csv')

joblib.dump(data_vectorized, 'data_vectorized.csv')
lda = LatentDirichletAllocation(n_components=50, random_state=0)

lda.fit(data_vectorized)

joblib.dump(lda, 'lda.csv')
def print_top_words(model, vectorizer, n_top_words):

    feature_names = vectorizer.get_feature_names()

    for topic_idx, topic in enumerate(model.components_):

        message = "\nTopic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)

    print()

print_top_words(lda, vectorizer, n_top_words=25)

doc_topic_dist = pd.DataFrame(lda.transform(data_vectorized))

doc_topic_dist.to_csv('doc_topic_dist.csv', index=False)

joblib.dump(dfpos, 'CovidMentioned.csv')

joblib.dump(dfneg, 'CovidNotMentioned.csv')
doc_topic_dist.head()
def get_k_common_docs(doc_dist, k=5, only_covid19=False, get_dist=False):

    temp = doc_dist

    distances = temp.apply(lambda x: jensenshannon(x, doc_dist))#, axis=1)

    k_nearest = distances[distances != 0].nsmallest(n=k).index

    if get_dist:

        k_distances = distances[distances != 0].nsmallest(n=k)

        return k_nearest, k_distances

    else:

        return k_nearest

def recommendation(paper_id, k=5, only_covid19=False, plot_dna=False):

    print(dfpos.body[dfpos.file.map(str).str.split('/').apply(lambda x : x[7]).map(str).str.split('.').apply(lambda x : x[0]) == paper_id].values[0])

    recommended, dist = get_k_common_docs(doc_topic_dist[dfpos.file.map(str).str.split('/').apply(lambda x : x[7]).map(str).str.split('.').apply(lambda x : x[0]) == paper_id].iloc[0], k,  only_covid19, get_dist=True)

    recommended = dfpos.iloc[recommended].copy()

    recommended['similarity'] = 1 - dist

    h = '<br/>'.join([ l +' '+ n +' (Similarity: ' + "{:.2f}".format(s) + ')' for l, n, s in recommended[['body','license', 'similarity']].values])

    display(HTML(h))
recommendation('90b5ecf991032f3918ad43b252e17d1171b4ea63', k=5, plot_dna=False)
symptoms = ['weight loss','chills','shivering','convulsions','deformity','discharge','dizziness','vertigo',

            'fatigue','malaise','asthenia','hypothermia','jaundice','muscle weakness','pyrexia','sweats',

            'swelling','swollen','painful lymph node','weight gain','arrhythmia','bradycardia','chest pain',

            'claudication','palpitations','tachycardia','dry mouth','epistaxis','halitosis','hearing loss',

            'nasal discharge','otalgia','otorrhea','sore throat','toothache','tinnitus','trismus','abdominal pain',

            'fever','bloating','belching','bleeding','blood in stool','melena','hematochezia', 'constipation',

            'diarrhea','dysphagia','dyspepsia','fecal incontinence','flatulence','heartburn','nausea','odynophagia',

            'proctalgia fugax','pyrosis','steatorrhea','vomiting','alopecia','hirsutism','hypertrichosis','abrasion',

            'anasarca','bleeding into the skin','petechia','purpura','ecchymosis and bruising','blister','edema',

            'itching','laceration','rash','urticaria','abnormal posturing','acalculia','agnosia','alexia','amnesia',

            'anomia','anosognosia','aphasia and apraxia','apraxia','ataxia','cataplexy','confusion','dysarthria',

            'dysdiadochokinesia','dysgraphia','hallucination','headache','akinesia','bradykinesia','akathisia',

            'athetosis','ballismus','blepharospasm','chorea','dystonia','fasciculation','muscle cramps','myoclonus',

            'opsoclonus','tic','tremor','flapping tremor','insomnia','loss of consciousness','syncope','neck stiffness',

            'opisthotonus','paralysis and paresis','paresthesia','prosopagnosia','somnolence','abnormal vaginal bleeding',

            'vaginal bleeding in early pregnancy', 'miscarriage','vaginal bleeding in late pregnancy','amenorrhea',

            'infertility','painful intercourse','pelvic pain','vaginal discharge','amaurosis fugax','amaurosis',

            'blurred vision','double vision','exophthalmos','mydriasis','miosis','nystagmus','amusia','anhedonia',

            'anxiety','apathy','confabulation','depression','delusion','euphoria','homicidal ideation','irritability',

            'mania','paranoid ideation','suicidal ideation','apnea','hypopnea','cough','dyspnea','bradypnea','tachypnea',

            'orthopnea','platypnea','trepopnea','hemoptysis','pleuritic chest pain','sputum production','arthralgia',

            'back pain','sciatica','Urologic','dysuria','hematospermia','hematuria','impotence','polyuria',

            'retrograde ejaculation','strangury','urethral discharge','urinary frequency','urinary incontinence',

            'urinary retention']

organs = ['mouth','teeth','tongue','salivary glands','parotid glands','submandibular glands','sublingual glands',

          'pharynx','esophagus','stomach','small intestine','duodenum','Jejunum','ileum','large intestine','liver',

          'Gallbladder','mesentery','pancreas','anal canal and anus','blood cells','respiratory system','nasal cavity',

          'pharynx','larynx','trachea','bronchi','lungs','diaphragm','Urinary system','kidneys','Ureter','bladder',

          'Urethra','reproductive organs','ovaries','Fallopian tubes','Uterus','vagina','vulva','clitoris','placenta',

          'testes','epididymis','vas deferens','seminal vesicles','prostate','bulbourethral glands','penis','scrotum',

          'endocrine system','pituitary gland','pineal gland','thyroid gland','parathyroid glands','adrenal glands',

          'pancreas','circulatory system','Heart','patent Foramen ovale','arteries','veins','capillaries',

          'lymphatic system','lymphatic vessel','lymph node','bone marrow','thymus','spleen','tonsils','interstitium',

          'nervous system','brain','cerebrum','cerebral hemispheres','diencephalon','the brainstem','midbrain','pons',

          'medulla oblongata','cerebellum','the spinal cord','the ventricular system','choroid plexus',

          'peripheral nervous system','nerves','cranial nerves','spinal nerves','Ganglia','enteric nervous system',

          'sensory organs','eye','cornea','iris','ciliary body','lens','retina','ear','outer ear','earlobe','eardrum',

          'middle ear','ossicles','inner ear','cochlea','vestibule of the ear','semicircular canals','olfactory epithelium',

          'tongue','taste buds','integumentary system','mammary glands','skin','subcutaneous tissue']
import re

import string

from wordcloud import WordCloud

from matplotlib import pyplot as plt



DataPosSymp = ' '.join(j for i in dfpos.body for j in i.lower().split() if j in symptoms)

wc = WordCloud(background_color="white",)

wc.generate(DataPosSymp)

plt.axis("off")

plt.imshow(wc, interpolation="bilinear")

plt.show()
DataNegSymp = ' '.join(j for i in dfneg.body for j in i.lower().split() if j in symptoms)

wc = WordCloud(background_color="white",)

wc.generate(DataNegSymp)

plt.axis("off")

plt.imshow(wc, interpolation="bilinear")

plt.show()
DataPosOrg = ' '.join(j for i in dfpos.body for j in i.lower().split() if j in organs)

wc = WordCloud(background_color="white",)

wc.generate(DataPosOrg)

plt.axis("off")

plt.imshow(wc, interpolation="bilinear")

plt.show()
DataNegOrg = ' '.join(j for i in dfneg.body for j in i.lower().split() if j in organs)

wc = WordCloud(background_color="white",)

wc.generate(DataNegOrg)

plt.axis("off")

plt.imshow(wc, interpolation="bilinear")

plt.show()
freqSympPos = {'symptoms' : symptoms, 'frequency' : [DataPosSymp.count(j) for j in symptoms]}

freqSympPos = pd.DataFrame().from_dict(freqSympPos).sort_values(by = 'frequency',ascending = False)

top = freqSympPos.head(10)

plt.barh(top.symptoms,top.frequency)

plt.show()

freqSympNeg = {'symptoms' : symptoms, 'frequency' : [DataNegSymp.count(j) for j in symptoms]}

freqSympNeg = pd.DataFrame().from_dict(freqSympNeg).sort_values(by = 'frequency',ascending = False)

top = freqSympNeg.head(10)

plt.barh(top.symptoms,top.frequency)

plt.show()

del(DataPosSymp,freqSympPos,DataNegSymp,freqSympNeg,top)
freqOrgPos = {'organs' : organs, 'frequency' : [DataPosOrg.count(j) for j in organs]}

freqOrgPos = pd.DataFrame().from_dict(freqOrgPos).sort_values(by = 'frequency',ascending = False)

top = freqOrgPos.head(10)

plt.barh(top.organs,top.frequency)

plt.show()

freqOrgNeg = {'organs' : organs, 'frequency' : [DataNegOrg.count(j) for j in organs]}

freqOrgNeg = pd.DataFrame().from_dict(freqOrgNeg).sort_values(by = 'frequency',ascending = False)

top = freqOrgNeg.head(10)

plt.barh(top.organs,top.frequency)

plt.show()

del(DataPosOrg,freqOrgPos,DataNegOrg,freqOrgNeg,top)