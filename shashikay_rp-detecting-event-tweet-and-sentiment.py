"Importing necessary packages"

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gensim

from gensim.models import LdaModel

from gensim.corpora import Dictionary

import pyLDAvis.gensim

pyLDAvis.enable_notebook()



#pd.set_option('display.expand_frame_repr', False)



import sklearn

from sklearn.externals import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score

from sklearn import preprocessing

from mean_w2v import MeanEmbeddingVectorizer #custom function



import ipywidgets as widgets

from ipywidgets import Button, Layout

from IPython.display import display, HTML, clear_output



from lime import lime_text

from lime.lime_text import LimeTextExplainer



import warnings

warnings.filterwarnings("ignore")



"Prep dataset for LDA topic modelling"

dfeng = pd.read_csv("../input/rp-manglish-tweets-on-kl-rapid-transit//Eng_traindata_processed.csv")

dfmal = pd.read_csv("../input/rp-manglish-tweets-on-kl-rapid-transit//Malay_traindata_processed.csv")

dfeng['textfin'] = dfeng['textfin'].astype('str')

dfmal['textfin'] = dfmal['textfin'].astype('str')

corpus_mal = dfmal['textfin'].tolist()

corpus_eng = dfeng['textfin'].tolist()

train_eng_texts = [doc.split(" ") for doc in corpus_eng]

train_mal_texts = [doc.split(" ") for doc in corpus_mal]

dfmal_txt = pd.DataFrame(dfmal[['text','textfin']]);df_comb = pd.DataFrame(dfeng[['text','textfin']])

df_comb = df_comb.append(dfmal_txt, ignore_index = True)



"pyldaviz function. pyldaviz provides interactive visualization of topic modelling results "

def show_pyldavis(docs,passes,num_topics,no_below=0):  

    bigram = gensim.models.Phrases(docs, min_count=5, threshold=100) # higher threshold fewer phrases.

    bigram_mod = gensim.models.phrases.Phraser(bigram)

    texts = [bigram_mod[doc] for doc in docs]  

    dictionary = Dictionary(texts)

    dictionary.filter_extremes(no_below=no_below)

    dictionary.compactify()  

    corpus = [dictionary.doc2bow(text) for text in texts]

    ldamodel = gensim.models.LdaMulticore(corpus=corpus, num_topics=num_topics, 

                                        id2word=dictionary,

                                        random_state=100,

                                        chunksize=100,

                                        passes=passes,

                                        alpha="asymmetric",

                                        eta=0.91)

    viz = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

    return pyLDAvis.display(viz)
df_comb.sample(10)
"Load the trained SVM (tuned) classifier and word2vec vectorizer"

senti_model_mal = joblib.load("../input/sentiment-function/svm_pipe_mal.joblib.dat")



"load pre-processed event tweets data"

df = pd.read_csv("../input/senti-data/senti_data.csv")

le = preprocessing.LabelEncoder()

df['sentiment_bin'] = le.fit_transform(df.sentiment.values)

df['textfin'] = df['textfin'].astype('str')

df_mal = df.loc[:,["text","textfin","sentiment","sentiment_bin"]]

df_mal = df_mal.reset_index(drop=True)



"Load Lime text explainer model"

class_names = ['negative', 'neutral']

explainer = LimeTextExplainer(class_names = class_names)



"Function to display sentiment prediction & lime text explainer"

def display_senti(idx):   

    #print(idx)

    ori_tweet = df_mal.text[idx];print("Original tweet:\n%s " % (ori_tweet),flush=True)

    processed_tweet = df_mal.textfin[idx];print("\nProcessed tweet:\n%s " % (processed_tweet),flush=True)

    pred = senti_model_mal.predict([processed_tweet]);print("\nSentiment prediction: %s" % (' '.join(le.inverse_transform(pred))))

    probas = senti_model_mal.predict_proba(processed_tweet);acc = probas[0][pred[0]];print("Accuracy: %.4f" %(acc))

    exp = explainer.explain_instance(processed_tweet, senti_model_mal.predict_proba,num_features=10, top_labels=1)

    exp.show_in_notebook(text=True)



"Function to create widget to generate random tweet & corresponding prediction"

btn = widgets.Button(description='Randomize tweet', layout=Layout(width='20%', height='40px'));btn.style.button_color = 'lightgreen'

out = widgets.Output(layout={'border': '3px solid green'})

def btn_eventhandler(obj):

    with out:

        clear_output()

        x = df_mal.textfin.sample(1)

        idx = x.index[0];ori_tweet = df_mal.text[idx];x_l = df_mal.textfin.sample(1).to_list()

        print(idx)

        #display('{}'.format(x))

        print("Original tweet: ");print(ori_tweet, "\n");print("Preprocessed tweet: ");print(' '.join(x), "\n")

        pred = senti_model_mal.predict(x);y = ' '.join(le.inverse_transform(pred));probas = senti_model_mal.predict_proba(x);acc = probas[0][pred[0]]

        print("Prediction: %s" %(y));print("Accuracy: %.4f" %(acc))
#print("Examples of Event tweet data")

#pd.options.display.max_colwidth = 8000



df_mal.sample(5)
print("Tweet 1")

display_senti(5)

print("Tweet 2")

display_senti(56)

print("Tweet 3")

display_senti(191)

print("Tweet 4")

display_senti(111)

print("Tweet 5")

display_senti(110)
#print("Click 'Randomize tweet' button")

display(btn);btn.on_click(btn_eventhandler);display(out)