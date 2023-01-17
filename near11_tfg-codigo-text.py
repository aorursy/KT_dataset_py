# Semilla
SEED = 333

# Exportar CSV
from IPython.display import HTML
def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# Preparamos el lematizado
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

# Lematizar un string
import re
def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [porter_stemmer.stem(word) for word in words]
    return words

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

## Columnas de guardado para los algortimos 
COLUMNS = ['mean_fit_time','std_fit_time','mean_test_neg_log_loss','std_test_neg_log_loss','rank_test_neg_log_loss',
           'mean_test_accuracy','rank_test_accuracy',
           'mean_test_f1_macro','rank_test_f1_macro',
           'mean_test_roc_auc_ovr','rank_test_roc_auc_ovr']

# Funcion de guardado de resultados que es un subconjunto de cv_results. 
# Guarda los resultados de los parametros del algoritmo y las metricas que le pasamos como parametro.
def save_results(gs,params_to_evaluate,columns=COLUMNS):
    aux = pd.DataFrame(gs.cv_results_)
    gs_res = pd.DataFrame()
    for col in params_to_evaluate:
        gs_res[col] = aux[col]
    for col in columns:
        gs_res[col] = aux[col]
    return gs_res


# Habilita que se pueda graficar directamente desde el dataframe
import cufflinks as cf
import plotly.express as px
cf.set_config_file(offline=True)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

train_variants_df = pd.read_csv("../input/data-c/training_variants", engine='python')
train_txt_df = pd.read_csv("../input/data-c/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train_txt_df['Class'] = train_variants_df['Class']
train_txt_df.sample(10,random_state=SEED)
# Inicializamos el dataframe que vamos a utilizar
W = pd.DataFrame()
word_clod = pd.DataFrame()

# Añadimos una columna que nos indica el tamaño del texto de cada instancia
W['Text_count']  = train_txt_df["Text"].apply(lambda x: len(str(x).split()))

# Copiamos la clase y el texto
W['Class'] = train_txt_df['Class'].copy()
W['Text'] = train_txt_df["Text"].copy()

# Aplicamos el lematizado a cada instancia de texto para mostrarlo
#word_clod['Text'] = train_txt_df["Text"].apply(lambda x: stemming_tokenizer(str(x)))
#word_clod['Class'] = train_txt_df['Class'].copy()

# Nos quedamos con las instancias que no tengan el texto nulo
W = W[W['Text_count']!=1]

# Mostramos el dataframe
W.sample(10,random_state=SEED)
#W.to_csv('W.csv')
#create_download_link(filename='W.csv')
"""fig = px.violin(W ,y="Text_count", color="Class")
fig.update_yaxes(automargin=True)
fig.show()"""
"""# data prepararion
from plotly import tools
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from textwrap import wrap

# Primero creamos una lista de palabras personalizadas las cuales no deben aportar mucha informarcion a la hora de clasificar
custom_words = ["fig", "figure", "et", "al", "al.", "also",
                "data", "analyze", "study", "table", "using",
                "method", "result", "conclusion", "author", 
                "find", "found", "show"]
stop_words = set(list(STOPWORDS) + custom_words)

def show_wordcloud(data,title):
    plt.subplots(figsize=(8,8))
    wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384,
                          stopwords = stop_words,
                          random_state = SEED
                         ).generate(str(data.Text))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('graph.png')
    plt.title(title,fontsize = 'x-large',color = 'w')
    plt.show()

show_wordcloud(word_clod,"Palabras Frecuentes")"""
"""def word_agroupped(data, att, value):
    subset = data[data[att] == value]
    show_wordcloud(subset,"Palabras Frecuentes"+ ' '+ att + ' ' + str(value))

for i in range(1,10):
    word_agroupped(word_clod,'Class',i)"""
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

def plot_tfidf(data_tfidf,x="Word",y="TF-IDF",title="Palabras Clave ",num_instances=50):
    fig = px.bar(data_tfidf.head(num_instances), x=x, y=y,title = title)
    fig.show()
    
def train_tfidf(data = W,max_features=1000,tokenizer = stemming_tokenizer,ngram_range=(1,1)):
    # Creamos el tfidf
    count_vectorizer = TfidfVectorizer(analyzer="word", tokenizer=tokenizer,stop_words= 'english', max_features=max_features,ngram_range=ngram_range)
    tfidf = count_vectorizer.fit_transform(data['Text'].values.astype('U'))
    
    # Lo transformamos a dataframe
    df_tfidf = pd.DataFrame(tfidf[0].T.todense(), columns=["TF-IDF"])
    df_tfidf["Word"] = count_vectorizer.get_feature_names()
    
    return df_tfidf.sort_values('TF-IDF', ascending=False)


def plot_subset(fixed_value,data_tfidf=W,att='Class',ngram_range=(1,2)):
    # Subconjunto a pintar
    subset = data_tfidf[data_tfidf[att] == fixed_value]
    
    # TF-IDF
    df_tfidf = train_tfidf(subset,ngram_range=ngram_range)
    
    # Pintar
    plot_tfidf(df_tfidf,title="Palabras Clave Clase"+" "+str(fixed_value))
#df_tfidf = train_tfidf(ngram_range=(1,1))
#plot_tfidf(df_tfidf)
"""for i in range(1,10):
    plot_subset(i,ngram_range=(1,1))"""
#df_tfidf = train_tfidf(ngram_range=(2,2))
#plot_tfidf(df_tfidf)
"""for i in range(1,10):
    plot_subset(i,ngram_range=(2,2))"""
#df_tfidf = train_tfidf(ngram_range=(1,2))
#plot_tfidf(df_tfidf)
"""for i in range(1,10):
    plot_subset(i,ngram_range=(1,2))"""
from sklearn.pipeline import Pipeline
def create_pipeline(clf,ngram_range=(1,1)):
    return Pipeline([('tfidf', TfidfVectorizer(analyzer="word", tokenizer=stemming_tokenizer,stop_words= 'english',ngram_range=ngram_range)),
                         ('clf', clf)])

from sklearn.model_selection import GridSearchCV
# Validacion Cruzada Stratificada(n_splits=5):
from sklearn.model_selection import StratifiedKFold
CV = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)

def create_gscv(pipeline,params,scoring = ["neg_log_loss","accuracy","f1_macro","roc_auc_ovr"],cv = CV):
    return GridSearchCV(
            pipeline,
            params,
            verbose = 1,
            cv = cv,
            n_jobs = -1,
            scoring = scoring,
            refit = "neg_log_loss" 
            )
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
parameters = {
    'tfidf__ngram_range': ((1, 1),(2, 2),(1,2)),
    #'tfidf__use_idf': (True, False),
    #'tfidf__max_features': (1000,3000,5000,7000,9000,None),
    'tfidf__max_features': (1000,None),
    #'tfidf__norm': ('l1', 'l2'),
 }
pipeline = create_pipeline(clf)
gs_NB_M = create_gscv(pipeline,parameters)
#gs_NB_M.fit(W['Text'],W['Class'])
params_to_evaluate = ["param_tfidf__max_features","param_tfidf__ngram_range"]
#NB_M_res = save_results(gs_NB_M,params_to_evaluate)
#NB_M_res = pd.DataFrame(gs_NB_M.cv_results_)
#NB_M_best = NB_M_res.sort_values(by='mean_test_score',ascending=False).head(1)
#NB_M_res.sort_values(by='mean_test_score',ascending=False).head(5)
# Exportamos los resultados
#NB_M_res.to_csv('NB_M.csv')
#create_download_link(filename='NB_M.csv')
from sklearn.naive_bayes import ComplementNB
clf2 = ComplementNB()
parameters = {
    'tfidf__ngram_range': ((1, 1),(2, 2),(1,2)),
    #'tfidf__use_idf': (True, False),
    #'tfidf__max_features': (1000,3000,5000,7000,9000,None),
    'tfidf__max_features': (1000,None),
    #'tfidf__norm': ('l1', 'l2'),
 }
pipeline2 = create_pipeline(clf2)
gs_NB_C = create_gscv(pipeline2,parameters)
#gs_NB_C.fit(W['Text'],W['Class'])
"""NB_C_res = save_results(gs_NB_C,params_to_evaluate)
NB_C_best = NB_C_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(1)
NB_C_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(5)"""
# Exportamos los resultados
#NB_C_res.to_csv('NB_C.csv')
#create_download_link(filename='NB_C.csv')
from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(random_state=SEED)
parameters = {
    #'tfidf__ngram_range': ((1, 1),(2, 2),(1,2)),
    #'tfidf__ngram_range': ((1, 1)),
    #'tfidf__use_idf': (True, False),
    #'tfidf__max_features': (1000,3000,5000,7000,9000,None),
    'tfidf__max_features': (1000,None),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__criterion':('giny','entropy')
 }
#pipeline3 = create_pipeline(clf3)
#gs_RF = create_gscv(pipeline3,parameters)
#gs_RF.fit(W['Text'],W['Class'])
'''params_to_evaluate = ["param_tfidf__max_features"]
RF_res = save_results(gs_RF,params_to_evaluate)
RF_best = RF_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(1)
RF_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(5)'''
# Exportamos los resultados
#RF_res.to_csv('RF.csv')
#create_download_link(filename='RF.csv')
'''from sklearn.svm import SVC
clf4 = SVC(probability=True)
parameters = {
    #'tfidf__ngram_range': ((1, 1),(2, 2),(1,2)),
    #'tfidf__use_idf': (True, False),
    #'tfidf__max_features': (1000,3000,5000,7000,9000,None),
    'tfidf__max_features': (1000,None),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__kernel':('linear','rbf','sigmoid','poly')
 }
pipeline4 = create_pipeline(clf4)
gs_SVC = create_gscv(pipeline4,parameters)'''
#gs_SVC.fit(W['Text'],W['Class'])
'''params_to_evaluate = ["param_tfidf__max_features"]
SVC_res = save_results(gs_SVC,params_to_evaluate)
SVC_best = SVC_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(1)
SVC_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(5)'''
# Exportamos los resultados
SVC_res.to_csv('SVC.csv')
create_download_link(filename='SVC.csv')
from sklearn.svm import SVC
clf4 = SVC(probability=True)
parameters = {
    #'tfidf__ngram_range': ((1, 1),(2, 2),(1,2)),
    #'tfidf__use_idf': (True, False),
    #'tfidf__max_features': (1000,3000,5000,7000,9000,None),
    'tfidf__max_features': (1000,None),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__kernel':('linear','rbf','sigmoid','poly')
 }
pipeline4 = create_pipeline(clf4,ngram_range=(2, 2))
gs_SVC = create_gscv(pipeline4,parameters)
gs_SVC.fit(W['Text'],W['Class'])
params_to_evaluate = ["param_tfidf__max_features"]
SVC_res = save_results(gs_SVC,params_to_evaluate)
SVC_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(5)
# Exportamos los resultados
SVC_res.to_csv('SVC.csv')
create_download_link(filename='SVC.csv')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=60)
parameters = {
    'tfidf__ngram_range': ((1, 1),(2, 2),(1,2)),
    #'tfidf__use_idf': (True, False),
    #'tfidf__max_features': (1000,3000,5000,7000,9000,None),
    'tfidf__max_features': (1000,None),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__n_neighbors': tuple(range(1,60,2))
 }
pipeline5 = create_pipeline(knn)
gs_KNN = create_gscv(pipeline5,parameters)
gs_KNN.fit(W['Text'],W['Class'])
params_to_evaluate = ["param_tfidf__max_features","param_tfidf__ngram_range"]
KNN_res = save_results(gs_KNN,params_to_evaluate)
KNN_best = KNN_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(1)
KNN_res.sort_values(by='mean_test_neg_log_loss',ascending=False).head(5)
# Exportamos los resultados
KNN_res.to_csv('SVC.csv')
create_download_link(filename='SVC.csv')

