#modulo de manejo de datos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#modulos de graficas
import seaborn as sns; sns.set()
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
datos = pd.read_csv('../input/winemag-data-130k-v2.csv')
#visualizar los primeros 5 registros
datos.head(20)
datos.info()
datos.describe()
datos[['country','variety','winery','taster_name','taster_twitter_handle','province','region_1','region_2']].describe()
datos.country.value_counts().head(15).plot.barh(width=0.9,figsize=(10,6),color='darkred');
#porcentaje de vinos por cada pais
(datos.country.value_counts(normalize=True)*100).head(6)
datos.variety.value_counts().head(70)
#Analizando US por tener el mayor %
US = datos[datos['country'] == 'US']
US.head()
years = US.title.str.extract('([1-2][0-9]{3})').astype(float)
years[years < 1990] = None
US = US.assign(year = years)
US=US.dropna(subset=['price'])
#En la data principal Pinot Noir es la variedad mas 
plt.scatter(x=US[US['variety'] == 'Pinot Noir']['points'],y=US[US['variety'] == 'Pinot Noir']['price'],c=US[US['variety'] == 'Pinot Noir']['year']);
US[US['variety'] == 'Pinot Noir']
sns.boxplot(x='variety', y='year', data = US[US['variety'] == 'Pinot Noir'])
sns.jointplot(x='year',y='price',data=US[US['variety'] == 'White Blend']);
US = US.drop_duplicates('description')
US.shape
US.variety.value_counts()
US = US.groupby('variety').filter(lambda x: len(x) >500)
wine_us =US.variety.unique().tolist()
wine_us.sort()
from sklearn.model_selection import train_test_split
X = US.drop(['Unnamed: 0','country','designation','points','province','taster_name',
       'taster_twitter_handle', 'title','region_1','region_2','variety','winery'], axis = 1)
y = US.variety
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
output = set()
for x in US.variety:
    x = x.lower()
    x = x.split()
    for y in x:
        output.add(y)
variety_list =sorted(output)
extras = ['.', ',', '"', "'", '?', '!', ':', ';','-' ,'(', ')', '[', ']', '{', '}', 'cab',"%"]
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop.update(variety_list)
stop.update(extras)
US.variety.value_counts()
US.head()
sum(US['year']>2002)
datos_=US[US['year']>2002] 
plt.figure(figsize=(10,4))
datos_.variety.value_counts().plot.bar()
rep_v = datos_.groupby('variety')['year'].agg('median')
for i in rep_v.index :
    datos_.loc[(datos_['year'].isna()) & (datos_['variety'] == i),'year']=rep_v[i]
datos_.variety.describe()
#data necesaria
#Tomando las provincias como dummy
X_ = datos_.drop(['Unnamed: 0','designation','country','taster_name',
       'taster_twitter_handle', 'title','region_1','region_2','variety','winery'], axis = 1)
X = pd.get_dummies(X_,columns=["province"])
y = datos_.variety
X.info()
X.head()
from sklearn.feature_extraction.text import CountVectorizer 
#generando variable de textmining
from scipy.sparse import hstack
vect = CountVectorizer(stop_words = stop)
X_dtm = vect.fit_transform(X.description)
Xs = X.drop(['description'],axis=1).as_matrix()
X_dtm = hstack((Xs,X_dtm))
X_dtm
X_dtm.shape,X.shape
len(y)
Xtrain,Xtest,ytrain,ytest = train_test_split(X_dtm,y,random_state=1)
wine=y.unique()
from sklearn.linear_model import LogisticRegression
models = {}
for z in wine:
    model = LogisticRegression()
    y = ytrain == z
    model.fit(Xtrain, y)
    models[z] = model
testing_probs = pd.DataFrame(columns = wine)
len(wine)
probs = pd.DataFrame(columns = wine)
for z in wine: 
    probs[z] = models[z].predict_proba(Xtest)[:,1]
probs.head()
probs_=probs.fillna(0)
pred = probs.idxmax(axis=1)
comparison = pd.DataFrame({'actual':ytest.values, 'predicted':pred.values})   
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")
comparison.head(50)
from  sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(comparison.actual, comparison.predicted)
fig, ax = plt.subplots(figsize=(15,13))
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
print (classification_report(comparison.actual, comparison.predicted))
from sklearn.naive_bayes import MultinomialNB
models = {}
for z in wine:
    model = MultinomialNB()
    y = ytrain == z
    model.fit(Xtrain, y)
    models[z] = model
testing_probs = pd.DataFrame(columns = wine)
probs = pd.DataFrame(columns = wine)
for z in wine: 
    probs[z] = models[z].predict_proba(Xtest)[:,1]
probs.head()

probs_=probs.fillna(0)
pred = probs.idxmax(axis=1)
comparison = pd.DataFrame({'actual':ytest.values, 'predicted':pred.values})   
print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")
from sklearn import preprocessing 
le  =  preprocessing.LabelEncoder() 
le.fit(datos_['variety'])  
le.transform(datos_['variety'])  
datos_['var_codes'] = le.fit_transform(datos_['variety'])
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(datos_['description'],datos_['var_codes'], test_size=0.33, random_state=42)
modelo_cvec = CountVectorizer(stop_words=stop)
X_train = modelo_cvec.fit_transform(X_train)
modelo_NB = MultinomialNB()
modelo_NB.fit(X_train, y_train)
# Predicción
X_test = modelo_cvec.transform(X_test)
labels_predichas = modelo_NB.predict(X_test)
accuracy_score(y_test, labels_predichas)
print (classification_report(y_test, labels_predichas))
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
ypred = model.predict(X_test)
print(metrics.classification_report(ypred, y_test))
#data necesaria
#sin dummies
X_ = datos_.drop(['Unnamed: 0','designation','country','taster_name','province',
       'taster_twitter_handle', 'title','region_1','region_2','variety','winery','var_codes'], axis = 1)

y = datos_.variety
X_.info()
#generando variable de textmining
from scipy.sparse import hstack
vect = CountVectorizer(stop_words = stop)
X_dtm = vect.fit_transform(X_.description)
Xs = X_.drop(['description'],axis=1).as_matrix()
X_dtm = hstack((Xs,X_dtm))
X_dtm
Xtrain,Xtest,ytrain,ytest = train_test_split(X_dtm,y,random_state=1)
wine=y.unique()
models = {}
for z in wine:
    model = LogisticRegression()
    y = ytrain == z
    model.fit(Xtrain, y)
    models[z] = model
testing_probs = pd.DataFrame(columns = wine)
probs = pd.DataFrame(columns = wine)
for z in wine: 
    probs[z] = models[z].predict_proba(Xtest)[:,1]
probs.head()
probs_=probs.fillna(0)
pred = probs.idxmax(axis=1)
comparison = pd.DataFrame({'actual':ytest.values, 'predicted':pred.values})   
print('Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%")
conf_mat = confusion_matrix(comparison.actual, comparison.predicted)
fig, ax = plt.subplots(figsize=(15,13))
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
import re
import nltk
import string
# Importando todo NLTK
import nltk

# Tambien se puede importar modulos
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer

#Downloading NLP corpus from NLTK 
nltk.download('stopwords')
nltk.download('wordnet')
# Stop words = Creacion de un conjunto unico desde el listado de stopwords en ingles que viene con el paquete NLTK. 

#stop = set(stopwords.words('english'))

# Idem punto anterior, se excluyen signos de puntuacion 

exclude = set(string.punctuation) 

# Lemmatizacion de las palabras 
lemma = WordNetLemmatizer()

def clean(doc):
    #pasar a minusculas separando por espacios
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
datos_.variety.value_counts()
wine_data=datos_.groupby('variety')
wine_data.get_group('Red Blend')
#bORRAMOS ALGUNAS PALABRAS MAS COMUNES EN LA DESCRIPTION
#borramos los nombres de las variety en la description
def clean_text(text):
    text = re.sub('rose', ' ', text)
    text = re.sub('cabernet sauvignon', ' ', text)
    text = re.sub('syrah', ' ', text)
    text = re.sub('red blend', ' ', text)
    text = re.sub('zinfandel', ' ', text)
    text = re.sub('sauvignon blanc', ' ', text)
    text = re.sub('bordeaux-style Red Blend', ' ', text)
    text = re.sub('zinfandel', ' ', text)
    text = re.sub('riesling', ' ', text)
    text = re.sub('cabernet franc', ' ', text)
    text = re.sub('merlot', ' ', text)
    text = re.sub('the', ' ', text)
    text = re.sub('wine', ' ', text)
    text = re.sub('drink', ' ', text)
    text = re.sub('acidity', ' ', text)
    text = re.sub('aroma', ' ', text)
    text = re.sub('aromas', ' ', text)
    text = re.sub('finish', ' ', text)
    text = re.sub('fruit', ' ', text)
    text = re.sub('palate', ' ', text)
    text = re.sub('sparkling blend', ' ', text)
    text = re.sub('pinot gris', ' ', text)
    text = re.sub('riesling', ' ', text)
    text = re.sub('petite sirah', ' ', text)

    return text
modelo_cvec = CountVectorizer(stop_words=stop)
datos_['description'] = datos_['description'].map(lambda com : clean(com))
datos_['description'] = datos_['description'].map(lambda com : clean_text(com))
datos_['description']= datos_['description'].apply(lambda x: re.sub(r'([^\s\w]|_)+', '', x))
datos_['variety'].value_counts(normalize=True)
datos_.shape
Cons = datos_['description']
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=0.05, max_df=0.9,stop_words=stop)
data_vectorized = vectorizer.fit_transform(Cons)
tf_feature_names = vectorizer.get_feature_names()
lda_model = LatentDirichletAllocation(n_topics=10, max_iter=30, learning_method='online')
lda_Z = lda_model.fit_transform(data_vectorized)
import pyLDAvis.sklearn
 
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
panel
lda_fit = lda_model.fit(data_vectorized)
# Ejecutamos un método de visualización de los tópicos

import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
from __future__ import print_function
pyLDAvis.save_html(panel, 'lda_topics_US10.html')
Cons = datos_['description']
# Guardamos en una nueva serie

Cons_clean = [clean(comment).split() for comment in Cons] 

from gensim import corpora, models 
import gensim
# converting the corpus into a document-term matrix. every unique term is assigned an index.

dictionary = corpora.Dictionary(Cons_clean)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above. Usamos doc2bow de gensim

doc_term_matrix = [dictionary.doc2bow(comment) for comment in Cons_clean]
X_lda = pd.DataFrame(lda_Z)
df_lda = X_lda.copy()
df_lda['Topico'] = df_lda.idxmax(axis=1)
df_lda.head()
lda_model = LatentDirichletAllocation(n_topics=20, max_iter=30, learning_method='online')
lda_Z = lda_model.fit_transform(data_vectorized)
pyLDAvis.enable_notebook()
panel20 = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
panel20
lda_fit = lda_model.fit(data_vectorized)
