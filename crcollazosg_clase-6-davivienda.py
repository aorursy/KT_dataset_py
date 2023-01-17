import pandas as pd
import numpy as np
df = pd.read_csv("sentiment.csv")
df.shape
df.head(10)
df.dtypes
df["clase"] = ["pos" if int(x)==4 else "neg" for x in df.target]
df.head()
df["clase"] = pd.Categorical(df["clase"])
df.dtypes
df.groupby("clase")["ids"].count()
palabras_unicas = list(set([palabra for frase in df["text"] for palabra in frase.split()]))
len(palabras_unicas)
from sklearn.model_selection import train_test_split
X = df["text"]
y = df["clase"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
type(y_test)
y_test.value_counts()
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=10000, random_state=42)
# sss.split da como resultado los índices de la lista, no la lista en sí
# es necesario usar un ciclo (que en este caso se ejecuta una sola vez)
# porque sss.split puede generar tantas combinaciones diferentes como se
# especifique en el parámetro n_splits
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
X_train[:20]
letras = "".join(sorted(set([letra for palabra in palabras_unicas for letra in palabra])))
letras
# no olvidar incluir el espacio
letras_quitar = "".join([l for l in letras if l.lower() not in "abcdefghijklmnopqrstuvwxyz "])
letras_quitar
dicc_quitar = {l:'' for l in letras_quitar}
tabla = str.maketrans(dicc_quitar)
prueba = "}³´µ·¸º»½¿ÃÆÊÖØÚÜßàáâãäåçèéêëìíïñòóôõEsta esőřşšũůžơưʻ˚πВИКЛПСХЦЫабвгдежзийклмнопртухцчшщ una ~\x7f¡£¤¥¨©ª«¬\xad®°cadena con ruido.89:;=?@[\\]^_`{|ö÷øùúüýăđēęěğİı"
prueba.translate(tabla)
def my_preprocessor(doc):
    doc_limpio = doc.translate(tabla)
    doc_limpio = doc_limpio.lower()
    return doc_limpio
my_preprocessor(prueba)
def bypass_preprocessor(doc):
    return doc
def my_tokenizer(doc):
    out = doc.split(' ')
    return out
def bypass_tokenizer(doc):
    return doc
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(preprocessor=my_preprocessor,tokenizer=my_tokenizer, min_df=10, ngram_range=(1,1))
X_train_vec = vec.fit_transform(X_train)
len(vec.vocabulary_)
list(vec.stop_words_)[:20]
%matplotlib inline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

sns.set(style="whitegrid")
sample = 20000

XX = X_train_vec[:sample,].todense()

n_components = 1000
pca = PCA(n_components=n_components, random_state=42)

# pca = PCA()
components = pca.fit_transform(XX)
components.shape
# reducción a 2 dimensiones para visualizar
fig, ax = plt.subplots(figsize=(15,10))
sns.scatterplot(x=components[:,0], y=components[:,1], hue=y_train[:sample], ax=ax)
ax.set_title('Reducción dimensional PCA - Sentiment Analysis');
ax.set_xlabel('X1');
ax.set_ylabel('X2');
fig.savefig('pca2d_sentiment.png')
# pca = PCA()
# pca.fit(XX)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(15,10))
ax.plot(range(1, exp_var_cumul.shape[0] + 1), exp_var_cumul)
ax.set_title('Varianza explicada PCA - Sentiment Analysis');
ax.set_xlabel('Dimensiones [n]');
ax.set_ylabel('Varianza');
fig.savefig('var_exp_full_sentiment.png')
from sklearn.linear_model import SGDClassifier
cls = SGDClassifier(loss="hinge", class_weight="balanced")
%%time
cls.fit(X_train_vec, y_train)
# Primero los datos de prueba tienen que pasar por el mismo
# proceso que pasaron los de entrenamiento 
# (no se debe entrenar de nuevo!)
X_test_vec = vec.transform(X_test)
y_pred = cls.predict(X_test_vec)
from sklearn.metrics import confusion_matrix, classification_report
class_names = cls.classes_
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = [i for i in class_names],
                     columns = [i for i in class_names])

df_cm
print(classification_report(y_test, y_pred))
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df_cm, annot=True, fmt='g', ax=ax)
ax.set_title('Matriz de confusión - Absoluta');
ax.set_xlabel('Predicción');
ax.set_ylabel('Realidad');
plt.show()
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

df_cm = pd.DataFrame(cm, index = [i for i in class_names],
                     columns = [i for i in class_names])
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df_cm, annot=True, fmt='.2f', ax=ax)
ax.set_title('Matriz de confusión - Normalizada');
ax.set_xlabel('Predicción');
ax.set_ylabel('Realidad');
plt.show()
vec_of = TfidfVectorizer()
X_train_of = vec_of.fit_transform(X_train)
len(vec_of.vocabulary_)
cls_of = SGDClassifier(loss="hinge", class_weight="balanced")
%%time
cls_of.fit(X_train_of, y_train)
X_test_of = vec_of.transform(X_test)
y_pred_of = cls_of.predict(X_test_of)
class_names = cls.classes_
cm_of = confusion_matrix(y_test, y_pred_of)
df_cm_of = pd.DataFrame(cm_of, index = [i for i in class_names],
                     columns = [i for i in class_names])
print(classification_report(y_test, y_pred_of))
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df_cm_of, annot=True, fmt='g', ax=ax)
ax.set_title('Matriz de confusión - Absoluta');
ax.set_xlabel('Predicción');
ax.set_ylabel('Realidad');
plt.show()
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('vec', TfidfVectorizer(preprocessor=my_preprocessor,tokenizer=my_tokenizer)), 
    ('cls', SGDClassifier())
    ],
    verbose=True
)
from sklearn.model_selection import GridSearchCV
param = {
    'cls__loss':['hinge', 'log'],
#     'cls__alpha':[0.0001, 0.001],
    'cls__penalty':['l1','l2'],
    'vec__min_df':[5, 10],
    'vec__ngram_range':[(1,1), (1,2)]
}
gs = GridSearchCV(pipe, param, cv=4, verbose=2, n_jobs=4)
gs.fit(X_train, y_train)
gs_df = pd.DataFrame(gs.cv_results_)
gs_df
gs.best_params_
y_pred_gs = gs.predict(X_test)
print(classification_report(y_test, y_pred_gs))
class_names = cls.classes_
cm_gs = confusion_matrix(y_test, y_pred_gs)
df_cm_gs = pd.DataFrame(cm_gs, index = [i for i in class_names],
                     columns = [i for i in class_names])
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df_cm_gs, annot=True, fmt='g', ax=ax)
ax.set_title('Matriz de confusión - Absoluta');
ax.set_xlabel('Predicción');
ax.set_ylabel('Realidad');
plt.show()
import nltk
stm = nltk.stem.SnowballStemmer("english")
def my_preprocessor(doc):
    doc_limpio = doc.translate(tabla)
    doc_limpio = doc_limpio.lower()
    return doc_limpio
def stemmed_tokenizer(doc):
    out = doc.split(' ')
    out = [stm.stem(w) for w in out]   # stemmer!
    return out
pipe = Pipeline([
    ('vec', TfidfVectorizer(preprocessor=my_preprocessor,
                            tokenizer=stemmed_tokenizer, 
                            min_df=10, 
                            ngram_range=(1,2))), 
    ('cls', SGDClassifier(loss='hinge', random_state=42))
    ],
    verbose=True
)
pipe.fit(X_train, y_train)
len(pipe.named_steps['vec'].vocabulary_)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))
class_names = cls.classes_
cm_pipe = confusion_matrix(y_test, y_pred)
df_cm_pipe = pd.DataFrame(cm_pipe, index = [i for i in class_names],
                     columns = [i for i in class_names])
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df_cm_pipe, annot=True, fmt='g', ax=ax)
ax.set_title('Matriz de confusión - Absoluta');
ax.set_xlabel('Predicción');
ax.set_ylabel('Realidad');
plt.show()
