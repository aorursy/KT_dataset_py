#Importar bibliotecas e criar funcoes de tokenizacao e svd



from nltk.corpus import nps_chat

import nltk

import numpy as np

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

import string

from nltk.corpus import wordnet

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn import neighbors

from sklearn.model_selection import RandomizedSearchCV

import scipy

from sklearn.metrics import *

import pickle

from nltk.corpus import sentence_polarity

nltk.download('sentence_polarity')



def my_tokenizer(doc):

    words = word_tokenize(doc)

    pos_tags = pos_tag(words)

    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]

    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]

    lemmas = []

    for w in non_punctuation:

        if w[1].startswith('J'):

            pos = wordnet.ADJ

        elif w[1].startswith('V'):

            pos = wordnet.VERB

        elif w[1].startswith('N'):

            pos = wordnet.NOUN

        elif w[1].startswith('R'):

            pos = wordnet.ADV

        else:

            pos = wordnet.NOUN

        lemmas.append(lemmatizer.lemmatize(w[0], pos))

    return lemmas



class SVDDimSelect(object):

    def fit(self, X, y=None):        

        try:

            self.svd_transformer = TruncatedSVD(n_components=round(X.shape[1]/2))

            self.svd_transformer.fit(X)

            cummulative_variance = 0.0

            k = 0

            for var in sorted(self.svd_transformer.explained_variance_ratio_)[::-1]:

                cummulative_variance += var

                if cummulative_variance >= 0.5:

                    break

                else:

                    k += 1

            self.svd_transformer = TruncatedSVD(n_components=k)

        except Exception as ex:

            print(ex)

        return self.svd_transformer.fit(X)

    def transform(self, X, Y=None):

        return self.svd_transformer.transform(X)

    def get_params(self, deep=True):

        return {}



    

# Obter Corpus e Dividir em datasets de treino e teste

x_data_neg = []

x_data_neg.extend([' '.join(post) for post in sentence_polarity.sents(categories=['neg'])])

y_data_neg = [0] * len(x_data_neg)

x_data_pos = []

x_data_pos.extend([' '.join(sent) for sent in sentence_polarity.sents(categories=['pos'])])

y_data_pos = [1] * len(x_data_pos)

x_data_full = x_data_neg[:500] + x_data_pos[:500]

y_data_full = y_data_neg[:500] + y_data_pos[:500]

x_data = np.array(x_data_full, dtype=object)

y_data = np.array(y_data_full)

train_indexes = np.random.rand(len(x_data)) < 0.80

x_data_train = x_data[train_indexes]

y_data_train = y_data[train_indexes]

x_data_test = x_data[~train_indexes]

y_data_test = y_data[~train_indexes]

stopwords_list = stopwords.words('english')

lemmatizer = WordNetLemmatizer()



# Criar um Pipeline e Treinar o classificador

clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')

my_pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer=my_tokenizer)), ('svd', SVDDimSelect()), ('clf', clf)])               

par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}

hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20) #njobs > 1 no kaggle nao permite pois nao consegue rodar mais de 1 thread

hyperpar_selector.fit(X=x_data_train, y=y_data_train)



#Imprimir a melçhor parametrização

print("Best score: %0.3f" % hyperpar_selector.best_score_)

print("Best parameters set:")

best_parameters = hyperpar_selector.best_estimator_.get_params()

for param_name in sorted(par.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))

    

y_pred = hyperpar_selector.predict(x_data_test)

print(accuracy_score(y_data_test, y_pred))
# Serializando o modelo

string_obj = pickle.dumps(hyperpar_selector)

model_file = open('model.pkl', 'wb')

model_file.write(string_obj)

model_file.close()

model_file = open('model.pkl', 'rb')

model_content = model_file.read()

obj_classifier = pickle.loads(model_content)

model_file.close()
res = obj_classifier.predict_proba(["take care of my cat offers a refreshingly different slice of asian cinema ."])

res2 = obj_classifier.predict_proba(["at heart the movie is a deftly wrought suspense yarn whose richer shadings work as coloring rather than substance ."])

print(res, res2)
x_data_pos