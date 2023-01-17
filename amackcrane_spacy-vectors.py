import spacy

try:

    nlp

except NameError:

    nlp = spacy.load('en_core_web_lg')



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.pipeline import Pipeline, FeatureUnion, make_union, make_pipeline

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.linear_model import RidgeClassifier

from sklearn.metrics.pairwise import cosine_similarity

from collections.abc import Iterable



import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None

import functools

import math



import matplotlib.pyplot as plt

%matplotlib inline
import pandas as pd

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train = train.sample(frac=1, replace=False)

train.shape

n=6350

for i in range(n, n+20):

    print(str(train.target[i]) + " -- " + train.keyword[i] + " -- " + train.text[i])

    print()


# Since we may have multiple parallel feature transformers wanting

#   spacy objects, we abstract out the spacy parsing

class SpacyParser(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X['doc'] = list(nlp.pipe(X.text, disable=['ner']))

        return X



# get first occurrence of given spacy dependency tag

# takes iterable dependencies argument

def get_dep(doc, dependencies):

    try:

        token = [tok for tok in doc if tok.dep_ in dependencies][0]

    except IndexError:

        token = None

    return token



# Grab token's syntactic dependents and sum vectors

def sum_children(token, vector_length):

    children = get_children(token)

    if children == None:

        return np.zeros(vector_length)

    else:

        vecs = [tok.vector for tok in children]

        vecs = np.array(vecs).reshape(-1, len(children))

        vec = np.sum(vecs, axis=1)

        return vec

    

def get_children(token):

    if token == None:

        return None

    if token.dep_ == "ROOT":

        # summing w/ dependency makes more sense for subj/obj phrases than ROOT

        return [token]

    else:

        tokens = list(token.children)

        tokens.append(token)    

        return tokens

    

    

class Embedder(BaseEstimator, TransformerMixin):

    def __init__(self, dep_list=["ROOT", "nsubj", ("dobj", "pobj", "obj")], 

                 norm=False, sum=False):

        self.dep_list = dep_list

        # Must have dependency sets wrapped in iterables for predictable 

        #   behavior when testing set membership in get_dep

        for i,deps in enumerate(self.dep_list):

            if isinstance(deps, str):

                self.dep_list[i] = (deps,)

                

        self.norm = norm

        self.sum = sum

    

    def fit(self, X, y=None):

        return self

    # in: data.frame w/ .text attr

    def transform(self, X):

        if 'doc' in X.columns:

            docs = X.doc.values

        else:

            #docs = [nlp(string) for string in X.text]

            # faster

            docs = list(nlp.pipe(X.text.values, disable=["ner"]))

        

        vector_length = len(docs[0][0].vector)

        

        matrix_list = []

        for deps in self.dep_list:

            toks = [get_dep(doc, deps) for doc in docs]

            if self.sum:

                vecs = [sum_children(tok, vector_length) for tok in toks]

            else:

                vecs = [tok.vector if tok else np.zeros(vector_length) 

                             for tok in toks]



            vecs = np.array(vecs).reshape(-1, vector_length)

            

            # unit norm?

            if self.norm:

                norms = [(1 / tok.vector_norm) if (tok and tok.vector_norm > 0) else 1

                             for tok in toks]

                # to multiply matrix cols by vector, make sure we brodacast along cols

                norms = np.array(norms).reshape(-1, 1)

                vecs = vecs * norms

            

            matrix_list.append(vecs)

        

        matrix = np.concatenate(matrix_list, axis=1)

        return matrix

    

# Features related to keyword variable and its dependency relations

#   w/in tweet

class KeywordFeatures(BaseEstimator, TransformerMixin):

    # Get dependency label for keyword

    # return np.array of strings

    def dependency(self, keywords, X):

        if 'doc' in X.columns:

            docs = X.doc

        else:

            docs = list(nlp.pipe(X.text, disable=['ner']))

        

        # filter nan's from keywords

        #keywords = np.where(pd.isnull(keywords), "", keywords)

        

        # for multi-word keyword, last word seems more useful than first

        keywords = [kw.split('%20')[-1] for kw in keywords]

        

        # find keyword instance(s) in doc

        kw_tok = [[tok for tok in doc if tok.text.lower() == kw.lower()]

                      for kw, doc in zip(keywords, docs)]

        # take first instance, if found, and get dependency relation

        kw_dep = [toks[0].dep_ if toks else "" for toks in kw_tok]

        kw_dep = np.array(kw_dep).reshape(-1, 1)

        return kw_dep

    

    # for multi-word keyword, check whether they occur in same part of sentence

    def distance(self, keywords, X):

        if 'doc' in X.columns:

            docs = X.doc

        else:

            docs = list(nlp.pipe(X.text, disable=['ner']))

        

        # grab roots

        roots = [get_dep(doc, "ROOT") for doc in docs]

        

        # tokenize keywords

        keywords = [kw.split('%20') for kw in keywords]

        kw_tok = []

        for kws, doc in zip(keywords, docs):

            kw_tok_temp = []

            if len(kws) == 1:

                # ignore singletons here

                kw_tok.append(None)

                continue

            for kw in kws:

                # append first match

                try:

                    kw_tok_temp.append([tok for tok in doc if tok.text.lower() == kw.lower()][0])

                except IndexError:

                    # Give up if a match failed

                    kw_tok.append(None)

                    continue

            kw_tok.append(kw_tok_temp)

        

        # code dependency proximity as 0/1, w/ -1 for singletons

        same_side = []

        for root, kws in zip(roots, kw_tok):

            # treat singletons

            if not kws:

                same_side.append(-1)

                continue

            # Collect dependency locations

            left = []

            for kw in kws:

                left.append(kw in list(root.lefts))

            # compute XNOR -- true if all on same side of root in parse tree

            if (functools.reduce(lambda x,y: x and y, left) or 

                not functools.reduce(lambda x,y: x or y, left)):

                same_side.append(1)

            else:

                same_side.append(0)

                

        return np.array(same_side).reshape(-1, 1)

        

    

    # Concatenate features in categorical encoding

    def get_categorical_features(self, X):

        # keep 'kw' one-dim for preprocessing

        kw = X.keyword.fillna("").values.reshape(-1)

        dep = self.dependency(kw, X)

        dist = self.distance(kw, X)

        # convert kw to 2-d as feature

        kw = kw.reshape(-1, 1)

        all = np.concatenate([kw, dep, dist], axis=1)

        return all

    

    def fit(self, X, y=None):

        cat = self.get_categorical_features(X)

        

        self.encoder = OneHotEncoder(handle_unknown='ignore')

        self.encoder.fit(cat)

        return self

        

    def transform(self, X):

        cat = self.get_categorical_features(X)

        return self.encoder.transform(cat)

        



def cosine_dist(x, y):

    #x = x.reshape(1, -1)

    #y = y.reshape(1, -1)

    return 1 - cosine_similarity(x,y)

    



class TfidfWrapper(TfidfVectorizer):

    def fit(self, X, y=None):

        return super().fit(X.text.values)

    def transform(self, X):

        return super().transform(X.text.values)

    def fit_transform(self, X, y=None):

        self.fit(X)

        return self.transform(X)

    

# Abstract class

# For making transformers out of estimators w/ continuous 'predict_proba'-style outputs

# written here for binary classification

class PredictorTransformer:

    def _fit(self, fit_method, predict_method, X, y):

        fit_method(X, y)

        self.ss = StandardScaler()

        self.ss.fit(self._pred(predict_method, X))

        

    def _pred(self, predict_method, X):

        z = predict_method(X)

        # in case of probabilities, avoid collinearity

        if len(z.shape) > 1 and z.shape[1] > 1:

            z = z[:,:-1]

        else:

            z = z.reshape(-1, 1)

        return z

        

    def _transform(self, predict_method, X):

        z = self._pred(predict_method, X)

        z = self.ss.transform(z)

        return z

    

class SVCTransformer(LinearSVC, PredictorTransformer):

    def fit(self, X, y):

        self._fit(super().fit, super().decision_function, X, y)

        return self

        

    def transform(self, X):

        df = self._transform(super().decision_function, X)

        return df



    

class DeprSVCTransformer(LinearSVC):

    def get_transform(self, X):

        return super().decision_function(X).reshape(-1, 1)



    def fit(self, X, y):

        super().fit(X, y)

        # persist scaling

        self.ss = StandardScaler()

        df = super().decision_function(X).reshape(-1, 1)

        self.ss.fit(df)

        return self

    

    def transform(self, X):

        df = super().decision_function(X).reshape(-1, 1)

        df = self.ss.transform(df)

        return df

    

class KernelSVCTransformer(SVC, PredictorTransformer):

    def fit(self, X, y):

        self._fit(super().fit, super().decision_function, X, y)

        return self

    def transform(self, X):

        df = self._transform(super().decision_function, X)

        return df

"""

class GNBTransformer(GaussianNB):

    def transform(self, X):

        probs = super().predict_proba(X)

        # drop a column to avoid redundancy

        return probs[:,:-1]

""" 



class MNBTransformer(MultinomialNB, PredictorTransformer):

    def fit(self, X, y):

        self._fit(super().fit, super().predict_proba, X, y)

        return self

    def transform(self, X):

        pr = self._transform(super().predict_proba, X)

        return pr

    

class RFTransformer(RandomForestClassifier, PredictorTransformer):

    def fit(self, X, y):

        self._fit(super().fit, super().predict_proba, X, y)

        return self

    def transform(self, X):

        pr = self._transform(super().predict_proba, X)

        return pr



class FeaturePasser(BaseEstimator, TransformerMixin):

    def get_length(self, X):

        length = list(map(lambda x: len(x.split()), X.text.values))

        length = np.array(length).reshape(-1, 1)

        return length

    def fit(self, X, y=None):

        self.ss = StandardScaler()

        self.ss.fit(self.get_length(X))

        return self

    def transform(self, X):

        length = self.ss.transform(self.get_length(X))

        return length

# Word embedding features

vectors = Pipeline([('emb', Embedder(sum=True, norm=True)),

                   ('transformer', KernelSVCTransformer(kernel=cosine_similarity))])



# Features from keyword column

keywords = Pipeline([('kwf', KeywordFeatures()),

                    ('transformer', RFTransformer())])



spacy_features = FeatureUnion([('vector', vectors),

                              ('keyword', keywords)])

spacy_features = Pipeline([('parser', SpacyParser()),

                          ('feature', spacy_features)])



# Features based on doc-word matrix

counts = Pipeline([('vectorizer', TfidfWrapper(min_df=1, max_df=.5,

                                                 strip_accents='unicode')),

                  ('transformer', make_union(SVCTransformer(C=.3),

                                            MNBTransformer()))])



features = FeatureUnion([('count', counts),

                        ('spacy', spacy_features),

                        ('aux', FeaturePasser())])



pipe = Pipeline([('feature', features),

                 ('classifier', KNeighborsClassifier())], verbose=False)

# save baseline parameters

baseline_params = pipe.get_params()



# GridSearch parameters

param_grid = {'feature__spacy__feature__keyword': [keywords],

              'feature__spacy__feature__vector': ['drop', vectors],

              'feature__count': [counts],

              'feature__aux': [FeaturePasser()]

              }

    

cv_params = {'cv': 2, 'scoring': 'f1', 

                     'verbose': 0, 'n_jobs': -1}
pipe.set_params(**baseline_params)

pass
""" # Check out dependency parsing

n=10

doc = nlp(train.text[n])



def print_dep(doc):

    print("-------------")

    for tok in doc:

        print(tok.text + " " + tok.dep_)



print_dep(doc)

#root = [tok for tok in doc if tok.dep_ == "ROOT"][0]

#print(root)

root = get_dep(doc, "ROOT")

print()

print(root)

print(list(root.lefts)[0])

print(list(root.rights)[0])



print(list(doc.noun_chunks))

"""

# Testing Silly Matrix Ideas

#left = get_dep(doc, "nsubj")

#right = get_dep(doc, "obj", "pobj")

#print(left)

#print(right)

#A = np.matmul(left.vector.reshape(-1, 1),

#              np.linalg.pinv(right.vector.reshape(-1, 1)))

#print(A.shape)

#print(np.linalg.eig(A)[1][:,0])

#print(right.vector - left.vector)





"""# Testing Keyword Features

n=3000

x = train.iloc[n:n+20]

kw = KeywordFeatures().get_categorical_features(x)

print(kw)

#print([kw[i] for i in range(len(kw)) if kw[i,2] == 0])

print(x)

#print(type(kw))

#print(kw.shape)

"""





""" # Testing Embedder

n=20

x = train[n:n+5]

deps = ["dobj"]

embed = Embedder(sum=False, dep_list=deps).transform(x)

embed2 = Embedder(sum=True, dep_list=deps).transform(x)



for i in range(5):

    got_dep = get_dep(nlp(x.iloc[i].text), *deps)

    print(str(deps) + ": " + str(got_dep))

    if got_dep:

        print("vector: " + str(got_dep.has_vector))

        print("children: " + str(get_children(got_dep)))

        print("base vec: ")

        print(got_dep.vector[:5])

        print("concat vec: ")

        print(sum_children(got_dep, len(got_dep.vector))[:5])



print(embed)

print(embed2)



for i in range(5):

    _

    #print_dep(nlp(x.iloc[i].text))

"""

pass
train['length'] = list(map(lambda x: len(x.split()), train.text))

# print(train.length.describe())

# print(train.target.astype('category').describe())

# print(train.keyword.astype('category').describe())

# print(train.keyword.value_counts()[:10])
labels = np.array(train.target)

labels.shape
doc_word = TfidfVectorizer().fit_transform(train.text)

doc_word = doc_word.toarray()

frequencies = np.add.reduce(doc_word, axis=0)

#plt.title("Distribution of Word Frequencies")

plt.hist(frequencies, bins=20, log=True)

pass


# hack hack hack

if isinstance(param_grid, list):

    nrows = len(param_grid)

else:

    nrows=1

    param_grid = [param_grid]



best_params = []

results = []

for i in range(nrows):

    gridsearch = GridSearchCV(pipe, param_grid=param_grid[i], refit=True,

                             **cv_params)

    gridsearch.fit(X=train, y=labels)

    results.append(pd.DataFrame(gridsearch.cv_results_))

    best_params.append(gridsearch.best_params_)



    # reset

    pipe.set_params(**baseline_params)
# if param_grid:

#     for res in results:

#         print(res.filter(items=['params', 'mean_fit_time', 

#                                 'mean_test_score']))
for i,res in enumerate(results):

    # pull out params_ columns...

    params = res.filter(like='param_')



    # pull out a transformer parameter...

    for i in range(params.shape[0]):

        try:

            x = params['param_feature__spacy__feature__keyword__transformer'].iloc[i]

            params['param_feature__spacy__feature__keyword__transformer'].iloc[i] = x.get_params()['max_depth']

        except:

            _



    # gross, would like mutate_if(is.???, as.character)

    params = params.applymap(lambda x: x if isinstance(x, (float, int))

                             else str(x))



    # drop trivial columns

    keep=[]

    for col in params.columns:

        _



    # how many params are there?

    cols = params.shape[1]



    if cols == 1:

        fig, ax = plt.subplots()







        try: # for numeric parameters...

            xlims = [params.values.min(), params.values.max()]

        except Exception:

            pass



        if params.dtypes[0] in {'int64', 'float64'}:

            ax.plot(params, res.mean_test_score, 'b-')

            ax.set_xlim(xlims)

        else:

            ax.bar(x=range(params.shape[0]), height=res.mean_test_score,

                  align='edge', width=.4)



        ax.set_xlabel(params.columns[0])



        ax2 = ax.twinx()

        if params.dtypes[0] in {'int64', 'float64'}:

            ax2.plot(params, res.mean_fit_time, 'r-')

        else:

            ax2.bar(x=np.linspace(.5, params.shape[0] + .5, 

                                  params.shape[0], endpoint=False), 

                    height=res.mean_fit_time, 

                   align='edge', width=.4, color='red')

        ax.set_ylabel("Test Score")

        ax2.set_ylabel("Fit Time")

    else:

        #print(res)

        #print(params.columns.values[1:])

        #print(res.groupby(params.columns.values, axis=0))

        table = res.pivot_table(columns=params.iloc[:,0], 

                              index=params.iloc[:,2],

                              #index=params.columns[1:],

                              values=['mean_test_score', 'mean_fit_time',

                                     'rank_test_score'])

        print(table)

        #axs[i].axis('off')

        #axs[i].table(cellText=table.values, rowLabels=table.index, 

        #             colLabels=table.columns, loc=9)
# collect best parameters, in case we did multiple grid searches

new_params = functools.reduce(lambda x,y: dict(x, **y), best_params, {})

pipe.set_params(**new_params)

pipe.fit(X=train, y=labels)

pass
features = pipe[:-1].transform(train)

try:

    features = features.toarray()

except Exception:

    pass

if features.shape[1] > 2:

    pca = PCA(n_components=2)

    doc_pc = pca.fit_transform(features)

else:

    doc_pc = features

features.shape
try:

    plt.scatter(x=doc_pc[:,0], y=doc_pc[:,1], c=labels)

except IndexError:

    # only one feature

    pass
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test.head(5)
test.shape
test_labels = pipe.predict(test)

test_labels.shape
submission = pd.DataFrame({'id': test.id, 'target': test_labels})

#submission['target'] = 1 - submission['target']

submission.to_csv('submission.csv', index=False)

submission