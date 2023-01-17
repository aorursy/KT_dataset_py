from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.pipeline import Pipeline, FeatureUnion, make_union, make_pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.decomposition import PCA

from sklearn.naive_bayes import MultinomialNB



import numpy as np

import pandas as pd

import functools

import math



import matplotlib.pyplot as plt

%matplotlib inline
class MNBTransformer(MultinomialNB):

    def transform(self, X):

        return super().predict_proba(X)

    

class SVCTransformer(LinearSVC):

    def transform(self, X):

        df = super().decision_function(X).reshape(-1, 1)

        df = StandardScaler().fit_transform(df)

        return df

    

# Convenience wrapper so that we can input the DataFrame to the pipeline

class TfidfWrapper(TfidfVectorizer):

    def fit(self, X, y=None):

        return super().fit(X.text)

    def transform(self, X):

        return super().transform(X.text)

    def fit_transform(self, X, y=None):

        return self.fit(X).transform(X)

    

# Make keyword and tweet length available as features

class FeaturePasser(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.onehot = OneHotEncoder(handle_unknown='ignore')

        kw = X.keyword.fillna("").values.reshape(-1, 1)

        self.onehot.fit(kw)

        return self

    

    def transform(self, X):

        kw = X.keyword.fillna("").values.reshape(-1, 1)

        kw = self.onehot.transform(kw).toarray()

        length = list(map(lambda x: len(x.split()), X.text))

        length = np.array(length).reshape(-1, 1)

        length = StandardScaler().fit_transform(length)

        feat = np.concatenate((length, kw), axis=1)

        return feat

    

    def fit_transform(self, X, y=None):

        return self.fit(X).transform(X)
# baseline parameters

vectorizer_args = {'features__topic__vectorizer': TfidfWrapper(),

                   'features__topic__vectorizer__tokenizer': None,

                   'features__topic__vectorizer__max_df': .5,

                   'features__topic__vectorizer__min_df': 1,

                    'features__topic__vectorizer__strip_accents': 'unicode'}

baseline_args = dict({'classifier': KNeighborsClassifier(), 

                      'features__topic__model': SVCTransformer(C=.3)},

                     **vectorizer_args)







# GridSearch parameters

param_grid = {'features__topic__model__C': [.1, .3, .5]}

              

gridsearch_params = {'refit': True, 'cv': 5, 'scoring': 'f1', 

                     'verbose': 0, 'n_jobs': -1}
topic = Pipeline([('vectorizer', TfidfWrapper()),

                  ('model', SVCTransformer())], verbose=True)

features = FeatureUnion([('topic', topic),

                         ('aux', FeaturePasser())])

pipe = Pipeline([('features', features),

                ('classifier', KNeighborsClassifier())], verbose=False)



pipe.set_params(**baseline_args)

pass
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train = train.sample(frac=1, replace=False)

train.shape
train.head(5)
train['length'] = list(map(lambda x: len(x.split()), train.text))

# print(train.length.describe())

# print(train.target.astype('category').describe())

# print(train.keyword.astype('category').describe())

# print(train.keyword.value_counts()[:20])
labels = np.array(train.target)

labels.shape

doc_word = TfidfVectorizer().fit_transform(train.text)

doc_word = doc_word.toarray()

frequencies = np.add.reduce(doc_word, axis=0)

plt.hist(frequencies, bins=20, log=True)

#plt.title("Distribution of Word Frequencies")

pass


# hack hack hack

if isinstance(param_grid, type([])):

    nrows = len(param_grid)

else:

    nrows=1

    param_grid = [param_grid]



best_params = []

results = []

for i in range(nrows):

    gridsearch = GridSearchCV(pipe, param_grid=param_grid[i],

                             **gridsearch_params)

    gridsearch.fit(X=train, y=labels)

    results.append(pd.DataFrame(gridsearch.cv_results_))

    best_params.append(gridsearch.best_params_)



    # reset

    pipe.set_params(**baseline_args)



# for res in results:

#     print(res.filter(items=['params', 'mean_fit_time', 

#                             'mean_test_score'])) 
for i,res in enumerate(results):

    # pull out params_ columns...

    params = res.filter(like='param_')



    # gross, would like mutate_if(is.???, as.character)

    params = params.applymap(lambda x: x if isinstance(x, (float, int))

                             else str(x)[:12])



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

                  align='edge', width=.4, label="Test Score")



        ax.set_xlabel(params.columns[0])



        ax2 = ax.twinx()

        if params.dtypes[0] in {'int64', 'float64'}:

            ax2.plot(params, res.mean_fit_time, 'r-', label="Fit Time")

        else:

            ax2.bar(x=np.linspace(.5, params.shape[0] + .5, 

                                  params.shape[0], endpoint=False), 

                    height=res.mean_fit_time, 

                   align='edge', width=.4, color='red', label="Fit Time")

        ax.set_ylabel("Score")

        ax2.set_ylabel("Time (s)")

        fig.legend()

    else:

        table = res.pivot_table(columns=params.iloc[:,0], 

                              index=params.iloc[:,1],

                              values=['mean_test_score', 'mean_fit_time',

                                     'rank_test_score'])

        print(table)

        #axs[i].axis('off')

        #axs[i].table(cellText=table.values, rowLabels=table.index, 

        #             colLabels=table.columns, loc=9)



# collect best parameters, in case we did multiple independent gridsearches

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

submission.to_csv('submission.csv', index=False)

submission