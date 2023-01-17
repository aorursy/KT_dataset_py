# Download and install all dependencies

# gpr_pub contains the heuristics models and supplementary code

!git clone https://github.com/sattree/gpr_pub.git



!pip install allennlp --ignore-installed greenlet



# !pip install ../input/neural-coref/en_coref_lg-3.0.0/en_coref_lg-3.0.0/

# Huggingface neuralcoref model has issues with spacy-2.0.18

# !conda install -y cymem==1.31.2 spacy==2.0.12



!pip install attrdict pyhocon



!wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip

!unzip stanford-corenlp-full-2018-10-05.zip

!rm stanford-corenlp-full-2018-10-05.zip



# setup berkeley coref system

!git clone https://github.com/gregdurrett/berkeley-entity.git

!curl -s http://nlp.cs.berkeley.edu/downloads/berkeley-entity-models.tgz | tar xvz -C berkeley-entity

!mkdir berkeley-entity/data

!wget http://www.cs.utexas.edu/~gdurrett/data/gender.data.tgz

!tar -xvf gender.data.tgz

!mv gender.data berkeley-entity/data/

!rm gender.data.tgz
from IPython.core.display import display, HTML

# Add css styles and js events to DOM, so that they are available to rendered html

display(HTML(open('gpr_pub/visualization/highlight.css').read()))

display(HTML(open('gpr_pub/visualization/highlight.js').read()))
# Required for Lee et al coref model

# details can be found here https://github.com/kentonl/e2e-coref



import tensorflow as tf

TF_CFLAGS = " ".join(tf.sysconfig.get_compile_flags())

TF_LFLAGS = " ".join(tf.sysconfig.get_link_flags())



# Linux (build from source)

!g++ -std=c++11 -shared gpr_pub/modified_e2e_coref/coref_kernels.cc -o coref_kernels.so -fPIC $TF_CFLAGS $TF_LFLAGS -O2
import os

import pyhocon

import sys

import logging



from attrdict import AttrDict

from collections import defaultdict



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from sklearn.metrics import classification_report, log_loss

from sklearn.externals.joblib import Parallel, delayed

from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser

from tqdm import tqdm, tqdm_notebook



from allennlp.predictors.predictor import Predictor

from allennlp.models.archival import load_archive



import spacy
import sys

sys.path.insert(0, 'gpr_pub/modified_e2e_coref/')

sys.path.insert(0, 'gpr_pub/')
from gpr_pub.models.coref import Coref

from gpr_pub.models.heuristics.stanford_base import StanfordModel

from gpr_pub.models.heuristics.spacy_base import SpacyModel



# Heuristics models implement coref resolution based on heuristics described in the paper

# Pronoun resolution is a simple wrapper to convert coref predictions into class-specific labels

from gpr_pub.models.heuristics.random_distance import RandomModel

from gpr_pub.models.heuristics.token_distance import TokenDistanceModel

from gpr_pub.models.heuristics.syntactic_distance import StanfordSyntacticDistanceModel

from gpr_pub.models.heuristics.parallelism import AllenNLPParallelismModel as ParallelismModel

from gpr_pub.models.heuristics.url_title import StanfordURLTitleModel as URLModel



from gpr_pub.models.pretrained.lee_et_al import LeeEtAl2017

from gpr_pub.models.pretrained.stanford import StanfordCorefModel

from gpr_pub.models.pretrained.allennlp import AllenNLPCorefModel

from gpr_pub.models.pretrained.huggingface import HuggingfaceCorefModel

from gpr_pub.models.pretrained.berkley_coref_system import BCS



from gpr_pub.models.pronoun_resolution import PronounResolutionModel, PronounResolutionModelV2
from gpr_pub import visualization

from gpr_pub.utils import CoreNLPServer
SPACY_MODEL = spacy.load('en_core_web_lg')



STANFORD_CORENLP_PATH = 'stanford-corenlp-full-2018-10-05/'

server = CoreNLPServer(classpath=STANFORD_CORENLP_PATH,

                        corenlp_options=AttrDict({'port': 9090, 

                                                  'timeout': '600000',

                                                  'thread': '2',

                                                  'quiet': 'true',

                                                  'preload': 'tokenize,ssplit,pos,lemma,parse,depparse,ner,coref'}))

server.start()

STANFORD_SERVER_URL = server.url

STANFORD_MODEL = CoreNLPParser(url=STANFORD_SERVER_URL)
model_url = 'https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz'

archive = load_archive(model_url, cuda_device=0)

ALLEN_COREF_MODEL = Predictor.from_archive(archive)



model_url = 'https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz'

archive = load_archive(model_url, cuda_device=0)

ALLEN_DEP_MODEL = Predictor.from_archive(archive)



# HUGGINGFACE_COREF_MODEL = en_coref_lg.load()
test = pd.read_csv('../input/gendered-pronoun-resolution/test_stage_1.tsv', sep='\t')



# normalizing column names

test.columns = map(lambda x: x.lower().replace('-', '_'), test.columns)

with pd.option_context('display.max_rows', 10, 'display.max_colwidth', 15):

    display(test)
lee_coref_model = LeeEtAl2017(SPACY_MODEL, config = {'name': 'final',

                                                     'log_root': '../input/e2e-coref-data/',

                                                    'model': 'gpr_pub/modified_e2e_coref/experiments.conf',

                                                    'context_embeddings_root': '../input/e2e-coref-data/',

                                                    'head_embeddings_root': '../input/e2e-coref-data/',

                                                    'char_vocab_root': '../input/e2e-coref-data/'

                                                    })
sample = test.loc[17]

data = ALLEN_COREF_MODEL.predict(sample.text)

print('{:-<100}'.format('Example where Coref resolves to both gold mentions: Pronoun={}, A={}, B={}'.format(sample.pronoun, sample.a, sample.b)))

visualization.render(data, allen=True, jupyter=True)



sample = test.loc[13]

data = ALLEN_COREF_MODEL.predict(sample.text)

print('{:-<100}'.format('Example where a single antecedent contains both gold mentions: Pronoun={}, A={}, B={}'.format(sample.pronoun, sample.a, sample.b)))

visualization.render(data, allen=True, jupyter=True)



sample = test.loc[13]

data = lee_coref_model.predict(**sample)

print('{:-<100}'.format('Example to compare the performance of Lee et al: Pronoun={}, A={}, B={}'.format(sample.pronoun, sample.a, sample.b)))

visualization.render({'document': data[0], 'clusters': data[1]}, allen=True, jupyter=True)
random_coref_model = RandomModel(SPACY_MODEL)

random_proref_model = PronounResolutionModel(random_coref_model)



token_distance_coref_model = TokenDistanceModel(SPACY_MODEL)

token_distance_proref_model = PronounResolutionModel(token_distance_coref_model)



syntactic_distance_coref_model = StanfordSyntacticDistanceModel(STANFORD_MODEL)

syntactic_distance_proref_model = PronounResolutionModel(syntactic_distance_coref_model, n_jobs=1)



parallelism_coref_model = ParallelismModel(ALLEN_DEP_MODEL, SPACY_MODEL)

parallelism_proref_model = PronounResolutionModel(parallelism_coref_model)



url_title_coref_model = URLModel(STANFORD_MODEL)

url_title_proref_model = PronounResolutionModel(url_title_coref_model, n_jobs=1)
# preds = random_proref_model.predict(test)

# test['random_a_coref'], test['random_b_coref'] = zip(*preds)



# preds = token_distance_proref_model.predict(test)

# test['token_distance_a_coref'], test['token_distance_b_coref'] = zip(*preds)



preds = syntactic_distance_proref_model.predict(test)

test['syntactic_distance_a_coref'], test['syntactic_distance_b_coref'] = zip(*preds)



preds = parallelism_proref_model.predict(test)

test['parallelism_a_coref'], test['parallelism_b_coref'] = zip(*preds)



preds = url_title_proref_model.predict(test)

test['parallelism_url_a_coref'], test['parallelism_url_b_coref'] = zip(*preds)
stanford_coref_model = StanfordCorefModel(STANFORD_MODEL, algo='clustering')

deterministic_stanford_proref_model = PronounResolutionModelV2(stanford_coref_model, n_jobs=1, multilabel=True)



stanford_coref_model = StanfordCorefModel(STANFORD_MODEL, algo='statistical')

statistical_stanford_proref_model = PronounResolutionModelV2(stanford_coref_model, n_jobs=1, multilabel=True)



stanford_coref_model = StanfordCorefModel(STANFORD_MODEL, algo='neural', greedyness=0.5)

neural_stanford_proref_model = PronounResolutionModelV2(stanford_coref_model, n_jobs=1, multilabel=True)



allen_coref_model = AllenNLPCorefModel(ALLEN_COREF_MODEL, SPACY_MODEL)

allen_proref_model = PronounResolutionModelV2(allen_coref_model, n_jobs=2, multilabel=True)



# huggingface_coref_model = HuggingfaceCorefModel(HUGGINGFACE_COREF_MODEL)

# hugginface_proref_model = PronounResolutionModelV2(huggingface_coref_model, multilabel=True)



lee_coref_model = LeeEtAl2017(SPACY_MODEL, config = {'name': 'final',

                                                     'log_root': '../input/e2e-coref-data/',

                                                    'model': 'gpr_pub/modified_e2e_coref/experiments.conf',

                                                    'context_embeddings_root': '../input/e2e-coref-data/',

                                                    'head_embeddings_root': '../input/e2e-coref-data/',

                                                    'char_vocab_root': '../input/e2e-coref-data/'

                                                    })

lee_proref_model = PronounResolutionModelV2(lee_coref_model, multilabel=True)



bcs_coref_model = BCS(STANFORD_MODEL)

bcs_proref_model = PronounResolutionModelV2(bcs_coref_model, multilabel=True)
# preds = hugginface_proref_model.predict(test)

# test['huggingface_ml_a_coref'], test['huggingface_ml_b_coref'] = zip(*preds)



preds = allen_proref_model.predict(test)

test['allen_ml_a_coref'], test['allen_ml_b_coref'] = zip(*preds)



preds = deterministic_stanford_proref_model.predict(test)

test['stanford_ml_deterministic_a_coref'], test['stanford_ml_deterministic_b_coref'] = zip(*preds)



preds = statistical_stanford_proref_model.predict(test)

test['stanford_ml_statistical_a_coref'], test['stanford_ml_statistical_b_coref'] = zip(*preds)



preds = neural_stanford_proref_model.predict(test)

test['stanford_ml_neural_a_coref'], test['stanford_ml_neural_b_coref'] = zip(*preds)



preds = lee_proref_model.predict(test)

test['lee_a_coref'], test['lee_b_coref'] = zip(*preds)



preds = bcs_proref_model.predict(test, preprocessor=BCS.preprocess)

test['bcs_a_coref'], test['bcs_b_coref'] = zip(*preds)
# Investigate diversity of the models

models = (

        'parallelism_url',

        'allen_ml', 

        # 'huggingface_ml', 

        'parallelism', 

        'stanford_ml_deterministic', 

        'syntactic_distance', 

        'stanford_ml_statistical',

        'stanford_ml_neural',

        'bcs',

        'lee',

       )



scores = []

for model in models[:-1]:

    score = mutual_info_score(test['{}_a_coref'.format(model)], test['lee_a_coref'])

    score2 = mutual_info_score(test['{}_b_coref'.format(model)], test['lee_b_coref'])

    scores.append((model, score, score2))

    

models = pd.DataFrame(scores, columns=['model', 'score_a', 'score_b']).set_index('model').sort_values('score_b')

models['parity(~diversity)'] = models.min(axis=1)

models
# shortlisted diverse set of models

models = (

        'parallelism_url',

        'allen_ml',

        'parallelism', 

        'syntactic_distance', 

        'stanford_ml_statistical',

        'lee',

        'bcs'

       )



models_a = [model+'_a_coref' for model in models]

models_b = [model+'_b_coref' for model in models]



test['votes_a'] = test[models_a].sum(axis=1)

test['votes_b'] = test[models_b].sum(axis=1)

test['votes_a_b'] = test[models_a+models_b].sum(axis=1)



plt.hist([test['votes_a'], test['votes_b']], label=['Class A', 'Class B'], bins=range(1, len(models)+2))

plt.legend()

plt.show()
# define proxy labels based on votes

mask_a = test['votes_a'] >=5

mask_b = test['votes_b'] >=4

mask_a_b = test['votes_a_b'] <= 1



true_proxy = test[mask_a | mask_b | mask_a_b]



true_proxy['label'] = 2

true_proxy.loc[mask_a, 'label'] = 0

true_proxy.loc[mask_b, 'label'] = 1
feats = models_a + models_b



X = true_proxy[feats]

y = true_proxy['label']



print(X.shape)



clf = LogisticRegression(multi_class='auto', solver='liblinear', penalty='l1', C=.05, max_iter=30)



scores = cross_val_score(clf, X, y, cv=StratifiedKFold(3, random_state=21))

scores
clf.fit(X, y)



X_tst = test[feats]

probabilties = clf.predict_proba(X_tst)



plt.hist(probabilties, range=(0,1), label=['A', 'B', 'NEITHER'])

plt.legend()

plt.title('Class-wise probability distributions from Confidence Model')

plt.show()
models_perfect = ('parallelism_url',

        'parallelism', 

        'lee')



models_a = [model+'_a_coref' for model in models_perfect]

models_b = [model+'_b_coref' for model in models_perfect]



mask_a_perfect = test[models_a].all(axis=1)

mask_b_perfect = test[models_b].all(axis=1)

print(test[mask_a_perfect].shape, test[mask_b_perfect].shape)



# set the lower bound, assuming 1% chance of failure

probabilties[mask_a_perfect] = [1,.02,.02]

probabilties[mask_b_perfect] = [.01,1,.01]



# Softmax of probabilities of joint model to convert them to labels for analysis

y_pred = np.zeros_like(probabilties)

y_pred[np.arange(len(probabilties)), probabilties.argmax(1)] = 1

y_pred = y_pred.astype(bool)



plt.hist(probabilties, range=(0, 1), label=['A', 'B', 'NEITHER'])

plt.title('Class-wise probability distributions from Confidence Model after adjustment')

plt.legend()

plt.show()
res = pd.concat([pd.DataFrame(y_pred, columns=['A', 'B', 'NEITHER']), 

                 pd.DataFrame(probabilties, columns=['prob_A', 'prob_B', 'prob_NEITHER'])], 

                axis=1)



plt.figure(figsize=(16,4))

plt.subplot(1,3,1)

plt.hist([res[res['A']]['prob_A'], res[~res['A']]['prob_A']], bins=10, rwidth=0.7, label=['True', 'False'])

plt.title('Distribution of probabilties over \nsamples predicted as class A')

plt.legend()



plt.subplot(1,3,2)

plt.hist([res[res['B']]['prob_B'], res[~res['B']]['prob_B']], bins=10, rwidth=0.7, label=['True', 'False'])

plt.title('Distribution of probabilties over \nsamples predicted as class B')

plt.legend()



plt.subplot(1,3,3)

plt.hist([res[res['NEITHER']]['prob_NEITHER'], res[~res['NEITHER']]['prob_NEITHER']], bins=10, rwidth=0.7, label=['True', 'False'])

plt.title('Distribution of probabilties over \nsamples predicted as class NEITHER')

plt.legend()



plt.show()
sub_df = pd.read_csv('../input/gendered-pronoun-resolution/sample_submission_stage_1.csv')

sub_df.loc[:, 'A'] = probabilties[:, 0]

sub_df.loc[:, 'B'] = probabilties[:, 1]

sub_df.loc[:, 'NEITHER'] = probabilties[:, 2]



sub_df.to_csv("submission.csv", index=False)



sub_df.head()
!rm -r stanford-corenlp-full-2018-10-05/

!rm -r gpr_pub/

!rm -r berkeley-entity/

!rm -r tmp/