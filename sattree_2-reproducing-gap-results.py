# Download and install all dependencies

# gpr_pub contains the heuristics models and supplementary code

!git clone https://github.com/sattree/gpr_pub.git

!wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip

!unzip stanford-corenlp-full-2018-10-05.zip

!pip install allennlp --ignore-installed greenlet

!pip install attrdict
import pandas as pd

from sklearn.metrics import log_loss, classification_report

from attrdict import AttrDict



import spacy



from allennlp.predictors.predictor import Predictor

from allennlp.models.archival import load_archive

from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser



from gpr_pub.utils import CoreNLPServer



# gap_scorer_ext has minor fixes for py3 and to take pandas df as input instead of filepaths

from gpr_pub.gap.gap_scorer_ext import read_annotations, calculate_scores, add_to_score_view
# Heuristic models implement coref resolution based on heuristics described in the paper

# Pronoun resolution is a simple wrapper to convert coref predictions into class-specific labels

# Multi pass sieve model implements backoff mechanism

from gpr_pub.models.heuristics.random_distance import RandomModel

from gpr_pub.models.heuristics.token_distance import TokenDistanceModel

from gpr_pub.models.heuristics.syntactic_distance import StanfordSyntacticDistanceModel

from gpr_pub.models.heuristics.parallelism import AllenNLPParallelismModel as ParallelismModel

from gpr_pub.models.heuristics.url_title import StanfordURLTitleModel as URLModel



from gpr_pub.models.pronoun_resolution import PronounResolutionModel



from gpr_pub.models.multi_pass_sieve import MultiPassSieveModel
# Instantiate stanford corenlp server

STANFORD_CORENLP_PATH = 'stanford-corenlp-full-2018-10-05/'

server = CoreNLPServer(classpath=STANFORD_CORENLP_PATH,

                        corenlp_options=AttrDict({'port': 9090, 

                                                  'timeout': '600000', 

                                                  'quiet': 'true',

                                                  'preload': 'tokenize,spplit,lemma,parse,deparse'}))

server.start()

STANFORD_SERVER_URL = server.url
# Instantiate base models

STANFORD_MODEL = CoreNLPParser(url=STANFORD_SERVER_URL)

SPACY_MODEL = spacy.load('en_core_web_lg')

model_url = 'https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz'

archive = load_archive(model_url, cuda_device=0)

ALLEN_DEP_MODEL = Predictor.from_archive(archive)
# Load data

train = pd.read_csv('gpr_pub/data/gap-development.tsv', sep='\t')

# normalizing column names

train.columns = map(lambda x: x.lower().replace('-', '_'), train.columns)

with pd.option_context('display.max_rows', 10, 'display.max_colwidth', 15):

    display(train)
# Instantiate heuristic models

random_coref_model = RandomModel(SPACY_MODEL)

random_proref_model = PronounResolutionModel(random_coref_model)



token_distance_coref_model = TokenDistanceModel(SPACY_MODEL)

token_distance_proref_model = PronounResolutionModel(token_distance_coref_model)



syntactic_distance_coref_model = StanfordSyntacticDistanceModel(STANFORD_MODEL)

syntactic_distance_proref_model = PronounResolutionModel(syntactic_distance_coref_model, n_jobs=-1)



parallelism_coref_model = ParallelismModel(ALLEN_DEP_MODEL, SPACY_MODEL)

parallelism_proref_model = PronounResolutionModel(parallelism_coref_model)



url_title_coref_model = URLModel(STANFORD_MODEL)

url_title_proref_model = PronounResolutionModel(url_title_coref_model, n_jobs=-1)
# Creates sieve pipeline of heuristic models, applying each new heuristic with appropriate backoff models

# Multi pass sieve - order of models provided as input is important

#    - left to right: recall increases

#    - right to left: precision increases

preds = MultiPassSieveModel(random_proref_model).predict(train)

score_df = add_to_score_view(preds, train, None, 'Random')



preds = MultiPassSieveModel(token_distance_proref_model).predict(train)

score_df = add_to_score_view(preds, train, score_df, 'Token Distance')



preds = MultiPassSieveModel(syntactic_distance_proref_model,

                           token_distance_proref_model).predict(train)

score_df = add_to_score_view(preds, train, score_df, 'Syntactic Distance')



preds = MultiPassSieveModel(parallelism_proref_model,

                            syntactic_distance_proref_model,

                           token_distance_proref_model).predict(train)

score_df = add_to_score_view(preds, train, score_df, 'Parallelism')



preds = MultiPassSieveModel(url_title_proref_model,

                            parallelism_proref_model,

                            syntactic_distance_proref_model,

                           token_distance_proref_model).predict(train)
score_df = add_to_score_view(preds, train, score_df, 'Parallelism+URL')
y_pred = pd.DataFrame(preds, columns=['A', 'B'])

y_pred['NEITHER'] = ~y_pred['A'] & ~y_pred['B']



y_true = pd.DataFrame(train[['a_coref', 'b_coref']].values, columns=['A', 'B'])

y_true['NEITHER'] = ~y_true['A'] & ~y_true['B']



print(classification_report(y_true, y_pred, target_names=['A', 'B', 'NEITHER']))
# shift the predictions by 0.25

log_loss(y_true, y_pred + 0.25)
test_df = pd.read_csv('../input/test_stage_1.tsv', sep='\t')

# normalizing column names

test_df.columns = map(lambda x: x.lower().replace('-', '_'), test_df.columns)

preds = MultiPassSieveModel(url_title_proref_model,

                            parallelism_proref_model,

                            syntactic_distance_proref_model,

                           token_distance_proref_model).predict(test_df)

preds = pd.DataFrame(preds, columns=['A', 'B'])



sub_df = pd.read_csv('../input/sample_submission_stage_1.csv')

sub_df.loc[:, 'A'] = preds['A'] + 0.25

sub_df.loc[:, 'B'] = preds['B'] + 0.25

sub_df.loc[:, 'NEITHER'] = (~preds['A'] & ~preds['B']) + 0.25



sub_df.to_csv("submission.csv", index=False)
sub_df.head()
!rm -r stanford-corenlp-full-2018-10-05/

!rm -r gpr_pub/

!rm stanford-corenlp-full-2018-10-05.zip