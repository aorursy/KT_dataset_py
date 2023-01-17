import pandas as pd

import numpy as np

import os

from tqdm import tqdm

from pathlib import Path
# We use scispaCy for UMLS entity recognition

!pip install scispacy==0.2.4 https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
!pip install flair==0.4.5
custom_data_path = Path('/kaggle/input/covid-19-risk-classification/')
from flair.models import TextClassifier

clf = TextClassifier.load(custom_data_path / 'final-model.pt')
clf.predict('Risk is higher for men than women')[0].labels
## Prepared results

df_covid_risk_sentences = pd.read_csv(custom_data_path / 'covid_risk_sentences.tsv', sep='\t', index_col=['art_index', 'sent_index'])

df_covid_risk_sentences.head()
pd.set_option('display.max_colwidth', 0)



def get_risk_sentences(term):

    res = df_covid_risk_sentences[df_covid_risk_sentences.canonical.str.lower() == term][['cui', 'sentence', 'title', 'cord_uid']]

    return res.groupby(res.index).first()
get_risk_sentences('chronic kidney diseases')
get_risk_sentences('nurses')
get_risk_sentences('alcohol consumption')
get_risk_sentences('bacteria')
os.mkdir('Risk Factor')
import regex as re



categories = ['comorbidity', 'population', 'behaviour', 'infection']



for category in categories:

    folder = 'Risk Factor/' + category

    if not os.path.exists(folder):

        os.mkdir(folder)



    df_risk = df_covid_risk_sentences[df_covid_risk_sentences.category == category]

    values = df_risk.groupby('canonical').sentence.count().sort_values(ascending=False)



    for factor in values.index:

        result = pd.DataFrame(columns=['Date','Study','Study Link', 'Journal', 'Severe','Severe Significant','Severe Age Adjusted',

                                       'Severe OR Calculated or Extracted','Fatality','Fatality Significant','Fatality Age Adjusted',

                                       'Fatality OR Calculated or Extracted','Design','Sample','Study Population'])

        result[['Date', 'Study', 'Study Link', 'Journal']] = df_risk[df_risk.canonical == factor].reset_index()[['publish_time', 'title', 'url', 'journal']].drop_duplicates()



        result.to_csv(folder + '/' + re.sub('\W', '_', factor) + '.csv')
import ipywidgets as widgets

from IPython.display import display, Markdown, clear_output



def get_risk_factor_widget(df_risk):

    buttons = []



    output = widgets.Output(layout=widgets.Layout(width='50%'))



    values = df_risk.groupby('canonical').sentence.count().sort_values(ascending=False)

    

    for i in values.index:

        b = widgets.Button(

            description='(' + str(values.loc[i]) + ') ' + i,

            disabled=False,

            tooltip='Click me',

        )

        b.item = i



        def on_button_clicked(b):

            vals = df_risk.loc[df_risk.canonical == b.item]

            with output:

                clear_output()

                display(Markdown('# ' + b.item))

                for i in vals.index.drop_duplicates().values:

                    #display(vals)

                    display(Markdown('** Article: ** ' + vals.loc[[i]].iloc[0].title))

                    display(Markdown('** Sentence: ** ' + vals.loc[[i]].iloc[0].sentence))

                    display(Markdown('** Terms: ** ' + ', '.join(vals.loc[[i]].term)))

                    display(Markdown('---'))



        b.on_click(on_button_clicked)

        buttons.append(b)



    return widgets.HBox(

        [widgets.Box(buttons, layout=widgets.Layout(width='50%',display='inline-flex',flex_flow='row wrap')), output],

    )
get_risk_factor_widget(df_covid_risk_sentences[df_covid_risk_sentences.category == 'population'])
get_risk_factor_widget(df_covid_risk_sentences[df_covid_risk_sentences.category == 'comorbidity'])
get_risk_factor_widget(df_covid_risk_sentences[df_covid_risk_sentences.category == 'behaviour'])
get_risk_factor_widget(df_covid_risk_sentences[df_covid_risk_sentences.category == 'infection'])
%%time



import en_core_sci_sm

import scispacy

from scispacy.umls_linking import UmlsEntityLinker

from scispacy.abbreviation import AbbreviationDetector

from scispacy.candidate_generation import CandidateGenerator



nlp = en_core_sci_sm.load()



abbreviation_pipe = AbbreviationDetector(nlp)

nlp.add_pipe(abbreviation_pipe)



candidate_generator = CandidateGenerator()



linker = UmlsEntityLinker(resolve_abbreviations=True, candidate_generator=candidate_generator)



nlp.add_pipe(linker)
# Read sentences that do not contain a risk factor at all

df_sentence_no_risk = pd.read_csv(custom_data_path / 'labelled_no_risk.tsv', sep='\t', index_col=0)

df_sentence_no_risk['label'] = 0

# Read sentences with candidate risk factor terms

df_dev_sentence = pd.read_csv(custom_data_path / 'labelled_prepared.tsv', sep='\t', index_col=0)

df_dev_sentence['label'] = df_dev_sentence['label'] == 'risk_factor' 

df_dev_sentence = pd.concat([df_dev_sentence, df_sentence_no_risk], sort=False)
df_dev_sentence.head()
term_attrs = ['term', 'cui', 'tui', 'type_name', 'category']



def flatten_sentence_df(sent_df):

    def reduce(item):

        if item.name == 'label':

            return any(list(item))

        if item.name in term_attrs:

            return list(item)

        return item.iloc[0]



    cols = ['tui', 'term', 'cui', 'type_name', 'category', 'sentence']

    if 'label' in sent_df.columns:

        cols += ['label']

    return sent_df.groupby(['art_index', 'sent_index'])[cols].aggregate(reduce)
df_dev = flatten_sentence_df(df_dev_sentence)

Y_dev = df_dev.label.values.astype(int)
# Fraction of positive instances

Y_dev.shape[0], Y_dev.sum(), Y_dev.sum() / Y_dev.shape[0]
def predict(clf, sentences):

    return clf.predict(sentences, embedding_storage_mode='none')



def apply_threshold(pred_clf, pos_threshold=0.5):

    def get_pos_prob(p):

        return p.labels[0].score if p.labels[0].value == '1' else (1 - p.labels[0].score)

        

    return np.array([(1 if get_pos_prob(p) >= pos_threshold else 0) for p in pred_clf])
%%time

dev_pred_sent = predict(clf, list(df_dev.sentence.values))

dev_pred = apply_threshold(dev_pred_sent, 0.5)

dev_pred.shape
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score



def print_metrics(y_true, y_pred):

    print('Precision %.2f' % precision_score(y_true, y_pred))

    print('Recall %.2f' % recall_score(y_true, y_pred))

    print('F1-Score %.2f' % f1_score(y_true, y_pred))

    print(confusion_matrix(y_true, y_pred)) 
print_metrics(Y_dev, dev_pred)
# UMLS TUIs we care about

root_tuis = dict(

    comorbidity = ['T046', 'T184'],

    population = ['T096', 'T032'],

    behaviour = ['T053'],

    infection = ['T004', 'T005', 'T007']

)
# Get UMLS subtypes as well

def expand_tree(semantic_type):

    tuis = [semantic_type]

    for c in semantic_type.children:

        tuis += expand_tree(c)

    return tuis



tui_mapping = {}

for k, v in root_tuis.items():

    for seed_tui in v:

        seed_type = linker.umls.semantic_type_tree.type_id_to_node[seed_tui]

        for semantic_type in expand_tree(seed_type):

            tui = semantic_type.type_id

            if tui in tui_mapping:

                print('Error', tui, seed_tui)

            tui_mapping[tui] = k, semantic_type
def get_umls_match(spacy_entity, conf_threshold=0.8):

    for umls_ent in spacy_entity._.umls_ents:

        if umls_ent[1] < conf_threshold:

            continue

        umls_match = linker.umls.cui_to_entity[umls_ent[0]]

        for tui in umls_match.types:

            if tui in tui_mapping:

                return umls_match, tui 

    return None, None
def get_sentence_term_df(df):

    candidates = []



    for i, article in tqdm(df.iterrows(), total=df.shape[0]):

        doc = nlp(article.abstract)

        for sid, s in enumerate(doc.sents):

            for ent in s.ents:

                umls_match, tui = get_umls_match(ent)

                if umls_match:

                    candidates.append({

                        'art_index' : i,

                        'sent_index' : sid,

                        'term' : ent.text,

                        'cui' : umls_match.concept_id,

                        'canonical': umls_match.canonical_name,

                        'span' : (ent.start, ent.end),

                        'tui' : tui,

                        'category' : tui_mapping[tui][0],

                        'type_name' : tui_mapping[tui][1].full_name,

                        'sentence' : s.text,

                        #'spacy_entity' : ent,

                        #'spacy_sentence' : s

                    })



    return pd.merge(pd.DataFrame(candidates), df, left_on='art_index', right_index=True)
data_path = Path('/kaggle/input/CORD-19-research-challenge')
df = pd.read_csv(os.path.join(data_path / 'metadata.csv'))
%run /kaggle/usr/lib/covid19_tools/covid19_tools.py
df, _ = count_and_tag(df, COVID19_SYNONYMS, 'disease_covid19')

novel_corona_filter = (abstract_title_filter(df, 'novel corona') &

                       df.publish_time.str.startswith('2020', na=False))

print(f'novel corona (published 2020): {sum(novel_corona_filter)}')

df.loc[novel_corona_filter, 'tag_disease_covid19'] = True
df_covid_abstract = df[~df.abstract.isna() & df.tag_disease_covid19]
%%time

df_covid_sentences = get_sentence_term_df(df_covid_abstract)
df_covid_sentence_index = df_covid_sentences.groupby(['art_index', 'sent_index']).sentence.first()

covid_sentences = list(df_covid_sentence_index.values)

len(covid_sentences)
%%time

sent_pred = predict(clf, covid_sentences)

labels = apply_threshold(sent_pred, 0.5)
# This will give you the filtered DataFrame we presented earlier

df_covid_risk_sentences = df_covid_sentences.set_index(['art_index', 'sent_index']).loc[df_covid_sentence_index.iloc[labels == 1].index]

df_covid_risk_sentences.to_csv('covid_risk_sentences.tsv', sep='\t')
labels.sum(), labels.shape[0], labels.sum() / labels.shape[0]
!pip install snorkel==0.9.5
sars_synonyms = [r'\bsars\b',

                 'severe acute respiratory syndrome']

mers_synonyms = [r'\bmers\b',

                 'middle east respiratory syndrome']

ards_synonyms = ['acute respiratory distress syndrome',

                 r'\bards\b']
df, _ = count_and_tag(df, sars_synonyms + mers_synonyms + ards_synonyms,

                      'disease_corona_general')
# Construct all large, unlabelled training set with articles about Covid-19, SARS and MERS

df_train_articles = df[(df.tag_disease_covid19 | df.tag_disease_corona_general) & ~df.abstract.isna() & ~df.cord_uid.isin(df_dev_sentence['cord_uid'].unique())]

df_train_articles.shape
%%time

df_train_sentence = get_sentence_term_df(df_train_articles)
df_train = flatten_sentence_df(df_train_sentence)

df_train.shape
del df_dev_sentence

del df_train_articles

del df_train_sentence

del df_covid_abstract



del abbreviation_pipe

del linker

del nlp

del candidate_generator



import gc; gc.collect()
from snorkel.labeling import labeling_function



POSITIVE = 1

NEGATIVE = 0

ABSTAIN = -1
risk_factor_synonyms = ['risk factor',

                        'risk model',

                        'risk by',

                        'comorbidity',

                        'comorbidities',

                        'coexisting condition',

                        'co existing condition',

                        'clinical characteristics',

                        'clinical features',

                        'demographic characteristics',

                        'demographic features',

                        'behavioural characteristics',

                        'behavioural features',

                        'behavioral characteristics',

                        'behavioral features',

                        'predictive model',

                        'prediction model',

                        'univariate', # implies analysis of risk factors

                        'multivariate', # implies analysis of risk factors

                        'multivariable',

                        'univariable',

                        'odds ratio', # typically mentioned in model report

                        'confidence interval', # typically mentioned in model report

                        'logistic regression',

                        'regression model',

                        'factors predict',

                        'factors which predict',

                        'factors that predict',

                        'factors associated with',

                        'underlying disease',

                        'underlying condition']
import regex as re



def zip_terms(item):

    for _, row in pd.DataFrame(np.array([item[t] for t in term_attrs]).T, columns=term_attrs).iterrows():

        yield row



def intersects(item_terms, terms):

    return len(set(item_terms) & set(terms)) > 0



def find_in_sentence(sent, search_terms):

    return [match for match in [re.search(term, sent) for term in search_terms] if match is not None]



def find_matches(x, search_terms):

    return find_in_sentence(x.sentence.lower(), search_terms)



def count(x, search_terms):

    return len(find_matches(x, search_terms))



def contains(x, search_terms):

    return any(find_matches(x, search_terms))
RISK_TERMS = ['risk', 'susceptib', 'likel', 'higher', 'incr']



GENERAL_PEOPLE_TERMS = [

    'C0027361','C0027567', 'C1257890', 'C0687744', 'C0599755', 'C0337611', 'C2700280', 'C0679646']

    

DIABETES = ['C0011860', 'C0011849', 'C0011847']

HYPERTENSION = ['C0020538']

CARDIOVASCULAR = ['C0007222', 'C0497243', 'C0034072', 'C0010068']

CANCER = ['C0006826', 'C0877578']

HEALTH_PROBLEMS = ['C2963170']

HOMELESSNESS = ['C0150041']



# Population terms that are usually false positives

POPULATION_FP = ['nation', 'passenger', 'peer', 

                 'patient', 'differences', 'adult', 

                 'animal', 'species', 'human',

                 'response', 'size', 'discharged', 'capabilit',

                 'characteristics', 'heterogeneity', 'temperature', 'months', 

                 'abilities', 'clarity', 'viral', 'clinical', 'patients', 

                 'findings', 'HR',

                 'problem', 'medication', 

                 'diagnosis', 'background',  'susceptib',

                 'city', 'report', 'market', 'infection', 'capacit', 'interpretation']



MEN = 'C0025266'

WOMEN = 'C0043210'
@labeling_function()

def lf_no_population(x):

    if 'population' in x.category:

        return ABSTAIN

    return NEGATIVE



@labeling_function()

def lf_population_age_synonyms(x):

    if contains(x, AGE_SYNONYMS):

        return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_population_sex_synonyms(x):

    if count(x, SEX_SYNONYMS) >= 2 or intersects(x.cui, [MEN, WOMEN]):

        return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_population_terms(x):

    for term in zip_terms(x):

        if term.tui in ['T100', 'T032', 'T201']:

            if (term.cui not in GENERAL_PEOPLE_TERMS) and not(len(find_in_sentence(term.term.lower(), POPULATION_FP)) > 0):

                return POSITIVE

    return ABSTAIN

    

@labeling_function()

def lf_known_comorbities(x):

    for t in DIABETES + HYPERTENSION + CARDIOVASCULAR + CANCER + HEALTH_PROBLEMS + HOMELESSNESS:

        if t in x.cui:

            return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_known_comorbities_synonmys(x):

    if contains(x, DIABETES_SYNONYMS + HYPERTENSION_SYNONYMS + 

            CANCER_SYNONYMS + IMMUNITY_SYNONYMS + IMMUNODEFICIENCY_SYNONYMS + 

               ['underlying disease', 'cardiac injury', 'chronic diseases', 'homelessness', 'mental health (care|problems)']):

        return POSITIVE

    return ABSTAIN



OUTCOME_TERMS = ['died', 'death', 'mortality']



def is_not_disease_fp(t):

    return len(find_in_sentence(t.term, COVID19_SYNONYMS + 

                                ['infect', 'virus', 'death', 'outcome',

                                 'diseas', 'epidemic', 

                                 'ARDS', 'acute respiratory distress syndrome',

                                 'pneum', 'fever'])) == 0



@labeling_function()

def lf_disease_1(x):

    if not contains(x, ['\d%', 'CI']):

        return ABSTAIN

    if len([t for t in zip_terms(x) if t.tui == 'T047' and is_not_disease_fp(t)]) > 1:

        return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_disease_0(x):

    if not contains(x, ['\d%', 'CI']):

        return ABSTAIN

    if len([t for t in zip_terms(x) if t.tui == 'T047' and is_not_disease_fp(t)]) > 0:

        return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_pregnancies(x):

    if contains(x, ['mothers', 'natal', 'born']):

        return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_first_sentence(x):

    if x.name[1] == 0:

        return NEGATIVE

    return ABSTAIN



@labeling_function()

def lf_smoking_synonyms(x):

    if contains(x, SMOKING_SYNONYMS):

        return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_bodyweigth_synonyms(x):

    if contains(x, BODYWEIGHT_SYNONYMS):

        return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_old(x):

    if contains(x, [r'\sold', 'elderly']):

        return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_workers(x):

    if 'T097' in x.tui and 'T047' in x.tui and any([t in x.sentence for t in OUTCOME_TERMS + risk_factor_synonyms + RISK_TERMS]):

        return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_short_sentences(x):

    if len(x.sentence.split(' ')) < 6:

        return NEGATIVE

    return ABSTAIN



@labeling_function()

def lf_hiv(x):

    if 'HIV' in x.sentence:

        return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_symptoms(x):

    if contains(x, ['symptom']):

        return NEGATIVE

    return ABSTAIN



@labeling_function()

def lf_structure(x):

    if contains(x, ['abstract', 'funding', 'design', 'objective', 'methods', 'importance', 'background', 'methods', 'collected']):

        return NEGATIVE

    return ABSTAIN



@labeling_function()

def lf_ethnicity(x):

    if contains(x, ['asian', 'white', 'ethnic']):

        return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_risk_specific(x):

    if contains(x, ['more patients', 

                    '(associated|correlated) with higher',

                    '(increased|higher) (morbidity|risk)',

                    'more likely to',

                    'worse outcome',

                    'associated with \w+ outcomes',

                    'patients who died',

                    'severity of COVID-19',

                    'who (have become|were) infected',

                    r'most of the \w+ patients',

                    'prone to',

                    'predisposing factor',

                    'association between',

                    'contributes? to susceptibility'

                    ]):

        return POSITIVE

    return ABSTAIN



@labeling_function()

def lf_risk_unspecific(x):

    if contains(x, risk_factor_synonyms):

        return POSITIVE

    return ABSTAIN
from snorkel.labeling import PandasLFApplier



in_lfs = [

    lf_risk_specific,

    lf_risk_unspecific,

    lf_population_terms,

    lf_population_age_synonyms,

    lf_population_sex_synonyms,

    lf_disease_0,

    lf_disease_1,

    lf_known_comorbities,

    lf_known_comorbities_synonmys,

    lf_workers,

    lf_old,

    lf_pregnancies,

    lf_hiv,

    lf_smoking_synonyms,

    lf_bodyweigth_synonyms,

    lf_ethnicity,

    lf_structure,

    lf_no_population,

    lf_short_sentences,

    lf_symptoms,

]



# Catch all labelling function - we assume all completely unlabelled datapoints are negatives

def lf_all(in_lf):

    @labeling_function()

    def lf_combine(x):

        for l in in_lf:

            res = l(x)

            if res == POSITIVE:

                return ABSTAIN

        return NEGATIVE

    return lf_combine



lfs = in_lfs + [lf_all(in_lfs)]



applier = PandasLFApplier(lfs)
L_dev = applier.apply(df_dev)
from snorkel.labeling import LFAnalysis

LFAnalysis(L_dev, lfs).lf_summary(Y_dev).sort_values(['Emp. Acc.', 'Coverage'], ascending=False)
%%time

L_train = applier.apply(df_train)
LFAnalysis(L_train, lfs).lf_summary().sort_values('Coverage', ascending=False)
from snorkel.labeling.model import LabelModel



label_model = LabelModel(cardinality=2, verbose=True)

label_model.fit(L_train, Y_dev, n_epochs=5000, log_freq=500, seed=12345)
from snorkel.analysis import metric_score

from snorkel.utils import probs_to_preds



probs_dev = label_model.predict_proba(L_dev)

preds_dev = probs_to_preds(probs_dev)



for m in ['precision', 'recall', 'f1', 'roc_auc']:

    print(

        f"Label model  {m} score: {metric_score(Y_dev, preds_dev, probs=probs_dev, metric=m)}"

    )
probs_train = label_model.predict_proba(L_train)

Y_train = probs_to_preds(probs_train)
ml_data = 'ml_data'



if not os.path.exists(ml_data):

    os.mkdir(ml_data)
def write_ml_data(df, labels, probs, fname):

    pd.DataFrame(np.vstack([df['sentence'].values, labels, probs]).T).to_csv(os.path.join(ml_data, fname), index=False, sep='\t', header=None)
write_ml_data(df_train, Y_train, probs_train[:,1].ravel(), 'train.csv')

write_ml_data(df_dev, Y_dev, Y_dev, 'dev.csv')
soft_label_map = {0 : 'text', 1 : '_', 2 : 'label'}

#hard_label_map = {0 : 'text', 1 : 'label', 2 : '_'}
from flair.datasets import CSVClassificationCorpus



ml_data = 'ml_data'



corpus = CSVClassificationCorpus(

    data_folder=ml_data, 

    column_name_map=soft_label_map, 

    train_file='train.csv', 

    dev_file='dev.csv',

    test_file='dev.csv',

    skip_header=True,

    in_memory=True,

    delimiter='\t',

)
from flair.data import Dictionary



label_dict = Dictionary(False)

label_dict.add_item('0')

label_dict.add_item('1')
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings

from flair.trainers import ModelTrainer

from flair.models import TextClassifier
import torch

import flair

from typing import List, Union, Callable, Dict

from flair.data import Dictionary, Sentence, Label, Token, space_tokenizer



from torch.utils.data import DataLoader

from flair.training_utils import (

    convert_labels_to_one_hot,

    Metric,

    Result,

    store_embeddings,

)





class SoftTextClassifier(TextClassifier):

    

    def __init__(self, soft_loss, *args, **kwargs):

        self.soft_loss = soft_loss

        super().__init__(*args, **kwargs)

    

    def evaluate(

        self,

        data_loader: DataLoader,

        out_path: Path = None,

        embedding_storage_mode: str = "none",

    ) -> (Result, float):



        with torch.no_grad():

            eval_loss = 0



            metric = Metric("Evaluation", beta=self.beta)



            lines: List[str] = []

            batch_count: int = 0

            for batch in data_loader:



                batch_count += 1



                labels, loss = self.forward_labels_and_loss(batch)



                eval_loss += loss



                sentences_for_batch = [sent.to_plain_string() for sent in batch]

                confidences_for_batch = [

                    [label.score for label in sent_labels] for sent_labels in labels

                ]

                predictions_for_batch = [

                    [label.value for label in sent_labels] for sent_labels in labels

                ]

                true_values_for_batch = [

                    sentence.get_label_names() for sentence in batch

                ]

                available_labels = self.label_dictionary.get_items()



                for sentence, confidence, prediction, true_value in zip(

                    sentences_for_batch,

                    confidences_for_batch,

                    predictions_for_batch,

                    true_values_for_batch,

                ):

                    eval_line = "{}\t{}\t{}\t{}\n".format(

                        sentence, true_value, prediction, confidence

                    )

                    lines.append(eval_line)



                for predictions_for_sentence, true_values_for_sentence in zip(

                    predictions_for_batch, true_values_for_batch

                ):



                    y_pred = predictions_for_sentence[0]

                    y_true = true_values_for_sentence[0]

                    

                    if y_true == '1':

                        if y_pred == y_true:

                            metric.add_tp('1')

                        else:

                            metric.add_fn('1')

                    else:

                        if y_pred == y_true:

                            metric.add_tn('1')

                        else:

                            metric.add_fp('1')

                            

                    #import pdb; pdb.set_trace()

                            

                store_embeddings(batch, embedding_storage_mode)



            eval_loss /= batch_count



            detailed_result = (

                f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"

                f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"

            )

            for class_name in metric.get_classes():

                detailed_result += (

                    f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "

                    f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "

                    f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "

                    f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "

                    f"{metric.f_score(class_name):.4f}"

                )



            result = Result(

                main_score=metric.micro_avg_f_score(),

                log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",

                log_header="PRECISION\tRECALL\tF1",

                detailed_results=detailed_result,

            )



            if out_path is not None:

                with open(out_path, "w", encoding="utf-8") as outfile:

                    outfile.write("".join(lines))



            return result, eval_loss

    

    def custom_cross_entropy(self, i, target, size_average=True):

        logsoftmax = torch.nn.LogSoftmax(dim=1)

        if size_average:

            return torch.mean(torch.sum(-target * logsoftmax(i), dim=1))

        else:

            return torch.sum(torch.sum(-target * logsoftmax(i), dim=1))



        def __init__(self, *args, **kwargs):

            super().__init__(*args, **kwargs)

    

    def _calculate_loss(self, scores, data_points):

        if not self.soft_loss:

            return super()._calculate_loss(scores, data_points)

        

        def get_prob_label(prob_str):

            prob = float(prob_str)

            return [1 - prob, prob]

        

        labels = torch.FloatTensor([get_prob_label(s.get_label_names()[0]) for s in data_points]).to(flair.device)

        return self.custom_cross_entropy(scores, labels, size_average=True)
word_embeddings = [

                    WordEmbeddings('glove'),

                    FlairEmbeddings('pubmed-forward'),

                    FlairEmbeddings('pubmed-backward'),

                   ]



document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=128)
classifier = SoftTextClassifier(soft_loss=True, 

                                document_embeddings=document_embeddings, 

                                label_dictionary=label_dict)



trainer = ModelTrainer(classifier, corpus)
#trainer.train('tagger',

#              learning_rate=1e-01,

#              mini_batch_size=32,

#              anneal_factor=0.5,

#              patience=10,

#              max_epochs=10,

#              embeddings_storage_mode='none')