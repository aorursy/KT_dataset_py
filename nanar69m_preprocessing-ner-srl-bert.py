!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz  #scispacy model
import pandas as pd

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import spacy

import scispacy

from scispacy.abbreviation import AbbreviationDetector

from spacy import displacy

from tqdm.notebook import tqdm

from transformers import AutoTokenizer, AutoModel

import torch

from allennlp.predictors.predictor import Predictor

import pyLDAvis
metadata = (

    pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")

    .assign(abstract=lambda df_: df_.abstract.replace(np.nan, '', regex=True))

    .assign(publish_time=lambda df_: pd.to_datetime(df_.publish_time, errors='coerce'))

)
MIN_YEAR = 2020

MIN_LENGTH = 50

new_metadata = metadata[(metadata.publish_time.dt.year >= MIN_YEAR) & (metadata.abstract.str.len() > MIN_LENGTH)]

len(new_metadata)
new_metadata.head()
# spacy.prefer_gpu()  # causes exception with allennlp srl predictor using GPU

nlp = spacy.load("en")

scinlp = spacy.load("en_core_sci_sm")

abbreviation_pipe = AbbreviationDetector(scinlp)

scinlp.add_pipe(abbreviation_pipe)
new_metadata['abstract_spacy'] = list(tqdm(nlp.pipe(new_metadata.abstract)))
new_metadata['abstract_scispacy'] = [scinlp(abstract) for abstract in tqdm(new_metadata.abstract)]
displacy.render(list(new_metadata.abstract_spacy.iloc[0].sents)[0], style="ent")
displacy.render(list(new_metadata.abstract_scispacy.iloc[0].sents)[0], style="ent")
# use scispacy as seems more accurate than spacy given this dataset

new_metadata['sentences'] = [[sent.text for sent in doc.sents] for doc in tqdm(new_metadata.abstract_scispacy)]
(new_metadata.sentences

 .groupby(new_metadata.sentences.str.len())

 .count()

 .plot(kind="bar", figsize=(20,5), title="number of sentences in abstract")

)
# BERT_MODEL = 'allenai/scibert_scivocab_uncased'

BERT_MODEL = "bert-base-uncased"

MAX_SEQ_LENGTH = 32

BERT_BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

POOLING = "cls"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

bert_model = AutoModel.from_pretrained(BERT_MODEL)

bert_model.eval()

bert_model = bert_model.to(device)

bert_dim = bert_model.config.to_dict()['hidden_size']
res = bert_model(tokenizer.encode("this is the text", max_length=MAX_SEQ_LENGTH, pad_to_max_length=True, return_tensors='pt').to(device))

res[0].shape, res[1].shape
def bertify(texts, model, tokenizer, batch_size, max_seq_length=None, device="cuda", pooling="avg"):

    input_ids = []

    for text in tqdm(texts, desc='tokenize'):

        input_ids.append(

            tokenizer.encode(text, max_length=max_seq_length, pad_to_max_length=True, return_tensors='pt')

        )  # return_tensors='pt' sets return type of shape [1, max_seq_length]

    # input_ids = torch.stack([torch.tensor(x) for x in input_ids])

    distilbert_embs = []

    with torch.no_grad():

        for i in tqdm(range(0, len(input_ids), batch_size), desc='embed'):

            # out: [batch_size, max_seq_length, distilbert_dim]

            batch_inputs = torch.cat(input_ids[i : i + batch_size])

            last_hidden, pooler = model(batch_inputs.to(device))

            if pooling == "avg":

                # out: [batch_size, distilbert_embs]

                # https://huggingface.co/transformers/model_doc/bert.html

                # [...] However, averaging over the sequence may yield better results than using the [CLS] token.

                out = torch.mean(last_hidden, dim=1)  # average word embeddings

            else:

                out = pooler  # [CLS] token

            out = out.cpu().detach()

            distilbert_embs.extend(out)

    return distilbert_embs
sentences = [sent for sents in new_metadata.sentences for sent in sents]

bert_sentences = bertify(sentences, bert_model, tokenizer, BERT_BATCH_SIZE, MAX_SEQ_LENGTH, device, POOLING)

bert_sentences = torch.stack(bert_sentences)

bert_sentences.shape
offset = 0

_bert_sentences = []

for num_sents in new_metadata.sentences.str.len():

    _bert_sentences.append(bert_sentences[offset: offset+num_sents])

    offset += num_sents

del bert_sentences  # limit OOM issues

new_metadata['abstract_sents_bert'] = _bert_sentences

del _bert_sentences
has_covid = lambda sent: any(_ in sent.lower() for _ in ['sars', 'covid-19'])

has_vaccine = lambda sent: any(_ in sent.lower() for _ in ['vaccin', 'treatment', 'drug'])

for idx, row in new_metadata.iterrows():

    abstract = row.abstract_scispacy 

    if has_covid(abstract.text) and has_vaccine(abstract.text):

        print(row.title)

        displacy.render(abstract, style="ent", options={'compact':True})

        break
text = "findings treatment virus"

text_distilbert = bertify([text], bert_model, tokenizer, BERT_BATCH_SIZE, MAX_SEQ_LENGTH, device, POOLING)

text_distilbert = torch.stack(text_distilbert)

sims = cosine_similarity(torch.cat(new_metadata.abstract_sents_bert.tolist()), text_distilbert)[:,0]

indices = np.argsort(sims)[::-1]

idx = 0

top_print = 10

for idx in indices:

    if top_print and len(sentences[idx]) >= 100:

        print(f"{sims[idx]:.3f} {idx:5d} {sentences[idx]}")

        top_print -= 1        
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz", cuda_device=0 if torch.cuda.is_available() else -1)

predictor.predict(sentence="this is a test and this is yet another one")
srl_sents = []

srl_docs = []

for doc in tqdm(new_metadata.abstract_scispacy):

    srl_doc = []

    #for sent in doc.sents:

    #    sent_srl_preds = predictor.predict(sentence=sent.text)

    #    srl_sents.append(sent_srl_preds)

    #    srl_doc.append(sent_srl_preds)

    srl_doc = predictor.predict_batch_json([{"sentence": sent.text} for sent in doc.sents])

    srl_sents.extend(srl_doc)

    srl_docs.append(srl_doc)
new_metadata['srl'] = srl_docs
verbs = []

for srl_pred in srl_sents:

    for _ in srl_pred['verbs']:

        verbs.append(_['verb'])
from collections import Counter

Counter(verbs).most_common(10)
del predictor
del nlp

del scinlp
def replace_abbr_with_json(spacy_doc): 

    new_abbrs = []

    for short in spacy_doc._.abbreviations:

        if type(short) == dict:  # already cast

            return

        short_text = short.text 

        short_start = short.start 

        short_end = short.end 

        long = short._.long_form 

        long_text = long.text 

        long_start = long.start 

        long_end = long.end 

        serializable_abbr = {"short_text": short_text, "short_start": short_start, "short_end": short_end, "long_text": long_text, "long_start": long_start, "long_end": long_end} 

        short._.long_form = None 

        new_abbrs.append(serializable_abbr) 

    spacy_doc._.abbreviations = new_abbrs

    

# cast otherwise pickling spacy docs won't work with Pandas, so save in different object

# https://github.com/allenai/scispacy/issues/205

for doc in new_metadata.abstract_scispacy:

    if doc is not None:

        replace_abbr_with_json(doc)
(new_metadata

# .drop('abstract_scispacy', axis=1)

 .drop('abstract_sents_bert', axis=1)

 .to_pickle("metadata_2020.pkl"))