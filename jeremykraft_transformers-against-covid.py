!cp -r ../input/github-files/covid-kaggle-master/* ./
!/opt/conda/bin/python3.7 -m pip install --upgrade pip
!pip install rank_bm25
!pip install -U sentence-transformers
import os
import json
from pprint import pprint
from copy import deepcopy
import numpy as np
import pandas as pd
import nltk
import re
from time import time
from scipy.stats import spearmanr, kendalltau
from sentence_transformers import SentenceTransformer
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi, BM25Plus # don't use BM25L, there is a mistake
import os
import torch
import numpy
from tqdm import tqdm, trange
from transformers import *
from tqdm import tqdm
import warnings

import os
import json
from pprint import pprint
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import nltk
import re
from time import time
from scipy.stats import spearmanr, kendalltau
from sentence_transformers import SentenceTransformer
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import math
import string

from covid import BM25_helper_functions as BM25_hf
from covid import transformer_helper_functions as transf_hf
!cp -r ../input/papers-12gb-prepared/* ./
# load the 50.000 papers, and filter them to contain covid 19 keywords, about ~ 1900 papers remaining
data_df_red = pd.read_csv("./all_doc_df_12gb_filtered.csv")

data_df_red
BIOBERT = True
S_BERT = True
COMPUTE_PARAGRAPH_SCORE = True
QA = True
BART = True
top_res = 50


# prepare models

if BIOBERT:
    # load model
    tokenizer_biobert = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)
    config = AutoConfig.from_pretrained('monologg/biobert_v1.1_pubmed', output_hidden_states=True)
    model_biobert = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed', config=config)
    assert model_biobert.config.output_hidden_states == True
if S_BERT:
    embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
if QA:
    # QA
    # tokenizer = AutoTokenizer.from_pretrained("ahotrod/albert_xxlargev1_squad2_512")
    # model = AutoModelForQuestionAnswering.from_pretrained("ahotrod/albert_xxlargev1_squad2_512")
    tokenizer_qa = AutoTokenizer.from_pretrained("clagator/biobert_squad2_cased")
    model_qa = AutoModelForQuestionAnswering.from_pretrained("clagator/biobert_squad2_cased")
    # tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    # model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
    # tokenizer = AutoTokenizer.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")
    # model = AutoModelForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")

if BART:
    
    if torch.cuda.is_available():
      device = 'cuda'
    else:
      device = 'cpu'

    model_name= "bart-large-xsum"
    model_name = "bart-large-cnn" # Use the cnn model. Otherwise the model hallucinates too many details.
    
    model_bart = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer_bart = BartTokenizer.from_pretrained(model_name)
    model_bart.to(device)
    

flat_query = 'Are the pregnant women more at risk?'
flat_query = 'Is there a seasonality in transmission?'
flat_query = 'How does temperature and humidity affect the transmission of 2019-nCoV?'
flat_query = 'What is the longest duration of viral shedding?'
# flat_query = 'What is the fatality rate?'
quest_text = [flat_query]
# remove \n, but has no impact, this is already done somewhere else
model = BM25Okapi
indices, scores, quest = BM25_hf.search_corpus_for_question(
    quest_text, data_df_red, model, len(data_df_red), 'cleaned_text')
# remove again docs without keywords if searched with Okapi BM25
if model == BM25Okapi:
    contain_keyword = np.array(indices)[scores>0]
    answers = data_df_red.iloc[contain_keyword, :].copy()
    answers['scores_BM25'] = scores[scores>0]
else:
    answers = data_df_red.iloc[indices, :].copy()
    answers['scores_BM25'] = scores
answers = answers.sort_values(['scores_BM25'], ascending=False)
answers = answers.reset_index(drop=True)
answers
def concat_title_abstract(ans):
    """
    Compute the title with the abstract
    to enrich the abstracts.
    Parameters
    ----------
    ans: pandas dataframe
        Dataframe with the results of BM25
    Returns
    -------
    ans: pandas dataframe
        Dataframe with the added column
    """

    # concatenate title and abstract
    title_list = list(ans['title'])
    abstract_list = list(ans['abstract'])
    ind_list = list(range(len(title_list)))
    title_abstr_list = list(map(
        lambda x: clean_hf.add_title_to_abstr(x, title_list, abstract_list),
        ind_list))
    ans['title_abstr'] = title_abstr_list

ans = answers[['scores_BM25', 'title', 'abstract', 'text']].copy()
ans = transf_hf.concat_title_abstract(ans)
# clean-up
# remove rows without titles and abstracts
ans = ans[~(ans['title_abstr'].isna() | (ans['title_abstr'] == ' '))].iloc[:top_res, :].copy()
ans = ans.reset_index(drop=True)
ans
def compare_scores(ans):
    """
    Compare the scores to the BM25 scores.
    Parameters
    ----------
    ans : pandas dataframe
        Dataframe containing the BM25 score and other algorithm's score
    """

    col_list = list(ans.columns)
    # find out which column have scores
    col_has_score = list(map(lambda x: ('score' in x) and not('BM25' in x), col_list))
    col_w_score = np.array(col_list)[col_has_score]

    # compute comparisons between methods
    for col in col_w_score:
        print('Similarity between ' + col + ' and BM25:')
        print(spearmanr(ans[col], ans['scores_BM25']))
        print(kendalltau(ans[col], ans['scores_BM25']))
def extract_scibert(text, tokenizer, model):
    """
    Compute the transformer embeddings.
    If the text is too long, chunk it.
    Parameters
    ----------
    text : string
        Input text
    tokenizer : Huggingface tokenizer
        Tokenizer of the model
    model : Huggingface model
        Model
    Returns
    -------
    text_ids : tensor
        Tensor of token ids of dimension 1, quantity of tokens + 2 ( for CLS, SEP)
    text_words : list of string
        List of tokens
    state : tensor
        Tensor of output embeddings of dimension 1, quantity of tokens, hidden size
    class_state : tensor
        Tensor of the CLS embedding of dimension hidden size
    layer_concat : tensor
        Tensor of the 4 last layers concatenated of dimension 1, quantity of tokens, hidden size * 4
        ordered (-4, ..., -1)
    """

    # check that the model has been configured to output the embeddings from all layers
    assert model.config.output_hidden_states == True

    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0])[1:-1]

    n_chunks = int(numpy.ceil(float(text_ids.size(1)) / 510))
    states = []
    class_states = []
    layer_concats = []

    # chunk the text into passages of maximal length
    for ci in range(n_chunks):
        text_ids_ = text_ids[0, 1 + ci * 510:1 + (ci + 1) * 510]
        text_ids_ = torch.cat([text_ids[0, 0].unsqueeze(0), text_ids_])
        if text_ids[0, -1] != text_ids[0, -1]:
            text_ids_ = torch.cat([text_ids_, text_ids[0, -1].unsqueeze(0)])

        with torch.no_grad():
            res = model(text_ids_.unsqueeze(0))
            # stock all embeddings and the CLS embeddings
            state = res[0][:, 1:-1, :]
            class_state = res[0][:, 0, :]
            # compute the concatenation of the last 4 layers
            # initialize the embeddings
            all_embed = res[2]
            layer_concat = all_embed[-4][:, 1:-1, :]
            for i in range(3):
                # take only regular tokens
                layer_concat = torch.cat((layer_concat, all_embed[3 - i][:, 1:-1, :]), dim=2)

        states.append(state)
        class_states.append(class_state)
        layer_concats.append(layer_concat)

    # give back the results as tensors of dim (# tokens, # fetaures)
    state = torch.cat(states, axis=1)[0]
    class_state = torch.cat(class_states, axis=1)[0]
    layer_concat = torch.cat(layer_concats, axis=1)[0]

    return text_ids, text_words, state, class_state, layer_concat


def cross_match(state1, state2, use_CLS=False):
    if not use_CLS:
        sim = torch.cosine_similarity(torch.mean(state1, 0), torch.mean(state2, 0), dim=0)
    else:
        sim = torch.cosine_similarity(state1, state2, dim=0)
    sim = sim.numpy()
    return sim

if BIOBERT:

    # BERT embeddings similarity

    # consider only top results from BM25
    use_CLS = False # TO DO: solve bug when abstract is longer than 512 tokens
    use_last_four = True
    search_field = 'title_abstr'

    # process query
    query_ids, query_words, query_state, query_class_state, query_layer_concat =\
        extract_scibert(flat_query, tokenizer_biobert, model_biobert)

    # compute similarity scores
    sim_scores = []
    for text in tqdm(ans[search_field]):
        text_ids, text_words, state, class_state, layer_concat = extract_scibert(text, tokenizer_biobert, model_biobert)
        if use_CLS:
            sim_score = cross_match(query_class_state, class_state, True) # try cosine on CLS tokens
        elif use_last_four:
            sim_score = cross_match(query_layer_concat, layer_concat, False)
        else:
            sim_score = cross_match(query_state, state, False)
        sim_scores.append(sim_score)

    # Store results in the dataframe
    end_ans = ans.copy()
    orig_col = list(ans.columns)
    end_ans['score_biobert'] = np.array(sim_scores)
    # reorder columns
    end_ans = end_ans[['score_biobert'] + orig_col]
    end_ans = end_ans.sort_values(['score_biobert'], ascending=False)\
                    .reset_index(drop=True)
    ans = end_ans.copy()
    # print scores
    compare_scores(ans)

def search_w_stentence_transformer(embedder, flat_query,
                                corpus_list,
                                show_progress_bar=True, batch_size=8):
    """
    Compute the similarity scores of Sentence transformer.
    Parameters
    ----------
    embedder : Sentence Transformer model
        Model to use
    flat_query : string
        Query text
    corpus_list: list of string
        Texts to search in
    show_progress_bar : boolean
        True to show the progress bar when computing the corpus embeddings
    batch_size : int
        batch size for Sentence Transformer inference
    Returns
    -------
    s_bert_res : list
        Results
    """

    # compute embeddings
    query_embedding = embedder.encode([flat_query])
    corpus_embeddings = embedder.encode(corpus_list, batch_size= batch_size,  show_progress_bar=show_progress_bar)

    # compute similarity
    sim_scores = cosine_similarity(query_embedding[0].reshape(1, -1),
                              np.array(corpus_embeddings))[0]
    s_bert_res = sim_scores

    return s_bert_res

if S_BERT:

    # Try sentence transformers

    search_field = 'title_abstr'

    res_col_name='score_S_Bert'
    corpus_list = list(ans[search_field])
    res = search_w_stentence_transformer(embedder, flat_query,
                                        corpus_list=corpus_list,
                                       show_progress_bar=True, batch_size=8)
    orig_col = list(ans.columns)
    ans[res_col_name] = res
    # reorder columns
    ans = ans[[res_col_name] + orig_col].copy()
    ans = ans.sort_values([res_col_name], ascending=False)\
                    .reset_index(drop=True)
    # print scores
    compare_scores(ans)

def split_paragraph(text, min_words=2):
    """
    Split the paper text into paragraphs.
    Parameters
    ----------
    text: string
        paper text
    min_words: int
        remove paragraphs with quantity of words <= min words
    Returns
    -------
    split_text_clean: list of string
        list of paragraphs
    """

    split_text = text.split('\n\n')
    # remove last ''
    split_text = split_text[:-1]
    # remove trash
    split_text_clean = [t for t in split_text if len(t.split()) > min_words]

    return split_text_clean

def compute_parag_scores(index, parag_list, embedder, flat_query):
    """
    Compute the similarity scores of Sentence transformer.
    Parameters
    ----------
    index : int
        row index
    parag_list : list of list of string
        list of paragraphs / row
    embedder : Sentence Transformer model
        Model to use
    flat_query : string
        Query text
    Returns
    -------
    res : matrix of float
        Similarity scores / paragraphs / row
    """

    parag_paper = parag_list[index]
    res = search_w_stentence_transformer(embedder, flat_query,
                                        corpus_list=parag_paper,
                                       show_progress_bar=False, batch_size=8)

    return res


if COMPUTE_PARAGRAPH_SCORE:
    # compute paragraph scores

    orig_col = list(ans.columns)
    # compute paragraphs
    paragraph_mat = list(map(lambda x: split_paragraph(x), list(ans['text'])))
    # coompute scores
    res = list(map(lambda x:compute_parag_scores(x, paragraph_mat, embedder, flat_query),
             trange(len(ans))))
    max_parag_score = list(map(lambda x: np.max(x), res))

    # store results in dataframe
    res_col_name = 'score_max_parag'
    ans[res_col_name] = max_parag_score
    # reorder columns
    ans = ans[[res_col_name] + orig_col].copy()
    ans = ans.sort_values([res_col_name], ascending=False)\
                    .reset_index(drop=True)

    compare_scores(ans)


ans
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def answer_question_batch(question, answer_text, tokenizer, model, squad2=True, tau=0.5, batch_size=4):
    """
    Search the answer of a question and do the computations in batch.
    Parameters
    ----------
    question : string
        The natural language question
    answer_text : list of string
        The corpus to search in
    tokenizer : Huggingface tokenizer
        The tokenizer to use
    model : Huggingface model
        The model to use
    squad2 : boolean
        True if the model is trained on Squad2, False for Squad 1.
        In Squad 1, the model presupposes that the answer exists, and might return garbage
        if there is no answer in the text. For Squad 2, the model also predicts if there is an answer in the text
    tau : float
        For squad 2. Buffer used to decide if there is an answer.
        is_answer_found = score > cls_score + tau, see Bert paper
    batch_size : int
        the batch size
    Returns
    -------
    ans_qa_batch : Pandas dataframe
        Dataframe with the columns
    """

    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.

    answer_l = []
    score_l = []

    all_embeddings = []
    length_sorted_idx = np.argsort([len(sen) for sen in answer_text])

    # chunks
    iterator = range(0, len(answer_text), batch_size)
    iterator = tqdm(iterator, desc="Batches")

    # compute batches
    for batch_idx in iterator:
        # process per batch
        start_scores_l = []
        end_scores_l = []

        batch_start = batch_idx
        batch_end = min(batch_start + batch_size, len(answer_text))
        batch = length_sorted_idx[batch_start: batch_end]

        # compute the longest length in the batch
        # assume that it is for the last element

        # solve bug in indices for last element
        if batch_end != len(answer_text):
            longest_text = answer_text[length_sorted_idx[batch_end]]
        else:
            longest_text = answer_text[length_sorted_idx[batch_end - 1]]
        longest_seq = tokenizer.encode_plus(question,
                              answer_text[length_sorted_idx[batch_end-1]],
                              max_length=model.config.max_position_embeddings,
                              pad_to_max_length=False,
                              return_attention_masks=True)['input_ids']
        max_len = len(longest_seq)


        token_dict_batch = [tokenizer.encode_plus(question,
                                                  answer_text[text_id],
                                                  max_length = min(max_len, model.config.max_position_embeddings),
                                                  pad_to_max_length=True,
                                                  return_attention_masks=True)
                            for text_id in batch]

        input_ids = torch.tensor([token_dict_batch[i]['input_ids'] for i in range(len(token_dict_batch))])
        token_ids = torch.tensor([token_dict_batch[i]['token_type_ids'] for i in range(len(token_dict_batch))])
        attention_mask = torch.tensor([token_dict_batch[i]['attention_mask'] for i in range(len(token_dict_batch))])

        # compute scores (as before the softmax)
        start_scores, end_scores = model(input_ids,
                                         token_type_ids=token_ids,
                                         attention_mask=attention_mask)

        # save scores
        start_scores = start_scores.data.numpy()
        end_scores = end_scores.data.numpy()

        start_scores_l.append(start_scores)
        end_scores_l.append(end_scores)

        # reconstruct the answers
        for i in range(len(start_scores_l[0])):
            start_scores = start_scores_l[0][i]
            end_scores = end_scores_l[0][i]

            # get the best start and end score, with start <= end

            # compute the upper triangular matrix of Mij = start_i + end_j

            mat_score = np.tile(start_scores, (len(start_scores), 1)).transpose()
            mat_score = mat_score + np.tile(end_scores, (len(end_scores), 1))
            # take the upper triangular matrix to make sure that the end index >= start index
            mat_score = np.triu(mat_score)

            # find the indices of the maximum
            arg_max_ind = np.argmax(mat_score.flatten())
            answer_start = arg_max_ind // len(mat_score)
            answer_end = arg_max_ind % len(mat_score)
            assert np.max(mat_score) == mat_score[answer_start, answer_end]
            score = mat_score.flatten()[arg_max_ind]

            # check if answer is found (score > CLS_score + tau, see paper)
            # otherwise return no answer
            if squad2:
                # check if answer exists
                cls_score = start_scores[0] + end_scores[0]
                is_answer_found = score > cls_score + tau
                # redefine answer
                score = score if is_answer_found else cls_score
                answer_start = answer_start if is_answer_found else 0
                answer_end = answer_end if is_answer_found else 0
            # stock the answer
            answer = tokenizer.decode(
                token_dict_batch[i]['input_ids'][int(answer_start): int(answer_end)+1],
                skip_special_tokens = False,
                clean_up_tokenization_spaces = True)
            answer = answer if answer != '[CLS]' else 'No answer found.'
            answer_l.append(answer)
            score_l.append(score)


    # create dataframe results
    ans_qa_batch = pd.DataFrame(zip(score_l, answer_l, length_sorted_idx), columns=['score_qa', 'answer', 'original_idx'])
    ans_qa_batch['original_idx'] = ans_qa_batch['original_idx'].astype(int)
    ans_qa_batch = ans_qa_batch.sort_values(['score_qa'], ascending=False) \
        .reset_index(drop=True)

    return ans_qa_batch
# TRY QA

if QA:

    # search articles

    # compute answer scores

    answer_text = list(ans['title_abstr'])
    ans_qa_batch = answer_question_batch(flat_query, answer_text, tokenizer_qa, model_qa, squad2=True, tau=5, batch_size=4)

    # merge qa results
    # drop qa answer columns to do another search
    if 'score_qa' in ans.columns:
        ans = ans.drop(['score_qa', 'answer', 'original_idx'], axis=1)
    orig_col = list(ans.columns)

    ans['original_idx'] = list(range(len(ans)))
    ans = ans.merge(ans_qa_batch, on='original_idx')
    ans = ans[['score_qa', 'answer', 'original_idx'] + orig_col]
    ans = ans.sort_values(['score_qa'], ascending=False) \
        .reset_index(drop=True)

    # remove rows with no answers
    no_answer = (ans['answer'] == 'No answer found.') | (ans['answer'] == '')
    ans_clean = ans[~no_answer].copy()
    ans_clean = ans_clean.reset_index(drop=True)
    
ans_clean
import gc

if BART:

    DEFAULT_DEVICE = "cpu"


    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def generate_summaries(model, tokenizer,
        df, col_name: str = 'title_abstr', batch_size: int = 8, device: str = DEFAULT_DEVICE,
        max_length_input: int = 1024):

        """
        Summarize with batch processing.

        Parameters
        ----------
        model : Huggingface model
            The model to use
        tokenizer : Huggingface tokenizer
            The tokenizer to use
        df : pandas dataframe
            The dataframe containing the paragraph to summarize
        col_name : string
            column to summarize
        batch_size : int
            the batch size
        device : str
            the device to use for running the network
        max_length_input : int
            Maximum length of input. Longer input will be truncated.

        Returns
        -------
        df : Pandas dataframe
            Input dataframe with the added columns ['summary']
        """

        examples = df[col_name]
        summ_l = []

        max_length = 50
        min_length = 10

        
        examples = df[col_name]
        summ_l = []

        max_length = 50
        min_length = 10

        # choose the batches

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in list(examples)])

        # chunks
        iterator = range(0, len(examples), batch_size)
        iterator = tqdm(iterator, desc="Batches")

        # compute batches
        for batch_idx in iterator:
            # process per batch

            batch_start = batch_idx
            batch_end = min(batch_start + batch_size, len(examples))
            batch = length_sorted_idx[batch_start: batch_end]

            # compute the longest length in the batch
            # assume that it is for the last element

            # solve bug in indices for last element
            if batch_end != len(examples):
                longest_text = examples[length_sorted_idx[batch_end]]
            else:
                longest_text = examples[length_sorted_idx[batch_end - 1]]

            longest_seq_dct = tokenizer.batch_encode_plus([longest_text], return_tensors="pt")
            max_len = len(longest_seq_dct['input_ids'].squeeze())

            # encode th whole batch
            dct = tokenizer.batch_encode_plus(examples[batch],
                                              max_length=min(max_len, max_length_input),
                                              return_tensors="pt", pad_to_max_length=True)
            # generate batch summaries
            summaries = model.generate(
                input_ids=dct["input_ids"].to(device),
                attention_mask=dct["attention_mask"].to(device),
                num_beams=5,
                temperature=1,
                length_penalty=5.0,
                max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                min_length=min_length + 1,  # +1 from original because we start at step=1
                no_repeat_ngram_size=3,
                early_stopping=True,
                decoder_start_token_id=model.config.eos_token_id,
            )
            summ = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]

            # store the results
            summ_l.append(summ)

        # restore order

        flat_summ_l = [item for sublist in summ_l for item in sublist]

        # create dataframe results
        summary_batch = pd.DataFrame(zip(flat_summ_l, length_sorted_idx),
                                    columns=['summary', 'original_idx'])
        summary_batch['original_idx'] = summary_batch['original_idx'].astype(int)
        summary_batch = summary_batch.sort_values(['original_idx'], ascending=True) \
            .reset_index(drop=True)

        # copy results in input dataframe
        orig_col = list(df.columns)
        df['summary'] = summary_batch['summary']
        # reorder columns
        df = df[['summary'] + orig_col]

        return df

res = generate_summaries(model_bart, tokenizer_bart, df=ans_clean, batch_size = 4, device= "cuda")
for sum in res['summary']:
    print('\n' + sum)

res