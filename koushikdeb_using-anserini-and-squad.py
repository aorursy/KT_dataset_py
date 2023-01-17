!curl -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz
%%capture
!mv openjdk-11.0.2_linux-x64_bin.tar.gz /usr/lib/jvm/; cd /usr/lib/jvm/; tar -zxvf openjdk-11.0.2_linux-x64_bin.tar.gz
!update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-11.0.2/bin/java 1
!update-alternatives --set java /usr/lib/jvm/jdk-11.0.2/bin/java
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/jdk-11.0.2"
!pip install pyserini==0.8.1.0
from pyserini.search import pysearch
%%capture
!wget https://www.dropbox.com/s/uvjwgy4re2myq5s/lucene-index-covid-2020-03-20.tar.gz
# !wget https://www.dropbox.com/s/evnhj2ylo02m03f/lucene-index-covid-paragraph-2020-03-20.tar.gz
!tar xvfz lucene-index-covid-2020-03-20.tar.gz
query = 'What collaborations are happening within 2019-nCoV research community'
keywords = 'inter-sectorial, international, collaboration, global, coronavirus, novel coronavirus, sharing'
from IPython.core.display import display, HTML

searcher = pysearch.SimpleSearcher('lucene-index-covid-2020-03-20/')
hits = searcher.search(query + '. ' + keywords)
n_hits = len(hits)

display(HTML('<div style="font-family: Times New Roman; font-size: 20px; padding-bottom:12px"><b>Query</b>: '+query+'</div>'))

# Prints the first 10 hits
for i in range(0, n_hits):
  display(HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px">' + 
               F'{i+1} {hits[i].docid} ({hits[i].score:1.2f}) -- ' +
               F'{hits[i].lucene_document.get("authors")} et al. ' +
               F'{hits[i].lucene_document.get("title")}. ' +
               F'<a href="https://doi.org/{hits[i].lucene_document.get("doi")}">{hits[i].lucene_document.get("doi")}</a>.'
               + '</div>'))
import json

for i in range(0, n_hits):
    doc_json = json.loads(hits[i].raw)
    print(doc_json.keys())
    
print('number of hits: ', n_hits)
import json
available_keys ={}
for i in range(0, n_hits):
    doc_json = json.loads(hits[i].raw)
    for k in doc_json.keys():
        if k in available_keys:
            available_keys[k]+=1
        else:
            available_keys[k]=1
            
print(available_keys)
        
hit_dictionary = {}
for i in range(0, n_hits):
    doc_json = json.loads(hits[i].raw)
    idx = int(hits[i].docid)
    hit_dictionary[idx] = doc_json
    hit_dictionary[idx]['title'] = hits[i].lucene_document.get("title")
    hit_dictionary[idx]['authors'] = hits[i].lucene_document.get("authors")
    hit_dictionary[idx]['doi'] = hits[i].lucene_document.get("doi")

available_keys ={}
for idx,v in hit_dictionary.items():
    for k in v.keys():
        if k in available_keys:
            available_keys[k]+=1
        else:
            available_keys[k]=1
print(available_keys)
limit_print = True
print_current = True
for idx,v in hit_dictionary.items():
    if print_current:
        print(idx)
        print(v['abstract'])
        print()
        if limit_print:
            print_current = False
print_current=True

for idx,v in hit_dictionary.items():
    abs_dirty = v['abstract']
    # looks like the abstract value can be an empty list
    v['abstract_paragraphs'] = []
    v['abstract_full'] = ''
    
    if abs_dirty:
        # looks like if it is a list, then the only entry is a dictionary wher text is in 'text' key
        # looks like it is broken up by paragraph if it is in that form.  lets make lists for every paragraph
        # and a new entry that is full abstract text as both could be valuable for BERT derrived QA
        
        
        if isinstance(abs_dirty, list):
            for p in abs_dirty:
                v['abstract_paragraphs'].append(p['text'])
                v['abstract_full'] += p['text'] + '\n\n'
            
        # looks like in some cases the abstract can be straight up text so we can actually leave that alone
        if isinstance(abs_dirty, str):
            v['abstract_paragraphs'].append(abs_dirty)
            v['abstract_full'] += abs_dirty + '\n\n'            
            
    if print_current:
        print(idx)
        print(v['abstract_paragraphs'])
        print(v['abstract_full'])
        if limit_print: print_current = False
%%capture
!git clone https://github.com/kamalkraj/BERT-SQuAD
%%capture
!cd BERT-SQuAD; pip install -r requirements.txt
%%capture
!cd BERT-SQuAD; wget https://www.dropbox.com/s/8jnulb2l4v7ikir/model.zip
!cd BERT-SQuAD;  unzip -o model.zip;
!rm BERT-SQuAD/model.zip
import collections
import logging
import math

import numpy as np
import torch
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer)

from pytorch_transformers.tokenization_bert import (BasicTokenizer,
                                                    whitespace_tokenize)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        return s

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 paragraph_len,
                 start_position=None,
                 end_position=None,):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position

def input_to_squad_example(passage, question):
    """Convert input passage and question into a SquadExample."""

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    paragraph_text = passage
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    qas_id = 0
    question_text = question
    start_position = None
    end_position = None
    orig_answer_text = None

    example = SquadExample(
        qas_id=qas_id,
        question_text=question_text,
        doc_tokens=doc_tokens,
        orig_answer_text=orig_answer_text,
        start_position=start_position,
        end_position=end_position)
                
    return example

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def squad_examples_to_features(example, tokenizer, max_seq_length,
                                 doc_stride, max_query_length,cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)
    example_index = 0
    features = []
    # if example_index % 100 == 0:
    #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []

        # CLS token at the beginning
        if not cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)

        # Query
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(sequence_a_segment_id)

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_a_segment_id)

        # Paragraph
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                    split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(sequence_b_segment_id)
        paragraph_len = doc_span.length

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_b_segment_id)

        # CLS token at the end
        if cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token)
            input_mask.append(0 if mask_padding_with_zero else 1)
            segment_ids.append(pad_token_segment_id)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        start_position = None
        end_position = None

        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                paragraph_len=paragraph_len,
                start_position=start_position,
                end_position=end_position))
        unique_id += 1

    return features

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

RawResult = collections.namedtuple("RawResult",["unique_id", "start_logits", "end_logits"])

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def get_answer(example, features, all_results, n_best_size,
                max_answer_length, do_lower_case):
    example_index_to_features = collections.defaultdict(list)
    for feature in features:
        example_index_to_features[feature.example_index].append(feature)
    
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    
    _PrelimPrediction = collections.namedtuple( "PrelimPrediction",["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    example_index = 0
    features = example_index_to_features[example_index]

    prelim_predictions = []

    for (feature_index, feature) in enumerate(features):
        result = unique_id_to_result[feature.unique_id]
        start_indexes = _get_best_indexes(result.start_logits, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, n_best_size)
        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index]))
    prelim_predictions = sorted(prelim_predictions,key=lambda x: (x.start_logit + x.end_logit),reverse=True)
    _NbestPrediction = collections.namedtuple("NbestPrediction",
                        ["text", "start_logit", "end_logit","start_index","end_index"])
    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[pred.feature_index]
        orig_doc_start = -1
        orig_doc_end = -1
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text,do_lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ""
            seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit,
                start_index=orig_doc_start,
                end_index=orig_doc_end))

    if not nbest:
        nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0,start_index=-1,
                end_index=-1))

    assert len(nbest) >= 1

    total_scores = []
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)

    probs = _compute_softmax(total_scores)
    
    answer = {"answer" : nbest[0].text,
               "start" : nbest[0].start_index,
               "end" : nbest[0].end_index,
               "confidence" : probs[0],
               "document" : example.doc_tokens
             }
    return answer

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])



class QA:

    def __init__(self,model_path: str):
        self.max_seq_length = 384
        self.doc_stride = 128
        self.do_lower_case = True
        self.max_query_length = 64
        self.n_best_size = 20
        self.max_answer_length = 30
        self.model, self.tokenizer = self.load_model(model_path)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        self.model.eval()


    def load_model(self,model_path: str,do_lower_case=False):
        config = BertConfig.from_pretrained(model_path + "/bert_config.json")
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
        model = BertForQuestionAnswering.from_pretrained(model_path, from_tf=False, config=config)
        return model, tokenizer
    
    def predict(self,passage :str,question :str):
        example = input_to_squad_example(passage,question)
        features = squad_examples_to_features(example,self.tokenizer,self.max_seq_length,self.doc_stride,self.max_query_length)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)
        all_results = []
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]  
                        }
                example_indices = batch[3]
                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id    = unique_id,
                                    start_logits = to_list(outputs[0][i]),
                                    end_logits   = to_list(outputs[1][i]))
                all_results.append(result)
        answer = get_answer(example,features,all_results,self.n_best_size,self.max_answer_length,self.do_lower_case)
        return answer

model = QA('/kaggle/working/BERT-SQuAD/model')
document = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the state's law-making body for matters coming under state responsibility. The Victorian Constitution can be amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian people in a referendum, depending on the provision."

q = 'When was Victorias colonial constitution written?'

def makeBERTSQuADPrediction(model, document, question):
    return model.predict(document,question)

answer = makeBERTSQuADPrediction(model, document, q)

print(answer['answer'])
print(answer.keys())
from tqdm import tqdm
def searchAbstracts(hit_dictionary, model, question):
    abstractResults = {}
    for k,v in tqdm(hit_dictionary.items()):
        
        abstract = v['abstract_full']
        emptyToken = -1
        if abstract:
            ans = makeBERTSQuADPrediction(model, abstract, question)
            confidence = ans['confidence']
            abstractResults[confidence]={}
            abstractResults[confidence]['answer'] = ans['answer']
            abstractResults[confidence]['start'] = ans['start']
            abstractResults[confidence]['end'] = ans['end']
            abstractResults[confidence]['idx'] = k
        else:
            abstractResults[emptyToken]={}
            abstractResults[emptyToken]['answer'] = []
            abstractResults[emptyToken]['start'] = []
            abstractResults[emptyToken]['end'] = []
            abstractResults[emptyToken]['confidence'] = k
            emptyToken -= 1
    return abstractResults


answers = searchAbstracts(hit_dictionary, model, query)
probs = list(answers.keys())
probs.sort(reverse=True)
for p in probs:
    print(answers[p]['answer'])
def displayResults(hit_dictionary, answers, question):
    display(HTML('<div style="font-family: Times New Roman; font-size: 28px; padding-bottom:12px"><b>Query</b>: '+question+'</div>'))
    confidence = list(answers.keys())
    confidence.sort(reverse=True)
    
    display(HTML('<div style="font-family: Times New Roman; font-size: 20px; padding-bottom:12px"><b>Highlights</b>:</div>'))
    confidence = list(answers.keys())
    confidence.sort(reverse=True)
    
    HTML_list_text = '<div style="font-family: Times New Roman; font-size: 20px; padding-bottom:12px"> <ul style="list-style-type:disc;">' 
    
    for i,c in enumerate(confidence):
        if i < 3:
            idx = answers[c]['idx']
            full_abs = hit_dictionary[idx]['abstract_full']
            bert_ans = answers[c]['answer']
            split_abs = full_abs.split(bert_ans)
            sentance_beginning = split_abs[0][split_abs[0].rfind('.')+1:]
            sentance_end_pos = split_abs[1].find('. ')+1
            if sentance_end_pos == 0:
                sentance_end = split_abs[1]
            else:
                sentance_end = split_abs[1][:sentance_end_pos]
                
            sentance_full = sentance_beginning + bert_ans+ sentance_end
            if c> 0.5:
                color = 'green'
            elif c > 0.25:
                color = '#CCCC00'
            else:
                color = 'red'
            HTML_list_text += '<li>'+sentance_full +"<font color='"+color+"'> score: " + str(c) +' </font> </li>'
    HTML_list_text += '</ul> </div>'
            
    display(HTML(HTML_list_text))
            
    
    
    for c in confidence:
        if c>0:
            idx = answers[c]['idx']
            title = hit_dictionary[idx]['title']
            authors = hit_dictionary[idx]['authors']
            doi = hit_dictionary[idx]['doi']

            display(HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:12px"><b>Document</b>: '+
                 F'{authors} et al. ' +
                 F'{title}. ' + 
                 F'<a href="https://doi.org/{doi}">{doi}</a>. Confidence: ' + F'{c}' +
                 '</div>'))

            
            full_abs = hit_dictionary[idx]['abstract_full']
            bert_ans = answers[c]['answer']
            
            
            split_abs = full_abs.split(bert_ans)
            sentance_beginning = split_abs[0][split_abs[0].rfind('.')+1:]
            sentance_end_pos = split_abs[1].find('. ')+1
            if sentance_end_pos == 0:
                sentance_end = split_abs[1]
            else:
                sentance_end = split_abs[1][:sentance_end_pos]
                
            sentance_full = sentance_beginning + bert_ans+ sentance_end
            
            split_abs = full_abs.split(sentance_full)
      
            display(HTML('<div style="font-family: Times New Roman; font-size: 16px; padding-bottom:12px"><b>Abstract</b>: '
                         +split_abs[0] + " <font color='red'>"+sentance_full+"</font> "+split_abs[1]+'</div>'))
    

displayResults(hit_dictionary, answers, query)
import time
def searchDatabase(question, keywords, pysearch, lucene_database, BERTSQuAD_Model):
    ## search the lucene database with a combination of the question and the keywords
    print('Starting Lucene Database Search')
    searcher = pysearch.SimpleSearcher(lucene_database)
    hits = searcher.search(question + '. ' + keywords)
    print('Done')
    
    ## collect the relevant data in a hit dictionary
    print('Building hit dictionary')
    hit_dictionary = {}
    for i in range(0, n_hits):
        doc_json = json.loads(hits[i].raw)
        idx = int(hits[i].docid)
        hit_dictionary[idx] = doc_json
        hit_dictionary[idx]['title'] = hits[i].lucene_document.get("title")
        hit_dictionary[idx]['authors'] = hits[i].lucene_document.get("authors")
        hit_dictionary[idx]['doi'] = hits[i].lucene_document.get("doi")
        
    print('Done')
    print('Scrubbing retrieved Abstracts')
    ## scrub the abstracts in prep for BERT-SQuAD
    for idx,v in hit_dictionary.items():
        abs_dirty = v['abstract']
        # looks like the abstract value can be an empty list
        v['abstract_paragraphs'] = []
        v['abstract_full'] = ''

        if abs_dirty:
            # looks like if it is a list, then the only entry is a dictionary wher text is in 'text' key
            # looks like it is broken up by paragraph if it is in that form.  lets make lists for every paragraph
            # and a new entry that is full abstract text as both could be valuable for BERT derrived QA


            if isinstance(abs_dirty, list):
                for p in abs_dirty:
                    v['abstract_paragraphs'].append(p['text'])
                    v['abstract_full'] += p['text'] + '\n\n'

            # looks like in some cases the abstract can be straight up text so we can actually leave that alone
            if isinstance(abs_dirty, str):
                v['abstract_paragraphs'].append(abs_dirty)
                v['abstract_full'] += abs_dirty + '\n\n'
    print('Done')
    ## Search collected abstracts with BERT-SQuAD
    print('Searching and ranking results with BERT SQuAD')
    answers = searchAbstracts(hit_dictionary, BERTSQuAD_Model, question)
    print('Finished: here are the results')
    
    ## display results in a nice format
    displayResults(hit_dictionary, answers, question)

searchDatabase(query, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "Is the virus transmitted by aerisol, droplets, food, close contact, fecal matter, or water"
keywords = "2019-nCoV, COVID-19, coronavirus, novel coronavirus, person to person, human to human, interpersonal contact, air, water,fecal, surfaces, aerisol, transmission"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "How long is the incubation period for the virus"
keywords = "2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, hours, days, period"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "Can the virus be transmitted asymptomatically or during the incubation period"
keywords = "2019-nCoV, COVID-19, coronavirus, novel coronavirus, asymptomatic, person to person, human to human, transmission"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "How does weather, heat, and humidity affect the tramsmission of 2019-nCoV"
keywords = "2019-nCoV, COVID-19, coronavirus, novel coronavirus, seasonality, temperature, warm, cold, heat, dry, wet, spread, rate, climate"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "How long can the 2019-nCoV virus remain viable on common surfaces"
keywords = "2019-nCoV, COVID-19, coronavirus, novel coronavirus, persistance, copper, stainless steel, plastic, touch"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "What drugs or therapies are being investigated"
keywords = "2019-nCoV,  COVID-19, coronavirus, novel coronavirus, drug, antiviral, testing, clinical trial, study"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "Are anti-inflammatory drugs recommended"
keywords = "2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, ibuprofen, advil, NSAID, anti-inflamatory, treatment"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "What diagnostic tests (tools) exist or are being developed to detect 2019-nCoV"
keywords = "2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, diagnosis, tools, detetion"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "What is the immune system response to 2019-nCoV"
keywords = "2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, immune, immune system, response, immunity, antibodies"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "Can personal protective equipment prevent the transmission of 2019-nCoV"
keywords = "2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, ppe, masks, gloves, face shields, gown, eye protection"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "What risk factors contribute to the severity of 2019-nCoV"
keywords = "2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, susceptible, smoking, smoker, neonates, pregnant, socio-economic, behavioral, age, elderly, young, old, children"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "What is the fatality rate of 2019-nCoV"
keywords = "2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, fatality, statistics, death"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "What public health policies prevent or control the spread of 2019-nCoV"
keywords = "2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, guidance, prevention measures, public health, community, prevention, administration, government, health department, policy, control measures, travel"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "Can 2019-nCoV infect patients a second time"
keywords = "2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, reinfected, multiple infections, second time, permenant immunity"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)
question = "What telemedicine and cybercare methods are most effective"
keywords = "2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, telemedicine, 5G, cell phone, cyber, cybercare, information technolog, remote, over the phone, internet, web"
searchDatabase(question, keywords, pysearch, 'lucene-index-covid-2020-03-20/', model)