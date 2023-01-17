import numpy as np
import pandas as pd
import glob
import os
import json
import string
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords as nltkstopwords
from nltk.tokenize import word_tokenize
import re
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric, remove_stopwords, strip_multiple_whitespaces
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, AutoTokenizer, BertForQuestionAnswering, AutoModelForQuestionAnswering
base_path = "../input/CORD-19-research-challenge"

biorxiv_medrxiv = "biorxiv_medrxiv/biorxiv_medrxiv/pdf_json"
comm_use_subset = "comm_use_subset/comm_use_subset/pdf_json"
noncomm_use_subset = "noncomm_use_subset/noncomm_use_subset/pdf_json"
custom_license = "custom_license/custom_license/pdf_json"
questions = {
    "questions": [
        "risk factors covid 19",
        "risk factors corona",
        "What risk factors contribute to the severity of 2019-nCoV?",
        "What do we know about COVID-19 risk factors?",
    ]
}
class Document:
    """
    Helper class to hold information about a document
    """
    def __init__(self, title, file_path, body_text, abstract):
        self.title = title
        self.file_path = file_path
        self.body_text = body_text
        self.abstract = abstract
%%time
files = []
file_names = []
documents = []
def read_files(directory):
    files_in_dir = [f for f in glob.glob(os.path.join(base_path, directory) + "/*.json", recursive=True)]
    files.extend(files_in_dir)
    for file_path in files_in_dir:
        data = json.load(open(file_path))
        body_text = ""
        for i in range(len(data["body_text"])):
            body_text += " " + data["body_text"][i]["text"]
        
        body_text = re.sub(' +', ' ', body_text)
        abstract_text = ""
        for i in range(len(data["abstract"])):
            abstract_text += " " + data["abstract"][i]["text"]
        title = data["metadata"]["title"]
        documents.append(Document(title, file_path, body_text, abstract_text))
    return len(files_in_dir)

print("Number of biorxiv_medrxiv documents: {}".format(read_files(biorxiv_medrxiv)))
print("Number of comm_use_subset documents: {}".format(read_files(comm_use_subset)))
print("Number of noncomm_use_subset documents: {}".format(read_files(noncomm_use_subset)))
print("Number of custom license documents: {}".format(read_files(custom_license)))
print("Total number of documents: {}".format(len(documents)))
%%time
# filter duplicates where text is exactly the same
counter = 0

abstract_lengths = {}
text_lengths = {}
duplicates = []
for i in range(len(documents)):
    text_a = documents[i].body_text
    for j in range(i+1, len(documents)):
        text_b = documents[j].body_text
        if text_a == text_b:
            duplicates.append(j)
print("Found {} duplicates".format(len(duplicates)))
filtered_documents = [doc for idx, doc in enumerate(documents) if idx not in duplicates]
# set up some constants
NUMBER_OF_FILES = len(filtered_documents)

#set up patterns, which might appear in the context of high risk factors
PATTERNS = [
    r"(factor(.){0,9}risk)", # for example for "factors of risk"
    r"(risk(.){0,4}factor)", # for example for "risk factors"
    r"(high(.){0,6}risk)", # for example for "high risk" or "highly risky"
    r"(advers(.){0,4}outcome)"
]

POTENTIAL_RISKS = [
    "smoking",
    "pulmonary diseas", 
    "elder",
    "diabetes",
    "old",
    "age",
    "cancer", 
    "cardiac",
    "cardio"]

CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation]
CUSTOM_FILTERS_EXCLUDE_NUMERIC = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_numeric]
%%time
def extract_windows_containing(docs, pattern, print_out=True):
    indices = []
    for idx in range(NUMBER_OF_FILES):
        body_text = docs[idx].body_text
        
        indices_of_file = [(m.start(0), m.end(0)) for m in re.finditer(pattern, body_text)]
        indices.append(indices_of_file)
    
    return indices

indices = [[] for _ in range(NUMBER_OF_FILES)]
for pattern in PATTERNS:
    indices_ = extract_windows_containing(filtered_documents, pattern)
    for i in range(len(indices_)):
        indices[i].extend(indices_[i])

print("Found {} candidate text extracts".format(len([1 for a in indices if len(a)!=0])))
%%time

WINDOW_SIZE = 500
def process_indices_for_file(file_number, indices_of_file, filters, docs):
    contexts = []
    for match in indices_of_file:
        start = max(0, match[0]-int(WINDOW_SIZE/2))
        end = min(len(docs[file_number].body_text), match[1]+int(WINDOW_SIZE/2))
        context = docs[file_number].body_text[start:end]
        contexts.append(context)

    return contexts

def remove_special_character(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def tokenize(text):
    words = nltk.word_tokenize(text)
    return [str(word).lower() for word in words if len(word) > 1 and not word.isnumeric()]

potential_contexts = []
length_of_longest_context = 0
for file_number, indices_of_file in enumerate(indices):
    if len(indices_of_file) != 0:
        for context in process_indices_for_file(file_number, indices_of_file, CUSTOM_FILTERS_EXCLUDE_NUMERIC, filtered_documents):
            processed_context = " ".join(tokenize(remove_special_character(context)))
            if len(processed_context) > length_of_longest_context:
                length_of_longest_context = len(processed_context)
            potential_contexts.append(processed_context)

print("Length of longest context:", length_of_longest_context)
print("Number of potential_contexts:", len(potential_contexts))
%%time
def refined_contexts():
    print_counter = 0
    overall_processed_indices_count = 0
    refined_potential_contexts = []
    for file_number, indices_of_file in enumerate(indices):
        if len(indices_of_file) != 0:
            matches = process_indices_for_file(file_number, indices_of_file, CUSTOM_FILTERS, filtered_documents)
            processed_indices_of_file = []
            indices_of_file = sorted(indices_of_file)
            # adjust indices to avoid big overlaps
            for i in range(len(indices_of_file)):
                if i != 0 and len(processed_indices_of_file) != 0:
                    if abs(processed_indices_of_file[-1][0] - indices_of_file[i][0]) > 100 and abs(processed_indices_of_file[-1][1] - indices_of_file[i][1]) > 100:
                        processed_indices_of_file.append(indices_of_file[i])
                    else:
                        min_ = min(indices_of_file[i][0], processed_indices_of_file[-1][0])
                        max_ = max(indices_of_file[i][1], processed_indices_of_file[-1][1])
                        del processed_indices_of_file[-1]
                        processed_indices_of_file.append((min_, max_))
                else:
                    processed_indices_of_file.append(indices_of_file[i])
            overall_processed_indices_count += len(processed_indices_of_file)

            for index, match in zip(processed_indices_of_file, matches):
                for pattern in POTENTIAL_RISKS:
                    if pattern in match:
                        windows_size = max(int(len(" ".join(match))/2), WINDOW_SIZE)
                        text = filtered_documents[file_number].body_text
                        start_idx = max(0, index[0]-windows_size)
                        end_idx = min(len(text), index[1]+windows_size)
                        refined_potential_contexts.append(text[start_idx:end_idx])
                        break
    return overall_processed_indices_count, refined_potential_contexts

overall_processed_indices_count, refined_potential_contexts = refined_contexts()
print("Total number of possible text passages about risk factors: {}".format(overall_processed_indices_count))
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_AND_TOKENIZER = "ktrapeznikov/scibert_scivocab_uncased_squad_v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_AND_TOKENIZER)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_AND_TOKENIZER)

model = model.to(torch_device)
model.eval()


def generate_answer(question, context):
    encoded_dict = tokenizer.encode_plus(
                        question, context,
                        add_special_tokens = True,
                        max_length = 256,
                        pad_to_max_length = True,
                        return_tensors = 'pt'
                   )
    
    input_ids = encoded_dict['input_ids'].to(torch_device)
    token_type_ids = encoded_dict['token_type_ids'].to(torch_device)
    
    start_scores, end_scores = model(input_ids, token_type_ids=token_type_ids)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    
    answer = tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index+1])
    answer = answer.replace('[CLS]', '')
    answer = answer.replace('[PAD]', '')
    return answer

def create_output_results(question, 
                          all_contexts, 
                          all_answers):
    
    def find_start_end_index_substring(context, answer):   
        search_re = re.search(re.escape(answer.lower()), context.lower())
        if search_re:
            return search_re.start(), search_re.end()
        else:
            return 0, len(context)
        
    output = {}
    output['question'] = question
    results = []
    for c, a in zip(all_contexts, all_answers):
        span = {}
        span['context'] = c
        span['answer'] = a
        span['start_index'], span['end_index'] = find_start_end_index_substring(c,a)
        results.append(span)
    
    output['results'] = results
        
    return output

    
def get_results(question,
                contexts):
    answers = []
    for context in contexts:
        answers.append(generate_answer(question, context))
    
    return create_output_results(question, contexts, answers)

from IPython.display import display, Markdown, Latex, HTML

def layout_style():
    style = """
        div {
            color: black;
        }
        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }
        .answer{
            color: #dc7b15;
        }
        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }      
        div.output_scroll { 
            height: auto; 
        }
    """
    return "<style>" + style + "</style>"

def dm(x): display(Markdown(x))
def dh(x): display(HTML(layout_style() + x))
    
def display_single_context(context, start_index, end_index):    
    before_answer = context[:start_index]
    answer = context[start_index:end_index]
    after_answer = context[end_index:]

    content = before_answer + "<span class='answer'>" + answer + "</span>" + after_answer

    return dh("""<div class="single_answer">{}</div>""".format(content))

def display_question_title(question):
    return dh("<h2 class='question_title'>{}</h2>".format(question.capitalize()))


def display_all_contexts(index, question):
    def answer_not_found(context, start_index, end_index):
        return (start_index == 0 and len(context) == end_index) or (start_index == 0 and end_index == 0)

    display_question_title(str(index + 1) + ". " + question['question'].capitalize())
    
    # display context
    for i in question['results']:
        if answer_not_found(i['context'], i['start_index'], i['end_index']):
            continue # skip not found questions
        display_single_context(i['context'], i['start_index'], i['end_index'])

def display_result(result):
    for i, question in enumerate(result):
        display_all_contexts(i, question)
%%time
contexts = potential_contexts[:100]
result_on_potential_contexts = []
for q in questions['questions']:
    res = get_results(q, contexts)
    result_on_potential_contexts.append(res)
display_result(result_on_potential_contexts)
%%time
contexts = refined_potential_contexts[:100]
result_on_refined_contexts = []
for q in questions['questions']:
    res = get_results(q, contexts)
    result_on_refined_contexts.append(res)
display_result(result_on_refined_contexts)
from transformers import BartTokenizer, BartForConditionalGeneration, AutoModelWithLMHead
torch_device = "cpu"
text = " ".join(potential_contexts[500:1000])

model_used = "bart-large-cnn"

tokenizer_summarize = BartTokenizer.from_pretrained(model_used)
model_summarize = BartForConditionalGeneration.from_pretrained(model_used)

model_summarize = model_summarize.to(torch_device)
model_summarize.eval()
print()
def generate_summary(text):
    out = tokenizer_summarize.batch_encode_plus(
        [text], return_tensors='pt', max_length=512
    )

    input_ids = out['input_ids'].to(torch_device)
    attention_mask = out['attention_mask'].to(torch_device)

    beam_outputs = model_summarize.generate(input_ids,
                                           attention_mask=attention_mask,
                                           num_beams=5,
                                           max_length=256,
                                           early_stopping=True,
                                          )

    summary = tokenizer_summarize.decode(beam_outputs[0],
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True)
    summary = summary.replace(u'\xa0', u' ')
    summary = summary.replace('[CLS]', '')
    return summary
for question in result_on_potential_contexts:
    print("Question:", question["question"])
    results = question["results"]
    answers = ""
    count = 0
    for result in results:
        answer = result["answer"]
        if answer != "":
            answers += " " + answer
            count += 1
    print("\tNumber of answers: {}".format(count))
    print("\tLength of concatenation: {}".format(len(answers)))   
    answers = answers.replace('[SEP]', '')
    print("\n\tConcatenated answers:", answers)
    summary = generate_summary(answers)
    print("\n\tSummary:", summary)
    print("\n\n")

for question in result_on_refined_contexts:
    print("Question:", question["question"])
    results = question["results"]
    answers = ""
    count = 0
    for result in results:
        answer = result["answer"]
        if answer != "":
            answers += " " + answer
            count += 1
    print("\tNumber of answers: {}".format(count))
    print("\tLength of concatenation: {}".format(len(answers)))   
    answers = answers.replace('[SEP]', '')
    print("\n\tConcatenated answers:", answers)
    summary = generate_summary(answers)
    print("\n\tSummary:", summary)
    print("\n\n")




