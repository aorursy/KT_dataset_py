%%capture
# install SciSpacy and download a full spaCy pipeline for biomedical data
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
from typing import List, Dict, Iterable, Tuple

import os
import json

from tqdm import tqdm

import spacy
from spacy.tokens import Span
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
%%capture 
# instantiate language ScispaCy model
full_nlp = spacy.load('en_core_sci_sm')

# Add the abbreviation pipe to the spacy pipeline.
abbreviation_pipe = AbbreviationDetector(full_nlp)
full_nlp.add_pipe(abbreviation_pipe)

# Add the entity linking pipe to the spacy pipeline
linker = UmlsEntityLinker(resolve_abbreviations=True, filter_for_definitions=False)
full_nlp.add_pipe(linker)
ROOT_PATH = os.path.join("/kaggle", "input", "CORD-19-research-challenge")
JSON_PATH = os.path.join(ROOT_PATH, "document_parses", "pdf_json")

def load_json_files_lazy(directory_path: str):
    """Load the json files from a directory """
    loaded_files = []
    for filename in os.listdir(directory_path):
        full_path = os.path.join(directory_path, filename)
        with open(full_path) as _json_file:
            loaded_file = json.load(_json_file)
            yield loaded_file

json_loaded_files = load_json_files_lazy(JSON_PATH)
def build_doc_with_entities(full_text: str, abstract_text: List, body_text: List):
    """Build a doc using mention spans from the input file, but the scispacy linker"""
    
    # disable entity linker and named entity recognition pipelines
    with full_nlp.disable_pipes(['UmlsEntityLinker', 'ner']):
        doc = full_nlp(full_text)
        
    entities = []
    character_offset = 0
    for paragraph in abstract_text:
        paragraph_text = paragraph["text"].strip()
        for entity in paragraph['entity_spans']:
            entity_start = character_offset + entity['start']
            entity_end = character_offset + entity['end']
            entity_span = doc.char_span(entity_start, entity_end)

            # just skip for now if the character span does not align
            if entity_span is not None:
                entities.append(entity_span)

        character_offset += len(paragraph_text) + 1

    for paragraph in body_text:
        paragraph_text = paragraph["text"].strip()
        for entity in paragraph['entity_spans']:
            entity_start = character_offset + entity['start']
            entity_end = character_offset + entity['end']
            entity_span = doc.char_span(entity_start, entity_end)

            # just skip for now if the character span does not align
            if entity_span is not None:
                entities.append(entity_span)

        character_offset += len(paragraph_text) + 1
    
    new_entity_spans = [Span(doc, entity_span.start, entity_span.end, label="Entity") for entity_span in entities]
    doc.ents = new_entity_spans
    doc = linker(doc)
    
    return doc

def add_entities_to_file(input_json: Dict, use_existing_mentions: bool = False):
    """Copies the input json and adds the linked entities to it. 
       If you want to use entity annotations already present in the input json file,
       set the use_existing_mentions flag, otherwise scispacy's base model will be used for NER"""
    
    body_text = input_json["body_text"]
    abstract_text = input_json['abstract']
    
    paragraph_char_spans = []
    char_span_index = 0
    paragraph_index = 0
    full_text = ""
    for paragraph in abstract_text:
        paragraph_text = paragraph["text"].strip()
        full_text += paragraph_text + " "
        paragraph_char_spans.append(("abstract", paragraph_index, char_span_index, char_span_index + len(paragraph_text)))
        char_span_index = char_span_index + len(paragraph_text) + 1
        paragraph_index += 1
    
    paragraph_index = 0
    for paragraph in body_text:
        paragraph_text = paragraph["text"].strip()
        full_text += paragraph_text + " "
        paragraph_char_spans.append(("body_text", paragraph_index, char_span_index, char_span_index + len(paragraph_text)))
        char_span_index = char_span_index + len(paragraph_text) + 1
        paragraph_index += 1
    
    full_text = full_text[:-1]
    
    if not use_existing_mentions:
        doc = full_nlp(full_text)
    else:
        doc = build_doc_with_entities(full_text, abstract_text, body_text)
    
    input_copy = input_json.copy()
    for i, (paragraph, (section, paragraph_index, start_char, end_char)) in enumerate(zip(abstract_text + body_text, paragraph_char_spans)):
        entities = []
        paragraph_span = doc.char_span(start_char, end_char)
        for mention_span in paragraph_span.ents:
            linked_cuis_and_scores = mention_span._.umls_ents
            # the definition, aliases, and type can be accessed via linker.umls.cui_to_entity[cui]
            entity = {}
            entity['start'] = mention_span.start_char - paragraph_span.start_char
            entity['end'] = mention_span.end_char - paragraph_span.start_char
            entity['text'] = mention_span.text
            
            # could filter out specific UMLS types here, if desired
            entity['links'] = [(cui, linker.umls.cui_to_entity[cui].types[0], score) for (cui, score) in linked_cuis_and_scores]
            entities.append(entity)

        input_copy[section][paragraph_index]['entity_spans'] = entities
    return input_copy
    
def write_json_file(directory_path: str, file_name: str, output_json: Dict):
    """Write a json file out"""
    with open(os.path.join(directory_path, file_name), 'w') as _json_file:
        json.dump(output_json, _json_file, indent=4)

def write_subset_directory_with_entities(directory_path: str, inputs: List[Dict], num_files_to_process: int = 0):
    """Write the transformed jsons for a full subset directory"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

    for i, file in tqdm(enumerate(inputs), desc=f"Processing {directory_path}"):
        if i >= num_files_to_process:
            break
        new_json = add_entities_to_file(file)
        write_json_file(directory_path, file['paper_id'] + '.json', new_json)

KAGGLE_OUTPUT_DIRECTORY = os.path.join("/kaggle", "working")
ROOT_OUTPUT_DIRECTORY = os.path.join(KAGGLE_OUTPUT_DIRECTORY, "CORD-19-with-entities")
JSON_OUTPUT_DIRECTORY = os.path.join(ROOT_OUTPUT_DIRECTORY, "comm_use_subset", "comm_use_subset")

# process a few samples
write_subset_directory_with_entities(JSON_OUTPUT_DIRECTORY, json_loaded_files, 10)
# interpret the output - links to UMLS 
with open(os.path.join(JSON_OUTPUT_DIRECTORY, list(os.listdir(JSON_OUTPUT_DIRECTORY))[0])) as _json_file:
    loaded_file = json.load(_json_file)

body = loaded_file['body_text']
first_paragraph = body[0]


print("[INFO] Take a look at the analysis of the first paragraph:\n")
print(first_paragraph['text'])
print()
for entity in first_paragraph['entity_spans']:
    top_link = entity['links'][0] if len(entity['links']) > 0 else None
    mention_text = entity['text']
    print(f"Mention: {mention_text}")
    print(linker.umls.cui_to_entity[top_link[0]] if top_link else "No links passed the threshold")
    print()