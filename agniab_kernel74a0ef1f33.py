
import json
import pickle
import os


articles = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(dirname, filename)) as json_file:
            json_data = json.load(json_file)
            current_article = {'paper_id': json_data['paper_id'], 'title': json_data['metadata'].get('title', ''), 'abstract': '', 'text': ''}
            for element in json_data.get('abstract', []):
                current_article['abstract'] += '\n' + element['text']
            for element in json_data.get('body_text', []):
                current_article['text'] += '\n' + element['text']
            articles.append(current_article)
                
pickle.dump(articles, open('articles.pkl', 'wb'))

articles = pickle.load(open('articles.pkl', 'rb'))
virus_names = {
    'covid19',
    'covid-19',
    'covid 19',
    '2019-ncov',
    '2019 ncov',
    '2019 novel coronavirus',
    '2019 coronavirus',
    'coronavirus disease 2019',
    'corona virus disease 2019',
    '2019-novel coronavirus',
    '2019 coronavirus',
    'corona-virus',
    'sars-cov-2'}

covid_articles = []
for article in articles:
    text = article['title'].lower() + article['abstract'].lower()
    if not text:
        text = article['text'].lower()
    if any([name for name in virus_names if name in text]):
        covid_articles.append(article)
print(f'{len(covid_articles)} of {len(articles)} seem to be about COVID-19')
pickle.dump(covid_articles, open('covid_articles.pkl', 'wb'))
import csv
import pickle

filenames = ['/kaggle/input/umls-concept-lists/umls_2017AB_drugs_filtered.csv', '/kaggle/input/umls-concept-lists/umls_2017AB_procedures_filtered.csv']

def csv_to_pkl(file_in, file_out, filter_code=""):
    umls_concepts = dict()
    with open(file_in, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if filter_code in row['SAB']:
                umls_concepts[row['CUI']] = row['STR']
    pickle.dump(umls_concepts, open(file_out, 'wb'))
    

csv_to_pkl(filenames[0], 'drugs.pkl')
csv_to_pkl(filenames[1], 'procedures.pkl', 'CPT')
csv_to_pkl(filenames[1], 'problems.pkl', 'ICD')

            
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
import pickle
import scispacy
import en_core_sci_lg
from scispacy.umls_linking import UmlsEntityLinker


drug_by_id = pickle.load(open('drugs.pkl', 'rb'))
procedure_by_id = pickle.load(open('procedures.pkl', 'rb'))
problem_by_id = pickle.load(open('problems.pkl', 'rb'))

nlp = en_core_sci_lg.load()
linker = UmlsEntityLinker()
nlp.add_pipe(linker)

def update_examples(concept_id, example_dict, concept_name, example, article_id):
    if concept_id not in example_dict:
        example_dict[concept_id] = (concept_name, [])
    example_dict[concept_id][1].append((example, article_id))

def find_entities_in_text(text, article_id):
    parse = nlp(text)
    drugs, procedures, problems = [], [], []
    for ent in parse.ents:
        if ent._.umls_ents:
            umls_id = ent._.umls_ents[0][0] # there can be multiple matches but they are sorted by confidence, so taking the first
            if umls_id in drug_by_id:
                update_examples(umls_id, drugs_with_examples, drug_by_id[umls_id], ent.sent.text, article_id)
            if umls_id in procedure_by_id:
                update_examples(umls_id, procedures_with_examples, procedure_by_id[umls_id], ent.sent.text, article_id)
            if umls_id in problem_by_id:
                update_examples(umls_id, problems_with_examples, problem_by_id[umls_id], ent.sent.text, article_id)
    

problems_with_examples = {}
drugs_with_examples = {}
procedures_with_examples = {}

covid_articles = pickle.load(open('/kaggle/input/outputs-cache/covid_articles.pkl', 'rb'))
for article in covid_articles:
    text = article.get('abstract') or article.get('text')[:3000]
    find_entities_in_text(text, article['paper_id'])
    
pickle.dump(problems_with_examples, open('problems_with_examples.pkl', 'wb'))
pickle.dump(drugs_with_examples, open('drugs_with_examples.pkl', 'wb'))
pickle.dump(procedures_with_examples, open('procedures_with_examples.pkl', 'wb'))

    
import pickle
import csv

problems_with_examples = pickle.load(open('/kaggle/input/entities-with-examples/problems_with_examples.pkl', 'rb'))
drugs_with_examples = pickle.load(open('/kaggle/input/entities-with-examples/drugs_with_examples.pkl', 'rb'))
procedures_with_examples = pickle.load(open('/kaggle/input/entities-with-examples/procedures_with_examples.pkl', 'rb'))

print(f'{len(problems_with_examples)} problems')
print(f'{len(drugs_with_examples)} drugs')
print(f'{len(procedures_with_examples)} procedures')

def sorted_to_tsv(examples_dict, filename):
    sorted_examples = sorted(list(examples_dict.values()), key = lambda item: len(item[1]), reverse=True)
    with open(filename, 'w') as tsv:
        writer = csv.writer(tsv, delimiter='\t')
        writer.writerow(['concept', 'paper_id', 'relevant_sentence'])
        for concept, examples in sorted_examples:
            for ex in examples:
                writer.writerow([concept, ex[1], ex[0]])

sorted_to_tsv(problems_with_examples, 'problems.tsv')
sorted_to_tsv(drugs_with_examples, 'drugs.tsv')
sorted_to_tsv(procedures_with_examples, 'procedures.tsv')
    

            
            
            
!pip install biobert_embedding
import spacy
import pickle
from scipy.spatial.distance import cosine
from biobert_embedding.embedding import BiobertEmbedding

biobert = BiobertEmbedding()
# questions = [
# 'COVID-19 risk factors',
# 'risk of COVID-19 in smokers',
# 'risk of COVID-19 with pre-existing pulmonary disease',
# 'co-infections make COVID-19 more transmissible',
# 'co-infections make COVID-19 more virulent',
# 'co-morbidities and COVID-19',
# 'drugs developed to treat COVID-19',
# 'drugs tried to treat COVID-19',
# 'therapeutic for COVID-19',
# 'extrapulmonary manifestations of COVID-19',
# 'COVID-19 affecting the heart'
# ]

# questions = [
# 'Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies',
# 'Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients',
# 'Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest.',
# 'Oral medications that might potentially work',
# 'Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)',
# 'Effectiveness of drugs being developed and tried to treat COVID-19 patients',
# 'pabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.',
# 'Data on potential risks factors such as smoking, pre-existing pulmonary disease, co-infections, and other co-morbidities'
# ]

questions = [
'risk factors',
'risk in smokers',
'risk with pre-existing pulmonary disease',
'co-infections make virus more transmissible',
'co-infections make virus more virulent',
'co-morbidities',
'drugs developed to treat virus',
'drugs tried to treat virus',
'therapeutic for virus',
'extrapulmonary manifestations',
'virus affecting the heart'
]

nlp = spacy.load("en_core_web_sm")
question_embeddings = [biobert.sentence_vector(q) for q in questions]

SIMILARITY_THRESHOLD = 0.6

covid_articles = pickle.load(open('/kaggle/input/outputs-cache//covid_articles.pkl', 'rb'))
for article in covid_articles:
#     parsed = nlp(article.get('title', '') + '. ' + article.get('abstract', ''))
#     if not article.get('abstract'):
#         parsed = nlp(article.get('text', '')[:3000])
    
    parsed = nlp(article.get('text', ''))
    if not article.get('text'):
        parsed = nlp(article.get('abstract', ''))
    for sent in parsed.sents:
        if len(sent.text) < 30:  # random things were popping up, like "1, 4"
            continue
        for i in range(len(questions)):
            sim = cosine(question_embeddings[i], biobert.sentence_vector(sent.text[:512]))
            if sim > SIMILARITY_THRESHOLD:
                print(sim, questions[i], sent.text)
