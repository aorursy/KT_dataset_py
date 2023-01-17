import spacy

spacy.prefer_gpu()

import json

from tqdm.auto import tqdm

from pathlib import Path

import pandas as pd

import numpy as np

import gzip

!pip install opentargets

from opentargets import OpenTargetsClient

!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bionlp13cg_md-0.2.4.tar.gz

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz
def extract_paper_annotations():

    """

    This function looks at all the papers in the CORD-19 dataset and extract entities

    """

    # Define the list of papers we will process

    #papers = [Path("/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/59eab95c43fdea01481fdbf9bae45dfe28ffc693.json")]

    papers = [p for p in Path('/kaggle/input/CORD-19-research-challenge').glob('biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/*.json')]

    #papers += [p for p in Path('/kaggle/input/CORD-19-research-challenge').glob('comm_use_subset/comm_use_subset/pdf_json/*.json')]

    #papers += [p for p in Path('/kaggle/input/CORD-19-research-challenge').glob('noncomm_use_subset/noncomm_use_subset/pdf_json/*.json')]

    #papers += [p for p in Path('/kaggle/input/CORD-19-research-challenge').glob('custom_license/custom_license/pdf_json/*.json')]

    print (len(papers)) 



    # Load the NLP models

    nlp_model_bionlp13cg = spacy.load('/opt/conda/lib/python3.6/site-packages/en_ner_bionlp13cg_md/en_ner_bionlp13cg_md-0.2.4') # For cells, genes, ...

    nlp_model_bc5cdr = spacy.load('/opt/conda/lib/python3.6/site-packages/en_ner_bc5cdr_md/en_ner_bc5cdr_md-0.2.4') # For diseases



    # The output will be one hashmap associating each paper to its annotations

    output = {}



    # Process all the papers

    for paper in tqdm(papers):

        try:

            # Load the document

            document = json.loads(paper.read_text())



            # Get the ID

            paper_id = document['paper_id']

            

            # Initialise its entry

            output[paper_id] = {}

            output[paper_id]['topics'] = {} # The different topic annotations grouped per type

            

            # Group the text by sections (took more than 9h to process!)

            #section_texts = {}

            #section_texts['abstract'] = []

            #for b in document['abstract']:

            #    section_texts['abstract'].append(b['text'])

            #for b in document['body_text']:

            #    section_texts.setdefault(b['section'], [])

            #    section_texts[b['section']].append(b['text'])



            # Retrieve all the text

            texts = []

            for b in document['abstract']:

                texts.append(b['text'])

            if 'body_text' in document:

                for b in document['body_text']:

                    texts.append(b['text'])

            

            # Process the different sections to extract entities

            #for section,texts in section_texts.items():

            text = '.'.join(texts)

            for nlp_model in [nlp_model_bionlp13cg, nlp_model_bc5cdr]:

                tokens = nlp_model(text)

                for entity in tokens.ents:

                    topic_type = entity.label_

                    topic_value = str(entity.text)

                    output[paper_id]['topics'].setdefault(topic_type, set())

                    output[paper_id]['topics'][topic_type].add(topic_value)

            

        except Exception as e:

            print ('Error with {}'.format(paper))

            print (e)



    # Turn the sets into lists to save them as JSON

    for paper_id in output.keys():

        for topic_type in output[paper_id]['topics'].keys():

            output[paper_id]['topics'][topic_type] = list(output[paper_id]['topics'][topic_type])



    return output
# Step 1 => get the keywords out of the paper abstract and content

annotations = extract_paper_annotations()

print (len(annotations.keys()))
def get_paper_annotations_graph(annotations):

    """

    This function is used to generate a graph from the paper annotations

    

    We will turn all the NLP annotations into concept identifiers using a list of terms extracted form Open Targets and the ontology MONDO. 

    This is done using a basic exact string matching and all the non matching strings are ignored.



    We extract a mapping "disease name" => "disease identifier" from Open Targets as the primary source, falling back on Mondo to fill the gaps. 

    In particular one of the missing value in Open Targets right now is Covid-19 ... ;-)

    """

    

    # Prepare a map to deal with all the different types of entities type recognized by Spacy and that may be found in the annotations

    ontology_map = {

        'DISEASE': {},

        'CANCER': {},

        'GENE_OR_GENE_PRODUCT': {}

    }



    # TODO: If we want to keep more of the annotations returned by Spacy we should align:

    # From https://allenai.github.io/scispacy/ en_ner_bionlp13cg_md

    #  CANCER, ORGAN, TISSUE, ORGANISM, CELL, AMINO_ACID, GENE_OR_GENE_PRODUCT, 

    #  SIMPLE_CHEMICAL, ANATOMICAL_SYSTEM, IMMATERIAL_ANATOMICAL_ENTITY, 

    #  MULTI-TISSUE_STRUCTURE, DEVELOPING_ANATOMICAL_STRUCTURE, 

    #  ORGANISM_SUBDIVISION, CELLULAR_COMPONENT

    # From https://allenai.github.io/scispacy/ en_ner_bc5cdr_md

    #  DISEASE, CHEMICAL



    ########################

    # Get mappings for DISEASE

    ########################

    

    # Load the file from Open Targets and fill the hashmap in

    disease_list = pd.read_csv('https://storage.googleapis.com/open-targets-data-releases/20.02/output/20.02_disease_list.csv.gz', compression='gzip')

    disease_list['disease_full_name'] = disease_list['disease_full_name'].str.lower()

    print('Number of keywords in open targets:', len(set(disease_list['disease_full_name'].values)))

    for index, row in disease_list.iterrows():

        full_name = row['disease_full_name'].lower()

        identifier = row['efo_id']

        ontology_map['DISEASE'][full_name] = identifier



    # Open Targets does not have Covid-19 in its list of diseases. We had it manually

    # To get the labels we ran the following query on http://www.ontobee.org/sparql

    #  select distinct ?s ?o where {

    #    {<http://purl.obolibrary.org/obo/MONDO_0100096> <http://www.geneontology.org/formats/oboInOwl#hasExactSynonym> ?o} 

    #    union

    #    {<http://purl.obolibrary.org/obo/MONDO_0100096> <http://www.w3.org/2000/01/rdf-schema#label> ?o}

    #  }

    ontology_map['DISEASE']['2019 novel coronavirus infection'.lower()] = 'MONDO_0100096'

    ontology_map['DISEASE']['2019-nCoV infection'.lower()] = 'MONDO_0100096'

    ontology_map['DISEASE']['severe acute respiratory syndrome coronavirus 2'.lower()] = 'MONDO_0100096'

    ontology_map['DISEASE']['SARS-CoV-2'.lower()] = 'MONDO_0100096'

    ontology_map['DISEASE']['SARS-coronavirus 2'.lower()] = 'MONDO_0100096'

    ontology_map['DISEASE']['coronavirus disease 2019'.lower()] = 'MONDO_0100096'

    ontology_map['DISEASE']['COVID-19'.lower()] = 'MONDO_0100096'

    

    # Debug output

    print ('Number of keywords in map for diseases: {}'.format(len(ontology_map['DISEASE'])))

    

    

    ########################

    # Get mappings for CANCER

    ########################



    # We will simply treat the "CANCER" annotations from Spacy as "DISEASE"

    ontology_map['CANCER'] = ontology_map['DISEASE']

    

    

    ########################

    # Get mappings for GENE_OR_GENE_PRODUCT

    ########################

    

    # Load a target list from Open Targets. It will be used to map gene keywords

    target_list = pd.read_csv('https://storage.googleapis.com/open-targets-data-releases/20.02/output/20.02_target_list.csv.gz', compression='gzip')

    target_list['hgnc_approved_symbol'] = target_list['hgnc_approved_symbol'].str.lower()

    print('Number of genes in open targets:', target_list['hgnc_approved_symbol'].nunique())

    for index, row in target_list.iterrows():

        full_name = row['hgnc_approved_symbol']

        identifier = row['ensembl_id']

        ontology_map['GENE_OR_GENE_PRODUCT'][full_name] = identifier

    



    ########################

    # Turn the paper annotations into a graph

    ########################

    graph = []

    predicates = {

        'DISEASE': 'isAboutDisease',

        'CANCER': 'isAboutDisease',

        'GENE_OR_GENE_PRODUCT': 'isAboutTarget'

    }

    # Go through all the papers

    for (paper_id, data) in annotations.items():

        # For each annotation topic try to find a match in the ontology map

        for (topic, values) in data['topics'].items():

            if topic in ontology_map:

                for value in values:

                    if value.lower() in ontology_map[topic]:

                        obj = ontology_map[topic][value.lower()]

                        graph.append([paper_id, predicates[topic], obj])

               

    return graph
def connect_targets_and_diseases(graph):

    """

    This function will use the association data from Open Target to 

    connect instances of Target and Disease in the graph.

    

    We at the same time connect diseases to therapeutic areas (instances of Disease)

    as this information is returned by the API

    """

    

    # Get a list of all the targets (genes) and diseases currently in the graph

    targets = list(set([t[2] for t in graph if t[1] == 'isAboutTarget']))

    diseases = list(set([t[2] for t in graph if t[1] == 'isAboutDisease']))

    

    # Prepare a map of target => disease relations

    ot_output_associations = {}

    

    # Query OpenTargets for Target => Disease associations

    ot = OpenTargetsClient()

    for target in tqdm(targets):

        ot_output_associations.setdefault(target, set())

        search_results = ot.get_associations_for_target(target)

        if len(search_results) > 0 and search_results[0]['target']['id'] == target:

            for search_result in search_results:

                if search_result['association_score']['overall'] > 0.8:

                    disease = search_result['disease']['id']

                    ot_output_associations[target].add(disease)

                        

    # Query OpenTargets for Disease => Target associations

    for disease in tqdm(diseases): 

        search_results = ot.get_associations_for_disease(disease)

        if len(search_results) > 0 and search_results[0]['disease']['id'] == disease:

            for search_result in search_results:

                if search_result['association_score']['overall'] > 0.8:

                    target = search_result['target']['id']

                    ot_output_associations.setdefault(target, set())

                    ot_output_associations[target].add(disease)



    # Turn the output into new edges in the graph

    for target, diseases in ot_output_associations.items():

        for disease in diseases:

            # Target -> Disease relation

            graph.append([target, 'isAssociatedTo', disease])            
def connect_diseases_to_diseases(graph):

    """

    This function leverages the disease similarity information computed by Open Targets

    to connect Diseases to each other. Those links will later be used to find risk factors.

    """

    

    # Get a list of all the diseases currently in the graph.

    # We do that by looking at the objects of triples we know link to Diseases

    diseases = set([t[2] for t in graph if t[1] == 'isAboutDisease']) 

    diseases = diseases | set([t[2] for t in graph if t[1] == 'isAssociatedTo']) 

    

    # Query OpenTargets

    ot_output_diseases = {}

    ot = OpenTargetsClient()

    for disease in tqdm(diseases):

        ot_output_diseases[disease] = set()

        search_results = ot.get_similar_disease(disease)

        for search_result in search_results:

            if search_result['subject']['id'] == disease: # Safe guard

                ot_output_diseases[disease].add(search_result['object']['id'])

                

    # Turn the output we received into edges

    for src_disease, target_diseases in ot_output_diseases.items():

        for target_disease in target_diseases:

            graph.append([src_disease, 'hasGeneticClue', target_disease])
def add_disease_classification(graph):

    """

    This function adds to the graph the disease classification tree.

    See, for example, https://www.targetvalidation.org/disease/EFO_0005774 .

    """



    # Get a list of all the diseases in the graph

    diseases = set([t[2] for t in graph if t[1] == 'isAboutDisease']) 

    diseases = diseases | set([t[2] for t in graph if t[1] == 'isAssociatedTo']) 

    diseases = diseases | set([t[2] for t in graph if t[1] == 'hasGeneticClue']) 



    # Query OpenTargets

    paths = set()

    ot = OpenTargetsClient()

    for disease in tqdm(diseases):

        search_results = ot.search(disease)

        if search_results != None and len(search_results) > 0:

            search_result = search_results[0]

            if search_result['id'] == disease:

                if 'efo_path_codes' in search_result['data']:

                    for path in search_result['data']['efo_path_codes']:

                        paths.add('=>'.join(path))

                        

    # Turn the output we received into edges

    for path_str in paths:

        path = path_str.split('=>')

        for index in range(0, len(path)-1):

            start = path[index]

            end = path[index+1]

            graph.append([end, 'isASpecific', start])
def add_disease_therapeutic_areas(graph):

    """

    This function query Open Targets for the therapeutic area of all the diseases

    """

    

    # Get a list of all the diseases in the graph

    diseases = set([t[2] for t in graph if t[1] == 'isAboutDisease']) 

    diseases = diseases | set([t[2] for t in graph if t[1] == 'isAssociatedTo'])

    diseases = diseases | set([t[2] for t in graph if t[1] == 'hasGeneticClue']) 

    diseases = diseases | set([t[2] for t in graph if t[1] == 'isASpecific']) 

    

    # Query OpenTargets

    ot_output = {}

    ot = OpenTargetsClient()

    for disease in tqdm(diseases):

        ot_output[disease] = set()

        search_results = ot.get_disease(disease)

        if search_results != None and len(search_results) > 0:

            search_result = search_results[0]

            if search_result['code'].endswith(disease) and 'therapeutic_codes' in search_result:

                for therapeutic_code in search_result['therapeutic_codes']:

                        ot_output[disease].add(therapeutic_code)

                        

    # Turn the output we received into edges

    for (disease, areas) in ot_output.items():

        for area in areas:

            graph.append([disease, 'belongsToTherapeuticArea', area])
def print_graph_stats(graph):

    resources = set([r[0] for r in graph]) | set([r[2] for r in graph])

    predicates = set([r[1] for r in graph])

    print ('Graph has {} edges, {} resources, {} predicates'.format(len(graph), len(resources), len(predicates)))

    display(pd.DataFrame([t for t in graph], columns=['Subject', 'Predicate', 'Object']))
# Step 2 => get the starting graph of paper annotations

graph = get_paper_annotations_graph(annotations)

print_graph_stats(graph)
# Step 3 => enrich the graph with Target - Disease links

connect_targets_and_diseases(graph)

print_graph_stats(graph)
# Step 4 => connect diseases to related diseases

connect_diseases_to_diseases(graph)

print_graph_stats(graph)
# Step 5 => add disease classification trees

add_disease_classification(graph)

print_graph_stats(graph)
# Step 6 => add disease therapeutic areas

add_disease_therapeutic_areas(graph)

print_graph_stats(graph)
# Finally, we do a last pass to remove duplicate statements

final_graph = [t.split('=>') for t in set(['=>'.join(t) for t in graph])]

print_graph_stats(final_graph)



# and we save the graph to disk

graph_df = pd.DataFrame(final_graph, columns=['subject', 'predicate', 'object'])

graph_df.to_csv('graph.csv', index=False)
def get_neighbours(resource):

    # Extract a disease and target code=>label

    to_name = {}

    target_list = pd.read_csv('https://storage.googleapis.com/open-targets-data-releases/20.02/output/20.02_target_list.csv.gz', compression='gzip')

    for row in target_list.itertuples():

        to_name[row.ensembl_id] = row.hgnc_approved_symbol

    disease_list = pd.read_csv('https://storage.googleapis.com/open-targets-data-releases/20.02/output/20.02_disease_list.csv.gz', compression='gzip')

    for row in disease_list.itertuples():

        to_name[row.efo_id] = row.disease_full_name

    

    # Extract edges we may be interested in

    triples = [t for t in final_graph if t[0] == resource or t[2] == resource]



    # Construct a dataframe

    tmp = []

    for t in triples:

        s = '{} ({})'.format(t[0], to_name.get(t[0], '?'))

        o = '{} ({})'.format(t[2], to_name.get(t[2], '?'))

        tmp.append([s,t[1],o])

        

    return pd.DataFrame(tmp, columns=['Subject', 'Predicate', 'Object'])
display(get_neighbours('MONDO_0008903'))
display(get_neighbours('MONDO_0100096'))
! conda install tensorflow-gpu'>=1.14.0,<2.0.0' -y

! pip install ampligraph
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'



import pandas as pd

import numpy as np

np.random.seed(117)

from ampligraph.latent_features import ComplEx, TransE, DistMult, RandomBaseline



from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score, mr_score

from ampligraph.utils import save_model, restore_model





DATASET_BASE_PATH = "/kaggle/"
triples = pd.read_csv("graph.csv")



paper_diseases = set(triples[triples.predicate == 'isAboutDisease'].object)

paper_targets = set(triples[triples.predicate == 'isAboutTarget'].object)

new_triples = []

for row in triples.itertuples():

    if row.predicate == 'isAssociatedTo':

        if row.subject in paper_targets or row.object in paper_diseases:

            new_triples.append([row.subject, row.predicate, row.object])

    if row.predicate == 'isAboutDisease' or row.predicate == 'isAboutTarget':

        new_triples.append([row.subject, row.predicate, row.object])

    if row.predicate == 'hasGeneticClue':

        if row.subject in paper_diseases or row.object in paper_diseases:

            new_triples.append([row.subject, row.predicate, row.object])

    if row.predicate == 'isASpecific':

        new_triples.append([row.subject, row.predicate, row.object])

    if row.predicate == 'belongsToTherapeuticArea':

        new_triples.append([row.subject, row.predicate, row.object])

print (len(new_triples))





graph_df = pd.DataFrame(new_triples, columns=['subject', 'predicate', 'object'])



# this line is added for making sure that the results are reproducible.

graph_df.sort_values(by=['subject', 'predicate', 'object'], inplace=True)



graph_df.to_csv('COVID_KG_sample.csv', index=False)



graph_df.head()



print('Size of the graph:', graph_df.shape)



print(graph_df.columns)



print(graph_df.predicate.value_counts())
genetic_clue_triples = graph_df[graph_df['predicate']=='hasGeneticClue']

train_set = graph_df[graph_df['predicate']!='hasGeneticClue'].values


disease_list =  np.unique(np.concatenate([

                    np.unique(train_set[train_set[:, 1]=='isAboutDisease'][:, 2]),

                    np.unique(train_set[train_set[:, 1]=='isAssociatedTo'][:, 2]),

                ], 0))



print('diseases in df:', len(disease_list))
import random



np.random.seed(117)



test_set_diseases = np.random.choice(list(disease_list), 100).tolist()



#test_set_diseases = set(np.random.choice(list(disease_list), 2).tolist())

print(test_set_diseases)


test_set = genetic_clue_triples[genetic_clue_triples["subject"].isin(test_set_diseases)]

train_genetic_clue_triples = genetic_clue_triples[~genetic_clue_triples["subject"].isin(test_set_diseases)]

train_set = np.concatenate([train_set, train_genetic_clue_triples], 0)

train_set = np.random.permutation(train_set)



print('Train set size:', train_set.shape)

print('Test set size:', test_set.shape)

print('Full Graph size:', graph_df.shape)

disease_list_full =  np.unique(np.concatenate([

                        np.unique(train_set[train_set[:, 1]=='isAboutDisease'][:, 2]),

                        np.unique(train_set[train_set[:, 1]=='isAssociatedTo'][:, 2]),

                        np.unique(train_set[train_set[:, 1]=='hasGeneticClue'][:, 0]),

                        np.unique(train_set[train_set[:, 1]=='hasGeneticClue'][:, 2]),

                    ], 0))



print('diseases in df:', len(disease_list_full))
filter_triples = genetic_clue_triples.values



random_model = RandomBaseline(seed=0)



random_model.fit(train_set)



ranks = evaluate_performance(test_set.values, 

                             random_model, 

                             filter_triples=filter_triples, 

                             corrupt_side='o', 

                             entities_subset=list(disease_list_full))



print('MRR with random baseline:', mrr_score(ranks))
filter_triples = genetic_clue_triples.values





model = ComplEx(batches_count=15, seed=0, epochs=1000, k=200, eta=20,

                optimizer='adam', optimizer_params={'lr':1e-4}, 

                verbose=True, loss='multiclass_nll',

                regularizer='LP', regularizer_params={'p':3, 'lambda':1e-3})







early_stopping = { 'x_valid': test_set.values,

                   'criteria': 'mrr', 

                  'x_filter': filter_triples, 

                  'stop_interval': 3, 

                  'burn_in': 50, 

                  'corrupt_side':'o',

                  'corruption_entities': list(disease_list_full),

                  'check_interval': 50 }



model.fit(train_set, True,early_stopping)



ranks = evaluate_performance(test_set.values, 

                             model, 

                             filter_triples=filter_triples, 

                             corrupt_side='o', 

                             entities_subset=list(disease_list_full))



print('MRR with trained ComplEx embedding model:', mrr_score(ranks))



model.calibrate(train_set, positive_base_rate=0.5, epochs=100)

save_model(model, 'output_graph.pth')
disease_id = 'MONDO_0100096' #covid-19



test_predicate = 'hasGeneticClue'



hypothesis = np.concatenate([np.array([[disease_id] * disease_list_full.shape[0]]), 

                             np.array([[test_predicate] * disease_list_full.shape[0]]),

                             disease_list_full[np.newaxis, :]],0).T

print(hypothesis.shape)



scores = model.predict_proba(hypothesis)
disease_mapping_list_df = disease_list = pd.read_csv('https://storage.googleapis.com/open-targets-data-releases/20.02/output/20.02_disease_list.csv.gz', 

                                                     compression='gzip')





disease_mapping_list_df.head()

tested_hypothesis = pd.DataFrame(np.concatenate([hypothesis, 

                                                 scores[:, np.newaxis]], 1), 

                                 columns=['s','p','o','score'])



tested_hypothesis = tested_hypothesis[tested_hypothesis['o'] != disease_id]



tested_hypothesis = tested_hypothesis.sort_values(by='score', 

                                                  ascending=False)



tested_hypothesis = tested_hypothesis.merge(disease_mapping_list_df, 

                                            how='left', 

                                            left_on='o', 

                                            right_on='efo_id')[['disease_full_name', 'score']]



tested_hypothesis.columns = ['Risk Factors', 'Score']



pd.set_option('display.max_rows', 101)



tested_hypothesis.head(100)
tested_hypothesis.to_csv('predicted_covid19_risk_factors.csv')