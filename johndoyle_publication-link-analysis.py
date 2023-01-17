#import utility functions 

from covid19_data_processing_module import *



import matplotlib.pyplot as plt

import networkx as nx

import copy 

import collections



from spacy import displacy

from spacy.matcher import PhraseMatcher

from spacy.tokens import Span

from spacy.tokens import Doc
# Run matcher on doc

def run_covid19_matcher(text):

    """

    The function expans the base spacy nlp entity model with custom covid-19 based entity models. 

    It does so while removeing any internal entity span conflicts within spacy

    

    :param text (string): Input text block to process

    :return: spacy nlp object with overwriten entity list

    

    """

    # clean the document

    text_clean = clean(text)

    

    # process using spacy

    text_nlp = nlp(text_clean)

    

    # extract spacy's default entities 

    nlp_ent = list(text_nlp.ents)

    

    # run the custome covid-19 entity matcher, extract spans

    matches = matcher(text_nlp)

    match_enties = []

    for match_id, start, end in matches:    

        #remove any enties which overlap with the new entity span

        nlp_ent = [e for e in nlp_ent if len(set(range(start,end)).intersection(range(e.start,e.end ))) == 0 ]

        

        #now that conflics are removed, add new covid-19 entity 

        nlp_ent += [Span(text_nlp, start, end, label=match_id)]

        

    # reset the entity list

    text_nlp.ents = nlp_ent

    

    return text_nlp
# Load in disease names associated to Cardiovascular Disease & Respiratory Disease 

dir_input_data = '/kaggle/input/diseases/'



# Load in Cardiovascular Diseases names

df = pd.read_csv(os.path.join(dir_input_data, "CardiovascularDiseases.csv"),header=None)

cardiovascular_diseases = df[0].to_list()



# Load in Respiratory Diseases names

df = pd.read_csv(os.path.join(dir_input_data, "RespiratoryTractDiseases .csv"),header=None)

respiratory_diseases = df[0].to_list()



# add the main cardiovascular and respiratory diseases risks - sourced from texasheart.org and WHO

common_risks = ["High Blood Pressure", "Hypertension", "High Blood Cholesterol",

                "Diabetes", "Obesity", "Overweight", "Smoking", "Inactivity", 

                "Heredity", "Male", "Stress", "Hormones", "Birth Control Pill",

                "Alcohol", "Age", "Unhealthy Diet", "Tobacco Use", "Air Pollution",

                "Pollution","Allergens", "Hand to Mouth"]



covid_19_names = ["covid-19", "2019 novel coronavirus disease", "2019 novel coronavirus infection", 

                  "2019-nCoV disease", "2019-nCoV infection", "COVID-19 pandemic",

                  "COVID-19 virus disease", "COVID-19 virus infection", "COVID19",

                  "SARS-CoV-2 infection", "coronavirus disease 2019", "coronavirus disease-19"]



# clean each set of risk factors

cardiovascular_diseases = [clean(r) for r in cardiovascular_diseases]

respiratory_diseases = [clean(r) for r in respiratory_diseases]

common_risks = [clean(r) for r in common_risks]

covid_19_names = [clean(r) for r in covid_19_names]
# Create a PhraseMatcher for custom covid-19 themes 

matcher = PhraseMatcher(nlp.vocab)

[matcher.add("Cardiovascular Disease", None, nlp(r.lower())) for r in cardiovascular_diseases]

[matcher.add("Respiratory Disease", None, nlp(r.lower())) for r in respiratory_diseases]

[matcher.add("Risk Factor", None, nlp(r.lower())) for r in common_risks]

[matcher.add("COVID-19", None, nlp(r.lower())) for r in covid_19_names]



index_lookup = {}

index_lookup["Cardiovascular Disease"] = matcher.vocab.strings["Cardiovascular Disease"]

index_lookup["Respiratory Disease"] = matcher.vocab.strings["Respiratory Disease"]

index_lookup["Risk Factor"] = matcher.vocab.strings["Risk Factor"]
dir_input_data = '/kaggle/input/load-and-process-data-abstracts'



files = []

import os

for dirname, _, filenames in os.walk(dir_input_data):

        filenames = [names for names in filenames if '.pickle' in names]

        if filenames != []:

            files.append({'dirpath':dirname, 'filenames':filenames})
# Test using a single corpus

directory = files[0]["dirpath"]

filenames = files[0]["filenames"]



corpus_documents = {}

document_linkage_df = []

risk_factor_publications = collections.defaultdict(list)

file = 'biorxiv_medrxiv_biorxiv_medrxiv_pdf_json.pickle'



# open 

doc_list = []  

with open(os.path.join(directory, file),"rb") as f:

    doc_list = pickle.load(f)



#extract all document abstract

[doc.combine_abstract() for doc in doc_list if doc]

doc_abstracts = [doc.abstract[0].text for doc in doc_list]



# run a custom covid-19 spacy entity model using spacy

text_nlp = [run_covid19_matcher(text) for text in doc_abstracts]
# Test over a set of N files 

for i in range(0,50):

    ents = text_nlp[i].ents

    for key in index_lookup.keys():

        if any([True for e in ents if e.label == index_lookup[key]]):

            risk_factor_publications[key].append(i)



for i in risk_factor_publications["Cardiovascular Disease"]:

    displacy.render(text_nlp[i], style="ent")
for i in risk_factor_publications["Respiratory Disease"]:

    displacy.render(text_nlp[i], style="ent")
for i in risk_factor_publications["Risk Factor"]:

    displacy.render(text_nlp[i], style="ent")
directory = files[0]["dirpath"]

filenames = files[0]["filenames"]



corpus_documents = {}

document_linkage_df = []

risk_factor_publications = collections.defaultdict(list)



doc_count = 0

for file in filenames:        

  

    with open(os.path.join(directory, file),"rb") as f:

        doc_list = pickle.load(f)

        

        for doc in doc_list:

               

            if doc:

                

                title = copy.deepcopy(doc.title)

                if title not in corpus_documents.keys():

                    doc_count += 1

                    corpus_documents.update({title: doc_count})

                

                doc_id = corpus_documents[title]

        

                ref_titles = []

                for b in doc.bib:

                    title = copy.deepcopy(b.title)

                    if title not in corpus_documents.keys():

                        doc_count += 1

                        corpus_documents.update({title: doc_count})

                                                

                    ref_titles.append(corpus_documents[title])

                

                df = pd.DataFrame({"object": [doc_id for r in ref_titles],

                                   'relation': ["has reference" for r in ref_titles],

                                   'subject': ref_titles})

                

                # segment documents based on risk factors

                if doc.abstract:

                    doc.combine_abstract()

                

                    # parse using spacy matcher

                    text_nlp = run_covid19_matcher(doc.abstract[0].text)

                    ents = text_nlp.ents

                    for key in index_lookup.keys():

                        if any([True for e in ents if e.label == index_lookup[key]]):

                            risk_factor_publications[key].append(doc_id)

                        

                document_linkage_df.append(df)

                doc_count += 1

                

document_linkage_df = pd.concat(document_linkage_df)
# save corpus_documents for cross reference later

corpus_file = 'corpus_documents_lookup.json'

with open(corpus_file, 'w', encoding='utf-8') as f:

    json.dump(corpus_documents, f, ensure_ascii=False, indent=4)

    

#del after save to save memory 

del corpus_documents
def create_kg(pairs):

    k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',edge_attr = ['relation'],

            create_using=nx.DiGraph())

    return k_graph
def get_corpus_labels(corpus_dir, index):

    with open(corpus_dir) as file:

                corpus = json.load(file)

    return {value:key for key, value in corpus.items() if value in index}        
def plot_graph(G, font = 12):

    """

    This function creates a graph from selected nodes and 

    plots the nodes, relationship and relative improtance.

    

    G: networkx graph object.

    nodes: a list of nodes to create a graph object.

    font: a variable to modify the size of the text and nodes in the graph. default at 12 for large sub graphs.

    """

    

    node_deg = nx.degree(G)

    layout = nx.spring_layout(G, k=0.25, iterations=20)

    plt.figure(num=None, figsize=(120, 90), dpi=80)

    nx.draw_networkx(

        G,

        node_size=[int(deg[1]) * 500*(font/12) for deg in node_deg],

        arrowsize=40,

        linewidths=2.5,

        pos=layout,

        edge_color='red',

        edgecolors='black',

        node_color='white',

        font_size = font, 

        )



    subject = []

    obj = []

    relation = []

    tasks = []            

    for element in list(G.edges()):

        subject.append(element[0])

        obj.append(element[1])

        relation.append(G.get_edge_data(element[0],element[1])['relation'])

        

    labels = dict(zip(list(zip(subject, obj)),relation))

    nx.draw_networkx_edge_labels(G_sub, pos=layout, edge_labels=labels,

                                     font_color='black', font_size=font)

    plt.axis('off')

    plt.show()
G = create_kg(document_linkage_df)

print(nx.info(G))
top_nodes_N = 1000

node_rank = nx.degree_centrality(G)

node_rank_sorted = {k: v for k, v in sorted(node_rank.items(), key=lambda item: item[1],reverse=True)}

top_nodes = [k for k in node_rank_sorted.keys()][1:top_nodes_N]



G_sub = G.subgraph(top_nodes)

mapping = get_corpus_labels(corpus_file, list(G_sub.nodes))

G_sub = nx.relabel_nodes(G_sub, mapping)



plot_graph(G_sub)
corpus_documents = document_linkage_df['object'].drop_duplicates().tolist()

pr = nx.pagerank(G)

pr_df = pd.DataFrame({'Publications':list(pr.keys()), 'Ranking':list(pr.values())})

pr_sub_df = pr_df[pr_df.Publications.isin(corpus_documents)]
pr_df.to_csv('Page_Rank.csv', index=False)

pr_sub_df.to_csv('Page_Rank_Sub.csv', index=False)
for k,v in risk_factor_publications.items():

    print(k, len(risk_factor_publications[k]))
top_nodes = [v_sub for k,v in risk_factor_publications.items() for v_sub in v]



G_sub = G.subgraph(top_nodes)

mapping = get_corpus_labels(corpus_file, list(G_sub.nodes))

G_sub = nx.relabel_nodes(G_sub, mapping)

G_sub.remove_node('')



plot_graph(G_sub)