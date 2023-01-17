!pip install qwikidata

!pip install "tensorflow-gpu>=1.14.0,<2.0" # ampligraph only works with tensorflow 1.

!pip install ampligraph
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pickle
KG_companies = ["Facebook", "Amazon", "Apple Inc.", "Netflix", "Google"]
from qwikidata.entity import WikidataItem, WikidataLexeme, WikidataProperty

from qwikidata.linked_data_interface import get_entity_dict_from_api

from tqdm.notebook import tqdm
def get_triples_from_wikidata(companies_list, predicate_list):

    """

    Inputs: companies_list - a list of companies, identified by their Q id.

            predicate_list - a list of predicates, identified by their P id.

    Outputs: (company, predicate, object) triples.

            E.g. (Tesla, CEO, Elon Musk)

    """

    subjects, predicates, objects = [], [], []

    for Q_id in tqdm(companies_list):

        Q_company = WikidataItem(get_entity_dict_from_api(Q_id))

        for predicate in predicate_list:

            for claim in Q_company.get_claim_group(predicate):

                object_id = claim.mainsnak.datavalue.value["id"]

                object_entity = WikidataItem(get_entity_dict_from_api(object_id))



                subjects.append(Q_company.get_label())



                predicate_property = WikidataProperty(get_entity_dict_from_api(predicate))

                predicates.append(predicate_property.get_label()) 



                objects.append(object_entity.get_label())



    return subjects, predicates, objects  
companies_list = ["Q355", "Q3884", "Q312", "Q907311", "Q95"]

predicate_list = ["P31", "P17", "P361", "P452", "P112", "P169", "P463", "P355", "P1830", "P1056"]
subjects, predicates, objects = get_triples_from_wikidata(companies_list, predicate_list)

wiki_triples_df = pd.DataFrame({"subject": subjects, "predicate": predicates, "object": objects})
wiki_triples_df.sample(10)
wiki_triples = []

for index, row in wiki_triples_df.iterrows():

    wiki_triples.append((row.subject, row.predicate, row.object))
import networkx as nx

def create_graph(col):

    graph = nx.from_pandas_edgelist(wiki_triples_df[wiki_triples_df.subject == str(col)], "subject", "object", edge_attr=True, create_using=nx.MultiDiGraph())

    plt.figure(figsize=(12,12))

    pos = nx.spring_layout(graph)

    nx.draw(graph, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)

    nx.draw_networkx_edge_labels(graph, pos=pos)

    plt.show()
for comp in KG_companies:

    create_graph(comp)