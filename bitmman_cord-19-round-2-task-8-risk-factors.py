# Importing Python Packages and Tools

import pandas as pd

import nltk

from tqdm import tqdm

import os

from IPython.display import display

from ipywidgets import widgets

from ipywidgets import interact, interactive



# Reading summary tables

summarytable_dir = '../input/summarytables/'

sum_table = {}

for file in os.listdir(summarytable_dir):

    fname = file.strip('csv').strip('.')

    full_file = summarytable_dir + file    

    sum_table[fname] = pd.read_csv(full_file)

sum_table.keys()
# Displaying summary tables

pd.set_option('display.max_rows', 100)

for fname, df in sum_table.items():

    print(fname,' : ' + str(df.shape[0]) +' rows/row')

    display(df.head())
# !python preprocess_get_ids.py

# !python preprocess_cord_data.py
# !pip install git+https://bitbucket.org/nmonath/befree.git

# !python entities_get_pubtator_annotation.py

# !python entities_post_tiabs_to_pubtator.py

# !python entities_retrieve_tiabs_from_pubtator.py

# !python entities_process_pubtator_annotation.py

# !python entities_additional_annotation.py
# !python data_aggregation.py

# !python data_nodes_relations.py

# !python data_indexing_time.py

# !python data_indexing_word.py
# data pathes

data_path = '/kaggle/input/cord-19-data-with-tagged-named-entities/data' # folder for system data

json_path = '/kaggle/input/cord-19-data-with-tagged-named-entities/data/json_files/json_files' # path of final json files

mapping_pnid = 'mapping_corduid2nid.json' # dictionary mapping cord_uid to numeric id for each paper



index_year = 'index_time_year.json' # dictionary of list of papers for each publish year

index_title = 'index_word_title.json' # dictionary of list of papers for each word in title

index_abstract = 'index_word_abstract.json' # dictionary of list of papers for each word in abstract

word_counts = 'paper_word_counts.json' # word counts by paper

index_table = 'index_word_table.json'

paper_tables = 'paper_tables.json'



entity_lists = 'entity_lists.json' # entity checking lists including disease list, blacklist etc.

entity_nodes = 'entity_nodes.json' # entities dictionary

entity_relations = 'entity_relations.json' # entity relation dictionary



mapping_sents = 'mapping_sents2nid.json' # mapping sent id to numeric id

index_sents = 'index_word_sents.json' # mapping word to a list of numeric sent id

sentences = 'sentences.json' # dictionary of all sentences with unique id

# packages

from utils import *

from mining_search_tool import *

csv_path = 'csv'

if not os.path.exists(csv_path): os.makedirs(csv_path)
# load dataset for search and display

papers = SearchPapers(data_path, json_path, mapping_pnid, index_year,

                      index_title, index_abstract, word_counts, index_table, paper_tables,

                      entity_lists, entity_nodes, entity_relations, index_sents, mapping_sents, sentences)
covid19_names = """covid-19, covid19, covid, sars-cov-2, sars-cov2, sarscov2,

                   novel coronavirus, 2019-ncov, 2019ncov, wuhan coronavirus

                """

papers_covid19 = papers.search_papers(covid19_names, section = None, publish_year = '2020')

print(f"{'Total papers related to COVID-19:':20}{len(papers_covid19):6}")
papers.display_papers(papers_covid19[:1])
# Defining extended keyword quries for risk factors

Age_query = """ age, ages, infant, infancy, child, children, adolescent, adolescents, young, youth, old, olds, elderly, senior, pediatric, middle-age, aging, senescence """

Asthma_query = """ asthma, asthma attack, bronchial asthma, allergy, allergic asthma """

CKD_query = """ ckd, chronic kidney, chronic kidney disease, chronic renal disease """

COPD_query =  """ copd, chronic obstructive pulmonary disease """

Cancer_query = """ malignant neoplastic disease, cancer, tumor, carcinogenesis, melanoma, leukemia, benign, terminal """

CardioCerebrovascular_query = """  cardio cerebrovascular disease, vascular disease, cerebrovascular disease, cardiovascular disease, hypercholesterolemia, CVD """

Cerebrovascular_query = """ cerebrovascular, stroke, ischemic, hemorragic """

ChronicDigestiveDisorder_query = """ chronic digestive,  digestion, absorption, celiac disease, ibs, irretable bowel syndrome """

ChronicLiverDisease_query = """ chronic liver,  chronic liver disease, cirrhosis """

ChronicRespiratoryDisease_query = """ chronic respiratory, chronic respiratory disease """

Dementia_query = """ dementia, alzheimer's disease """

Diabete_query = """ diabetes, diebete, insulin resistance, prediabetes, diabetic, diabetic complications, blood glucose, fasting blood glucose, insulin sensitivity, hyperglycemia """

Drinking_query = """ alcohol, alcohol intake, alcoholic, alcoholic drinks, alcoholic beverage, alcoholic consumption, intoxicant, inebriant, binge drinking"""

Endocrine_query = """ hormone, endocrine, endocrine gland, endocrine, endocrinal, endocrinal disorder """

HeartDisease_query = """ heart, heart disease, chd,  coronary heart disease, arrhythmia, atherosclerosis, ischemia, angina """

HeartFailure_query = """ heart failure"""

Hispani_query = """ spanish american, hispanic american,hispanic, latino """

Hypertension_query = """high blood pressure, hypertension, metabolic syndrome, blood pressure """

Immune_query = """ immune system, immune system disorder, autoimmune disease, autoimmne thyroiditis, inflammation, inflammatory, gout, arthritis """

Male_query = """ male, man, gender, female, sex """

Obese_query =  """ bmi, heavy, obese, obesity, body mass index, fat, overweight, abdominal obesity, pear shape, apple shape """

Race_black_query = """ black, white, african american, afro-american, race, caucasian, caucasoid race, negroid race """

RespiratorySystemDisease_query = """ respiratory system, respiratory system disease, pneumonia """

Smoking_query = """ smoke,smoking, tobacco """
# Collecting the query name as a list for future usage.

query_list =  ['Age_query' ,'Asthma_query', 'CKD_query','COPD_query','Cancer_query' ,'CardioCerebrovascular_query' ,'Cerebrovascular_query',

               'ChronicDigestiveDisorder_query','ChronicLiverDisease_query','ChronicRespiratoryDisease_query','Dementia_query','Diabete_query' ,

               'Drinking_query','Endocrine_query','HeartDisease_query','HeartFailure_query','Hispani_query','Hypertension_query',

               'Immune_query' ,'Male_query','Obese_query' ,'Race_black_query','RespiratorySystemDisease_query','Smoking_query']
# Defining extended keyword list for risk factors

syn_list = []

for query_name in query_list:

    syn_name = query_name.strip('query')+'syn'

    syn_list.append(syn_name)    

    globals()[syn_name] = [sent.strip() for sent in globals()[query_name].split(',')]
# Defining extended keyword lists for severity terms and mortality terms

severe_syn = ['mild', 'moderate', 'severe', 'varied', 'critical', 'icu', 'non-icu','positive','positive testing','hospitalization','hospitalized']

fatality_syn = ['fatality','mortality','mortalities','death','deaths','dead','casualty']

combined_syn = severe_syn + fatality_syn
Age_query
# Getting papers related with 'age', searching over titles and abstracts, published in 2020

riskfactor_papers = papers.search_papers(Age_query, section = 'tiabs', publish_year = '2020')

print('There are ' + str(len(riskfactor_papers)) + ' papers in the dataset that are related to age.')

# Get the subset of Covid-19 related papers

riskfactor_papers = list(set(riskfactor_papers) & set(papers_covid19))

print('Among them, ' + str(len(riskfactor_papers)) + ' papers are related to COVID-19.')
ratios = ['OR', 'AOR', 'HR', 'AHR', 'RR', 'RH', 'odds ratio', 'hazard ratio', 'relative ratio','odds']

# The following pattern is looking for pattern: ( ratio keywords + numbers )

extract_pattern = '|'.join([f'\([^()]*\\b{ratio}\\b\s?[=:]?\s?\d+\.\d+.*?\)' for ratio in ratios])
# Search over full text to get papers with pattern ( ratios such as 'OR', 'AOR', 'HR', 'AHR', 'RR', 'RH' )

# and save these to a dictionary

def get_odds(riskfactor_papers):

    or_riskf = {}

    for paper_id in riskfactor_papers:

        paper = papers.get_paper(str(paper_id))

        date = paper['publish_time']

        study = paper['title']['text']

        study_link = paper['url']

        journal = paper['journal']

        doi = paper['doi']

        cord_uid = paper['cord_uid']

        pmc_link = paper['pmcid']

        abstract = paper['abstract']['text']



        if pmc_link != '':

            pmc_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_link}/"



        if abstract != '':

            rate = re.findall(extract_pattern, abstract)

            if rate != []:

                # if any word in combined synonyms list is in abstract, then we store the paper

                if (any(word in nltk.word_tokenize(abstract.lower()) for word in combined_syn)):

                    or_riskf[f'{str(paper_id)}|a|0|0'] = {'Date': date,

                                                          'Study': study,

                                                          'Study Link': study_link,

                                                          'Journal': journal,                                                        

                                                          'DOI': doi,

                                                          'CORD_UID': cord_uid,

                                                          'PMC_Link': pmc_link,

                                                          'Abstract': abstract,                                                         

                                                          'Text':abstract,

                                                          'Ratio':rate}                

#                 print(rate)

        bodytext = paper['body_text']

        if bodytext != []:

            for sec_id, section in enumerate(bodytext):

                for txt_id, text in enumerate(section['text']):

                    rate = re.findall(extract_pattern, text)

                    if rate != []:

                        # if any word in combined synonyms list is in body text, then we store the paper

                         if (any(word in nltk.word_tokenize(abstract.lower()) for word in combined_syn)):

                            or_riskf[f'{str(paper_id)}|b|{sec_id}|{txt_id}'] = {'Date': date,

                                                                                'Study': study,

                                                                                'Study Link': study_link,

                                                                                'Journal': journal,                                                        

                                                                                'DOI': doi,

                                                                                'CORD_UID': cord_uid,

                                                                                'PMC_Link': pmc_link,

                                                                                'Abstract': abstract,                                                         

                                                                                'Text':text,

                                                                                'Ratio':rate}

#                         print(rate)

    return or_riskf
# Transfering dictionary to a dataframe

riskfactor_dic = get_odds(riskfactor_papers)

for k,v in riskfactor_dic.items():

    df = pd.DataFrame.from_dict(riskfactor_dic,orient='index')

df = df.reset_index().rename(columns={'index':'Ratio_ID'})

df.head(2)
keep_pattern = '|'.join([f'(\([^\(\)]*\\b{ratio}\\b([^\(\)]*\(\w+\)[^\(\)]*)*\))|(\([^\(\)]*\\b{ratio}\\b[^\(\)]*\))' for ratio in ratios])
# If there are more than one sentences matching the pattern, then store it in another column (Sentence2)

df['Sentence'] = ''; df['Sentence2'] = ''; 

for idx,text in df.Text.items():

    sents = nltk.sent_tokenize(text)  

    for sent in sents:

        if (re.findall(keep_pattern,sent,re.I)):

            if any(word in nltk.word_tokenize(sent.lower()) for word in Age_syn) and any(word in sent.lower() for word in combined_syn):

                if df['Sentence'][idx] == '':

                    df['Sentence'][idx] += sent + ' '   

                else:

                    df['Sentence2'][idx] += sent + ' '

                    

# Combining column 'Setence' and 'Sentence2'

df1 = df.loc[:,'Ratio_ID':'Sentence']

df2 = df.loc[:,'Ratio_ID':'Sentence2'].drop(columns='Sentence')

df2 = df2.rename(columns = {'Sentence2':'Sentence'})

dfn = pd.concat([df1,df2])

dfn = dfn[dfn['Sentence'] != '']

dfn = dfn.sort_values('Study')

dfn = dfn.drop_duplicates(subset = 'Sentence',ignore_index =True)

dfn.shape
dfn.head(2)
# Here we use df_dic to store the dataframe.

df_dic = {}

for riskfactor_query in query_list:

    riskfactor_name = riskfactor_query.strip('query').strip('_')

    riskfactor_query = globals()[riskfactor_query]

    riskfactor_papers = papers.search_papers(riskfactor_query, section = 'tiabs', publish_year = '2020')

    riskfactor_papers = list(set(riskfactor_papers) & set(papers_covid19))

    print('-------------------------------------------------------------------------------------------')

    print('There are '+str(len(riskfactor_papers)) + ' Covid-19 papers in the dataset related to ' + riskfactor_name )      

    

    # Transfer dictionary to a dataframe

    riskfactor_dic = get_odds(riskfactor_papers)

    for k,v in riskfactor_dic.items():

        df = pd.DataFrame.from_dict(riskfactor_dic,orient='index')

    df = df.reset_index().rename(columns={'index':'Ratio_ID'})

    

    # Adding column 'Sentence' to store the sentences with pattern ( ratio keywords )

    df['Sentence'] = ''; df['Sentence2'] = ''; 

    for idx,text in df.Text.items():

        sents = nltk.sent_tokenize(text)  

        for sent in sents:

            if (re.findall(keep_pattern,sent,re.I)):

                if any(word in nltk.word_tokenize(sent.lower()) for word in Age_syn) and any(word in sent.lower() for word in combined_syn):

                    if df['Sentence'][idx] == '':

                        df['Sentence'][idx] += sent + ' '   

                    else:

                        df['Sentence2'][idx] += sent + ' '

                        

    # Modifying the dataframe                    

    df1 = df.loc[:,'Ratio_ID':'Sentence']

    df2 = df.loc[:,'Ratio_ID':'Sentence2'].drop(columns='Sentence')

    df2 = df2.rename(columns = {'Sentence2':'Sentence'})

    dfn = pd.concat([df1,df2])

    dfn = dfn[dfn['Sentence'] != '']

    dfn = dfn.sort_values('Study')

    dfn = dfn.drop_duplicates(subset = 'Sentence',ignore_index =True)

    

    print('The shape of the dataframe of ' + riskfactor_name + ' is :')

    print(dfn.shape)

    

    # Save dataframes as csv files

#     dfn.to_csv('csv/' + riskfactor_name + '.csv')

    

    # Save dataframes in a dictionary

    df_dic[riskfactor_name] = dfn
# Get dataframe by risk factor name

df_dic['Dementia']
# Study type extend keywords

sys_review = ['systematic review', 'meta-analysis',

              'search: pubmed, pmc, medline, embase, google scholar, pptodate, web of science',

              'searched: pubmed, pmc, medline, embase, google scholar, uptodate, web of science',

              'in: pubmed, pmc, medline, embase, google scholar, uptodate, web of science']

retro_study = ['record review','retrospective', 'observational cohort', 'scoping review']

simulation = ['modelling','model','molecular docking','modeling','immunoinformatics', 'simulation', 'in silico', 'in vitro']
# Regex for extracting sample size

ss_patient = re.compile(r'(\s)([0-9,]+)(\s|\s[^0-9,\.\s]+\s|\s[^0-9,\.\s]+\s[^0-9,\.\s]+\s)(patients|persons|cases|subjects|records)')

ss_review = re.compile(r'(\s)([0-9,]+)(\s|\s[^0-9,\.\s]+\s|\s[^0-9,\.\s]+\s[^0-9,\.\s]+\s)(studies|papers|articles|publications|reports|records)')
df_dic.keys()
s_table = {}

for name,dfs in df_dic.items():

    dfs['Severity of Disease'] = ''

    dfs['Fatality'] = ''

    dfs['Study Type'] = ''

    dfs['Sample Size'] = ''      

    

    for idx,row in dfs.iterrows():  

    

        abstract = row['Abstract']

        sentence = row['Sentence']

        ratio = row['Ratio']

        

        # Filling Study Type

        for pharase in sys_review:

            if(pharase in abstract):

                dfs.loc[idx,'Study Type'] = 'Systematic Review'

        for pharase in retro_study:

            if(pharase in abstract):

                dfs.loc[idx,'Study Type'] = 'Retrospective Study'

        for pharase in simulation:

            if(pharase in abstract):

                dfs.loc[idx,'Study Type'] = 'Simulation'



        #Filling Sample Size

        study_type = dfs.loc[idx,'Study Type']

        sample_size = ''

        if study_type == 'Systematic Review':

            matches = re.findall(ss_review, abstract)

            for match in matches:

                if match[1].isdigit() and int(match[1]) != 2019:

                    dfs.loc[idx, 'Sample Size'] = sample_size + ''.join(match[1:]) + '; '

        elif study_type == 'Retrospective Study' or study_type == 'Other' :

            matches = re.findall(ss_patient, abstract)

            for match in matches:

                if match[1].isdigit() and int(match[1]) != 2019:

                    dfs.loc[idx, 'Sample Size'] = sample_size + ''.join(match[1:]) + '; '



        # Filling Ratios

        if (any(word in nltk.word_tokenize(sentence.lower()) for word in severe_syn )):

            dfs.loc[idx,'Severity of Disease'] = ratio

        elif (any(word in nltk.word_tokenize(sentence.lower()) for word in fatality_syn )):

            dfs.loc[idx,'Fatality'] = ratio



    cols = ['Date','Study','Study Link','Journal','Study Type','Severity of Disease','Fatality','Sample Size','DOI','CORD_UID']

    dfw = dfs[cols]

    s_table[name] = dfw

    display(name + ' : ' + str(dfw.shape[0]) + ' rows/row' )

    display(dfw.head())    
# Saving risk factor dataframes into csv files

for fname, df in s_table.items():

    df.to_csv(f'{fname}.csv')
import json

import csv

risk_factor_json_dir = '../input/riskfactorjson/'
# Storing the files in a list and a dictionary

file_dic = {}

for fname in os.listdir(risk_factor_json_dir):

    filename = fname.strip('json').strip('.')

    full_fname = risk_factor_json_dir + fname    

    files = json.load(open(full_fname, 'rb'))    

    file_dic[filename] = files
# Get the file from the dictionary

file_dic['Age']['0']
# Defining Graph data structure

from collections import defaultdict

class Graph():

    def __init__(self):

        """

        self.edges is a dict of all possible next nodes

        e.g. {'X': ['A', 'B', 'C', 'E'], ...}

        self.weights has all the weights between two nodes,

        with the two nodes as a tuple as the key

        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}

        self.dep sotres the dependency between two nodes,

        with the two nodes as a tuple as the key

        e.g. {('X', 'A'): 'nsubj'}

        """

        self.edges = defaultdict(list)

        self.weights = {}

        self.dep = {}

    

    def add_edge(self, from_node, to_node, weight, dep):

        # Note: assumes edges are bi-directional

        self.edges[from_node].append(to_node)

        self.edges[to_node].append(from_node)

        self.weights[(from_node, to_node)] = weight

        self.weights[(to_node, from_node)] = weight

        self.dep[(from_node, to_node)] = dep

        self.dep[(to_node, from_node)] = dep



# Defining dijsktra to get the shortest path from initial node to target node        

def dijsktra(graph, initial, end):

    # shortest paths is a dict of nodes

    # whose value is a tuple of (previous node, weight)

    shortest_paths = {initial: (None, 0)}

    current_node = initial

    visited = set()

    

    while current_node != end:

        visited.add(current_node)

        destinations = graph.edges[current_node]

        weight_to_current_node = shortest_paths[current_node][1]



        for next_node in destinations:

            weight = graph.weights[(current_node, next_node)] + weight_to_current_node

            if next_node not in shortest_paths:

                shortest_paths[next_node] = (current_node, weight)

            else:

                current_shortest_weight = shortest_paths[next_node][1]

                if current_shortest_weight > weight:

                    shortest_paths[next_node] = (current_node, weight)

        

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}

        if not next_destinations:

            return "Route Not Possible"

        # next node is the destination with the lowest weight

        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    

    # Work back through destinations in shortest path

    path = []

    while current_node is not None:

        path.append(current_node)

        next_node = shortest_paths[current_node][0]

        current_node = next_node

    # Reverse path

    path = path[::-1]

    return path



# Switching the order of a tuple

def switch(x):

    return(x[1],x[0])



# Getting the intersection of two lists

def intersection(lst1, lst2): 

    return list(set(lst1) & set(lst2)) 



# Getting the shortest path from initial node to target node, with the dependecies

def shortest_path(file,from_node,to_node):

    

    print(file['text'])

    edges = file['edges']

    edge_list = []

    for idx,edge in edges.items():

        edge_list.append((edge['target'],edge['source'],1,edge['dep']))

        

    g = Graph()

    for edge in edge_list:

        g.add_edge(*edge)

    

    path = dijsktra(g, from_node, to_node)

    

    tpl = []

    for i in range(len(path) - 1):

        value = tuple(path[i:i+2])

        tpl.append(value)



    new_path = []

    for chunk in tpl:

        if chunk in g.dep.keys():

            if chunk[0] not in new_path:

                new_path.append(chunk[0])

            new_path.append(g.dep[chunk])

            new_path.append(chunk[1])

        elif switch(chunk) in g.dep.keys():

            if switch(chunk)[1] not in new_path:

                new_path.append(switch(chunk)[1])

            new_path.append(g.dep[switch(chunk)])

            new_path.append(switch(chunk)[0])

    

    return new_path
npaths = []

for i in range(len(files)):

    file = files[str(i)]

    sent_list = nltk.word_tokenize(file['text'].lower())

    if intersection(sent_list, Age_syn):

        age_related =intersection(sent_list, Age_syn)[0]

    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')  

#     print(intersection(sent_list, combined_syn))

   

    for severe_related in intersection(sent_list, combined_syn):

        print(age_related,severe_related)

        path = shortest_path(file,age_related,severe_related)        

        print(path)

        print('----------------------')

        npaths.append(path) 
npaths
file_dic.keys()
# short_path dictionary: take file name as a key and list of path as values

# gen_dic dictionary: take touple (from_node, to_node, file index) as a key and path as a value; 

#                     file index is to distinguish those same (from_node, to_node) 

short_path = {} ; gen_dic ={}

for riskfactor_query in query_list:

    

    fname = riskfactor_query.strip('query').strip('_')

    files = file_dic[fname]    

    fname_syn = globals()[fname + '_syn']

    

    npaths = []

    for i in range(len(files)):

        file = files[str(i)]

        sent_list = nltk.word_tokenize(file['text'].lower())

        

        if intersection(sent_list, fname_syn):

            fname_related = intersection(sent_list, fname_syn)[0] 

            

            if (intersection(sent_list, combined_syn)):

                print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*') 

#                 print(intersection(sent_list, combined_syn))

                

                for severe_related in intersection(sent_list, combined_syn):

                    tpl = (fname_related,severe_related, str(i))

                    path = shortest_path(file,fname_related,severe_related) 

                    if path:

                        print(path)

                        npaths.append(path) 

                        gen_dic[tpl] = [path, file['text']]



                            

    short_path[fname] = npaths
short_path['Asthma']
# Saving gen_dic into a csv file

import csv

with open('shortest_path.csv', 'w') as f:

    writer = csv.writer(f)

    writer.writerow(['Sentence','Nodes','Shortest Path'])

    for key,value in gen_dic.items():

        writer.writerow([value[1], key, value[0]])