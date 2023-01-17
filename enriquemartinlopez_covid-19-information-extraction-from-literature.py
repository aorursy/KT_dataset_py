import pandas as pd



import json

from tqdm.notebook import tqdm



%matplotlib inline



%load_ext autoreload

%autoreload 2
# PARAMETERS



mode = 0 # 0 to run in kaggle, 1 to run locally



corpus_paths = ['/kaggle/input/CORD-19-research-challenge/', '../data/CORD-19-research-challenge/']

results_paths = ['/kaggle/input/covid19-ie-from-literature-precomputed-results/', '../results/']

covid_master_df_filtered_paths = ['/kaggle/input/covid19-ie-from-literature-processed-triplets/', '../results/']



corpus_path = corpus_paths[mode]

results_path = results_paths[mode]

covid_master_df_file = results_path+'covid_master_df.csv/covid_master_df.csv'

covid_master_df_filtered_file = covid_master_df_filtered_paths[mode]+'covid_master_df_filtered.csv'



# SAVING LITERATURE REVIEW TABLES

def save_to_csv(df, filename): 

    foldername = [

        '/kaggle/working/',

        '../results/'

    ]

    foldername = foldername[mode]

    

    df.to_csv(foldername+filename)
# from create_corpus_dataframe import create_dataframes



# df = create_dataframes(corpus_path, results_path)
# Loading pre-computed dataframe

df = pd.read_csv(results_path+'all_articles_content.csv/all_articles_content.csv', index_col=False)

df.sort_index().head()
def create_entity_df_dict(entity_list, corpus_df):

    entity_dfs = {}

    for entity in entity_list:

        entity_dfs[entity] = corpus_df[\

                                       (corpus_df['content'].str.lower().str.contains(entity))|\

                                       (corpus_df['section'].str.lower().str.contains(entity))|\

                                       (corpus_df['title'].str.lower().str.contains(entity))

                                      ].copy()

        print('{}: {} articles found'.format(entity, len(entity_dfs[entity])))

    return entity_dfs



def combine_df_dict(entity_dfs):

    combined_df = pd.DataFrame()

    for df in entity_dfs.values():

        combined_df = pd.concat([combined_df, df])

    combined_df.drop_duplicates(inplace=True)

    print('Total articles: {}'.format(len(combined_df)))

    return combined_df
covid_ent_list = ['coronavirus disease', 'covid-19', 'severe acute respiratory syndrome coronavirus 2', 'sars-cov-2']
# Storing the dataframes for each concept of interest

covid_dfs = create_entity_df_dict(covid_ent_list, df)
# Combining the dataframes

covid_combined_df = combine_df_dict(covid_dfs)

covid_combined_df.sort_index().head()
# from openie import StanfordOpenIE



# def trim(annotations):

#     for sent in annotations['sentences']:

#         del sent['basicDependencies']

#         del sent['enhancedDependencies']

#         del sent['enhancedPlusPlusDependencies']

#     return annotations



# def create_annotations_list(text_list):

#     annotations_list = []

#     with StanfordOpenIE() as client:

#         for text in tqdm(text_list):

#             if isinstance(text, str):

#                 annotations = client.annotate(text, simple_format=False)

#                 annotations = trim(annotations)

#                 annotations_list.append(annotations)

#             elif isinstance(text, list):

#                 body_annots = []

#                 for paragraph in text:

#                     annotations = client.annotate(paragraph, simple_format=False)

#                     annotations = trim(annotations)

#                     body_annots.append(annotations)

#                 annotations_list.append(body_annots)

#             else:

#                 print('Wrong object passed to Information Extractor')

#     return annotations_list





# def create_covid19_triplets(text_list, covid_combined_df, covid_full_triplet_file):



#     annotations_list = create_annotations_list(text_list)

#     print(len(annotations_list))

#     with open(covid_full_triplet_file, 'w') as f:

#         json.dump(annotations_list, f)



#     # Creating a lite version of the dictionary including just triplets as a column for the dataframe

#     annotations_column = []

#     for row in annotations_list:

#         row_ann_list = []

#         for sentence in row['sentences']:

#             for triplet_d in sentence['openie']:

#                 row_ann_list.append({k: triplet_d[k] for k in ('subject', 'relation', 'object')})

#         annotations_column.append(row_ann_list)

#     # Adding to the dataframe

#     covid_combined_df['triplets'] = annotations_column

#     # Exporting

#     covid_combined_df.to_csv(covid_master_df_file)



#     return covid_combined_df





# df = create_covid19_triplets(covid_combined_df['content'], covid_combined_df, covid_full_triplet_file)
# Loading precomputed dataframe including all triplets for each section of the journal articles

covid_master_df = pd.read_csv(covid_master_df_file)

covid_master_df.head()
import covid as cv
# Create `ArticleFilter` instance and showing some relevant information

af = cv.ArticleFilter(covid_master_df_file)

af.info()
# af.removeDuplicateTriplets()

# af.info()

# af.cdf.to_csv(covid_master_df_filtered_file)
# Importing dataframe with filtered triplets

af = cv.ArticleFilter(covid_master_df_filtered_file)

af.info()
af.cdf.sort_index().head()
af.generateTripletsDB()
af.triplets.sort_index().head()
filter = {'section': 'history', 'triplet':'disease|care|public|prevention|transmission|infection|trials'}

af.search(filter, html=True, max_items=7)
filter = {'content': 'replication', 'subject':'naproxen|clarithromycin|minocycline|viral inhibitor'}

st_df = af.search(filter, html=True)
filename = 'Effectiveness of drugs being developed and tried to treat COVID-19 patients.csv'

save_to_csv(st_df,filename)

st_df
filter = {'content': 'ADE' , 'subject':'antibody-dependent enhancement'}

st_df = af.search(filter, html=True)
filename = 'Methods evaluating potential complication of Antibody-Dependent Enhancement in vaccine recipients.csv'

save_to_csv(st_df,filename)

st_df
filter = {'title':'vaccine', 'triplet':'animal|dog|mouse|cat|vamp|bird'}

st_df = af.search(filter, html=True)
filename = 'Exploration of use of best animal models and their predictive value for a human vaccine.csv'

save_to_csv(st_df,filename)

st_df
filter = {'content': 'therapeutic|antiviral', 'subject':'therapeutic'}

st_df = af.search(filter, html=True)
filename = 'Capabilities to discover a therapeutic (not vaccine) for the disease.csv'

save_to_csv(st_df,filename)

st_df
filter = {'content': 'therapeutic|antiviral|distribute|production', 'subject':'new therapeutic|scarce therapeutic'}

st_df = af.search(filter, html=True)
filename = 'Alternative models to aid decision makers in determining how to prioritize and distribute scarce newly proven therapeutics.csv'

save_to_csv(st_df,filename)

st_df
filter = {'content': 'vaccination|vaccine|universal vaccine',\

          'subject':'vaccination|vaccine|universal vaccine'}

st_df = af.search(filter, html=True)
filename = 'Efforts targeted at a universal coronavirus vaccine.csv'

save_to_csv(st_df,filename)

st_df
filter = {'content': 'animal model|challenge study|challenge studies|standardize|standardise',\

          'subject':'animal models|challenge study|challenge studies'}

st_df = af.search(filter, html=True)
filename = 'Efforts to develop animal models and standardize challenge studies.csv'

save_to_csv(st_df,filename)

st_df
filter = {'content': '', 'subject':'healthcare worker|doctor|nurse|front-line|prophylaxis'}

st_df = af.search(filter, html=True)
filename = 'Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers.csv'

save_to_csv(st_df,filename)

st_df
filter = {'content': 'risk after vaccination|enhanced disease', 'subject':'risk|vaccine|enhanced disease'}

st_df = af.search(filter, html=True)
filename = 'Approaches to evaluate risk for enhanced disease after vaccination.csv'

save_to_csv(st_df,filename)

st_df
filter = {'content': 'vaccine', 'subject':'assay'}

st_df = af.search(filter, html=True)
filename = 'Assays to evaluate vaccine immune response and process development for vaccines.csv'

save_to_csv(st_df,filename)

st_df