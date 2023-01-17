#Import pandas and the read in the metadata csv to a Dataframe

import pandas as pd

metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
#Utilizing matplotlib for display

%matplotlib inline

import matplotlib.pyplot as plt



# Show the percentage of each column that is present or not NULL

col_present_pct = metadata.notnull().sum() / len(metadata)



#Bar Plot

col_present_pct.sort_values().plot.bar()
id_col_list = ['doi','pmcid', 'pubmed_id']

def null_ids_graph(df):

    #Group by the various IDs and count their permutations

    id_types_present = df.notnull().groupby(id_col_list).size()

    #Plot with bar chart

    chart = id_types_present.plot.bar()

    for p in chart.patches:

        chart.annotate('{:,}'. format(p.get_height()), (p.get_x() * 1.00, p.get_height() * 1.01))

    return chart

null_ids_graph(metadata)
#URL Lib to query API

from urllib.request import urlopen



#ElementTree to parse XML response

import xml.etree.ElementTree as ET



#Import Tracker

from tqdm.notebook import tqdm

    

def chunks(lst, n):

    """

    Yield successive n-sized chunks from lst.

    src: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks

    

    """

    for i in range(0, len(lst), n):

        yield lst[i:i + n]



def ncbi_api(pmcids):

    #Base string is the API end point

    api_base_string = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids="

    #List of PMCIDS to send to API end point

    ids_string = ",".join(list(pmcids))

    #Call API with query

    api_query = api_base_string + ids_string



    #Get API response which is a list of dictionaries

    with urlopen(api_query) as response:

        response_content = response.read()

    root = ET.fromstring(response_content)

    

    #Return

    return [child.attrib for child in root[1:]]



def get_ncbi_results(file_name, pmcids):

    #Batch the results as API can only take 10 at a time

    batch_pmcids = chunks(pmcids, 10)

    batch_results = []



    #For each batch run against API

    for batch in tqdm(batch_pmcids):

        batch_results.extend(ncbi_api(batch))



    #Collect results into a Dataframe

    ncbi_results = pd.DataFrame(batch_results).drop("requested-id", axis=1)



    #Send dataframe to csv

    ncbi_results.to_csv(file_name)

    return ncbi_results



def load_ncbi_results(file_name):

    ncbi_results = pd.read_csv(file_name, usecols=["doi", "pmcid", "pubmed_id"])

    return ncbi_results



def get_pmcids(metadata):

    has_pmcid_no_doi = metadata[metadata.pmcids.notnull() & metadata.doi.isnull()]



#Get the NCBI Results

file_name = "ncbi_metadata.csv"

has_pmcid_no_doi = metadata[metadata.pmcid.notnull() & metadata.doi.isnull()]

pmcids_list = list(has_pmcid_no_doi.pmcid)



#Get from API

#You must also enable internet in the options to the right

ncbi_results = get_ncbi_results(file_name, pmcids_list)



#Load from Public Data Set

#ncbi_results = load_ncbi_results(file_name)
#Update metadata with new values from NCBI results

metadata_v2 = metadata.copy()

metadata_v2.update(ncbi_results)



#Graph update

id_count_v1 = metadata.notnull().groupby(id_col_list).size()

id_count_v2 = metadata_v2.notnull().groupby(id_col_list).size()

updated_counts = id_count_v2 - id_count_v1

updated_counts
pd.concat([pd.DataFrame(id_count_v1, columns = ["Before"]).T,

           pd.DataFrame(updated_counts, columns = ["After"]).T]).T.plot.bar(stacked=True)
null_ids_graph(metadata_v2)
#Output for future work

metadata_v2.to_csv("metadata_v2.csv")