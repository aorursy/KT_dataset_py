#PyPI has a wraper for the api: https://pypi.org/project/altmetric/

#quite pip install

!pip -q install 'altmetric'

from altmetric import Altmetric

a = Altmetric()
#Load the metadata file that was the output of notebook 1

#https://www.kaggle.com/dannellyz/cord-19-metadata-enrichment-1-x

#Found at URL: https://www.kaggle.com/dannellyz/cord19-metadata-enrichment

import pandas as pd

import numpy as np

base_file_path = "/kaggle/input/cord19-metadata-enrichment/"

metadata_v2 = pd.read_csv(base_file_path + "metadata_v2.csv")



#Split the data in order to run batch API requests

#This is incase the conneciton to the API breaks

metadata_batches = np.array_split(metadata_v2,500)
from tqdm.notebook import tqdm

import glob

import uuid



#Setup the funciton calls to the API

def altmetric_doi(doi):

    return a.doi(doi)

def altmetric_pmid(pmid):

    return a.pmid(pmid)



#Main funciton to query the altmetric service

def altmetric_query(journal):

    function_type_dict = {"doi": altmetric_doi,

                        "pmid": altmetric_pmid}

    #Get query type

    #If doi present start with that

    if pd.notnull(journal.doi):

        query_type = "doi"

        journal_id = str(journal.doi)

    #Try pmid next

    elif pd.notnull(journal.pubmed_id): 

        query_type = "pmid"

        journal_id = str(journal.pubmed_id)

    #Skip if neither id present

    else:

        return {}

    

    #Run query based on ids present

    response = function_type_dict[query_type](journal_id)

    if response:

        response["query_type"] = query_type

        return response

    #If empty fill in the info that is available

    else:

        return {"doi":str(journal.doi), "pmid":str(journal.pubmed_id)}

    

def batch_altmetric(metadata_batch):

    return metadata_batch.apply(altmetric_query, axis=1)



#Only run for two branches jsut to show funcitonality

#Running for all takes about 2 hours

#Already processed data is available in the Cord 19 Metadata Enrichment Dataset

#Found at URL: https://www.kaggle.com/dannellyz/cord19-metadata-enrichment

def make_batches(batch_folder):

    for batch in tqdm(metadata_batches[:2]):

        #Unique File Name

        unique_id = str(uuid.uuid4())

        batch_df = pd.DataFrame(list(batch_altmetric(batch)))

        batch_df.to_csv(batch_folder + "batch-"+unique_id+".csv")



make_batches("altmetric_data")
#Code to load the batches

def load_batches(batch_folder):

    all_batches = glob.glob(batch_folder + "/*.csv")

    batch_df_list = []

    for filename in tqdm_notebook(all_batches):

        batch_df = pd.read_csv(filename, header=0, index_col=0)

        batch_df_list.append(batch_df)

    if len(batch_df_list) == 0:

        return pd.DataFrame()

    return pd.concat(batch_df_list, axis=0, ignore_index=True, sort=False)
#Read Altmetric data

altmetric_metadata = pd.read_csv(base_file_path + "altmetric_metadata.csv", index_col=False)

#Available Data

val_counts = pd.DataFrame(altmetric_metadata.notna().sum(axis=0), columns=["present_count"])

val_counts["pct_avail_altmetric"] = val_counts["present_count"] / len(altmetric_metadata)

val_counts["pct_avail_all_data"] = val_counts["present_count"] / len(metadata_v2)

val_counts[val_counts["present_count"] > 0].sort_values(by="present_count", ascending=False)