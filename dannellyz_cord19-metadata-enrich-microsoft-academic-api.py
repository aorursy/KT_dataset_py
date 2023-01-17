import pandas as pd

import numpy as np



base_file_path = "/kaggle/input/CORD-19-research-challenge/"

enrich_file_path = "/kaggle/input/cord19-metadata-enrichment/"



#Can either go ahead with the enriched metdata or the base

metadata = pd.read_csv(base_file_path + "metadata.csv", index_col="cord_uid")



#Grab those that have DOIs as its the selector for the API

meta_has_doi = metadata[metadata.doi.notnull()]

meta_has_doi.shape
#Get the API Features of the Microsoft Academic API

micsoft_api_feats = pd.read_csv(enrich_file_path + "microsoft_api_features.csv")

micsoft_api_feats
#Get a list of the attributes to add to the search strings

micro_attrib_list = list(micsoft_api_feats.Attribute)



#Drop those you are not interested in

#E: Extended metadata has been depreciated

micro_attrib_list.remove("E")



#Take the attributes and format the API search string

search_string = ",".join(list(micsoft_api_feats.Attribute))

search_string
#Load API key from Kaggle Secret

#Can add-on secret at top of notebook

#Can apply for a free API key here: https://msr-apis.portal.azure-api.net/products



from kaggle_secrets import UserSecretsClient

secret_label = "microsoft_api_key"

secret_value = UserSecretsClient().get_secret(secret_label)

api_key = secret_value
import requests

import json

import concurrent.futures

import itertools

from tqdm.notebook import tqdm



#Base API Url

base_url = "https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate"



#Format the DOIs for query

#Each should read "DOI = '[DOI]'"

doi_list = list(meta_has_doi.doi)

query_list = ["DOI = '" + doi +"'" for doi in doi_list]



#Code to split the DOI list into sublsists len = 10

def chunks(lst, n):

    """Yield successive n-sized chunks from lst."""

    for i in range(0, len(lst), n):

        yield lst[i:i + n]      

split_querys = list(chunks(query_list,10))



#DOI Query is all DOIs joined enclosed with OR(...)

def make_doi_query(dois):

    prefix = "OR("

    dois = ",".join(dois)

    suffix = ")"

    return prefix + dois + suffix



api_responses = []

headers = {'Ocp-Apim-Subscription-Key': api_key}



def api_call(query_list):

    params = {"expr": make_doi_query(query_list), "attributes": search_string}

    r = requests.get(base_url, headers=headers, params=params)

    if r.ok:

        return r.json()["entities"]

    else:

        return 



with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

    map_obj = tqdm(executor.map(api_call, split_querys), total=len(split_querys))

    microsoft_api_df = pd.DataFrame(list(itertools.chain(*map_obj)))
#Available Data

val_counts = pd.DataFrame(microsoft_api_df.notna().sum(axis=0), columns=["present_count"])

val_counts["pct_avail_microsoft"] = val_counts["present_count"] / len(microsoft_api_df)

val_counts["pct_avail_all_data"] = val_counts["present_count"] / len(metadata)

val_counts[val_counts["present_count"] > 0].sort_values(by="present_count", ascending=False)
#Save to csv

microsoft_api_df.to_csv("microsoft_academic_metadata.csv")