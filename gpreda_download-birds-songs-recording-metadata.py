import numpy as np

import pandas as pd

import json

import requests

from tqdm import tqdm
def get_first_page_per_country(country):

    """

    @country: the country for which we download metadata content 

    @returns: the content downloaded

    """

    api_search = f"https://www.xeno-canto.org/api/2/recordings?query=cnt:{country}"

    response = requests.get(api_search)

    if response.status_code == 200:

        response_payload = json.loads(response.content)

        return response_payload

    else:

        return None



def get_page_per_country(country, page):

    """

    @country: the country for which we download metadata content 

    @page: the current page to be downloaded

    @returns: the content downloaded

    """

    api_search = f"https://www.xeno-canto.org/api/2/recordings?query=cnt:{country}&page={page}"

    response = requests.get(api_search)

    if response.status_code == 200:

        response_payload = json.loads(response.content)

        return response_payload

    else:

        return None



def inspect_json(json_data):

    """

    @json_data: json data to be inspected

    """

    print(f"recordings: {json_data['numRecordings']}")

    print(f"species: {json_data['numSpecies']}")

    print(f"page: {json_data['page']}")

    print(f"number pages: {json_data['numPages']}")



def get_recordings(payload):

    """

    @payload: json data from which we extract the bird recordings metadata collection

    @returns: birds recordings metadata collection

    """

    return payload["recordings"]



def download_suite_from_country(country, country_initial_payload):

    """

    @country: the country for which we download metadata content 

    @country_initial_payload: the initial downloaded payload for the country (1st page). We download all the other pages.

    @returns: the content recordings (all pages, including the original one)

    """

    pages = country_initial_payload["numPages"]

    

    all_recordings = []

    all_recordings = all_recordings + get_recordings(country_initial_payload)

    for page in tqdm(range(2,pages+1)):

        payload = get_page_per_country(country, page)

        recordings = get_recordings(payload)

        all_recordings = all_recordings + recordings

    

    return all_recordings
def download_save_all_meta_for_country(country):

    # download first batch. From here we extract the number of pages

    birds = get_first_page_per_country(country)

    # let's inspect the first batch

    inspect_json(birds)

    print(f"recordings in first batch: {len(get_recordings(birds))}")

    # download entire suite (all pages)

    suite = download_suite_from_country(country, birds)

    # convert the collection in a dataFrame

    data_df = pd.DataFrame.from_records(suite)

    # export the dataframe as a csv

    data_df.to_csv(f"birds_{country}.csv", index=False)

    print(f"suite length: {data_df.shape[0]}")

    return data_df
data_df = download_save_all_meta_for_country('france')
pd.set_option('max_columns', 30)

pd.set_option('max_colwidth', 100)

data_df.head()
data_df = download_save_all_meta_for_country('romania')
data_df.head()
data_df = download_save_all_meta_for_country('bulgaria')
data_df.head()
data_df = download_save_all_meta_for_country('italy')
data_df.head()
data_df = download_save_all_meta_for_country('india')
data_df.head()
data_df = download_save_all_meta_for_country('brazil')
data_df.head()