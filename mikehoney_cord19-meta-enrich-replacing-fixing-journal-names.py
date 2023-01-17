import pandas as pd

import numpy as np



base_file_path = "/kaggle/input/CORD-19-research-challenge/"

enrich_file_path = "/kaggle/input/cord19-metadata-enrichment/"



#Can either go ahead with the enriched metdata or the base

metadata = pd.read_csv(base_file_path + "metadata.csv", index_col="cord_uid")

journal_rankings = pd.read_csv(enrich_file_path + "scimago_journal_rankings.csv")
import requests

from bs4 import BeautifulSoup

import concurrent.futures

from tqdm.notebook import tqdm



def get_journal_name(search):

    if pd.notnull(search):

        #repalce spaces with +

        prepared = search.replace(" ", "+")

        r = requests.get("https://www.scimagojr.com/journalsearch.php?q=" + prepared)

        soup = BeautifulSoup(r.text, 'html.parser')

        #Find first and return

        for_return = soup.find("span", {"class": "jrnlname"})

        if for_return:

            return search.lower(),for_return.text.lower(),

        else:

            return None

    else:

        return None



#Get list of all journals not found in the Scimago Rankings but in the Metadata

missing_ranking = list(set(metadata["journal"]) - set(journal_rankings["Title"]))

jrnl_names_dict = {}



#Use threading to speed up process

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

    map_obj = tqdm(executor.map(get_journal_name, missing_ranking), total=len(missing_ranking))

    jrnl_names_dict = {item[0]:item[1] for item in map_obj if item}

pd.DataFrame.from_dict(jrnl_names_dict, orient="index").to_csv("journal_abrv_replace.csv")
#Send both journal columns to lower in case of odd caps

metadata["journal"] = metadata["journal"].str.lower()

journal_rankings["Title"] = journal_rankings["Title"].str.lower()



#Merge before request

paper_significance = metadata.merge(journal_rankings[["Title","SJR"]], left_on="journal", right_on="Title", how="left")

print(paper_significance.notnull().groupby(["journal", "SJR"]).size())
paper_significance[paper_significance.journal.notnull() & paper_significance.SJR.isnull()].journal.value_counts()
#Replace and rerun statistics

metadata["journal"] = metadata["journal"].replace(jrnl_names_dict)

paper_significance = metadata.merge(journal_rankings[["Title","SJR"]], left_on="journal", right_on="Title", how="left")

print(paper_significance.notnull().groupby(["journal", "SJR"]).size())

print(paper_significance[paper_significance.journal.notnull() & paper_significance.SJR.isnull()].journal.value_counts())
#Load from Enrichment Dataset example

jrnl_names = pd.read_csv(enrich_file_path + "journal_abrv_replace.csv", names=["metadata_name", "sjr_name"])

jrnl_dict = dict(zip(jrnl_names.metadata_name, jrnl_names.sjr_name))

len(jrnl_dict)