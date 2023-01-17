import os

import logging

import requests

import numpy as np

import pandas as pd

from requests.exceptions import HTTPError, ConnectionError

import json



logging.getLogger()

logging.basicConfig(level=logging.DEBUG)



'''

Generate paras, image text, image/text OCR text

'''

class DataPuller():

    def __init__(self, input_dir, metadata_path):

        self.input_dir = input_dir

        self.metadata_path = metadata_path

        self.paper_details_df = None



    def load(self):

        '''

        Loads all the data and

        flattens the data structure by joining metadata with paragraphs

        Drops papers without full text. # TODO

        '''        

        metadata = pd.read_csv(self.metadata_path)

        

        # read paper details

        paper_texts = []

        for (dirpath, dirnames, filenames) in os.walk(self.input_dir):

            for file in filenames:

                if(file.endswith(".json")):

                    with open(os.path.join(dirpath, file), 'r') as f:

                        paper_texts.append(json.loads(f.read()))

        

        # merge paper details with metadata

        paper_details_df = pd.DataFrame.from_records(paper_texts)

        paper_details_df = paper_details_df.drop(labels=['abstract'], axis=1)

        logging.info("Loaded paper details for {} papers".format(len(paper_details_df)))

        metadata['paper_id'] = metadata['sha']

        paper_details_df = paper_details_df.merge(metadata, on='paper_id', how='left')

        paper_details_df = paper_details_df[paper_details_df['title'].notna()]

        self.paper_details_df = paper_details_df.reset_index()

        logging.info("Successfully merged paper details with metadata . Total records after merging {}".format(len(self.paper_details_df)))



        records = []

        count = 0  

        # Get all abstract first

        for i in metadata.index:            

            row = metadata.iloc[i]

            records.append(self.get_record(row=row, 

                                        idx=count, 

                                        doc=row['abstract'],

                                        doc_type='abstract'))

        

            count += 1

        logging.info("Loaded abstract for {} papers".format(count))

        for i in self.paper_details_df.index:            

            row = self.paper_details_df.iloc[i]

            

            for para in row['body_text']:

                records.append(self.get_record(row=row, 

                                                idx=count, 

                                                doc=para['text'],

                                                doc_type='para'))

                count += 1    



            for k, fig in row['ref_entries'].items():

                records.append(self.get_record(row=row, 

                                                idx=count, 

                                                doc=fig['text'],

                                                doc_type='caption'))

                count += 1



        docs_df = pd.DataFrame.from_records(records)

        logging.info("Data puller total records : {}".format(len(docs_df)))

        docs_df = docs_df.drop_duplicates(subset='id')

        logging.info("Data puller total records after dropping dupes : {}".format(len(docs_df)))

        return docs_df



    def get_record(self, row, idx, doc, doc_type):

        return {

            'id': str(row['paper_id']) + '_' + str(idx),

            'paper_id': row['paper_id'],

            'paper_title': row['title'],

            'display_text': doc,

            'text': str(row['title']) + ' . ' + str(doc),

            'doc_type': doc_type,

            'link': row['url'],

            'date': row['publish_time'],

            'authors': row['authors'],

            'journal': row['journal']

        }
# Format:

# {

# 'id': <str>,

# 'paper_id': <str>,

# 'paper_title': <str>,

# 'display_text': <str>,

# 'text': <str>,

# 'doc_type': <str>, #['abstract', 'para', 'caption']

# 'link': <str>,

# 'date': <str>,

# 'authors': <str>,

# 'journal': <str>

# }
from urllib.request import urlopen

from IPython.display import IFrame

from urllib.parse import quote



def query_to_url(query):

    return "https://sfr-med.com/search?q=" + quote(query)





def render_url(url):

    html = urlopen(url).read()

    html = html.decode("utf-8")



    # Eliminate header and search bar

    html = html.split("<div class=\"slds-col slds-large-size_2-of-12\">")[0] + html.split("</form>")[1]

    

    filename = "./tmp.html"

    open(filename, "w").write(html)

    return IFrame(src=filename, width=1300, height=600)





def display_query(query):

    try:

        url = query_to_url(query)

        display(render_url(url))

    except:

        print("Error rendering...")



        

queries = """

- Effectiveness of drugs being developed and tried to treat COVID-19 patients.

- Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.

- Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.

- Exploration of use of best animal models and their predictive value for a human vaccine.

- Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.

- Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.

- Efforts targeted at a universal coronavirus vaccine.

- Efforts to develop animal models and standardize challenge studies

- Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers

- Approaches to evaluate risk for enhanced disease after vaccination

- Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]

""".strip().split("\n")
for query in queries[:1]:

    query = query.strip()

    print(query)

    display_query(query)