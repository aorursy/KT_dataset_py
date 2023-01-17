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

- How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures.

- Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).

- Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.

- Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.

- National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).

- Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.

- Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).

- Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.

- Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.

- Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.

- Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.

- Policies and protocols for screening and testing.

- Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.

- Technology roadmap for diagnostics.

- Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.

- New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.

- Coupling genomics and diagnostic testing on a large scale.

- Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.

- Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.

- One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.

""".strip().split("\n")
for query in queries[:1]:

    query = query.strip()

    print(query)

    display_query(query)