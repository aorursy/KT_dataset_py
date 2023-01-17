# Install cord19q project

!pip install git+https://github.com/neuml/cord19q
import csv

import os

import shutil



import pandas as pd



from cord19q.models import Models

from cord19q.query import Query



# Workaround for mdv terminal width issue

os.environ["COLUMNS"] = "80"



def uids():

    # Entry date mapping sha id to date

    uids = {}

    

    # Load in memory date lookup

    with open("../input/cord-19-article-entry-dates/entry-dates.csv", mode="r") as csvfile:

        for row in csv.DictReader(csvfile):

            uids[row["sha"]] = row["cord_uid"]



    return uids



# Copy vectors locally for predictable performance

shutil.copy("../input/cord19-fasttext-vectors/cord19-300d.magnitude", "/tmp")



# Load sha - cord id mapping

idmap = uids()



submission = []

topn = 50



# Load model

embeddings, db = Models.load("../input/trec-covid-search-index/cord19q")

cur = db.cursor()



with open("../input/trec-covid-information-retrieval/topics-rnd3.csv", mode="r") as csvfile:

    for topic in csv.DictReader(csvfile):

        # Run the search

        results = Query.search(embeddings, cur, topic["query"], topn)

        

        # Get results grouped by document

        documents = Query.documents(results, topn)   



        for uid in documents:

            # uid is third element, lookup cord_uid from shas:

            submission.append((topic["topic-id"], idmap[uid]))

            

df = pd.DataFrame(submission, columns=["topic-id", "cord-id"])

df.to_csv("submission.csv", index=False)