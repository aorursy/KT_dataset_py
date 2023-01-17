import os
import urllib.request
import tarfile

from whoosh import fields, index
from whoosh.analysis import StemmingAnalyzer
import os.path
import csv

from whoosh.qparser import QueryParser
from whoosh.qparser import MultifieldParser
from whoosh import scoring
from whoosh.index import open_dir
import sys
from whoosh import qparser

import pandas as pd
import json
import numpy as np
#Download and unpack the collection
def getData():
    urls = ['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/comm_use_subset.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/noncomm_use_subset.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/custom_license.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/biorxiv_medrxiv.tar.gz']

    # Create data directory
    try:
        os.mkdir('./data')
        print('Directory created')
    except FileExistsError:
        print('Directory already exists')

    #Download all files
    for i in range(len(urls)):
        urllib.request.urlretrieve(urls[i], './data/file'+str(i)+'.tar.gz')
        print('Downloaded file '+str(i+1)+'/'+str(len(urls)))
        tar = tarfile.open('./data/file'+str(i)+'.tar.gz')
        tar.extractall('./data')
        tar.close()
        print('Extracted file '+str(i+1)+'/'+str(len(urls)))
        os.remove('./data/file'+str(i)+'.tar.gz')
getData()
#Iterate through the collection and extract key information from each article (Task 1)
def extract():
    #Iterate through all files in the data directory
    count = 0
    set = None
    # dataCol = ['paper_id','abstract', 'title', 'authors', 'body_text']
    dataCol = ['paper_id','abstract', 'title', 'authors']
    newDF = pd.DataFrame(columns=dataCol)

    for subdir, dirs, files in os.walk('./data'):
        for file in files:
            with open(os.path.join(subdir, file)) as f:
                #TODO: Extract, compute, infer any information you like
                data = json.load(f)

                abstract = ''
                for i in data['abstract']:
                    abstract += i['text']
                    #abstract += " "
                #abstract = process_document(abstract)

                str = " "
                #abstract = str.join(abstract)
#                 print(abstract)

                title = data['metadata']['title']
                #title = process_document(title)
                str = " "
                #title = str.join(title)

                authors = ''
                for i in data['metadata']['authors']:
                    authors += i['first']
                    authors += " "
                    if len(i['middle']) != 0:
                        authors += i['middle'][0]
                        authors += " "
                    authors += i['last']
                    authors += " "
#                 print(authors)
                #authors = process_document(authors)
                str = " "
                #authors= str.join(authors)

                # body_text = ''
                # for i in data['body_text']:
                #     body_text += i['text']
                #     body_text += " "
                # body_text = process_document(body_text)

                paper_id = data['paper_id']

                l = len(newDF)
                # newDF.loc[l] = [paper_id,abstract, title, authors, body_text]
                newDF.loc[l] = [paper_id,abstract, title, authors]
    #Create csv with the corpus (only abstact, id, title, authors. NOT body)
    newDF.to_csv('not_parsed.csv', header = False, index = False)
    print(newDF.head)
    return newDF
#run script
extract()
  
#Organize the collection (Task 2)
def organize():
    #TODO: Organize the collection
    # This list associates a name with each position in a row
    columns = ["id", "abstract", "title", "authors"]

    stem_ana = StemmingAnalyzer()
    schema = fields.Schema(abstract=fields.TEXT(analyzer=stem_ana),
                            id=fields.ID(stored=True),
                            title=fields.TEXT(analyzer=stem_ana, stored=True),
                            authors=fields.TEXT)

    # Create the Whoosh index
    indexname = "index"
    if not os.path.exists(indexname):
       os.mkdir(indexname)
    ix = index.create_in(indexname, schema)

    # Open a writer for the index
    with ix.writer() as writer:
       # Open the CSV file
        with open("not_parsed.csv", "rt") as csvfile:
         # Create a csv reader object for the file
            csvreader = csv.reader(csvfile)

         # Read each row in the file
            for row in csvreader:

           # Create a dictionary to hold the document values for this row
                doc = {}

           # Read the values for the row enumerated like
           # (0, "name"), (1, "quantity"), etc.
                for colnum, value in enumerate(row):

             # Get the field name from the "columns" list
                    fieldname = columns[colnum]

             # Put the value in the dictionary
                    doc[fieldname] = value

           # Pass the dictionary to the add_document method
                writer.add_document(**doc)
organize()
def retrieve(queries):
    #TODO: perform information retrieval

    #TODO: For each query, produce an ordered list of 100 results from most relevant to least relevant and store each list in the overall results object (a list of lists).
    results_list = [] 
    for q in queries:
        ix = open_dir("index")

        # query_str is query string
        query_str = q
        # Top 'n' documents as result
        topN = int(5)

        with ix.searcher() as searcher:
            query = MultifieldParser(["title","abstract"], ix.schema, group=qparser.OrGroup).parse(query_str)
            results = searcher.search(query,limit=topN)
            results_id = [results[i]['title'] for i in range(topN)]
            results_list.append(results_id)
            #for i in range(topN):
                #print(results[i]['id'])#, str(resun
                
    #Output results
    for query in range(len(results_list)):
        for rank in range(len(results_list[query])):
            print(str(rank+1)+'\t'+str(results_list[query][rank]))


queries = ["coronavirus origins"]
retrieve(queries)

queries = ["coronavirus genetics"]
retrieve(queries)
queries = ["coronavirus evolution"]
retrieve(queries)