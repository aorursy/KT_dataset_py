import os

import urllib.request

import tarfile

import json

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

import numpy as np

from copy import deepcopy

nltk.download('punkt')

nltk.download('stopwords')

import csv





paper_to_citations = {} # dictionary from paper id -> list of paper ids

words_to_docs_to_freqs = {} # dictionary from word -> set(paper)

paper_to_tf = {} # dictionary from paper id -> dictionary from  words -> frequency

page_to_pr = {} # dictionary from paper id -> page rank scores

paper_to_title = {} # dictionary from paper id -> titles



doc_count = 0
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





def reset():

  paper_to_citations.clear()

  words_to_docs_to_freqs.clear()

  paper_to_tf.clear()

  page_to_pr.clear()

  paper_to_title.clear()
#Iterate through the collection and extract key information from each article (Task 1)

def extract(extractAll, pageRank):

    global doc_count

    reset()

    #Iterate through all files in the data directory

    for subdir, dirs, files in os.walk('./data'):

        doc_count = 0

        for dir in dirs:

            print("currently on: ", dir)

            dir_path = './data/' + dir

            dir_files = os.listdir(dir_path)

            counter = 0

            for file in dir_files:

              doc_count += 1

              with open(os.path.join(dir_path, file)) as f:

                file_data = json.load(f) ## dictionary for a single file containing json data

                paper_id = file_data["paper_id"]



                title = file_data["metadata"]["title"]

                authors = file_data["metadata"]["authors"] #list of author dictionaries

                abstract_paragraphs = file_data["abstract"] # list of abstracts

                body_paragraphs = file_data["body_text"] # list of body text paragraphs

                citations = file_data["bib_entries"]



                organize(paper_id, title, authors, abstract_paragraphs, body_paragraphs, citations, extractAll)



    output_to_txt(pageRank)
#Organize the collection (Task 2)

def organize(paper_id, title, authors, abstract_paragraphs, body_paragraphs, citations, extractAll):

                paper_to_citations[paper_id] = [] # paper id -> list of paper ids        

                paper_to_tf[paper_id] = {} # paper id -> (words -> frequency)

                paper_to_title[paper_id] = title



                words_in_doc = 0

                stemmed_title = tokenize(title)



                for word in stemmed_title:



                    if word in paper_to_tf[paper_id]:

                      paper_to_tf[paper_id][word] += 10

                    else:

                      paper_to_tf[paper_id][word] = 10



                    if word not in words_to_docs_to_freqs:

                      words_to_docs_to_freqs[word] = set()

                    if paper_id not in words_to_docs_to_freqs[word]:

                      words_to_docs_to_freqs[word].add(paper_id)



                    words_in_doc += 1



                

                for abstract in abstract_paragraphs:                

                  stemmed = tokenize(abstract["text"]) # str                



                  #add to your dictionaries

                  for word in stemmed:



                    if word in paper_to_tf[paper_id]:

                      paper_to_tf[paper_id][word] += 5

                    else:

                      paper_to_tf[paper_id][word] = 5



                    if word not in words_to_docs_to_freqs:

                      words_to_docs_to_freqs[word] = set()

                    if paper_id not in words_to_docs_to_freqs[word]:

                      words_to_docs_to_freqs[word].add(paper_id)



 

                    words_in_doc += 1

                

                if extractAll:

                  for ph in body_paragraphs:

                    stemmed = tokenize(ph["text"])



                    for word in stemmed:



                      if word in paper_to_tf[paper_id]:

                        paper_to_tf[paper_id][word] += 1

                      else:

                        paper_to_tf[paper_id][word] = 1



                      if word not in words_to_docs_to_freqs:

                        words_to_docs_to_freqs[word] = set()

                      if paper_id not in words_to_docs_to_freqs[word]:

                        words_to_docs_to_freqs[word].add(paper_id)

                      words_in_doc += 1





                #computes TF

                for (word, freq) in paper_to_tf[paper_id].items():

                  paper_to_tf[paper_id][word] = paper_to_tf[paper_id][word]/words_in_doc

                                                



                #fill up our list of citations

                for (bib,cit_info) in citations.items():

                  cit_id = cit_info["ref_id"]

                  paper_to_citations[paper_id].append(cit_id)







def distance(r1, r2):

  point_a = np.array(list(r2.values()))

  point_b = np.array(list(r1.values()))

  distance = np.linalg.norm(point_a - point_b)

  return distance





def page_rank():

  print("Starting page rank ...")

  ids = paper_to_tf.keys()

  num_docs = len(ids)

  epsilon = .00001

  weights = {}

  for page in ids:

    weights[page] = {}

    for page2 in ids:

      num_page2_links = len(paper_to_citations[page2])

      if page in paper_to_citations[page2]:

        weights[page][page2] = epsilon/num_docs + (1 - epsilon)*num_page2_links

      else:

        weights[page][page2] = epsilon/num_docs



   

  real_scores = {k: 0 for k in ids}

  delta = {k: 1/num_docs for k in ids}



  while(distance(real_scores, delta) > epsilon):

    real_scores = deepcopy(delta)

    for page_id in ids: 

      delta[page_id] = 0

      for (pid, score) in delta.items():

        delta[page_id] = delta[page_id] + (weights[page_id][pid] * real_scores[pid])

  



  for k in real_scores:

    page_to_pr[k] = real_scores[k]

 



def output_to_txt():

  with open('pr_index.csv', 'w') as pr_file:

    writer = csv.writer(pr_file)

    for page in page_to_pr:

      writer.writerow([page, page_to_pr[page]])

  with open('tf_index.csv', 'w') as tf_file:

    writer = csv.writer(tf_file)

    for page in paper_to_tf:

      for word in paper_to_tf[page]:

        writer.writerow([page, word, paper_to_tf[page][word]])

  with open('wdf_index.csv', 'w') as wdf_file:

    writer = csv.writer(wdf_file)

    for word in words_to_docs_to_freqs:

      lst = [word]

      for doc in words_to_docs_to_freqs[word]:

        lst.append(doc)

      writer.writerow(lst)

  with open('titles.csv', 'w') as titles_file:

    writer = csv.writer(titles_file)

    for paper in paper_to_title:

      writer.writerow([paper, paper_to_title[paper]])



def tokenize(text):

  tokens = word_tokenize(text.lower())

  stop_words = set(stopwords.words('english'))

  stemmer = PorterStemmer()

  without_stop = [word for word in tokens if word not in stop_words]

  stemmed_text = [stemmer.stem(word) for word in without_stop]

  return stemmed_text
#Answer a set of textual queries (Task 3)

def retrieve(queries, usePageRank=False):

  global doc_count

  r_page_to_pr = {}

  r_paper_to_tf = {}

  r_words_to_docs_to_freqs = {}

  r_paper_to_title = {}



  with open('pr_index.csv', 'r') as idf_file:

    reader = csv.reader(idf_file)

    for row in reader:

      paper = row[0]

      pr = float(row[1])

      r_page_to_pr[paper] = pr

  with open('tf_index.csv', 'r') as tf_file:

    reader = csv.reader(tf_file)

    for row in reader:

      paper = row[0]

      word = row[1]

      tf = float(row[2])

      if paper not in r_paper_to_tf:

        r_paper_to_tf[paper] = {}

      r_paper_to_tf[paper][word] = tf



  with open('wdf_index.csv', 'r') as wdf_file:

      reader = csv.reader(wdf_file)

      for row in reader:

        word = row[0]

        docs = set(row[1:])

        r_words_to_docs_to_freqs[word] = docs



  with open('titles.csv', 'r') as titles_file:

    reader = csv.reader(titles_file)

    for row in reader:

      paper = row[0]

      title = row[1]

      r_paper_to_title[paper] = title

  results = []



  for query in queries:

    ids_to_scores = {}

    stemmed_query = tokenize(query)

    for word in stemmed_query:

      if word not in r_words_to_docs_to_freqs:

        continue

      for paper in r_words_to_docs_to_freqs[word]:

        idf = Math.log(doc_count / len(r_words_to_docs_to_freqs[word]))

        change = r_paper_to_tf[paper][word] * idf

        if paper in ids_to_scores:

          ids_to_scores[paper] += change

        else:

          ids_to_scores[paper] = change

    if usePageRank:

      for (id, score) in ids_to_scores.items():

        ids_to_scores[id] = score * r_page_to_pr[id]

    

    results.append([pair[0] for pair in sorted(ids_to_scores.items(), key=lambda pair : pair[1], reverse=True)][:100])

    ids_to_scores.clear()

  

  #Output results

  for query in range(len(results)):

      for rank in range(len(results[query])):

        result = results[query][rank]

        print(str(query+1)+'\t'+str(rank+1)+'\t'+str(result) + "\t" + r_paper_to_title[result])
getData()
extract(True)
retrieve(['COVID19', 'coronavirus vaccine', 'coronavirus mortality', 'how can I protect myself against coronavirus'], True)