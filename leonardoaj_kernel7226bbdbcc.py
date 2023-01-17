import os
from functions import get_filename_list
import json
import networkx as nx
import re
import pickle
import pygraphviz
from networkx.drawing.nx_pydot import write_dot
from copy import deepcopy
# returns a list of authors for a given paper
def get_authors(json_file):
    return [re.sub("[\s\-\.\~\d\#\(\)\?\/\:\;\$\*\]\[€•†]|(&apos;)|(&amp;)|(&quot;)", "", f"{x['first']} {x['last']}".strip()).lower() for x
     in json_file["metadata"]["authors"]
            if len(x["first"]) > 2 and len(x["last"]) > 2]


# a simple method to clear non alphanumeric characters
def quick_clean(text):
    aux = re.sub('[^0-9a-zA-Z\s]+', '', text).lower()
    return re.sub('\s{2,}', ' ', aux)

papers = {}
authors = {}
citations = {}
titles = {}

for k, filename in enumerate(get_filename_list()):

    json_file = json.loads(open(filename).read())
    paper_id = json_file["paper_id"]

    title = quick_clean(json_file["metadata"]["title"])
    titles[title] = paper_id
    citations[paper_id] = [quick_clean(x["title"]) for x in json_file["bib_entries"].values()]

    paper_authors = set(get_authors(json_file))

    papers[paper_id] = paper_authors

    for a in paper_authors:
        if not a in authors.keys():
            authors[a] = []

        authors[a].append(paper_id)
for paper_id, title_list in citations.items():
    citation_paperid[paper_id] = [titles[title] for title in title_list if title in titles.keys()]
author_citations = {}  

for paper_id, author_list in papers.items():

    citations = citation_paperid[paper_id]
    cited_authors = set()

    for citation in citations:
        authors = papers[citation]
        cited_authors.update(authors)

    for author in author_list:
        if author not in author_citations.keys():
            author_citations[author] = set()

        author_citations[author].update(cited_authors)
G = nx.DiGraph()

for author, cited_authors in author_citations.items():

    for other_author in cited_authors:
        G.add_edge(author, other_author)
small_G = deepcopy(G)

to_delete = [node for node in G.nodes
             if len(small_G.out_edges(node)) > 20
             or len(G.in_edges(node)) > 20
             or len(G.in_edges(node)) < 10
             or len(G.out_edges(node)) < 10]

small_G.remove_nodes_from(to_delete)
write_dot(small_G, "small_g.dot")
from functions import get_filename_list, read_json
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import multiprocessing
def stemm_sentence(sentence):
    return [ps.stem(word) for word in word_tokenize(sentence.lower()) if word not in stop_words]


def process_corpus(sentences, rank):

    corpus = []

    for sentence in sentences:
        stemmized_sentence = stemm_sentence(sentence)
        corpus.append(stemmized_sentence)

    pickle.dump(corpus, open(f"sentences-{rank}.p", "wb"))
ps = PorterStemmer()
stop_words = [ps.stem(w) for w in stopwords.words('english')]

files = get_filename_list()
total_files = len(files)
num_cores = multiprocessing.cpu_count()
chunk_size = total_files // num_cores

processes = []
manager = multiprocessing.Manager()

for rank in range(num_cores):

    if rank + 1 == num_cores:
        file_list_chunk = files[rank*chunk_size:]
    else:
        file_list_chunk = files[rank*chunk_size: (rank+1)*chunk_size]

    print(f"Reading chunk {rank}...")

    file_paragraph_list = [read_json(x, as_list=True) for x in file_list_chunk]  # paragraph list

    sentences = []
    for file in file_paragraph_list:
        for paragraph in file:
            sentences.extend(paragraph.split("."))

    p = multiprocessing.Process(target=process_corpus, args=(sentences, rank))
    p.start()
    processes.append(p)


for k, p in enumerate(processes):
    p.join()
    print(f"{k} has finished")
from sklearn.feature_extraction.text import TfidfVectorizer

main_corpus = []

for i in range(num_cores):
    print(i)
    corpus_piece = pickle.load(open(f"sentences-{i}.p", "rb"))
    main_corpus.extend(corpus_piece)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(main_corpus)
index = {}

k = 0

for x, filename in enumerate(get_filename_list()):

    print(x)

    paragraphs = read_json(filename, as_list=True)
    
    for paragraph in paragraphs: 
        
        sentences = paragraph.split(".")
        
        for i, s in enumerate(sentences):
            index[k] = (filename, i)
            k += 1
questions = [
    "Are there geographic variations in the rate of COVID-19 spread?",
    "Are there geographic variations in the mortality rate of COVID-19?",
    "Is there any evidence to suggest geographic based virus mutations?"
]
question_vector = vectorizer.transform([stemm_paragraph(q) for q in questions])

similarities = cosine_similarity(question_vector, X)
for row, question in zip(similarities, questions):

    print(question)

    for i in range(1, 4):

        result = index[row.argsort()[-i]]

        file_contents = json.loads(open(result[0], "r").read())

        print(file_contents["metadata"]["title"])
        print(file_contents["body_text"][result[1]]["text"])
