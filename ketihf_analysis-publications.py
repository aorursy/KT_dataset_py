#import utility functions 

from covid19_data_processing_module import *



import matplotlib.pyplot as plt

import networkx as nx

import copy 

import collections



# key word linkage

risk_factors = ['smoking', 'reduced lung capacity', 'hand to mouth', 'lung disease', 'oxygen', 

                'chronic', 'cancer', 'high blood pressure', 'diabetes', 'cardiovascular', 'heart', 

                'chronic respiratory disease', 'heart disease']
# recreate the schema from "json_schema.txt"

class author():

    

    def __init__(self, input_dict=None):

        

        self.first = ""

        self.middle = []

        self.last = ""

        self.suffix = ""

        self.affiliation = {}

        self.email = ""

        

        if input_dict:

            for key in input_dict.keys():

                if "first" in key:

                    self.first = input_dict[key]

                if "middle" in key:

                    self.middle = input_dict[key]

                if "last" in key:

                    self.last = input_dict[key]

                if "suffix" in key:

                    self.suffix = input_dict[key]

                if "affiliation" in key:

                    self.affiliation = input_dict[key]

                if "email" in key:

                    self.email = input_dict[key]    

    

    def print_items(self):

        

        print("first: " + str(self.first) +  

              ", middle: " + str(self.middle) + 

              ", last: " + str(self.last) + 

              ", suffix: " + str(self.suffix) +

              ", email: " + str(self.email) + 

              ", affiliation: " + json.dumps(self.affiliation, indent=4, sort_keys=True)

             )





class inline_ref_span():

    

    def __init__(self, input_dict=None):

        

        self.start = 0

        self.end = 0

        self.text = ""

        self.ref_id = ""

        

        if input_dict:

            for key in input_dict.keys():

                if "start" in key:

                    self.start = input_dict[key]

                if "end" in key:

                    self.end = input_dict[key]

                if "text" in key:

                    self.text = input_dict[key]

                if "ref_id" in key:

                    self.ref_id = input_dict[key]

    

    def print_items(self):

        

        print("Text: " + str(self.text) + ", Start: " + 

              str(self.start) + ", End: " + str(self.end) + 

              ", Ref_id: " + str(self.ref_id))



    def step_index(self, n):

        

        self.start += n

        self.end += n

        

        

class text_block():

    

    def __init__(self, input_dict=None):

        

        self.text = ""

        self.cite_spans = []

        self.ref_spans = []

        self.eq_spans = []

        self.section = ""

        

        if input_dict:

            for key in input_dict.keys():

                if "text" in key:

                    self.text = input_dict[key]

                if "cite_spans" in key:

                    self.cite_spans = [inline_ref_span(c) for c in input_dict[key]]                

                if "ref_spans" in key:

                    self.ref_spans = [inline_ref_span(r) for r in input_dict[key]] 

                if "eq_spans" in key:

                    self.eq_spans = [inline_ref_span(e) for e in input_dict[key]]

                if "section" in key:

                    self.section = input_dict[key]

        

    def clean(self, swap_dict=None):

            

        self.text = clean(self.text, swap_dict)

    

    def print_items(self):

        

        print("\ntext: " + str(self.text))

        print("\nsection: " + str(self.section))

        print("\ncite_spans: ")

        [c.print_items() for c in self.cite_spans]

        print("\nref_spans: ")

        [r.print_items() for r in self.ref_spans]

        print("\neq_spans: ")

        [e.print_items() for e in self.eq_spans]





def combine_text_block(text_block_list):

    

    if text_block_list:

        

        combined_block = text_block_list[0]

        block_length = len(combined_block.text)

        

        for i in range(1,len(text_block_list)):

            combined_block.text += " " + text_block_list[i].text

            block_length += 1

            

            # update spans start & stop index

            [ref.step_index(block_length) for ref in text_block_list[i].cite_spans]

            [ref.step_index(block_length) for ref in text_block_list[i].ref_spans]

            [ref.step_index(block_length) for ref in text_block_list[i].eq_spans]

            

            # combine spans

            combined_block.cite_spans += text_block_list[i].cite_spans

            combined_block.ref_spans += text_block_list[i].ref_spans

            combined_block.eq_spans += text_block_list[i].eq_spans           

            combined_block.section += ", " + str(text_block_list[i].section)           

            

            block_length += len(text_block_list[i].text)

                       

        return [combined_block]

    else:

        return [text_block()]

      



class bib_item():

    

    def __init__(self, input_dict=None):

        

        self.ref_id: ""

        self.title: ""

        self.authors = []

        self.year = 0

        self.venue = ""

        self.volume = ""

        self.issn = ""

        self.pages = ""

        self.other_ids = {}

        

        if input_dict:

            for key in input_dict.keys():

                if "ref_id" in key:

                    self.ref_id = input_dict[key]

                if "title" in key:

                    self.title = input_dict[key]

                if "authors" in key:

                    self.authors = [author(a) for a in input_dict[key]]

                if "year" in key:

                    self.year = input_dict[key]

                if "venue" in key:

                    self.venue = input_dict[key]

                if "volume" in key:

                    self.volume = input_dict[key]

                if "issn" in key:

                    self.issn = input_dict[key]

                if "pages" in key:

                    self.pages = input_dict[key]

                if "other_ids" in key:

                    self.other_ids = input_dict[key]

    

    def print_items(self):

        

        print("\nBib Item:")

        print("ref_id: " + str(self.ref_id))

        print("title:" + str(self.title))

        print("Authors:")

        [a.print_items() for a in self.authors]

        print("year: " + str(self.year))

        print("venue:" + str(self.venue))

        print("issn:" + str(self.issn))

        print("pages:" + str(self.pages))

        print("other_ids:" + json.dumps(self.other_ids, indent=4, sort_keys=True))

        

        

class ref_entries():

    

    def __init__(self, ref_id=None, input_dict=None):

        

        self.ref_id = ""

        self.text = ""

        self.latex = None

        self.type = ""

        

        if ref_id:

            self.ref_id = ref_id

            

            if input_dict:

                for key in input_dict.keys():

                    if "text" in key:

                        self.text = input_dict[key]

                    if "latex" in key:

                        self.latex = input_dict[key]

                    if "type" in key:

                        self.type = input_dict[key]

    

    def print_items(self):

        

        print("ref_id: " + str(self.ref_id))

        print("text:" + str(self.text))

        print("latex: " + str(self.latex))

        print("type:" + str(self.type))

        

                    

class back_matter():

    

    def __init__(self, input_dict=None):

        

        self.text = ""

        self.cite_spans = []

        self.ref_spans = []

        self.section = ""

        

        if input_dict:

            for key in input_dict.keys():

                if "text" in key:

                    self.text = input_dict[key]

                if "cite_spans" in key:

                    self.cite_spans = [inline_ref_span(c) for c in input_dict[key]]                

                if "ref_spans" in key:

                    self.ref_spans = [inline_ref_span(r) for r in input_dict[key]] 

                if "section" in key:

                    self.section = input_dict[key]

    

    def print_items(self):

        

        print("text: " + str(self.text))

        print("cite_spans: ")

        [c.print_items() for c in self.cite_spans]

        print("ref_spans: ")

        [r.print_items() for r in self.ref_spans]        

        print("section:" + str(self.section))



        

# The following Class Definition is a useful helper object to store various 

# different covid-19 data types.

class document():

    

    def __init__(self, file_path=None):

        

        self.doc_filename = ""

        self.doc_language = {}

        self.paper_id = ""

        self.title = ""

        self.authors = []

        self.abstract = []

        self.text = []

        self.bib = []

        self.ref_entries = []

        self.back_matter = []

        self.tripples = {}

        self.key_phrases = {}

        self.entities = {}

        

        # load content from file on obj creation

        self.load_file(file_path)

     

    def _load_paper_id(self, data):

        

        if "paper_id" in data.keys():

            self.paper_id = data['paper_id']

    

    def _load_title(self, data):

        

        if "metadata" in data.keys():

            if "title" in data['metadata'].keys():

                self.title = data['metadata']["title"]

    

    def _load_authors(self, data):

        

        if "metadata" in data.keys():

            if "authors" in data['metadata'].keys():

                self.authors = [author(a) for a in data['metadata']["authors"]]

                

    def _load_abstract(self, data):

        

        if "abstract" in data.keys():

            self.abstract = [text_block(a) for a in data["abstract"]]

    

    def _load_body_text(self, data):

        

        if "body_text" in data.keys():

            self.text = [text_block(t) for t in data["body_text"]]

    

    def _load_bib(self, data):

        

        if "bib_entries" in data.keys():

            self.bib = [bib_item(b) for b in data["bib_entries"].values()]

    

    def _load_ref_entries(self, data):

        

        if "ref_entries" in data.keys():

            self.ref_entries = [ref_entries(r, data["ref_entries"][r]) for r in data["ref_entries"].keys()]

            

    def _load_back_matter(self, data):

        

        if "back_matter" in data.keys():

            self.back_matter = [back_matter(b) for b in data["back_matter"]]

        

    def load_file(self, file_path):

        

        if file_path:

            

            with open(file_path) as file:

                data = json.load(file)

                

                # call inbuilt data loading functions

                self.doc_filename = file_path

                self._load_paper_id(data)

                self._load_title(data)

                self._load_authors(data)

                self._load_abstract(data)

                self._load_body_text(data)

                self._load_bib(data)

                self._load_ref_entries(data)

                self._load_back_matter(data)

    

    def combine_data(self):

        

        self.data = {'doc_filename': self.doc_filename,

                     'doc_language': self.doc_language,

                     'paper_id': self.paper_id,

                     'title': self.title,

                     'authors':self.authors,

                     'abstract': self.abstract,

                     'text': self.text,

                     'bib_entries':self.bib,

                     'ref_entries': self.ref_entries,

                     'back_matter': self.back_matter,

                     'tripples': self.tripples,

                     'key_phrases': self.key_phrases,

                     'entities': self.entities}



    def extract_data(self):

        

        self.doc_filename = self.data['doc_filename']

        self.doc_language = self.data['doc_language']

        self.paper_id = self.data['paper_id']

        self.title = self.data['title']

        self.authors = self.data['authors']

        self.abstract = self.data['abstract']

        self.text = self.data['text']        

        self.bib = self.data['bib_entries']

        self.ref_entries = self.data['ref_entries']

        self.back_matter = self.data['back_matter']

        self.tripples = self.data['tripples']

        self.key_phrases = self.data['key_phrases']

        self.entities = self.data['entities']



    def save(self, dir):

        

        self.combine_data()



        if not os.path.exists(os.path.dirname(dir)):

            try:

                os.makedirs(os.path.dirname(dir))

            except OSError as exc:  # Guard against race condition

                if exc.errno != errno.EEXIST:

                    raise



        with open(dir, 'w') as json_file:

            json_file.write(json.dumps(self.data))



    def load_saved_data(self, dir):

        

        with open(dir) as json_file:

            self.data = json.load(json_file)

        self.extract_data()

    

    def print_items(self):

         

        print("---- Document Content ----") 

        print("doc_filename: " + str(self.doc_filename))

        print("doc_language: " + str(self.doc_language))

        print("paper_id: " + str(self.paper_id))

        print("title: " + str(self.title))

        print("\nAuthors: ")

        [a.print_items() for a in self.authors]

        print("\nAbstract: ")

        [a.print_items() for a in self.abstract]

        print("\nText: ")

        [t.print_items() for t in self.text]

        print("\nBib_entries: ")

        [b.print_items() for b in self.bib]

        print("\nRef_entries: ")

        [r.print_items() for r in self.ref_entries]

        print("\nBack_matter: ")

        [b.print_items() for b in self.back_matter]

        

        print("\nTripples: ")

        print(json.dumps(self.tripples, indent=4, sort_keys=True))

        print("\nKey Phrases: ")

        print(json.dumps(self.key_phrases, indent=4, sort_keys=True))        

        print("\nEntities: ")

        print(json.dumps(self.entities, indent=4, sort_keys=True))



    def clean_text(self, swap_dict=None):

        

        # clean all blocks of text

        [t.clean(swap_dict) for t in self.text]

    

    def clean_abstract(self, swap_dict=None):

        

        [t.clean(swap_dict) for t in self.abstract]

    

    def combine_text(self):

        

        # this function takes all text blocks within document.text and combines them into a single text_block object

        self.text = combine_text_block(self.text)

    

    def combine_abstract(self):

        

        self.abstract = combine_text_block(self.abstract)   

        

    def set_abstract_tripples(self):

                

        abstract_tripples = {}

        for i in range(0, len(self.abstract)):

            #for every block in the abstract, extract entity tripples

            self.abstract[i].clean()                       

            pairs, entities = get_entity_pairs(self.abstract[i].text)

            

            #if any tripples found

            if pairs.shape[0]>0:

                abstract_tripples["abstract_" + str(i)] = pairs.to_json()

                       

        self.tripples.update(abstract_tripples)

        

    def set_text_tripples(self):

        

        text_tripples = {}

        for i in range(0, len(self.text)):

            

            self.text[i].clean()                       

            pairs, entities = get_entity_pairs(self.text[i].text)

            if pairs.shape[0]>0:

                text_tripples["text_" + str(i)] = pairs.to_json()

                       

        self.tripples.update(text_tripples)

        

    def set_ref_tripples(self):

        

        ref_tripples = {}

        for r in self.ref_entries:

            pairs, entities = get_entity_pairs(r.text)

            if pairs.shape[0]>0:

                ref_tripples["ref_" + r.ref_id] = pairs.to_json()

        

        self.tripples.update(ref_tripples)

        

    def set_doc_language(self):

        # set the doc language based on the analysis of the first block within the abstract

        self.doc_language = get_text_language(self.text[0].text)

    
dir_input_data = '/kaggle/input/load-and-process-data-abstracts'



files = []

import os

for dirname, _, filenames in os.walk(dir_input_data):

        filenames = [names for names in filenames if '.pickle' in names]

        if filenames != []:

            files.append({'dirpath':dirname, 'filenames':filenames})
directory = files[0]["dirpath"]

filenames = files[0]["filenames"]



corpus_documents = {}

document_linkage_df = []

risk_factor_publications = collections.defaultdict(list)



doc_count = 0

for file in filenames:        

  

    with open(os.path.join(directory, file),"rb") as f:

        doc_list = pickle.load(f)

        

        for doc in doc_list:

               

            if doc:

                

                title = copy.deepcopy(doc.title)

                if title not in corpus_documents.keys():

                    doc_count += 1

                    corpus_documents.update({title: doc_count})

                

                doc_id = corpus_documents[title]

        

                ref_titles = []

                for b in doc.bib:

                    title = copy.deepcopy(b.title)

                    if title not in corpus_documents.keys():

                        doc_count += 1

                        corpus_documents.update({title: doc_count})

                                                

                    ref_titles.append(corpus_documents[title])

                

                

                df = pd.DataFrame({"object": [doc_id for r in ref_titles],

                                   'relation': ["has reference" for r in ref_titles],

                                   'subject': ref_titles})

                

                for risk in risk_factors:

                    if risk in doc.abstract:

                        risk_factor_publications[risk].append(doc_id)

                        

                document_linkage_df.append(df)

                doc_count += 1

                

document_linkage_df = pd.concat(document_linkage_df)



# save corpus_documents for cross reference later

with open('corpus_documents_lookup.json', 'w', encoding='utf-8') as f:

    json.dump(corpus_documents, f, ensure_ascii=False, indent=4)

    

# del after save to save memory 

del corpus_documents
def create_kg(pairs):

    k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',edge_attr = ['relation'],

            create_using=nx.DiGraph())

    return k_graph
G = create_kg(document_linkage_df)

print(nx.info(G))
def get_corpus_labels(corpus_dir, index):

    with open(corpus_dir) as file:

                corpus = json.load(file)

    return {value:key for key, value in corpus.items() if value in index}

            
node_rank = nx.degree_centrality(G)

node_rank_sorted = {k: v for k, v in sorted(node_rank.items(), key=lambda item: item[1],reverse=True)}

top_nodes = [k for k in node_rank_sorted.keys()][1:1000]



G_sub = G.subgraph(top_nodes)

mapping = get_corpus_labels('corpus_documents_lookup.json', list(G_sub.nodes))

G_sub = nx.relabel_nodes(G_sub, mapping)





subject = []

obj = []

relation = []

tasks = []

for element in list(G_sub.edges()):

        subject.append(element[0])

        obj.append(element[1])

        relation.append(G_sub.get_edge_data(element[0],element[1])['relation'])

        

node_deg = nx.degree(G_sub)

layout = nx.spring_layout(G_sub, k=0.25, iterations=20)

plt.figure(num=None, figsize=(120, 90), dpi=80)

nx.draw_networkx(

    G_sub,

    node_size=[int(deg[1]) * 500 for deg in node_deg],

    arrowsize=20,

    linewidths=1.5,

    pos=layout,

    edge_color='red',

    edgecolors='black',

    node_color='white',

    )







labels = dict(zip(list(zip(subject, obj)),relation))

nx.draw_networkx_edge_labels(G_sub, pos=layout, edge_labels=labels,

                                 font_color='black')

plt.axis('off')

plt.show()
corpus_documents = document_linkage_df['object'].drop_duplicates().tolist()

pr = nx.pagerank(G)

pr_df = pd.DataFrame({'Publications':list(pr.keys()), 'Ranking':list(pr.values())})

pr_sub_df = pr_df[pr_df.Publications.isin(corpus_documents)]
pr_df.to_csv('Page_Rank.csv', index=False)

pr_sub_df.to_csv('Page_Rank_Sub.csv', index=False)
for k,v in risk_factor_publications.items():

    print(k, len(risk_factor_publications[k]))
top_nodes = [v_sub for k,v in risk_factor_publications.items() for v_sub in v]



G_sub = G.subgraph(top_nodes)

mapping = get_corpus_labels('corpus_documents_lookup.json', list(G_sub.nodes))

G_sub = nx.relabel_nodes(G_sub, mapping)

G_sub.remove_node('')





subject = []

obj = []

relation = []

tasks = []

for element in list(G_sub.edges()):

        subject.append(element[0])

        obj.append(element[1])

        relation.append(G_sub.get_edge_data(element[0],element[1])['relation'])

        

node_deg = nx.degree(G_sub)

layout = nx.spring_layout(G_sub, k=0.25, iterations=20)

plt.figure(num=None, figsize=(120, 90), dpi=80)

nx.draw_networkx(

    G_sub,

    node_size=[int(deg[1]) * 500 for deg in node_deg],

    arrowsize=20,

    linewidths=1.5,

    pos=layout,

    edge_color='red',

    edgecolors='black',

    node_color='white',

    )







labels = dict(zip(list(zip(subject, obj)),relation))

nx.draw_networkx_edge_labels(G_sub, pos=layout, edge_labels=labels,

                                 font_color='black')

plt.axis('off')

plt.show()