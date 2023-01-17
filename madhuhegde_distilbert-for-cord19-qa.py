!pip install rank_bm25
!pip install transformers==2.3.0
import os
import re
import json
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import glob
from nltk import tokenize
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from rank_bm25 import BM25Okapi
from collections import Counter
import pickle

base_dir = '/kaggle/input/cord19researchchallenge/CORD-19-research-challenge/'
def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)

def format_body_text(body_text):
    
    body = ""

    for di in body_text:
        text = di['text']
        body += text
    return body
    
def format_corpus_text(body_text, min_len=18, max_len=128):
    junk_text = "copyright"
    
    def remove_braces_brackets(body_text):
        body_text = re.sub(r'\([0-9]+\)', '', body_text)
        body_text = re.sub(r'\[[^)]*\]', '', body_text)
        return(body_text)
        
    body_text = remove_braces_brackets(body_text)
    text_lines = []
    token_lines = tokenize.sent_tokenize(body_text)
    for line in token_lines:
      
        words = line.split()
        if junk_text not in words:
             max_word_len = len(max(words, key=len))
             if (len(words) > min_len) and (len(words) < max_len) and max_word_len > 5:
                 text_lines.append(line)
    
    return(text_lines)


def find_filenames(folder):
    all_files = glob.glob(f'{folder}/**/*.json', recursive=True)
    print("Number of articles retrieved from the folder:", len(all_files))
    files = []

    for filename in all_files:
        with open(filename) as f:
            file = json.load(open(filename))
            files.append(file)
    return(files) 


def find_file_index(folder):
    all_files = glob.glob(f'{folder}/**/*.json', recursive=True)
    path_name = []
    path_dict = {}
    path_dict_inv = {}
    file_index = []


    for filename in all_files:
        filename_split = filename.split('/')
        last = filename_split[-1]
        first = filename_split[-4]+'/'+filename_split[-3]+'/'+filename_split[-2]+'/'
  
        if first not in path_name:
            path_name.append(first)
            path_dict[first] = len(path_name)-1
            path_dict_inv[len(path_name)-1] = first
        file_index.append((path_dict[first], last))   
        
    print(len(file_index))
    return file_index, path_dict_inv 
def generate_clean_data(files):
    cleaned_text = []

    for file in tqdm(files):
        body_text = format_body_text(file['body_text'])
        body_text = body_text.replace('\n',' ')

        features = [
           file['metadata']['title'],
           format_authors(file['metadata']['authors'], with_affiliation=True),
           body_text]
        cleaned_text.append(features)
    
    col_names = [
       'title',
       'authors',
       'paragraphs']

    clean_df = pd.DataFrame(cleaned_text, columns=col_names)
    return(clean_df)

def find_index_text(base_dir, file_index, path_dict, index):
    indexed_files = []
    
    for i in index:
        filename = base_dir+path_dict[file_index[i][0]]+file_index[i][1]

        with open(filename) as f:
            file = json.load(open(filename))
            indexed_files.append(file)
        
    frame = generate_clean_data(indexed_files)
    return(frame)
class BM25Retriever(BM25Okapi):
    def __init__(self, lowercase=True, tokenizer=None, top_n=10, k1=1.5, b=0.75, epsilon=0.25):
        super().__init__("dummy", tokenizer=None, k1=k1, b=b, epsilon=epsilon)
        self.lowercase = lowercase
        self.top_n = top_n
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer
        self.num_doc = 0
        self.corpus_size = 0
        self.nd = Counter({})
        
    def fit_retriever(self, documents):
        doc_list = [document for document in documents]
        #print(len(doc_list))
        if self.tokenizer:
            tokenized_text = [self.tokenizer(document) for document in doc_list]
        else:
            tokenized_text = [document.split(" ") for document in doc_list]
   
        #print(tokenized_text[0])
        self.corpus_size = self.corpus_size+len(tokenized_text)
        num_doc = 0
        for doc_tokens in tokenized_text:
            num_doc += len(doc_tokens)
        self.num_doc = self.num_doc+num_doc   
        self.avgdl = self.num_doc/self.corpus_size
        
        #print(self.corpus_size, self.num_doc, self.avgdl)
        nd = Counter(self._initialize(tokenized_text))
        self.nd = self.nd + nd      
        
        
    def compute_params(self):    
        self._calc_idf(self.nd)
        
    def compute_scores(self, query):
        if(self.tokenizer == None):
           tokenized_query = query.split(" ")
        else:
           tokenizer = self.tokenizer
           tokenized_query = tokenizer(query)
      
        doc_scores = self.get_scores(tokenized_query)

        #return top_n indices and scores as list
        sorted_scores = np.argsort(doc_scores)
        top_n = self.top_n
        out = zip(sorted_scores[-1:-top_n-1:-1],doc_scores[sorted_scores[-1:-top_n-1:-1]])
        return list(out)   
                
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
class DocReader():
    """
    Uses Hungging Face DistilBertForQuestionAnswering
    Parameters
    ----------
    model     : path to folder containing pytorch.bin, bert_config.json and vocab.txt
                or pretrained model
    lowercase : boolean
        Convert all characters to lowercase before tokenizing. (default is True)
    tokenizer : default is DistilBertTokenizer
    """
    
    
    def __init__(self, model:str=None, lowercase=True, tokenizer=DistilBertTokenizer):

        self.lowercase = lowercase
        self.tokenizer = tokenizer.from_pretrained(model)
        self.model = DistilBertForQuestionAnswering.from_pretrained(model)
        if torch.cuda.is_available():
            self.device =torch.device("cuda")
            self.model.cuda()
            print("PyTorch on CUDA")
        else:
            self.device =torch.device("cpu")
        
    def predict(self, 
                df: pd.DataFrame = None,
                query: str = None,
                n_best: int =3):
    
        doc_text = df['paragraphs']
        self.n_best = n_best
        
        if(self.lowercase):
            query = query.lower()
        
        # num docs_index must be equal to top_n
        doc_index = list(doc_text.index)
        
        #prepare the model for validation
        self.model.eval()
        answers = []
        for df_index in doc_index:      
            if(self.lowercase):
                doc_lines = doc_text[df_index].lower()
            else:
                doc_lines = doc_text[df_index]
                 
            #doc_lines = tokenize.sent_tokenize(doc_lines)
            doc_lines = format_corpus_text(doc_lines)
            doc_answers = []
            for lines in doc_lines:
                input_ids  = self.tokenizer.encode(query, lines)
              
                input_ids_device = torch.tensor([input_ids]).to(self.device)
                with torch.no_grad():
                    start_scores, end_scores = self.model(input_ids_device)
                   
                all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
                entry = {}   
                start_scores = start_scores.detach().cpu().numpy()
                end_scores = end_scores.detach().cpu().numpy()
                entry['answer'] = answer
                entry['score'] = (max(start_scores[0]), max(end_scores[0]))
                entry['index'] = df_index
                entry['sentence'] = lines
                doc_answers.append(entry)
                
            best_doc_ans = [entry['score'][0]+entry['score'][1]  for entry in doc_answers]
            ans_index = np.argsort(best_doc_ans)    
            # take n_best answers per document based on max(start_scores+end_scores)
            #it is possible to improve by taking different metric
            for ans in range(1,self.n_best+1):
                answers.append(doc_answers[ans_index[-ans]])    
                
                
        best_ans = [entry['score'][0]+entry['score'][1]  for entry in answers]
        ans_index = np.argsort(best_ans)     
        
        n_best_answers = []
        for ans in range(1,self.n_best+1):
            n_best_answers.append(answers[ans_index[-ans]])
        
        return(n_best_answers)          
    
    
    
    def best_answer(self, answers):
        ans_dict = {}
        final_answer = {}
        ANS_THRESH = 2.0
        max_score = answers[0]['score'][0]+answers[0]['score'][1]
    
        for ans in answers:
           score = ans['score'][0]+ans['score'][1]
           if score > max_score - ANS_THRESH:
              start_end = ans['answer'].split()
              if(len(start_end)>0):
                 ans_key = (start_end[0], start_end[-1])
                 ans_dict[ans_key] = ans_dict.get(ans_key,0)+1
                
        inverse = [value for key, value in ans_dict.items()]
        inverse.sort()
        
        ans_list = [key for key, value in ans_dict.items() if(value == inverse[-1])]
    
        max_score = float('-inf')
        for ans in answers:
           start_end = ans['answer'].split()
           ans_key = (start_end[0], start_end[-1])
           for item in ans_list:
              if(ans_key == item):
                 score = ans['score'][0]+ans['score'][1]
                 if(score > max_score):
                    ans_ids = self.tokenizer.convert_tokens_to_ids(start_end)
                    final_answer['answer'] = self.tokenizer.decode(ans_ids)
                    final_answer['index'] = ans['index']
                    final_answer['sentence'] = ans['sentence']
        
        
        return(final_answer)
            
            
if 0:
  # Use Wordpiece tokenizer
    bert_tokenizer =  BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    retriever = BM25Retriever(tokenizer=bert_tokenizer.tokenize)

    base_dir = '/kaggle/input/CORD-19-research-challenge/'
    sub_folders = glob.glob(base_dir)
    for folder in sub_folders:
        files = find_filenames(folder)
        if(len(files) > 0):
            frame = generate_clean_data(files)
            retriever.fit_retriever(frame['paragraphs'])
   
    #Compute TF-IDF paramas
    retriever.compute_params()
    with open('/kaggle/output/retriever.pkl', 'wb') as f:
        pickle.dump(retriever, f, pickle.HIGHEST_PROTOCOL)
        
    #Save file index in a dictionary    
    file_index, path_dict_inv = find_file_index(base_dir)    
    datafile_dict['file_index'] = file_index
    datafile_dict['path_dict_inv'] = path_dict_inv
    
    with open('/kaggle/output/datafile_dict.pkl', 'wb') as f:
        pickle.dump(datafile_dict, f, pickle.HIGHEST_PROTOCOL)
    
        
# Use Wordpiece tokenizer
bert_tokenizer =  BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
retriever = BM25Retriever(tokenizer=bert_tokenizer.tokenize)

retriever_pkl_file = '/kaggle/input/bm25-dictionary/retriever.pkl'
with open(retriever_pkl_file, 'rb') as file_in:
    retriever = pickle.load(file_in)

print(retriever.corpus_size, retriever.num_doc, retriever.avgdl)    
datafile_pkl_file = '/kaggle/input/datafile-dict/datafile_dict.pkl'
with open(datafile_pkl_file, 'rb') as file_in:
    datafile_dict = pickle.load(file_in)

file_index = datafile_dict['file_index']
path_dict_inv = datafile_dict['path_dict_inv']

#Initialize Distilbert based document reader.
reader = DocReader('distilbert-base-uncased-distilled-squad')
# Find top_n documents based on BM250 for a given query 
query = "what is covid-19"
doc_scores = retriever.compute_scores(query)

#Select top_n documents
index = [score[0] for score in doc_scores]

#Retrieve document texts for top_n documents
#text['paragraphs'] = Entire text in the document
#text['title'] = title of the document
#text['authors'] = authors
text = find_index_text(base_dir, file_index, path_dict_inv, index)
#find best answer from n_best documents returned by the document retriever
ans = reader.predict(df=text, query=query, n_best=5)
b_answer = reader.best_answer(ans)

print('query: {}\n'.format(query))
print('answer: {}\n'.format(b_answer['answer']))
print('title: {}\n'.format(text['title'][b_answer['index']]))
print('paragraph: {}\n'.format(b_answer['sentence']))