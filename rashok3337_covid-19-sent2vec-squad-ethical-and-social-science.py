!pip install transformers



import torch

from transformers import BertForQuestionAnswering

from transformers import BertTokenizer



bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#bert_model.save_weights("bert_model.h5")

import pickle

pkl_filename = "bert_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(bert_model, file)



bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#tokenizer.save_weights("tokenizer.h5")

import pickle

pkl_filename = "bert_tokenizer.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(bert_tokenizer, file)



import os

#%%capture

!curl -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz

!mv openjdk-11.0.2_linux-x64_bin.tar.gz /usr/lib/jvm/; cd /usr/lib/jvm/; tar -zxvf openjdk-11.0.2_linux-x64_bin.tar.gz

!update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-11.0.2/bin/java 1

!update-alternatives --set java /usr/lib/jvm/jdk-11.0.2/bin/java

os.environ["JAVA_HOME"] = "/usr/lib/jvm/jdk-11.0.2"
#%%capture

!pip install pyserini==0.8.1.0

from pyserini.search import pysearch
#%%capture

!wget -O lucene.tar.gz https://www.dropbox.com/s/j55t617yhvmegy8/lucene-index-covid-2020-04-10.tar.gz?dl=0

!tar xvfz lucene.tar.gz

minDate = '2020/04/10'  

luceneDir = '/kaggle/working/lucene-index-covid-2020-04-10/'
from IPython.core.display import display, HTML

import json

def show_query(query):

    """HTML print format for the searched query"""

    return HTML('<br/><div style="font-family: Times New Roman; font-size: 20px;'

                'padding-bottom:12px"><b>Query</b>: '+query+'</div>')



def show_document(idx, doc):

    """HTML print format for document fields"""

    have_body_text = 'body_text' in json.loads(doc.raw)

    body_text = ' Full text available.' if have_body_text else ''

    return HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px">' + 

               f'<b>Document {idx}:</b> {doc.docid} ({doc.score:1.2f}) -- ' +

               f'{doc.lucene_document.get("authors")} et al. ' +

              f'{doc.lucene_document.get("journal")}. ' +

              f'{doc.lucene_document.get("publish_time")}. ' +

               f'{doc.lucene_document.get("title")}. ' +

               f'<a href="https://doi.org/{doc.lucene_document.get("doi")}">{doc.lucene_document.get("doi")}</a>.'

               + f'{body_text}</div>')



def show_query_results(query, searcher, top_k=10):

    """HTML print format for the searched query"""

    output_query = searcher.search(query)

#    print("output_query",output_query)

    display(show_query(query))

    for i, k in enumerate(output_query[:top_k]):

        display(show_document(i+1, k))

    return output_query[:top_k]   
import json

def query_id(query_result):

    #print(len(query_result))

    #print("query_result",query_result)

    files_list=[]

    for i in range(len(query_result)):

      doc_json = json.loads(query_result[i].raw)

      #print("doc_json",doc_json)

      paper_id = 'paper_id' in doc_json    

      if paper_id :

        #print(doc_json['paper_id'])

        files_list.append(doc_json)

        #print(doc_json)

      #else:

        #print(doc_json['sha'])

    #print("files_list",files_list)

    return files_list
import glob

import json

import pandas as pd

from tqdm import tqdm

import re

#all_json=files_list

#print("all_json type", type(all_json))

#print("length of json:",len(all_json))



class FileReader:

    def __init__(self, file):        

 #         print("file is=",file)

          content = file

 #         print("content is", type(content))

          self.paper_id = content['paper_id']

          self.abstract = []

          self.body_text = []

          self.abstract_section=[]

          self.body_section=[]

          # Abstract

          try:

              for entry in content['abstract']:

                  self.abstract.append(entry['text'])

                  self.abstract_section.append(entry['section'])

          except KeyError:pass    

          # Body text

          for entry in content['body_text']:

              self.body_text.append(entry['text'])

              self.body_section.append(entry['section'])

          self.abstract = '\n'.join(self.abstract)

          self.body_text = '\n'.join(self.body_text)

          self.abstract_section = '\n'.join(self.abstract_section)

          self.body_section = '\n'.join(self.body_section)

            

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}++++++ {self.body_text[:200]}+++++'

        

#first_row = FileReader(all_json[0])

#print("first_row===",first_row)



def get_breaks(content, length):

    data = ""

    words = content.split(' ')

    total_chars = 0

    # add break every length characters

    for i in range(len(words)):

        total_chars += len(words[i])

        if total_chars > length:

            data = data + "<br>" + words[i]

            total_chars = 0

        else:

            data = data + " " + words[i]

    return data



def generate_clean_df(files_list):

    all_json=files_list

    dict_ = {'paper_id': [], 'abstract': [], 'body_text': [],'body_section':[],'abstract_section':[], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': [],'source_x':[],'publish_time':[]}



    for idx, entry in enumerate(all_json):

        content = FileReader(entry) 

        dict_['paper_id'].append(content.paper_id)

        dict_['abstract'].append(content.abstract)

        dict_['body_text'].append(content.body_text)

        dict_['body_section'].append(content.body_section)

        dict_['abstract_section'].append(content.abstract_section)

    df_covid = pd.DataFrame(dict_, columns=['paper_id','abstract','abstract_section','body_section','body_text'])

    #print(df_covid.head())

    text_dict = df_covid.to_dict()

    len_text = len(text_dict["paper_id"])

    paper_id_list  = []

    body_text_list = []

    body_section_list =[]

    section_list =[]

    for i in tqdm(range(0,len_text)):

      paper_id = df_covid['paper_id'][i]

      body_text = df_covid['body_text'][i].split("\n")

      body_section = df_covid['body_section'][i].split('\n')

      for i in tqdm(range(0,len(body_text))):

        paper_id_list.append(paper_id)

        body_text_list.append(body_text[i])

        body_section_list.append(body_section[i])

        section_list.append("BODY")



    df_paragraph = pd.DataFrame({"paper_id":paper_id_list,"section":section_list,"paragraph":body_text_list,"subsection":body_section_list})

    df_paragraph.to_csv("paragraph.csv")

    return df_paragraph

#files_list=query_id(query_result)

#cl_df=generate_clean_df(files_list)
!wget https://github.com/facebookresearch/fastText/archive/v0.9.1.zip

!unzip v0.9.1.zip

!cd fastText-0.9.1

! ls -l

! pip install /kaggle/working/fastText-0.9.1/.
! git clone https://github.com/epfml/sent2vec.git

#! cd /content/sent2vec/

! pip install /kaggle/working/sent2vec/.
pkl_filename = "/kaggle/working/bert_model.pkl"

with open(pkl_filename, 'rb') as file:

    bert_model = pickle.load(file)

    

pkl_filename = "/kaggle/working/bert_tokenizer.pkl"

with open(pkl_filename, 'rb') as file:

    tokenizer = pickle.load(file)

import numpy as np

def bertsquadpred(bert_model, text, query):

    input_ids = tokenizer.encode(query, text)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    sep_index = input_ids.index(tokenizer.sep_token_id)

    num_seg_a = sep_index + 1

    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    assert len(segment_ids) == len(input_ids)

    n_ids = len(segment_ids)

    #print("n_ids",n_ids)

    #print(n_ids)

    if n_ids < 512:

        start_scores, end_scores = bert_model(torch.tensor([input_ids]), 

                                 token_type_ids=torch.tensor([segment_ids]))

    else:        

        start_scores, end_scores = bert_model(torch.tensor([input_ids[:512]]), 

                                 token_type_ids=torch.tensor([segment_ids[:512]]))

    #print("start_scores",start_scores)

    #print("end_scores",end_scores)

    start_scores = start_scores[:,1:-1]

    end_scores = end_scores[:,1:-1]

    answer_start = torch.argmax(start_scores)

    answer_end = torch.argmax(end_scores)

    #print("answer_start, answer_end",answer_start, answer_end)

    answer = ''

    

    t_count = 0    

    for i in range(answer_start, answer_end + 2):

        if tokens[i] == '[SEP]' or tokens[i] == '[CLS]':

            continue

        if tokens[i][0:2] == '##':

            answer += tokens[i][2:]

        else:

            if t_count == 0:

                answer +=  tokens[i]

            else:

                answer += ' ' + tokens[i]

        t_count+=1

            

    full_txt = ''

    for t in tokens:

        if t[0:2] == '##':

            full_txt += t[2:]

        else:

            full_txt += ' ' + t

            

    abs_returned = full_txt.split('[SEP]')[1]

            

    #print(abs_returned)

    ans={}

    ans['answer'] = answer

    #print(answer)

    if answer.startswith('[CLS]') or answer_end.item() < sep_index or answer.endswith('[SEP]'):

        ans['confidence'] = -1.0

    else:

        confidence = torch.max(start_scores) + torch.max(end_scores)

        confidence = np.log(confidence.item())

        ans['confidence'] = confidence/(1.0+confidence)

    ans['start'] = answer_start

    ans['end'] = answer_end

    ans['paragraph_bert'] = abs_returned

    return ans
import sent2vec

#from nltk import word_tokenize

#from nltk.corpus import stopwords

from string import punctuation

from scipy.spatial import distance



#model_path = "/content/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"

#model_path = "/kaggle/input/biosentvec/BioSentVec_CORD19-bigram_d700.bin"

model_path = "/kaggle/input/covid-sent2vec-ver2/BioSentVec_CORD19-bigram_d700_v2.bin"

#model_path = "/content/BioSentVec_CORD19-bigram_d700.bin"

model = sent2vec.Sent2vecModel()

try:

    model.load_model(model_path)

except Exception as e:

    print(e)

print('model successfully loaded')
import warnings

warnings.filterwarnings("ignore")

#from IPython.display import Markdown, display

from IPython.display import Markdown

def test_query(cl_df):

    para_vector_dict = {}

    subsection_vector_dict={}

    para_list = []

    #data=[]

    f_data=[]

    count = 0

    n_return = 2

    for id, group in cl_df.groupby(['paper_id']):

        para = group["paragraph"].values

        paper_id1=group["paper_id"].values

        paper_id=group["paper_id"].values

        paper_id=paper_id.tolist()       

        #print("\n paper_id",set(paper_id)) 

        #display(Markdown("""#Paper_id"""#Paper_id:",set(paper_id)))

        subsection = group["subsection"].values

        paras = [p for p in para if isinstance(p,str)]

        subsections = [s for s in subsection if isinstance (s, str)]

        

        paras_count = len(paras)

        #print(paras_count)    

        if paras_count==0:

            continue

        #paras_text = " ".join(paras)

        paragraph_dict={}

        paragraph_list=[]

        k=0

        for sub_para in paras:

          paragraph_dict[k]=model.embed_sentence(sub_para)

          paragraph_list.append(sub_para)

          k+=1    

        

        keys = list(paragraph_dict.keys())

        

        vectors = np.array(list(paragraph_dict.values()))

        #print("vectors are",vectors)



        nsamples, nx, ny = vectors.shape

        para_vectors = vectors.reshape((nsamples,nx*ny))



        from scipy.spatial import distance

        from sklearn.metrics.pairwise import cosine_similarity



        para_matrix_query = cosine_similarity(para_vectors, query_vector.reshape(1,-1))

        para_indexes = np.argsort(para_matrix_query.reshape(1,-1)[0])[::-1][:5]        

        short_listed_paras = [paragraph_list[i] for i in para_indexes]        

        subsection_head=[subsections[i] for i in para_indexes]

        

        #display(Markdown("query:",query))

        

        from itertools import chain 

        answer_squad_list=[]

        answer_squad_dict={}

        for sub_head,sub_para in zip(subsection_head,short_listed_paras):         

            answer_vector = model.embed_sentence(sub_para)

            cosine_sim = 1 - distance.cosine(query_vector, answer_vector)            

            answer_squad = bertsquadpred(bert_model, sub_para, query)

            answer_squad_list.append(answer_squad['answer'])

        

        display(Markdown("""Paper_id:"""+str(set(paper_id))+" "))

        #display(Markdown("""Paper_id:"""+group['paper_id'].unique()+" "))

        display(Markdown("""Query   :"""+query))

        display(Markdown("""Answer  :"""+" ".join(answer_squad_list)))

        #print("-"*90)

        #data={'paper_id':group['paper_id'].unique(),'answer':" ".join(answer_squad_list)}

        #data={'paper_id':(group['paper_id'].nunique()),'subsection_consider':",".join(subsection_head),'answer':" ".join(answer_squad_list)}

        #data={'paper_id':set(paper_id),'subsection_consider':",".join(subsection_head),'answer':" ".join(answer_squad_list)}

        #f_data.append(data)dispgroup['paper_id'].unique()lay(Markdown("""Paper_id:"""+str(set(paper_id))+" "))

    #df = pd.DataFrame(f_data) 

    #print(df)

    #display(Markdown("---Answer---"))

    #display(df[['paper_id', 'answer']])

    #df.to_csv("/kaggle/working/t.csv", index=False)

    #print(df)                                                                                                              

                                                                                                                  

        

    
question_list=["What has been published concerning ethical considerations for research",

              "What has been published concerning social sciences at the outbreak response",

              "Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019",

              "Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight",

              "Efforts to support sustained education, access, and capacity building in the area of ethics",

              "Efforts to establish a team at WHO that will be integrated within multidisciplinary research and oper,ational platforms and that will connect with existing and expanded global networks of social sciences",

              "Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)",

              "Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed",

              "Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media."]
#query="Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood)."

for query in question_list:    

    #query="What do we know about virus genetics, origin, and evolution"

    from pyserini.search import pysearch

    searcher = pysearch.SimpleSearcher(luceneDir)

    query_result = show_query_results(query, searcher, top_k=10)

    files_list=query_id(query_result)

    cl_df=generate_clean_df(files_list)

    query_vector = model.embed_sentence(query)

    test_query(cl_df)

#cl_df.to_csv("/kaggle/working/cl_df.csv",index=False)

#query_vector = model.embed_sentence(query)