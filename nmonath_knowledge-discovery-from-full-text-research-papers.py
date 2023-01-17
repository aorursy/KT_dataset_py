!pip install cython 

!pip install numpy

!pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

!pip install git+git://github.com/nmonath/kdcovid.git

!pip install git+https://bitbucket.org/nmonath/befree.git
metadata_file = '../input/CORD-19-research-challenge/metadata.csv'

input_file_list = '../input/cord19filelist/kaggle-file-list.txt'
from kdcovid.setup_corpus import DocumentLoader



doc_loader = DocumentLoader(input_file_list, metadata_file, max_files_processed=100)

documents = doc_loader.all_sections
import sent2vec

from kdcovid.encode_sentences import encode

import os

import numpy as np

import torch



model_path = '../input/scriptmodel/script_model.bin'

model = sent2vec.Sent2vecModel()



if os.path.exists(model_path):

    model.load_model(model_path)

else:

    print("Not found")



sentence_vectors, sentence_metadata = encode(documents, model=model)

sentence_vectors = torch.from_numpy(np.vstack(sentence_vectors).astype(np.float32))
from befree.src.ner import BeFree_NER_cord19

from kdcovid.parse_befree_output import parse_befree_output



identified_genes, identified_diseases = BeFree_NER_cord19.parse_cord19(documents)

combined_mentions = parse_befree_output(documents, identified_diseases, identified_genes)
from kdcovid.search_tool import SearchTool

# to use high quality cached results from the entire dataset

search_tool = SearchTool(all_vecs=sentence_vectors, all_meta=sentence_metadata, 

                         model=model, metadata_file='../input/CORD-19-research-challenge/metadata.csv', 

                         documents=documents, entity_links=combined_mentions, cached_result_file='../input/cachedresultsv7/cached_results.pkl', use_object=False,

                                 gv_prefix='http://kdcovid.nl/')



# to search on the toy subset uncomment the below line

# search_tool = SearchTool(all_vecs=sentence_vectors, all_meta=sentence_metadata, 

#                          model=model, metadata_file='../input/CORD-19-research-challenge/metadata.csv', 

#                          documents=documents, entity_links=combined_mentions, cached_result_file=None, use_object=False,

#                                  gv_prefix='http://kdcovid.nl/')
from IPython.core.display import display, HTML

display(HTML(search_tool.format_single_page_with_css(search_tool.get_search_results('Animal models for viral infection'))))