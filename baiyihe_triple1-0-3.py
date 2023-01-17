import mxnet as mx 

def gpu_device(gpu_number=0):

    try:

        _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))

    except mx.MXNetError:

        return None

    return mx.gpu(gpu_number)

if not gpu_device():

    print('No GPU device found!')
# !pip install bert-embedding
# import os

# import csv

# import mxnet as mx

# from bert_embedding import BertEmbedding
# ctxx = mx.gpu(0)

# bert_embedding = BertEmbedding(ctx=ctxx, model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
# read in triples

csv.field_size_limit(10000000)

triple = []

with open("../input/triple_3.csv") as file_r:

    file_reader = csv.reader(file_r)

    csv_tags = next(file_reader)

    # ["target_link", "api_full_name", "api_description", "api_description_html"]

    

    for record in file_reader:

        triple.append([[record[0]],[record[1]],record[2].split()])
rearrange_triple_list = []

triple_list = []

for i, record in enumerate(triple):

    if i % 200 == 0:

        rearrange_triple_list.append(triple_list)

        triple_list = []

    triple_list.append(record)

rearrange_triple_list.append(triple_list)

rearrange_triple_list = rearrange_triple_list[1:]
import pandas as pd

from IPython.display import HTML

import pandas as pd

import numpy as np



def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)
# train_list = []

# for list_ in rearrange_triple_list[270:300]:

#     for record in list_:

#         train_list.append(record)

# embedding_D1_list = []

# embedding_D2_list = []

# embedding_C_list = []

# for record_id, record in enumerate(train_list):

#     embedding_D1_list.append(bert_embedding(record[0]))

#     embedding_D2_list.append(bert_embedding(record[1]))

#     embedding_C_list.append([bert_embedding([word]) for word in record[2]])

#     print(record_id)

# path = 'triple_3_embedding_'+str(38)+'.csv'

# dictt = {"D1": embedding_D1_list,"D2": embedding_D2_list,"C":embedding_C_list}

# df = pd.DataFrame(dictt)

# df.to_csv(path, index = False)
# create_download_link(filename=path)
# train_list = []

# for list_ in rearrange_triple_list[300:]:

#     for record in list_:

#         train_list.append(record)

# embedding_D1_list = []

# embedding_D2_list = []

# embedding_C_list = []

# for record_id, record in enumerate(train_list):

#     embedding_D1_list.append(bert_embedding(record[0]))

#     embedding_D2_list.append(bert_embedding(record[1]))

#     embedding_C_list.append([bert_embedding([word]) for word in record[2]])

#     print(record_id)

# path = 'triple_3_embedding_'+str(39)+'.csv'

# dictt = {"D1": embedding_D1_list,"D2": embedding_D2_list,"C":embedding_C_list}

# df = pd.DataFrame(dictt)

# df.to_csv(path, index = False)
# create_download_link(filename=path)