!pip install transformers

!pip install faiss-gpu
!wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar -O scibert.tar

! tar -xvf scibert.tar
!pip install sentence-transformers
import torch

import transformers

import numpy as np 

import pandas as pd



from sklearn.metrics.pairwise import cosine_similarity



#globals 

MODEL = 'scibert_scivocab_uncased'



#load the model

sciBert = transformers.BertModel.from_pretrained(MODEL)



#create a transformer tokenizer for BERT

tokenizer = transformers.BertTokenizer.from_pretrained(MODEL, do_lower_case=True)



print(type(sciBert))



sciBert.eval()

sciBert.cuda(0)
def embedding_fn(model, text) :



  if not isinstance(model, transformers.modeling_bert.BertModel) :

    print('Model must be of type transformers.modeling_bert.BertModel, but got ', type(model))

    return



  with torch.no_grad():

    #generate tokens :

    tokens = tokenizer.encode(text)

    #expand dims : 

    batch_tokens = np.expand_dims(tokens, axis = 0)

    batch_tokens = torch.tensor(batch_tokens).cuda()

    #print(type(batch_tokens))

    #generate embedding and return hidden_state : 

    return model(batch_tokens)[0].cpu()



def embedding_fn_cpu(model, text) :



  if not isinstance(model, transformers.modeling_bert.BertModel) :

    print('Model must be of type transformers.modeling_bert.BertModel, but got ', type(model))

    return



  with torch.no_grad():

    #generate tokens :

    tokens = tokenizer.encode(text, max_length = 512)

    #expand dims : 

    batch_tokens = np.expand_dims(tokens, axis = 0)

    batch_tokens = torch.tensor(batch_tokens)

    #print(type(batch_tokens))

    #generate embedding and return hidden_state : 

    return model(batch_tokens)[0]



def compute_mean(embedding):



  if not isinstance(embedding, torch.Tensor):

    print('Embedding must be a torch.Tensor')

    return 

  

  return embedding.mean(1)





def compute_cosine_measure(x1, x2):



  #given two points in vector space, measure cosine distance

  return cosine_similarity(x1, x2)





def compute_distance(x1, x2):

  #replace this with your own measure

  return compute_cosine_measure(x1.detach().numpy(), x2.detach().numpy())
!ls /kaggle/input/CORD-19-research-challenge
dataset = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')





import json



#We generate Index File and of key-value pair , each dict has 2 values : cord_uid and title. This we save in a separate CSV file

def generate_mapping_index(dataframe):



  index_map = {}



  for index, row in dataframe.iterrows():

    index_map[index] = {

        "cord_uid" : row['cord_uid'],

        "title" : row['title'],

        "abstract" : row['abstract'],

        "url" : row['url']

    }

  

  return index_map





index_map = generate_mapping_index(dataset)

open('index.json', 'w').write(json.dumps(index_map))

dataset.head()
CHUNK_SIZE_EACH = 44000







def __embedding(text):

  return compute_mean(embedding_fn(sciBert, text))







def compute_bert_embeddings(dataframe_chunk, current_index, end_marker):



  np_chunk = __embedding(dataframe_chunk.loc[current_index * end_marker]['title']).detach().numpy()

  #np_chunk = np_chunk.reshape(np_chunk.shape[1])



  for idx in range(1, end_marker):



    try:

      embedding = __embedding(dataframe_chunk.loc[(current_index * end_marker) + idx]['title']).detach().numpy()

      #embedding = embedding.reshape(embedding.shape[1])

      np_chunk = np.append(np_chunk, embedding, axis = 0)

      print('\r {}'.format(np_chunk.shape), end = '')

    except Exception as e:

      print(e)

      np_chunk = np.append(np_chunk, np.zeros(shape = (1, 768)), axis = 0)

      continue 



  print(np_chunk.shape)

  np.savez_compressed('title_{}'.format(current_index), a = np_chunk)





def compute_embeddings_and_save(dataframe):



  n_rows = len(dataframe)

  

  chunk_sizes = n_rows // CHUNK_SIZE_EACH

  remaining = n_rows - chunk_sizes * CHUNK_SIZE_EACH



  for i in range(1):



    compute_bert_embeddings(dataframe[i * CHUNK_SIZE_EACH : (i * CHUNK_SIZE_EACH) + CHUNK_SIZE_EACH ], i, CHUNK_SIZE_EACH)





#Un-comment this if you want to regenerate embeddings.

#compute_embeddings_and_save(dataset)

!ls
embeddings = np.load('/kaggle/input/cord-19-title-embeddings/title_0.npz')['a']

embeddings.shape
import time



#print(index_map.keys())

def index_to_title(indexes):

    

    for i, idx in enumerate(indexes) :

        print('{}. {}'.format(i, index_map[idx]['title']))



def do_consine_search(embeddings, query_text, model, top_k):

    

    n_embeddings = embeddings.shape[0]

    

    embedding_q = compute_mean(embedding_fn(sciBert, query_text)).detach().numpy()

    

    #lets do the search and time the process

    st = time.time()

    distances = []

    for em in embeddings :

        

        em = np.expand_dims(em, axis = 0)

        distances.append(compute_cosine_measure(em, embedding_q)[0][0])

        

    top_k_arguments = np.argsort(np.array(distances))[::-1][:top_k]

    et = time.time()

    

    return et - st, top_k_arguments



    

time_cosine, indexes_top = do_consine_search(embeddings, "Middle East Virus", sciBert, 20)

print('Cosine search time :  ', time_cosine, ' seconds')



index_to_title(indexes_top)

        
import faiss
n_dimensions = embeddings.shape[1] #Number of dimensions (764)



fastIndex = faiss.IndexFlatL2(n_dimensions) # We will create an index of type FlatL2, there are many kinds of indexes, you can look at it in their repo.

fastIndex.add(embeddings.astype('float32')) # Add the embedding vector to faiss index, it should of dtype 'float32'
def do_faiss_lookup(fastIndex, query_text, model, top_k):

    n_embeddings = embeddings.shape[0]

    embedding_q = compute_mean(embedding_fn(sciBert, query_text)).detach().numpy()

    

    #let it be float32

    embedding_q = embedding_q.astype('float32')

    

    #perform the search

    st = time.time()

    matched_em, matched_indexes = fastIndex.search(embedding_q, top_k) # it returns matched vectors and thier respective indexes, we are interested only in indexes.

    

    #indexes are already sorted wrt to closest match

    et = time.time()

    

    return et - st, matched_indexes[0]



time_faiss_cpu, indexes_top_faiss = do_faiss_lookup(fastIndex, "Middle East Virus", sciBert, 20)

print('Faiss index lookup time :  ', time_faiss_cpu, ' seconds')



index_to_title(indexes_top_faiss)





    
import faiss



n_dimensions = embeddings.shape[1] #Number of dimensions (764)



fastIndex_gpu = faiss.IndexFlatL2(n_dimensions) # We will create an index of type FlatL2, there are many kinds of indexes, you can look at it in their repo.



#copy the index to GPU 

res = faiss.StandardGpuResources()



fastIndex_gpu = faiss.index_cpu_to_gpu(res, 0, fastIndex_gpu)



fastIndex_gpu.add(embeddings.astype('float32')) # Add the embedding vector to faiss index, it should of dtype 'float32'
def do_faiss_lookup_gpu(fastIndex, query_text, model, top_k):

    n_embeddings = embeddings.shape[0]

    embedding_q = compute_mean(embedding_fn(sciBert, query_text)).detach().numpy()

    

    #let it be float32

    embedding_q = embedding_q.astype('float32')

    

    #perform the search

    st = time.time()

    matched_em, matched_indexes = fastIndex.search(embedding_q, top_k) # it returns matched vectors and thier respective indexes, we are interested only in indexes.

    

    #indexes are already sorted wrt to closest match

    et = time.time()

    

    return et - st, matched_indexes[0]



time_faiss_gpu, indexes_top_faiss = do_faiss_lookup_gpu(fastIndex, "Middle East Virus", sciBert, 20)

print('Faiss index lookup time :  ', time_faiss_gpu, ' seconds')



index_to_title(indexes_top_faiss)

print('CPU based cosine-distance metric lookup : (Brute-force method : )', time_cosine)

print('CPU based FAISS index lookup : ', time_faiss_cpu)

print('GPU based FAISS index lookup : ', time_faiss_gpu)