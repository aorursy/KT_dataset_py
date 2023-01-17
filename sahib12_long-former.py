# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# visualisation

import seaborn as sns





#Transformer

import torch

from transformers import LongformerModel, LongformerTokenizer





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
news=pd.read_csv('/kaggle/input/news_articles.csv')# reading the file
print(news.columns)

print('\n')

print(news.dtypes)

news.head()
news['len_content']=news['Content'].apply(lambda x : len(x))
sns.distplot(news['len_content'])
model = LongformerModel.from_pretrained('allenai/longformer-base-4096',output_hidden_states = True)

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')



# Put the model in "evaluation" mode, meaning feed-forward operation.

model.eval()

all_content=list(news['Content'][:5])

#doing only for first 5 contents

# because doing for complete takes 2 hours on CPU



def sentence_bert():

    list_of_emb=[]

    for i in range(len(all_content)):

        SAMPLE_TEXT = all_content[i]  # long input document

        input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1



        # Attention mask values -- 0: no attention, 1: local attention, 2: global attention

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention

        attention_mask[:, [0,-1]] = 2  # Set global attention based on the task. For example,

                                            # classification: the <s> token

                                            # QA: question tokens

                                            # LM: potentially on the beginning of sentences and paragraphs



        # we have set <s> and </s> token's attention mask =2

        # because acc to Longformer documentation these tokens must be given 

        # global attention when we are doing sentence classification

        # Run the text through BERT, and collect all of the hidden states produced





        # from all 12 layers. 

        with torch.no_grad():



            outputs = model(input_ids, attention_mask=attention_mask)



            # Evaluating the model will return a different number of objects based on 

            # how it's  configured in the `from_pretrained` call earlier. In this case, 

            # becase we set `output_hidden_states = True`, the third item will be the 

            # hidden states from all layers. See the documentation for more details:

            # https://huggingface.co/transformers/

            hidden_states = outputs[2]

            

            #outputs[0] gives us sequence_output

            #outputs[1] gives us pooled_output

            #outputs[2] gives us Hidden_output

            





        # Concatenate the tensors for all layers. We use `stack` here to

        # create a new dimension in the tensor.

        token_embeddings = torch.stack(hidden_states, dim=0)



        # Remove dimension 1, the "batches".

        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Swap dimensions 0 and 1.

        token_embeddings = token_embeddings.permute(1,0,2)





        token_vecs_sum = []







        # For each token in the sentence...

        for token in token_embeddings:



            





            #but preferrable is

            sum_vec=torch.sum(token[-4:],dim=0)



            # Use `sum_vec` to represent `token`.

            token_vecs_sum.append(sum_vec)



#         print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))



        h=0

        for i in  range(len(token_vecs_sum)):

            h+=token_vecs_sum[i]

            

            

        list_of_emb.append(h)

    return list_of_emb

f=sentence_bert()

print(len(f))

    
f[0].shape  #  embedding  for first sentence
import pickle

with open("/kaggle/working/emeddings_bert_mean.txt", "wb") as fp:

    pickle.dump(f, fp)