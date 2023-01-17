# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packagesto load in 

!pip install ../input/sacremoses/sacremoses-master/ > /dev/null



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import torch

import re

from sklearn.preprocessing import OneHotEncoder

from urllib.parse import urlparse

from scipy.stats import rankdata



sys.path.insert(0, "../input/transformers/transformers-master/")

import transformers as ppb

import tensorflow as tf

import tensorflow_hub as hub

import keras.backend as K





import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



pd.options.display.max_columns = 100



import os

import gc

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# from IPython.display import FileLink

# FileLink('saved_vectors.npy')
# from IPython.display import FileLink

# FileLink('use_vectors.npy')
train = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

test = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

sample = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
train_dtypes = train.dtypes

target_columns= train_dtypes[train_dtypes==np.float64].index.tolist()



data = train[test.columns].append(test)



text_columns = ['question_title', 'question_body', 'answer']
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, 

                                                    ppb.DistilBertTokenizer, 

                                                   "../input/distilbertbaseuncased/")



tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

model = model_class.from_pretrained(pretrained_weights)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model.to(device)
class TextDataset(Dataset):

    

    def __init__(self, data ,col_name, max_len=512):

    

        self.col_name = col_name

        self.data = data

        self.max_len = max_len

        

    def __len__(self):

        

        return len(self.data)

        

    def __getitem__(self,index):

        

        tempData = self.data.iloc[index,:]

        text = tempData[self.col_name]

        sequence = tokenizer.encode(text, 

                add_special_tokens=True,

                max_length = self.max_len)

        

        if len(sequence)<self.max_len:

            sequence = sequence + [0]*(self.max_len-len(sequence))

           

            

        attention_mask = torch.tensor(np.where(np.array(sequence)!=0,1,0))

        

        padded_sequence = torch.tensor(sequence)

            

        

        return padded_sequence,attention_mask

    



    



# def my_collate_func(batch):

    

#     sequence_tensor = torch.stack([i[0] for i in batch])

#     mask_tensor = torch.stack([i[1] for i in batch])

    

#     return [sequence_tensor,mask_tensor]





    

    

text_dataset_dict = dict()

text_dataloader_dict = dict()



BATCH_SIZE = 10

for col in text_columns:

    text_dataset_dict[col] = TextDataset(data.copy(), col)

    text_dataloader_dict[col] = DataLoader(text_dataset_dict[col],

                                           batch_size= BATCH_SIZE)

        

        

        
def get_vectors(dataloader,dim=768, verbose=False):

    

    vectors = np.zeros((len(dataloader.dataset),dim))

    

    if verbose:

        print("Running for column: %s"%(dataloader.dataset.col_name))

    j = 0

       

    for i,batch in enumerate(dataloader):

        

        

        temp_input = batch[0].to(device)

        temp_mask = batch[1].to(device)

        if verbose:

            print("Running Batch Number: %d"%i)

            

        with torch.no_grad():

            

            train_hidden_state = model(temp_input, temp_mask)

            train_hidden_state = train_hidden_state[0].cpu()

        

        vectors[j:j+len(temp_input),:] = train_hidden_state[:,0,:].numpy()

        

        j+=len(temp_input)

        

        

    return vectors

        

                        
vectors_dict=dict()

for col in text_columns:

     vectors_dict[col] = get_vectors(text_dataloader_dict[col])

        
bert_vectors = np.hstack(list(vectors_dict.values()))



del vector_dict

gc.collect()

torch.cuda.empty_cache()
# bert_vectors = np.load("../input/distilbertembedding/saved_vectors.npy")
module_url = "../input/universalsentenceencoderlarge4/"

embed = hub.load(module_url)

embed_dim = 512 

embeddings ={}

embed_verbose = False



for text in text_columns:

    if embed_verbose:

        print("Running for column: %s"%text)

    data_text = (data[text]

                 .str.replace('?', '.')

                 .str.replace('!', '.')

                 .tolist())

    

    curr_data_emb = []

    batch_size = 2

   

    text_embedding = np.zeros((len(data_text),embed_dim))

    

    ind = 0

    

    while True:

        if embed_verbose:

            print("Running Batch: %d"%ind)

        if (ind+1)*batch_size< len(data_text):

            text_embedding[ind*batch_size:(ind+1)*batch_size,:]= (embed(data_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())

        else:

            

            text_embedding[ind*batch_size:,:]= (embed(data_text[ind*batch_size:])["outputs"].numpy())

            break

        ind += 1

        

        

    embeddings[text+"_USE"] = text_embedding

      

del embed

K.clear_session()

gc.collect()

torch.cuda.empty_cache()
use_vectors = np.hstack(list(embeddings.values()))
# use_vectors = np.load("../input/use-vectors/use_vectors.npy")
l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)



cos_dist = lambda x, y: (x*y).sum(axis=1)





def create_dist_features(vectors,dim=None,stack_array=True):

    

    features= np.array(

        [l2_dist(vectors[:,:dim],vectors[:,dim*2:]),

         l2_dist(vectors[:,dim:dim*2],vectors[:,dim*2:]),

         l2_dist(vectors[:,dim:dim*2],vectors[:,:dim]),

         

         cos_dist(vectors[:,:dim],vectors[:,dim*2:]),

         cos_dist(vectors[:,dim:dim*2],vectors[:,dim*2:]),

         cos_dist(vectors[:,dim:dim*2],vectors[:,:dim])

         ]).T

    

    if stack_array:

        

        return np.hstack([vectors,features])

    

    return features

    
bert_vectors = create_dist_features(bert_vectors, dim = 768)

use_vectors = create_dist_features(use_vectors, dim = 512)
find = re.compile(r"^[^.]*")

data.loc[:,'netloc'] = data['url'].apply(lambda x: 

                                         re.findall(find, urlparse(x).netloc)[0])



ohe  = OneHotEncoder(sparse = False)



topic_features = ohe.fit_transform(data[['netloc','category']],)

tfidf_vec_dict = {}

for col in text_columns:

    tfidf = TfidfVectorizer(strip_accents = 'ascii', 

                        stop_words = 'english',

                        ngram_range = (1,3),

                        max_features = 300,

                       )

    vec_train = tfidf.fit_transform(train['question_title']).toarray()

    vec_test = tfidf.transform(test['question_title']).toarray()

    tfidf_vec_dict[col] = np.vstack([vec_train, vec_test])

    
tfidf_vectors = np.hstack(list(tfidf_vec_dict.values()))
all_vectors = np.hstack([bert_vectors, use_vectors, topic_features, tfidf_vectors])
# x_train, x_test, y_train, y_test = train_test_split(all_vectors[:len(train),:],

#                                                     train[target_columns].values,

#                                                     test_size = 0.20, random_state = 42)



# x_train = torch.tensor(x_train).float()

# x_test = torch.tensor(x_test).float()

# y_train = torch.tensor(y_train).float()

# y_test = torch.tensor(y_test).float()
class LogCoshLoss(torch.nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, y_t, y_prime_t):

        ey_t = y_t - y_prime_t

        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


criterion = nn.BCELoss()

logCosh = LogCoshLoss()



class Classifier(nn.Module):

    

    def __init__(self,input_dim, output_dim, drop_out=0.2):

        

        super().__init__()

        

        self.drop_out = drop_out

        #self.layer = nn.Linear(input_dim, output_dim)

        self.layer1 = nn.Linear(input_dim, 512)

        self.layer2 = nn.Linear(512, output_dim)

        

    def forward(self,x):

        

        x = F.gelu(self.layer1(x))

        x = nn.Dropout(p=self.drop_out)(x)

        x = self.layer2(x)

        x = torch.sigmoid(x)

        

        return x

        

   

def SpearmanCorr(output, target):

    

    x = output

    y = target

    

    vx = x - torch.mean(x,axis=0,keepdim=True)

    vy = y - torch.mean(y, axis =0,keepdim=True)

    

    

    cost = 1.0*torch.sum(vx * vy,axis=0,keepdim=True) / (torch.sqrt(torch.sum(vx ** 2,axis=0,keepdim=True)) * torch.sqrt(torch.sum(vy ** 2,axis=0,keepdim=True)))

    

    return cost



def train_model(x_train,y_train, x_test, y_test,num_epoch = 20, thresh = 1e-3):

    

    prev_val_cor = 0

    model_state_dict = None

    

    for i in range(num_epoch):

        

        classifier.train()

        

        output = classifier(x_train)

        

        with torch.no_grad():

            spearman_loss = -1*SpearmanCorr(output,y_train).mean()

        

        loss = criterion(output, y_train) + logCosh(y_train,output)

        

        print("Spearman Loss",spearman_loss)

#         print("BCE Loss",bce_loss)

        

#         loss = (spearman_loss + bce_loss)

        

        print("Running Epoch: %d"%i)

        

        print("Training Loss: %f"%(loss.item()))

        

        optimizer.zero_grad()

        

        loss.backward()

        

        optimizer.step()

        

        with torch.no_grad():

            classifier.eval()

            val_cor = SpearmanCorr(classifier(x_test),

                                    y_test).mean()

            

            if (val_cor-prev_val_cor> thresh):

                prev_val_cor = val_cor

                model_state_dict = classifier.state_dict()

            

            print("Val Corr: %f"%val_cor.item()) 

            

    return model_state_dict

    
n_fold = 5



kf = KFold(n_fold, shuffle=True, random_state=42)

X_train = all_vectors[:len(train),:]

X_test = torch.tensor(all_vectors[len(train):,:]).float()

Y_train = train[target_columns].values



final_predictions = np.zeros((n_fold,len(test),len(target_columns)))



i_fold = 0

for train_index, test_index in kf.split(X_train):

    

    print("Running Fold: %d"%(i_fold))

    x_train, x_test = X_train[train_index], X_train[test_index]

    y_train, y_test = Y_train[train_index], Y_train[test_index]

    

    x_train = torch.tensor(x_train).float()

    x_test = torch.tensor(x_test).float()

    y_train = torch.tensor(y_train).float()

    y_test = torch.tensor(y_test).float()

    

    classifier = Classifier(all_vectors.shape[1],len(target_columns),

                        drop_out =0.3)



    optimizer = optim.Adam(classifier.parameters(), lr = 1e-3)



    model_state_dict=train_model(x_train,

                        y_train,

                        x_test,

                        y_test,

                        num_epoch = 150,

                        thresh = 1e-3)



    final_classifier = Classifier(all_vectors.shape[1],len(target_columns),

                            drop_out =0.3)



    final_classifier.load_state_dict(model_state_dict)

    final_classifier.eval()

    

    with torch.no_grad():

        

        Y_test = final_classifier(X_test)

        

    final_predictions[i_fold,:,:] = Y_test.numpy()

    

    i_fold+=1

# classifier = Classifier(all_vectors.shape[1],len(target_columns),

#                         drop_out =0.5)



# optimizer = optim.Adam(classifier.parameters(), lr = 1e-3)



# model_state_dict=train_model(x_train,

#                     y_train,

#                     x_test,

#                     y_test,

#                     num_epoch = 100)



# final_classifier = Classifier(all_vectors.shape[1],len(target_columns),

#                         drop_out =0.5)



# final_classifier.load_state_dict(model_state_dict)

# final_classifier.eval()
# preds = np.array([[rankdata(row) for row in fold] for fold in final_predictions]).mean(axis=0)

# max_val = preds.max() + 1

# final_preds = preds/max_val +1e-12
sample.loc[:,target_columns] = np.mean(final_predictions, axis=0)
sample.to_csv("submission.csv",index=False)