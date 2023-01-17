import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import  pickle

import spacy



nlp = spacy.load('en_core_web_sm')

!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv

!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv

!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv

!pip install bert-embedding

!pip install https://github.com/dmlc/gluon-nlp/tarball/master

!pip install bert
import mxnet as mx

from bert_embedding import BertEmbedding
train_df = pd.read_csv('gap-test.tsv', sep='\t')

test_df = pd.read_csv('gap-development.tsv', sep='\t')

dev_df = pd.read_csv('gap-validation.tsv', sep='\t')
embed_name = 'bert_12_768_12_uncased'

#this runs much faster on GPU but I could not get it working in kernel even with GPU on.

#Should just uninstall mxnet and install mxnet_cu92

#ctx = mx.gpu(0)

#bert = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased', max_seq_length=512, batch_size=8, ctx=ctx)

bert = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased', max_seq_length=512, batch_size=16)
%%time

cache_file =  'embed_dev_{0}_orig.pkl'.format(embed_name)      

if not os.path.isfile(cache_file):

    dev_bert_orig = bert.embedding(dev_df.Text.values[:])  

    with open(cache_file, 'wb') as f:

        pickle.dump(dev_bert_orig, f)

else:        

    with open(cache_file, 'rb') as f:

        dev_bert_orig = pickle.load(f)
%%time



cache_file =  'embed_train_{0}_orig.pkl'.format(embed_name)     

if not os.path.isfile(cache_file):

    train_bert_orig = bert.embedding(train_df.Text.values[:])  

    with open(cache_file, 'wb') as f:

        pickle.dump(train_bert_orig, f)

else:        

    with open(cache_file, 'rb') as f:

        train_bert_orig = pickle.load(f)
%%time



cache_file =  'embed_test_{0}_orig.pkl'.format(embed_name)

if not os.path.isfile(cache_file):

    test_bert_orig = bert.embedding(test_df.Text.values[:])  

    with open(cache_file, 'wb') as f:

        pickle.dump(test_bert_orig, f)

else:        

    with open(cache_file, 'rb') as f:

        test_bert_orig = pickle.load(f)


def get_embedding(df, embed_orig):    

    bert_embed = []

    for i, row in enumerate(df.iloc[:].iterrows()):

        

        text = row[1]['Text']

        #There is only one double space in the GAP data but it does throw off the spacy indexing. 

        #It might be better to not do this line and work out making an extra index in the embedding

        #Otherwhise you need to do this replace everytime you use the data in your model as well

        doc = nlp(text.replace('  ', ' '))

        doc_embed = embed_orig[i];

        doc_embed_new = [];

        offset = 0

        spacy_offset = 0

        #print(text)

        for w_i in range(len(doc)):

            if (w_i + spacy_offset) >= len(doc): break

            w = doc[w_i+spacy_offset]

            if (w.i + offset) >= len(doc_embed[1]):

                print (i, 'need longer embedding')         

                break

                

            spacy_text = w.text.lower()

            

            embed_word = doc_embed[0][w.i + offset]

            embed_vector = doc_embed[1][w.i + offset]

            part_count = 1

            

            #print(i, spacy_text, embed_word, len(embed_word) ,len(spacy_text), spacy_offset)

            while len(embed_word) > len(spacy_text):

                spacy_offset+=1



                embed_vector/=part_count            

                doc_embed_new.append(np.array(embed_vector))

                w = doc[w_i+spacy_offset] 

                spacy_text += w.text.lower()   

                offset-=1

            

            while(embed_word != spacy_text and len(embed_word) <= len(spacy_text)):

                offset+=1     

                part_count+=1

                

                if (w.i + offset) >= len(doc_embed[1]): 

                    print('Should not happend', w.text)

                    break

                embed_word += doc_embed[0][w.i + offset]

                embed_vector += doc_embed[1][w.i + offset]



            #if you have issues on new data try just running these two while loops again.

            

            embed_vector/=part_count            

            doc_embed_new.append(np.array(embed_vector))

            if (spacy_text != embed_word):

                print(i, 'Should not happend', spacy_text, embed_word, len(embed_word) ,len(spacy_text))

                

            

        bert_embed.append(np.array(doc_embed_new))    

    return np.array(bert_embed)
%%time

cache_file =  'embed_train_{0}_spacy.pkl'.format(embed_name)

if not os.path.isfile(cache_file):

    embed_train = get_embedding(train_df, train_bert_orig)

    with open(cache_file, 'wb') as f:

        pickle.dump(embed_train, f)

else:        

    with open(cache_file, 'rb') as f:

        embed_train = pickle.load(f)
%%time

cache_file =  'embed_test_{0}_spacy.pkl'.format(embed_name)

if not os.path.isfile(cache_file):

    embed_test = get_embedding(test_df, test_bert_orig)

    with open(cache_file, 'wb') as f:

        pickle.dump(embed_test, f)

else:        

    with open(cache_file, 'rb') as f:

        embed_test = pickle.load(f)
%%time

cache_file =  'embed_dev_{0}_spacy.pkl'.format(embed_name)

if not os.path.isfile(cache_file):

    embed_dev = get_embedding(dev_df, dev_bert_orig)

    with open(cache_file, 'wb') as f:

        pickle.dump(embed_dev, f)

else:        

    with open(cache_file, 'rb') as f:

        embed_dev = pickle.load(f)