! pip install embedding-as-service
from embedding_as_service.text.encode import Encoder  

en = Encoder(embedding='bert', model='bert_base_cased', max_seq_length=90) 
import pickle

import numpy as np



def process_data(file,output):

    

    y_fp = open('label_'+output,'wb')

    sen_fp = open('sen_'+output,'wb')

    ent_fp = open('ent_'+output,'wb')

    mask_fp = open('mask_'+output,'wb')

    

    y_list = []

    sen_list = []

    mask_list = []

    ent_list = []

    

    lines = file.readlines()

    

    for line in lines:

        group = line.split('\t')

        

        y_one = np.array([int(i) for i in group[0].split(',')])

        sen = group[2]

        word_one = group[4].replace('\n','').replace(' ','').split(',')

        

        entity = []

        for index,item in enumerate(word_one):

            if item == '1':

                entity.append(sen.split()[index])

                

        sen_vec = en.encode(texts=[sen])[0]

        sen_mask = [False] * 90

        for i in range(len(sen.split())):

            sen_mask[i] = True

        ent_vecs = en.encode(texts=entity,pooling='reduce_mean')

        

        y_list.append(y_one)

        sen_list.append(sen_vec)

        mask_list.append(sen_mask)

        ent_list.append(ent_vecs)

    

    pickle.dump(np.array(y_list),y_fp)

    pickle.dump(np.array(sen_list),sen_fp)

    pickle.dump(np.array(mask_list),mask_fp)

    pickle.dump(np.array(ent_list),ent_fp)

    

    y_fp.close()

    sen_fp.close()

    ent_fp.close()

    



with open('../input/word_testlable.txt','r') as test:

    process_data(test,'test.pkl')

    

with open('../input/word_trainlable.txt','r') as train:

    process_data(train,'train.pkl')



    