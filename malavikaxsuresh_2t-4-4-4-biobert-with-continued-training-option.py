run_type='load' #save/load



add_tokens=0



method = '2' #1/2 - defines how to use manconcorpus

mancon_data_to_use = 'all' #all/equal - defines how many training pairs to use from mancon corpus

multinli_data_to_use = 100000 #0/number- 0 implies all, defines how many trainings pairs to use from multinli corpus

multinli_iteration = 3 #defines current iteration of continued training

mednli_data_to_use = 0 #0/number- 0 implies all, defines how many trainings pairs per class to use from mednli corpus

model_name = 'allenai/biobert-roberta-base' #"allenai/biobert-roberta-base"/"deepset/covid_bert_base"



model_continue = 0 #0/1 - whether load & continue fine-tuning of model

model_continue_sigmoid_path = "/kaggle/input/biobertmultinlipart2/sigmoid.pickle"

model_continue_transformer_path = "/kaggle/input/biobertmultinlipart2/transformer"



finetune_multinli = 0

finetune_mednli = 0

finetune_mancon = 0
#!pip install transformers
import os

import shutil

import json

import numpy as np

import pandas as pd

import xml.etree.ElementTree as et 

from itertools import permutations



from keras.utils import np_utils

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.callbacks import ModelCheckpoint

import transformers

from transformers import AutoModel

from transformers import TFAutoModel, AutoTokenizer, AutoModelWithLMHead

from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping

import pickle
multinli_data = pd.read_csv('/kaggle/input/multinli/multinli_1.0_train.txt', sep='\t', error_bad_lines=False)

multinli_test_data = pd.read_csv('/kaggle/input/multinli-dev/multinli_1.0_dev_matched.txt', sep='\t', error_bad_lines=False)



multinli_data['gold_label'] = [2 if l=='contradiction' else 1 if l=='entailment' else 0 for l in multinli_data.gold_label]

multinli_test_data['gold_label'] = [2 if l=='contradiction' else 1 if l=='entailment' else 0 for l in multinli_test_data.gold_label]



if multinli_data_to_use!=0:

        print('Using only a subset of multiNLi for training')

        #temp = multinli_data[multinli_data.gold_label==2].head(multinli_data_to_use).append(multinli_data[multinli_data.gold_label==1].head(multinli_data_to_use)).reset_index(drop=True)

        #multinli_data = temp.append(multinli_data[multinli_data.gold_label==0].head(multinli_data_to_use)).reset_index(drop=True)

        multinli_data = multinli_data.head(multinli_data_to_use*multinli_iteration).tail(multinli_data_to_use)



x_train = '[CLS]'+multinli_data.sentence1+'[SEP]'+multinli_data.sentence2

x_test = '[CLS]'+multinli_test_data.sentence1+'[SEP]'+multinli_test_data.sentence2

y_train = np_utils.to_categorical(multinli_data.gold_label)

y_test = np_utils.to_categorical(multinli_test_data.gold_label)
mednli_data1 = pd.DataFrame()

mednli_data2 = pd.DataFrame()

mednli_test_data = pd.DataFrame()



with open('/kaggle/input/mednli/mli_train_v1.jsonl', 'r', encoding='utf-8') as f:

    for line in f:

        mednli_data1 = mednli_data1.append(json.loads(line.rstrip('\n|\r')),ignore_index=True)

with open('/kaggle/input/mednli/mli_dev_v1.jsonl', 'r', encoding='utf-8') as f:

    for line in f:

        mednli_data2 = mednli_data2.append(json.loads(line.rstrip('\n|\r')),ignore_index=True)

mednli_data = mednli_data1.append(mednli_data2, ignore_index=True).reset_index(drop=True)



with open('/kaggle/input/mednli/mli_test_v1.jsonl', 'r', encoding='utf-8') as f:

    for line in f:

        mednli_test_data = mednli_test_data.append(json.loads(line.rstrip('\n|\r')),ignore_index=True)



mednli_data['gold_label'] = [2 if l=='contradiction' else 1 if l=='entailment' else 0 for l in mednli_data.gold_label]

mednli_test_data['gold_label'] = [2 if l=='contradiction' else 1 if l=='entailment' else 0 for l in mednli_test_data.gold_label]



if mednli_data_to_use!=0:

        print('Using only a subset of multiNLi for training')

        temp = mednli_data[mednli_data.gold_label==2].head(mednli_data_to_use).append(mednli_data[mednli_data.gold_label==1].head(mednli_data_to_use)).reset_index(drop=True)

        mednli_data = temp.append(mednli_data[mednli_data.gold_label==0].head(mednli_data_to_use)).reset_index(drop=True)



x_train_3 = '[CLS]'+mednli_data.sentence1+'[SEP]'+mednli_data.sentence2

x_test_3 = '[CLS]'+mednli_test_data.sentence1+'[SEP]'+mednli_test_data.sentence2

y_train_3 = np_utils.to_categorical(mednli_data.gold_label)

y_test_3 = np_utils.to_categorical(mednli_test_data.gold_label)
if method=='1':

    xtree = et.parse('/kaggle/input/manconcorpus/ManConCorpus.xml')

    xroot = xtree.getroot() 



    manconcorpus_data = pd.DataFrame(columns = ['claim','assertion','question'])



    for node in xroot:

        for claim in node.findall('CLAIM'):

            manconcorpus_data = manconcorpus_data.append({'claim':claim.text,\

                                                        'assertion':claim.attrib.get('ASSERTION'),\

                                                        'question':claim.attrib.get('QUESTION')},

                                                         ignore_index=True)

    print(len(manconcorpus_data))
if run_type=='save' and method=='1':

    questions = list(set(manconcorpus_data.question))

    con = pd.DataFrame(columns=['claim1','claim2','label'])

    ent = pd.DataFrame(columns=['claim1','claim2','label'])



    for q in questions:

        claim_yes = pd.DataFrame(manconcorpus_data.loc[(manconcorpus_data.question==q) & (manconcorpus_data.assertion=='YS'),'claim'])

        claim_no = pd.DataFrame(manconcorpus_data.loc[(manconcorpus_data.question==q) & (manconcorpus_data.assertion=='NO'),'claim'])

        temp = claim_yes.assign(key=1).merge(claim_no.assign(key=1), on='key').drop('key', 1)

        temp1 = temp.rename(columns={'claim_x':'claim1','claim_y':'claim2'})

        con = con.append(temp1)

        #Swap claim1 & claim2 to generate more examples. This will handle directionality during fine-tuning.

        temp2 = temp.rename(columns={'claim_x':'claim2','claim_y':'claim1'})

        con = con.append(temp2)

        con['label'] = 1   

        con.drop_duplicates(inplace=True)



        for i,j in list(permutations(claim_yes.index, 2)):

            ent = ent.append({'claim1':claim_yes.claim[i],\

                        'claim2':claim_yes.claim[j],\

                        'label':0},\

                       ignore_index=True)



        for i,j in list(permutations(claim_no.index, 2)):

            ent = ent.append({'claim1':claim_no.claim[i],\

                        'claim2':claim_no.claim[j],\

                        'label':0},\

                       ignore_index=True)



    transfer_data = pd.concat([con,ent]).reset_index(drop=True)

    transfer_data['label'] = transfer_data.label.astype('float')

    print(len(con))

    print(len(ent))
if run_type=='save' and method=='1':

    x_train_2,x_test_2,y_train_2,y_test_2=train_test_split('[CLS]'+transfer_data.claim1+'[SEP]'+transfer_data.claim2,transfer_data['label'],test_size=0.2)

    print(y_train_2.sum())

    print(y_test_2.sum())
# if run_type=='save' and method=='2':

transfer_data = pd.read_csv('/kaggle/input/manconcorpus-sent-pairs/manconcorpus_sent_pairs_200516.tsv', sep ='\t')

transfer_data['label'] = [2 if l=='contradiction' else 1 if l=='entailment' else 0 for l in transfer_data.label]

transfer_data['label'] = transfer_data.label.astype('float')

print(len(transfer_data[transfer_data.label==2]))

print(len(transfer_data[transfer_data.label==1]))

print(len(transfer_data[transfer_data.label==0]))
if run_type=='save' and method=='2':

    if mancon_data_to_use=='equal':

        temp = transfer_data[transfer_data.label==2].append(transfer_data[transfer_data.label==1].head(1000)).reset_index(drop=True)

        transfer_data = temp.append(transfer_data[transfer_data.label==0].head(1000)).reset_index(drop=True)

    print(len(transfer_data[transfer_data.label==2]))

    print(len(transfer_data[transfer_data.label==1]))

    print(len(transfer_data[transfer_data.label==0]))
if run_type=='save' and method=='2':

    x_train_2,x_test_2,y_train_2,y_test_2=train_test_split('[CLS]'+transfer_data.text_a+'[SEP]'+transfer_data.text_b,transfer_data['label'],test_size=0.2)

    print(len(y_train_2[transfer_data.label==2]))

    print(len(y_train_2[transfer_data.label==1]))

    print(len(y_train_2[transfer_data.label==0]))

    print(len(y_test_2[transfer_data.label==2]))

    print(len(y_test_2[transfer_data.label==1]))

    print(len(y_test_2[transfer_data.label==0]))

    y_train_2 = np_utils.to_categorical(y_train_2)

    y_test_2 = np_utils.to_categorical(y_test_2)
if run_type=='save' and method=='3':

    x_train_2,x_test_2,y_train_2,y_test_2=train_test_split('[CLS]'+manconcorpus_data.question+'[SEP]'+manconcorpus_data.claim1,manconcorpus_data['assertion'],test_size=0.2)
def regular_encode(texts, tokenizer, maxlen=512):

    """ Function to encode many sentences"""

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen,

        sep_token='[SEP]'

    )

    return np.array(enc_di['input_ids'])
def build_model(transformer, max_len=512):

    """

    Require a transformer of type TFAutoBert

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    if method=='1':

        out = Dense(1, activation='sigmoid', name='sigmoid')(cls_token)

    if method=='2':

        out = Dense(3, activation='softmax', name='softmax')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)

    if method=='1':

        model.compile(Adam(lr=1e-6), loss='binary_crossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), 'accuracy'])

    if method=='2':

        model.compile(Adam(lr=1e-6), loss='categorical_crossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.CategoricalAccuracy()])

    return model
def save_model(model, transformer_dir='transformer'):

    """

    Special function to save a keras model that uses a transformer layer

    """

    transformer = model.layers[1]

    !mkdir transformer

    transformer.save_pretrained(transformer_dir)

    sigmoid = model.get_layer(index=3).get_weights()

    pickle.dump(sigmoid, open('sigmoid.pickle', 'wb'))



def load_model(pickle_path, transformer_dir='transformer', max_len=512):

    """

    Special function to load a keras model that uses a transformer layer

    """

    transformer = TFAutoModel.from_pretrained(transformer_dir)

    model = build_model(transformer, max_len=max_len)

    sigmoid = pickle.load(open(pickle_path, 'rb'))

    if method=='1':

        model.get_layer('sigmoid').set_weights(sigmoid)

    if method=='2':

        model.get_layer('softmax').set_weights(sigmoid)



    return model
if run_type=='save':

    print(int(int(x_train.str.len().max())))

    print(int(x_train.str.len().median()))

    print(int(int(x_train_2.str.len().max())))

    print(int(x_train_2.str.len().median()))

    print(int(int(x_train_3.str.len().max())))

    print(int(x_train_3.str.len().median()))
# Configuration params

EPOCHS = 3

MAX_LEN = 512

BATCH_SIZE = 32
drug_names = pd.read_csv('/kaggle/input/drugnames/DrugNames.txt',header=None)

drug_names = list(drug_names[0])

print('Full list of drugs:',len(drug_names))

if method=='1':

    text = ' '.join(list(set(transfer_data.claim1)))

if method=='2':

    text = ' '.join(list(set(transfer_data.text_a)))

drug_names = [drug for drug in drug_names if drug in text]

print('List of drugs in training & testing corpus:',len(drug_names))

virus_names = pd.read_csv('/kaggle/input/virus-words/virus_words.txt',header=None)

virus_names = list(virus_names[0])
if model_name == 'deepset/covid_bert_base':

    MODEL = "deepset/covid_bert_base"

else:

    MODEL = "allenai/biomed_roberta_base"



# First load the real tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL)

if add_tokens==1:

    tokenizer.add_tokens(drug_names+virus_names)
len(tokenizer)
%%time

if run_type=='save':

    x_train_str = []

    for x in x_train:

        x_train_str.append(str(x))



    x_test_str = []

    for x in x_test:

        x_test_str.append(str(x))

    

    x_train = regular_encode(x_train_str, tokenizer, maxlen=MAX_LEN)

    x_test = regular_encode(x_test_str, tokenizer, maxlen=MAX_LEN)

    

    x_train_3_str = []

    for x in x_train_3:

        x_train_3_str.append(str(x))



    x_test_3_str = []

    for x in x_test_3:

        x_test_3_str.append(str(x))

    

    x_train_3 = regular_encode(x_train_3_str, tokenizer, maxlen=MAX_LEN)

    x_test_3 = regular_encode(x_test_3_str, tokenizer, maxlen=MAX_LEN)

    

    x_train_2 = regular_encode(x_train_2.values, tokenizer, maxlen=MAX_LEN)

    x_test_2 = regular_encode(x_test_2.values, tokenizer, maxlen=MAX_LEN)
es = EarlyStopping(monitor='val_accuracy', 

                    min_delta=0.001, 

                    patience=3,

                    verbose=1, 

                    mode='max', 

                    restore_best_weights=True)
# !pip install wandb

# !wandb login

# import wandb

# from wandb.keras import WandbCallback

# wandb.init(project="vt-relation-extract", sync_tensorboard=True)
if run_type=='save' and model_continue==0:

    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    if model_name == 'deepset/covid_bert_base':

        model = AutoModelWithLMHead.from_pretrained("deepset/covid_bert_base")

        model.resize_token_embeddings(len(tokenizer))

        !mkdir covid_bert_base

        model.save_pretrained("covid_bert_base")

        with strategy.scope():

          model = TFAutoModel.from_pretrained("covid_bert_base", from_pt=True)

          #model.resize_token_embeddings(len(tokenizer))

          model = build_model(model)

        !rm -r covid_bert_base

    else:

        model = AutoModel.from_pretrained("allenai/biomed_roberta_base")

        model.resize_token_embeddings(len(tokenizer))

        !mkdir biomed_roberta_base

        model.save_pretrained("biomed_roberta_base")

        with strategy.scope():

          model = TFAutoModel.from_pretrained("biomed_roberta_base", from_pt=True)

          #model.resize_token_embeddings(len(tokenizer))

          model = build_model(model)

        !rm -r biomed_roberta_base

    BATCH_SIZE = 2 * strategy.num_replicas_in_sync
if run_type=='save' and model_continue==1:

    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    with strategy.scope():

        model = load_model(model_continue_sigmoid_path, model_continue_transformer_path)

    BATCH_SIZE = 2 * strategy.num_replicas_in_sync
if run_type=='save':

    model.summary()
# Fine tune on MultiNLI



if run_type=='save' and finetune_multinli==1:

    train_history = model.fit(

                        x_train, y_train,

                        batch_size = BATCH_SIZE,

                        validation_data=(x_test, y_test),

                        #callbacks=[es, WandbCallback()],

                        callbacks=[es],

                        epochs=EPOCHS

                        )
# Fine tune on MedNLI



if run_type=='save' and finetune_mednli==1:

    train_history = model.fit(

                        x_train_3, y_train_3,

                        batch_size = BATCH_SIZE,

                        validation_data=(x_test_3, y_test_3),

                        #callbacks=[es, WandbCallback()],

                        callbacks=[es],

                        epochs=EPOCHS

                        )
# Fine tune on Manconcorpus



if run_type=='save' and finetune_mancon==1:

    train_history = model.fit(

                        x_train_2, y_train_2,

                        batch_size = BATCH_SIZE,

                        validation_data=(x_test_2, y_test_2),

                        #callbacks=[es, WandbCallback()],

                        callbacks=[es],

                        epochs=EPOCHS

                        )
# from google.colab import auth

# from datetime import datetime

# auth.authenticate_user()

# !gsutil cp -r best_epoch_roberta gs://coronaviruspublicdata/temp_data/snapshots
#import pickle 

#save model, input: sentence, output: binary

#pickle.dump(model, open( "bioERT_model1.pickle", "wb" ) )

# !gsutil cp model.pickle gs://coronaviruspublicdata/model.pickle
if run_type=='save':

    save_model(model)

    shutil.make_archive('biobert_output', 'zip', '/kaggle/working/')
if run_type=='load':

    model = load_model("/kaggle/input/biobertfinalmodelv2/sigmoid.pickle", "/kaggle/input/biobertfinalmodelv2/transformer")
# model.summary()

# model.get_layer(index=3)
# !gsutil cp -r transformer3 gs://coronaviruspublicdata/re_final_best2/s

# !gsutil cp sigmoid3.pickle gs://coronaviruspublicdata/re_final_best2/s
output_data = pd.read_excel('/kaggle/input/pilotannotations/Pilot_Contra_Claims_Annotations_06_30 - Copy.xlsx',sheet_name='All_phase2')

output_data = output_data.dropna().reset_index(drop=True)
ls = []

for i in range(len(output_data)):

    ls.append(str('[CLS]'+output_data.loc[i,'claim_1']+'[SEP]'+output_data.loc[i,'claim_2']))

    

test_example = regular_encode(ls, tokenizer, maxlen=MAX_LEN)

predictions = model.predict(test_example)

if method=='1':

    output_data['BioBERT_Prediction'] = [p[0] for p in predictions]

if method=='2':

    output_data['BioBERT_Prediction_con'] = [p[0] for p in predictions]

    output_data['BioBERT_Prediction_ent'] = [p[1] for p in predictions]

    output_data['BioBERT_Prediction_neu'] = [p[2] for p in predictions]
print(len(output_data))

print(len(output_data.loc[output_data.label=='Contradiction',:]))

print(len(output_data.loc[output_data.label=='Entailment',:]))

print(len(output_data.loc[output_data.label=='Neutral',:]))
if method=='1':

    print(max(output_data.BioBERT_Prediction))

if method=='2':

    print(max(output_data.BioBERT_Prediction_con))

    print(max(output_data.BioBERT_Prediction_ent))

    print(max(output_data.BioBERT_Prediction_neu))
if method=='1':

    output_data['label'] = [1 if a=='contradiction' else 0 for a in output_data.annotation]

    output_data['BioBERT_Prediction_class'] = [1 if p>=0.375 else 0 for p in output_data.BioBERT_Prediction]

    

    print('Overall accuracy: '\

      + str(accuracy_score(output_data['label'], output_data['BioBERT_Prediction_class'] )))

    print('Precision: '\

          + str(precision_score(output_data['label'], output_data['BioBERT_Prediction_class'] )))

    print('Recall: '\

          + str(recall_score(output_data['label'], output_data['BioBERT_Prediction_class'] )))

    print('F1 score: '\

          + str(f1_score(output_data['label'], output_data['BioBERT_Prediction_class'] )))



if method=='2':

    output_data['label'] = output_data.label

    output_data['BioBERT_Prediction_class'] = output_data[['BioBERT_Prediction_con','BioBERT_Prediction_ent','BioBERT_Prediction_neu']].idxmax(axis=1)

    output_data['BioBERT_Prediction_class'].replace(to_replace={'BioBERT_Prediction_con':'Contradiction','BioBERT_Prediction_ent':'Entailment','BioBERT_Prediction_neu':'Neutral'}\

                                                   ,inplace=True)

    

    print('Overall accuracy: '\

      + str(accuracy_score(output_data['label'], output_data['BioBERT_Prediction_class'] )))

    print('Precision: '\

          + str(precision_score(output_data['label'], output_data['BioBERT_Prediction_class'], average = None)))

    print('Recall: '\

          + str(recall_score(output_data['label'], output_data['BioBERT_Prediction_class'], average = None)))

    print('F1 score: '\

          + str(f1_score(output_data['label'], output_data['BioBERT_Prediction_class'], average = None)))
output_data.to_csv('bioBERT_Output.csv',header=True)
# !ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi

# !pip install gputil

# !pip install psutil

# !pip install humanize

# import psutil

# import humanize

# import os

# import GPUtil as GPU

# GPUs = GPU.getGPUs()

# # XXX: only one GPU on Colab and isn’t guaranteed

# gpu = GPUs[0]

# def printm():

#  process = psutil.Process(os.getpid())

#  print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))

#  print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

# printm()
# from google.colab import auth

# from datetime import datetime

# auth.authenticate_user()

# !gsutil cp -r transformer gs://coronaviruspublicdata/re_snapshot/4_13_2020

# !gsutil cp sigmoid.pickle gs://coronaviruspublicdata/re_snapshot/4_13_2020