!pip install nltk -q

import nltk

from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag

sent= '''Prime Minister Jacinda Ardern has claimed that New Zealand had won a big 

battle over the spread of coronavirus. Her words came as the country begins to exit from its lockdown.'''

words= word_tokenize(sent)

postags=pos_tag(words)

ne_tree = nltk.ne_chunk(postags,binary=False)

print(ne_tree)
ne_tree = nltk.ne_chunk(postags,binary=True)

print(ne_tree)
!pip install spacy -q

!python -m spacy download en_core_web_sm
from pprint import pprint

import spacy 

from spacy import displacy

  

nlp = spacy.load('en_core_web_sm') 

  

sentence = '''Prime Minister Jacinda Ardern has claimed that New Zealand had won a big 

battle over the spread of coronavirus. Her words came as the country begins to exit from its lockdown.'''

 

  

entities= nlp(sentence)

#to print all the entities with iob tags 

pprint([ (X, X.ent_iob_, X.ent_type_) for X in entities] )

 

#to print just named entities use this code

print("Named entities in this text are\n")

for ent in entities.ents: 

    print(ent.text,ent.label_)

 

# visualize named entities

displacy.render(entities, style='ent', jupyter=True)
import random

from spacy.gold import GoldParse

#the training data with named entity NE needs to be of this format 

# ( "training example",[(strat position of NE,end position of NE,"type of NE")] )

train_data = [

    ("Uber blew through $1 million a week", [(0, 4, 'ORG')]),

    ("Android Pay expands to Canada", [(0, 11, 'PRODUCT'), (23, 30, 'GPE')]),

    ("Spotify steps up Asia expansion", [(0, 8, "ORG"), (17, 21, "LOC")]),

    ("Google Maps launches location sharing", [(0, 11, "PRODUCT")]),

    ("Google rebrands its business apps", [(0, 6, "ORG")]),

    ("look what i found on google! 😂", [(21, 27, "PRODUCT")])]

#An optimizer is set to update the model’s weights.

optimizer = nlp.begin_training()

for itn in range(100):

    random.shuffle(train_data)

    for raw_text, entity_offsets in train_data:

        doc = nlp.make_doc(raw_text)

        gold = GoldParse(doc, entities=entity_offsets)

        nlp.update([doc], [gold], drop=0.25, sgd=optimizer)

#setting drop makes it harder for the model to just memorize the data.

nlp.to_disk("/model")
from pprint import pprint

sentence_1 = '''i use google to search. Google is a large IT company'''

sentence_2='''Prime Minister Jacinda Ardern has claimed that New Zealand had won a big 

battle over the spread of coronavirus. Her words came as the country begins to exit from its lockdown.'''

  

entities_1= nlp(sentence_1)

entities_2=nlp(sentence_2)

#to print just named entities use this code

print("Named entities in this text are\n")

for ent in entities_1.ents: 

    print(ent.text,ent.label_)

for ent in entities_2.ents: 

    print(ent.text,ent.label_)
#Install Apex for Mix Precision

!cd ../input/apex-master/apex/ && pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
!pip install simpletransformers -q
from simpletransformers.ner import NERModel





# Create a NERModel

model = NERModel('bert', 'bert-base-cased')

# model.args = {

#     "output_dir": "/kaggle/working/",

#     "cache_dir": "/kaggle/working/",



#     "fp16": True,

#     "fp16_opt_level": "O1",

#     "max_seq_length": 128,

#     "train_batch_size": 8,

#     "gradient_accumulation_steps": 1,

#     "eval_batch_size": 8,

#     "num_train_epochs": 1,

#     "weight_decay": 0,

#     "learning_rate": 4e-5,

#     "adam_epsilon": 1e-8,

#     "warmup_ratio": 0.06,

#     "warmup_steps": 0,

#     "max_grad_norm": 1.0,



#     "logging_steps": 50,

#     "save_steps": 2000,



#     "overwrite_output_dir": False,

#     "reprocess_input_data": False,

#     "evaluate_during_training": False,



#     "process_count": 2,

#     "n_gpu": 1,

# }

# model = NERModel('bert', 'bert-base-cased', args={'learning_rate': 2e-5, 'overwrite_output_dir': True, 'reprocess_input_data': True})

model.train_model('../input/conll003-englishversion/train.txt')



#read model

#model = NERModel('bert', 'outputs/')



#evaluate

results, model_outputs, predictions = model.eval_model('../input/conll003-englishversion/valid.txt')



# Check predictions

print(predictions[:5])