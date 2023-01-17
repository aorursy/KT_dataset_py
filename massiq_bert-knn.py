!pip install sentence-transformers

!pip install annoy

!pip install bert-extractive-summarizer
from sentence_transformers import SentenceTransformer

from annoy import AnnoyIndex

import pandas as pd

import torch

import os

import warnings

warnings.filterwarnings("ignore")
input_file = '/kaggle/input/combining-csvfiles/combined_dataset.csv'

data = pd.read_csv(input_file)

data = data.fillna('')

title = data['combined_title']

abstract = data['abstract']

text = data['text']

paper_id = data['paper_id']

authors = data['authors']

print(data.info())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_model.to(device)

print()
sentence_embeddings = sentence_model.encode(title.values.tolist())

print('Dimensionality of the embeddings: ', len(sentence_embeddings[0]))
embed_dim = 768

tree = AnnoyIndex(embed_dim, "dot")



for i, vec in enumerate(sentence_embeddings):

    tree.add_item(i, vec)



tree.build(20)

tree.save('/kaggle/working/titles_bert_emb.ann')

del sentence_embeddings[:]
questions = {

    'q_1' : ["what are the effects of COVID-19 or coronavirus on pregnant women?"],

    'q_2' : ["what are the effects of COVID-19 or coronavirus on new born babies?"],

    'q_3' : ['what are the effects of COVID-19 or coronavirus on cancer patients?'],

    'q_4' : ['Which age group is more vulnerable to covid-19?'],

    'q_5' : ['what are most common underlying diseases in covid-19 patients?'],

    'q_6' : ['What are the effects of social distancing?'],

    'q_7' : ['What are the psychological effects of covid-19 on medical staff?'],

    'q_8' : ['What are the control strategies to curtail transmission of covid-19?'],

    'q_9' : ['what are the public health mitigation measures that could be effective for control of covid-19?'],

    'q_10' : ['What are the economic and behavioral impacts of covid-19 pandemic or coronavirus, what are different socio-economic and behavioral factors arised as a result of covid-19 that can affect economy? What is the difference between groups for risk for COVID-19 by education level? by income? by race and ethnicity? by contact with wildlife markets? by occupation? household size? for institutionalized vs. non-institutionalized populations (long-term hospitalizations, prisons)?'],

    'q_11' : ['what are the transmission dynamics of the covid-19, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors?'],

    'q_12' : ['what are the public measures to control the spread of covid-19?']

}
tree = AnnoyIndex(embed_dim, 'dot')

tree.load('/kaggle/working/titles_bert_emb.ann')
output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])

question = questions['q_1']

question_emb = sentence_model.encode(question)

title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results

print('QUESTION: ',questions['q_1'][0])

title_list = title.values.tolist()

for i, o in enumerate(title_output):

  print('-------')

  print(i)

  print('Title: ',title_list[o])

  print('Paper Id: ', paper_id[o])

  print('Authors: ', authors[o])

  print('\n')

  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])

  output = output.append(df, ignore_index=True)



output_file = 'pregnant_women.csv'

output.to_csv(output_file, index=False)
output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])

question = questions['q_2']

question_emb = sentence_model.encode(question)

title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results

print('QUESTION: ',questions['q_2'][0])

title_list = title.values.tolist()

for i, o in enumerate(title_output):

  print('-------')

  print(i)

  print('Title: ',title_list[o])

  print('Paper Id: ', paper_id[o])

  print('Authors: ', authors[o])

  print('\n')

  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])

  output = output.append(df, ignore_index=True)



output_file = 'neonates.csv'

output.to_csv(output_file, index=False)
output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])

question = questions['q_3']

question_emb = sentence_model.encode(question)

title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results

print('QUESTION: ',questions['q_3'][0])

title_list = title.values.tolist()

for i, o in enumerate(title_output):

  print('-------')

  print(i)

  print('Title: ',title_list[o])

  print('Paper Id: ', paper_id[o])

  print('Authors: ', authors[o])

  print('\n')

  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])

  output = output.append(df, ignore_index=True)



output_file = 'cancer_patients.csv'

output.to_csv(output_file, index=False)
output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])

question = questions['q_4']

question_emb = sentence_model.encode(question)

title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results

print('QUESTION: ',questions['q_4'][0])

title_list = title.values.tolist()

for i, o in enumerate(title_output):

  print('-------')

  print(i)

  print('Title: ',title_list[o])

  print('Paper Id: ', paper_id[o])

  print('Authors: ', authors[o])

  print('\n')

  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])

  output = output.append(df, ignore_index=True)



output_file = 'vulnerable_groups.csv'

output.to_csv(output_file, index=False)
output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])

question = questions['q_5']

question_emb = sentence_model.encode(question)

title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results

print('QUESTION: ',questions['q_5'][0])

title_list = title.values.tolist()

for i, o in enumerate(title_output):

  print('-------')

  print(i)

  print('Title: ',title_list[o])

  print('Paper Id: ', paper_id[o])

  print('Authors: ', authors[o])

  print('\n')

  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])

  output = output.append(df, ignore_index=True)



output_file = 'underlying_diseases.csv'

output.to_csv(output_file, index=False)
output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])

question = questions['q_6']

question_emb = sentence_model.encode(question)

title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results

print('QUESTION: ',questions['q_6'][0])

title_list = title.values.tolist()

for i, o in enumerate(title_output):

  print('-------')

  print(i)

  print('Title: ',title_list[o])

  print('Paper Id: ', paper_id[o])

  print('Authors: ', authors[o])

  print('\n')

  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])

  output = output.append(df, ignore_index=True)



output_file = 'social_distancing.csv'

output.to_csv(output_file, index=False)
output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])

question = questions['q_7']

question_emb = sentence_model.encode(question)

title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results

print('QUESTION: ',questions['q_7'][0])

title_list = title.values.tolist()

for i, o in enumerate(title_output):

  print('-------')

  print(i)

  print('Title: ',title_list[o])

  print('Paper Id: ', paper_id[o])

  print('Authors: ', authors[o])

  print('\n')

  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])

  output = output.append(df, ignore_index=True)



output_file = 'medical_staff.csv'

output.to_csv(output_file, index=False)
output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])

question = questions['q_8']

question_emb = sentence_model.encode(question)

title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results

print('QUESTION: ',questions['q_8'][0])

title_list = title.values.tolist()

for i, o in enumerate(title_output):

  print('-------')

  print(i)

  print('Title: ',title_list[o])

  print('Paper Id: ', paper_id[o])

  print('Authors: ', authors[o])

  print('\n')

  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])

  output = output.append(df, ignore_index=True)



output_file = 'controlling_spread.csv'

output.to_csv(output_file, index=False)
output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])

question = questions['q_9']

question_emb = sentence_model.encode(question)

title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results

print('QUESTION: ',questions['q_9'][0])

title_list = title.values.tolist()

for i, o in enumerate(title_output):

  print('-------')

  print(i)

  print('Title: ',title_list[o])

  print('Paper Id: ', paper_id[o])

  print('Authors: ', authors[o])

  print('\n')

  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])

  output = output.append(df, ignore_index=True)



output_file = 'public_health_measures.csv'

output.to_csv(output_file, index=False)
output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])

question = questions['q_10']

question_emb = sentence_model.encode(question)

title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results

print('QUESTION: ',questions['q_10'][0])

title_list = title.values.tolist()

for i, o in enumerate(title_output):

  print('-------')

  print(i)

  print('Title: ',title_list[o])

  print('Paper Id: ', paper_id[o])

  print('Authors: ', authors[o])

  print('\n')

  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])

  output = output.append(df, ignore_index=True)



output_file = 'economic_impacts.csv'

output.to_csv(output_file, index=False)
output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])

question = questions['q_11']

question_emb = sentence_model.encode(question)

title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results

print('QUESTION: ',questions['q_11'][0])

title_list = title.values.tolist()

for i, o in enumerate(title_output):

  print('-------')

  print(i)

  print('Title: ',title_list[o])

  print('Paper Id: ', paper_id[o])

  print('Authors: ', authors[o])

  print('\n')

  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])

  output = output.append(df, ignore_index=True)



output_file = 'transmission_dynamics.csv'

output.to_csv(output_file, index=False)
output = pd.DataFrame(columns=['paper_id', 'authors', 'title'])

question = questions['q_12']

question_emb = sentence_model.encode(question)

title_output = tree.get_nns_by_vector(question_emb[0], 10) #top 10 results

print('QUESTION: ',questions['q_12'][0])

title_list = title.values.tolist()

for i, o in enumerate(title_output):

  print('-------')

  print(i)

  print('Title: ',title_list[o])

  print('Paper Id: ', paper_id[o])

  print('Authors: ', authors[o])

  print('\n')

  df = pd.DataFrame([[paper_id[o], authors[o], title[o]]], columns=['paper_id', 'authors', 'title'])

  output = output.append(df, ignore_index=True)



output_file = 'public_measures_controlling_spread.csv'

output.to_csv(output_file, index=False)