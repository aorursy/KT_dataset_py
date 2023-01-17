# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow_hub as hub

import tensorflow as tf

from scipy import spatial



module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

module_url2 = "https://tfhub.dev/google/universal-sentence-encoder/4"



embed = hub.load(module_url2)



df_biorxiv = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv',header=0)

df_comm_use = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv',header=0)

df_non_comm_use = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv',header=0)

df_pmc = pd.read_csv('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv',header=0)



### Concat all the dataframes

df_all = pd.concat([df_biorxiv,df_comm_use,df_non_comm_use,df_pmc])
main_questions = ['What is known about transmission, incubation, and environmental stability?','What do we know about natural history, transmission, and diagnostics for the virus?','What have we learned about infection prevention and control?']

sub_questions = ['Range of incubation periods for the disease in humans', 'how this varies across age and health status','how long individuals are contagious, even after recovery','Prevalence of asymptomatic shedding and transmission, particularly children.',

    'Seasonality of transmission.','Physical science of the coronavirus, charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding',

    'Persistence and stability on a multitude of substrates and sources, e.g., nasal discharge, sputum, urine, fecal matter, blood.','Persistence of virus on surfaces of different materials, e,g., copper, stainless steel, plastic.',

    'Natural history of the virus and shedding of it from an infected person',

    'Implementation of diagnostics and products to improve clinical processes',

    'Disease models, including animal models for infection, disease and transmission',

    'Tools and studies to monitor phenotypic change and potential adaptation of the virus',

    'Immune response and immunity',

    'Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings',

    'Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings',

    'Role of the environment in transmission']
def obtain_embeddings(X,batch_size=4096):

    X_embeddings = []

    idx = 0



    while True:

        last_idx = len(X) if (idx+batch_size)>len(X) else idx+batch_size

        #print("Debug:last_idx: {}\n\n\n".format(last_idx))

        embeddings = embed(X[idx:last_idx])

        message_embeddings = embeddings

        X_embeddings.extend(message_embeddings)

        idx = idx+batch_size

        if idx > len(X):

            break;



    return X_embeddings;
df_all['title'].fillna("",inplace=True)
title_embeddings = np.array(obtain_embeddings(df_all['title'].values[:], batch_size=256))
main_question_embeddings = np.array(obtain_embeddings(main_questions, batch_size=256))

sub_question_embeddings = np.array(obtain_embeddings(sub_questions, batch_size=256))
distance_matrix = np.zeros((len(main_question_embeddings),len(title_embeddings)))

for i in range(len(main_question_embeddings)):

    for j in range(len(title_embeddings)):

        distance_matrix[i][j] = spatial.distance.cosine(title_embeddings[j], main_question_embeddings[i])

    df_all['MQ'+str(i)]= distance_matrix[i]

    

distance_matrix = np.zeros((len(sub_question_embeddings),len(title_embeddings)))

for i in range(len(sub_question_embeddings)):

    for j in range(len(title_embeddings)):

        distance_matrix[i][j] = spatial.distance.cosine(title_embeddings[j], sub_question_embeddings[i])

    df_all['SQ'+str(i)]= distance_matrix[i]

n = 10
### Retrieve top n paper titles by ignoring the NaNs



for i in range(len(main_question_embeddings)):

    print("Question:", main_questions[i])

    indices = np.argsort(df_all['MQ'+str(i)])

    j=0

    while(j<n):

        if not pd.isnull(df_all['title'].iloc[indices.iloc[j]]):

            print(df_all['title'].iloc[indices.iloc[j]])

            j+=1;

    print("\n\n")

    

for i in range(len(sub_question_embeddings)):

    print("Question:", sub_questions[i])

    indices = np.argsort(df_all['SQ'+str(i)])

    j=0

    while(j<n):

        if not pd.isnull(df_all['title'].iloc[indices.iloc[j]]):

            print(df_all['title'].iloc[indices.iloc[j]])

            j+=1;

    print("\n\n")

            