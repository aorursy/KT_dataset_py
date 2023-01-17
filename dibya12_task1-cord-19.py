
"""
@author: Dibya, Noor and Rajdeep
"""

import spacy 

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import PhraseMatcher

#Statement to read the metadata.
meta = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
#The suggestion from https://towardsdatascience.com/how-to-get-started-analyzing-covid-19-data-808822437c32 was very helpful. We modified as per our requirement.
nlp = spacy.load("en_core_web_sm")

vectorabst_dict = {}
vectortitle_dict = {}
# We took title and abstract to search for relevant articles related to COVID-19
for sha, title, abstract in tqdm(meta[["sha","title","abstract"]].values):
    if (isinstance(abstract, str)):
        vectorabst_dict[sha] = nlp(abstract).vector
    if (isinstance(title,str)):
        vectortitle_dict[sha] = nlp(title).vector         

#Like shown below other values can also be created. Because in spome metadata information, title has some information even though abstract does not and vice versa.
#Some of the licenses also have the keyword covid which allows to eliminate the records which does not have mention of covid. However we did not want to exclude some of the articles at this stage.
values = list(vectorabst_dict.values())
values1 = list(vectortitle_dict.values())
#cosine similarity matrix to find nearest similar matrix
cosine_sim_matrix_abst = cosine_similarity(values, values)
cosine_sim_matrix_title = cosine_similarity(values1, values1)
keys = list(vectorabst_dict.keys())


n_return = 50

#finding similar articles as per the query in abstracts. We have added 2 keywords such as coronavisrus and covid to narrow the search     
queriestogether = ("Coronavirus, covid ,Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery,Prevalence of asymptomatic shedding and transmission (e.g., particularly children),Seasonality of transmission,Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding),Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood),Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic),Natural history of the virus and shedding of it from an infected person,Implementation of diagnostics and products to improve clinical processes,Disease models, including animal models for infection, disease and transmission, Tools and studies to monitor phenotypic change and potential adaptation of the virus, Immune response and immunity, Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings, Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings, Role of the environment in transmission")  
query_vector = nlp(queriestogether).vector
cosine_sim_matrix_query = cosine_similarity(values, query_vector.reshape(1,-1))
query_sim_indexes = np.argsort(cosine_sim_matrix_query.reshape(1,-1)[0])[::-1][:n_return]
query_shas = [keys[i] for i in query_sim_indexes]
meta_info_query = meta[meta.sha.isin(query_shas)]

print(f"----Similar articles-----")
for abst in meta_info_query.abstract.values:
    print(abst)
    print("---------") 