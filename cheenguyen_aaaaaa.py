# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import spacy

import os



#for dirname, _, filenames in os.walk('/kaggle/input'):

#   for filename in filenames:

#       print(os.path.join(dirname, filename)

        

# Any results you write to the current directory are saved as output.
!ls /kaggle/input/CORD-19-research-challenge/
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

datata = pd.read_csv(metadata_path)



datata.columns
datata.head()
#Get the abstract

features = ['title', 'abstract']

cool_data = datata[features]

#Remove rows with missing values

cool_data.dropna(axis=0, subset=['abstract'], inplace=True)



cool_data.head()
#For finding paper on transmission and incubation

#key_phrases is an abstract of a paper about the topic of the task

key_phrases = "Respiratory viruses can cause a wide spectrum of pulmonary diseases, ranging from mild, upper respiratory tract infections to severe and life-threatening lower respiratory tract infections, including the development of acute lung injury (ALI) and acute respiratory distress syndrome (ARDS). Viral clearance and subsequent recovery from infection require activation of an effective host immune response; however, many immune effector cells may also cause injury to host tissues. Severe acute respiratory syndrome (SARS) coronavirus and Middle East respiratory syndrome (MERS) coronavirus cause severe infection of the lower respiratory tract, with 10% and 35% overall mortality rates, respectively; however, >50% mortality rates are seen in the aged and immunosuppressed populations. While these viruses are susceptible to interferon treatment in vitro, they both encode numerous genes that allow for successful evasion of the host immune system until after high virus titres have been achieved. In this review, we discuss the importance of the innate immune response and the development of lung pathology following human coronavirus infection."
# Need to load the large model to get the vectors

nlp = spacy.load('en_core_web_lg')



##Testing 

#text_to_test = "Respiratory viruses can cause a wide spectrum of pulmonary diseases, ranging from mild, upper respiratory tract infections to severe and life-threatening lower respiratory tract infections, including the development of acute lung injury (ALI) and acute respiratory distress syndrome (ARDS). Viral clearance and subsequent recovery from infection require activation of an effective host immune response; however, many immune effector cells may also cause injury to host tissues. Severe acute respiratory syndrome (SARS) coronavirus and Middle East respiratory syndrome (MERS) coronavirus cause severe infection of the lower respiratory tract, with 10% and 35% overall mortality rates, respectively; however, >50% mortality rates are seen in the aged and immunosuppressed populations. While these viruses are susceptible to interferon treatment in vitro, they both encode numerous genes that allow for successful evasion of the host immune system until after high virus titres have been achieved. In this review, we discuss the importance of the innate immune response and the development of lung pathology following human coronavirus infection."

#Vectorize the testing text

#test_vec = nlp(text_to_test).vector



#Vectorizing key phrases

key_vec = nlp(key_phrases).vector 



#Function to get the similarity between two vectors

def cosine_similarity(a, b):

    return a.dot(b)/np.sqrt(a.dot(a)*b.dot(b))



#cosine_similarity(test_vec, key_vec)

#Vectorizing all abstracts in the cool_data

#

a_body = cool_data 

with nlp.disable_pipes():

    vectors = np.array([nlp(a_bone.abstract).vector for idx, a_bone in a_body.iterrows()])



vectors.shape
##Center the abstracts in the vectors

#Calculate the mean for vectors

vec_mean = vectors.mean(axis=0)

#Subtract the mean from the vectors

centered = vectors - vec_mean



# Calculate similarities for each abstract

#Recalling key_vec is the vectorized key_phrases 

sims = np.array([cosine_similarity(key_vec - vec_mean, vec) for vec in centered])



print(sims)

most_similar = sims.argmax()

print(sims[most_similar])
results = pd.DataFrame(columns=['Title', 'Abstract', 'Match'])



for value in sims:

    if value > 0.6:

        my_index = np.where(sims == value)

        results = results.append({'Title': datata.iloc[my_index].title, 'Abstract': datata.iloc[my_index].abstract, 'Match': value}, ignore_index=True)

        

results.head()
results.to_csv('submission.csv', index=False)