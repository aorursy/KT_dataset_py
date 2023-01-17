# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
print(tf.__version__)
df_covid_NPI = pd.read_excel('../input/excel1/df_covid_NPI.xlsx')
df_covid_NPI
import nltk
import torch
!pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer
nltk.download('punkt')
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
embeddings_distilbert2 = model.encode(df_covid_NPI.abstract.values)

# Number of top news
K=5
def find_similar(vector_representation, all_representations, k=1):
    similarity_matrix = cosine_similarity(vector_representation, all_representations)
    np.fill_diagonal(similarity_matrix, 0)
    similarities = similarity_matrix[0]
    if k == 1:
        return [np.argmax(similarities)]
    elif k is not None:
        return np.flip(similarities.argsort()[-k:][::1])
df_covid_NPI.head()
descriptions = ["Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases",
                "Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments",
                "Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.",
                "Methods to control the spread in communities, barriers to compliance and how these vary among different populations",
                "Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.",
                "Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.",
                "Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high)",
                "Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay"
               ]

for description in descriptions:
    print("Description: {}".format(description))
    print()
     
        
    distilbert_similar_indexes = find_similar(model.encode([description]), embeddings_distilbert2, K)
    print("5 most similar descriptions using Bert")
    for idx,index in enumerate(distilbert_similar_indexes):
        print("Result ",str(idx))
        print("[Title]: ",df_covid_NPI.title[index])
        print("[Abstract]: ", df_covid_NPI.abstract[index])
        print()
        #print(df_covid_NPI.abstract[index])
    print()
    
