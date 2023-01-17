!pip install spacy
import numpy as np
import pandas as pd
import spacy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances
from tqdm.notebook import tqdm

import spacy.cli
spacy.cli.download("en_core_web_md")
import en_core_web_md
nlp = en_core_web_md.load()
# load the data
data_df = pd.read_csv("/kaggle/input/jobposts/data job posts.csv")
data_df.head()
# check the missing values
data_df.isna().sum()
# drop the rows without descriptions or titles
data_df = data_df.dropna(subset=['Title', 'JobDescription'])
data_df = data_df.drop("jobpost", axis=1)
# let's get the job title and describtion
titles = data_df['Title'].values
describtions = data_df['JobDescription'].values
# let's build the vectors for the describtion
describtion_vectors = np.zeros((len(describtions), 300))
for i, desc in enumerate(tqdm(nlp.pipe(describtions), total=len(describtions))):
    vector = np.zeros(300,)
    valid_tokens = 0
    for token in desc:
        if not token.is_stop and not token.is_punct and token.has_vector:
            vector += token.vector
            valid_tokens += 1
    vector = vector/valid_tokens if valid_tokens > 1 else vector 
    describtion_vectors[i, :] = vector
print("all jobs were vectorized !")
# export the vectors and the new data frame (if you need to)
np.save("jobs_vectors.npy", describtion_vectors)
data_df.to_csv("cleaned_data.csv", index=False)
# now let's build a KNN model
knn = KNeighborsClassifier(weights='distance', metric=lambda v1, v2: cosine_distances([v1], [v2])[0])
knn.fit(describtion_vectors, titles)
def sent2vect(text):
    vector = np.zeros(300,)
    valid_tokens = 0
    for token in nlp(text):
        if not token.is_stop and not token.is_punct and token.has_vector:
            vector += token.vector
            valid_tokens += 1
    vector = vector/valid_tokens if valid_tokens > 1 else vector
    return vector
# let's test it !
new_job_desc = "Machine learning engineer"

vector = sent2vect(new_job_desc)
knn.predict(vector.reshape(1, -1))
# let's try to build it ourself
def get_top_similar(job_desc, k=5):
    vector = sent2vect(job_desc)
    # get similarity scores
    distances = cosine_distances([vector], describtion_vectors)
    most_similar = np.argsort(distances).flatten()[:k]
    return data_df.iloc[most_similar].to_dict(orient='records')
new_job_desc = "fashion designer"

get_top_similar(new_job_desc)