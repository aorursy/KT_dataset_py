# Imports

import pandas as pd

import os
def get_duty(filename):

    description = open(filename,"r").read()

    

    title = description.lstrip().split("\n")[0]

    

    duties = description.split("DUTIES")[1]    

    duties = duties.split("REQUIREMENT")[0]

    # A second split is required because the file format varies

    duties = duties.split("NOTE")[0]

    

    return title,duties 
rows = []





files = os.listdir("../input/cityofla/CityofLA/Job Bulletins/")

    

for filename in files:

    try:

        title, duties = get_duty("../input/cityofla/CityofLA/Job Bulletins/" + filename)

        rows.append({"title":title,"duties":duties})



    except:

        print("No Duties:", filename)
df = pd.DataFrame(rows)

df.head()
# Import and declare the vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()



# Fit the vectorizer on our DUTIES strings

duties_vectors = vec.fit_transform(df['duties'])
# Import and create an empty list

from sklearn.metrics.pairwise import cosine_similarity

similar_jobs = []



# Loop through every job

for index, row in df.iterrows():

    # Calculate the cosine similarity between the current job's vector and all other job vectors.

    sim = cosine_similarity(duties_vectors[index], duties_vectors)

    

    # We are interested in the most similar job, so we must sort the cosine_similarity matrix

    # Convert to Pandas data frame

    temp_df = pd.DataFrame(sim.reshape(-1,1))

    # Sort it, then access the index value, this will be the index of the most similar job

    similar_job_index = temp_df.sort_values(0,ascending=False).iloc[1:2].index[0]

    

    # Finally, access the most similar job via its index, then add that record to our final data structure

    similar_job = df.iloc[similar_job_index]['title']

    similar_jobs.append(similar_job)
df['similar_job'] = similar_jobs

df.head(10)
df[['title','similar_job']].to_csv("Job_Similarity.csv", index=False)
# Entire output for the public notebook

df[['title','similar_job']]