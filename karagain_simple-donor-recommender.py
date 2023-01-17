import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import sklearn
import math
import random

import os
print(os.listdir("../input"))

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
donations = pd.read_csv('../input/Donations.csv')
donations.columns
# Pull out information to get the minimal dataframe to Join information
id_df = donations.loc[:,['Project ID', 'Donation ID', 'Donor ID']]
id_df.head()
# multi-index matrix with the count of donations to a project as eventStrength
sample = id_df.groupby(['Donor ID', 'Project ID']).count().head(20)
sample
# grab donor id's donations
sample.loc['00002d44003ed46b066607c5455a999a']
# grabs the count of occurances
sample.loc['00002d44003ed46b066607c5455a999a', '2f53e5f31890e647048ac217cda3b83f']
sample.iloc[1:5]
projects = pd.read_csv('../input/Projects.csv', nrows=500)

# # Do you need to split?
# features_train, features_test, labels_train, labels_test = train_test_split(word_data, 
#                                                                                authors, 
#                                                                                test_size=0.2, 
#                                                                                random_state=42)
vectorizer = TfidfVectorizer(analyzer='word', 
                                 ngram_range=(1,2), 
                                 sublinear_tf=True, 
                                 max_df=0.5,
                                 lowercase=True,
                                 max_features=1000, # The lower the features, the more specific the words will be to a category.
                                 stop_words='english')

tfidf_matrix = vectorizer.fit_transform(projects['Project Essay'])
# features_test_transformed  = vectorizer.transform(features_test)
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix
tfidf_matrix[0]
# This is the number of donations made by a person, this can use to weigh the donations. 
id_df.groupby('Donor ID').count().head(10)
# left join this onto donations, and then divide to get the event strength. 
sum_df = donations.groupby('Donor ID')['Donation Amount'].sum()
sum_df = pd.DataFrame(sum_df)

sum_df = sum_df.unstack().reset_index()

sum_df = sum_df[['Donor ID', 0]]

sum_df['Donation Sum'] = sum_df[0]

sum_df = sum_df[['Donor ID', 'Donation Sum']]

donations = donations.merge(sum_df, on='Donor ID', how='left')
donations['eventStrength'] = donations['Donation Amount'] / donations['Donation Sum']
donations.head()
# Create copy with donor ID, Project ID, and eventStrength
don_df = donations[['Donor ID', 'Project ID', 'eventStrength']].copy()
don_df.head()
project_ids = list(projects['Project ID'])
project_ids[0:10]
don_df = don_df.iloc[:100000]
# [1] 
# replace user with donor
# contentid with Project ID
# interactions with donations
# item with project


def get_project_profile(project_id):
    idx = project_ids.index(project_id)
    project_profile = tfidf_matrix[idx:idx+1]
    return project_profile

def get_project_profiles(ids):
    project_profiles_list = [get_project_profile(x) for x in ids]
    project_profiles = scipy.sparse.vstack(project_profiles_list)
    return project_profiles

def build_donors_profile(donor_id, don_indexed_df):
    donations_donor_df = don_indexed_df[don_indexed_df['Donor ID'] == donor_id]
    donor_project_profiles = get_project_profiles(list(donations_donor_df['Project ID']))
    donor_project_strengths = np.array(donations_donor_df['eventStrength']).reshape(-1,1)
    #Weighted average of project profiles by the donations strength
    donor_project_strengths_weighted_avg = np.sum(donor_project_profiles.multiply(donor_project_strengths),
                                                  axis=0) / np.sum(donor_project_strengths)
    donor_profile_norm = sklearn.preprocessing.normalize(donor_project_strengths_weighted_avg)
    return donor_profile_norm

def build_donors_profiles(): 
    don_indexed_df = don_df[don_df['Project ID'].isin(projects['Project ID'])]
    donor_profiles = {}
    for donor_id in don_indexed_df['Donor ID'].unique():
        donor_profiles[donor_id] = build_donors_profile(donor_id, don_indexed_df)
    return donor_profiles

donor_profiles = build_donors_profiles()
donor_id = don_df[don_df['Project ID'].isin(projects['Project ID'])]['Donor ID'].unique()[0]

myprofile = donor_profiles[donor_id]
print(myprofile.shape)
print(donor_id)
pd.DataFrame(sorted(zip(tfidf_feature_names, 
                        donor_profiles[donor_id].flatten().tolist()), key=lambda x: -x[1])[:20],
             columns=['token', 'relevance'])






