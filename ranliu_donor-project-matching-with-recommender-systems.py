import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Set up test mode to save some time
test_mode = True

# Read datasets
projects = pd.read_csv('../input/Projects.csv')
donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv')

#this piece of code converts Project_ID which is a 32-bit Hex int digits 10-1010
# create column "project_id" with sequential integers
f=len(projects)
projects['project_id'] = np.nan
g = list(range(10,f+10))
g = pd.Series(g)
projects['project_id'] = g.values

# Merge datasets
donations = donations.merge(donors, on="Donor ID", how="left")
df = donations.merge(projects,on="Project ID", how="left")

# only load a few lines in test mode
if test_mode:
    df = df.head(10000)

donations_df = df
# Deal with missing values
donations["Donation Amount"] = donations["Donation Amount"].fillna(0)

# Define event strength as the donated amount to a certain project
donations_df['eventStrength'] = donations_df['Donation Amount']

def smooth_donor_preference(x):
    return math.log(1+x, 2)
    
donations_full_df = donations_df \
                    .groupby(['Donor ID', 'Project ID'])['eventStrength'].sum() \
                    .apply(smooth_donor_preference).reset_index()
        
# Update projects dataset
project_cols = projects.columns
projects = df[project_cols].drop_duplicates()

print('# of projects: %d' % len(projects))
print('# of unique user/project donations: %d' % len(donations_full_df))
donations_full_df.head()
donations_train_df, donations_test_df = train_test_split(donations_full_df,
                                   test_size=0.20,
                                   random_state=42)

print('# donations on Train set: %d' % len(donations_train_df))
print('# donations on Test set: %d' % len(donations_test_df))

#Indexing by Donor Id to speed up the searches during evaluation
donations_full_indexed_df = donations_full_df.set_index('Donor ID')
donations_train_indexed_df = donations_train_df.set_index('Donor ID')
donations_test_indexed_df = donations_test_df.set_index('Donor ID')
# Preprocessing of text data
textfeats = ["Project Title","Project Essay"]
for cols in textfeats:
    projects[cols] = projects[cols].astype(str) 
    projects[cols] = projects[cols].astype(str).fillna('') # FILL NA
    projects[cols] = projects[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
 
text = projects["Project Title"] + ' ' + projects["Project Essay"]
vectorizer = TfidfVectorizer(strip_accents='unicode',
                             analyzer='word',
                             lowercase=True, # Convert all uppercase to lowercase
                             stop_words='english', # Remove commonly found english words ('it', 'a', 'the') which do not typically contain much signal
                             max_df = 0.9, # Only consider words that appear in fewer than max_df percent of all documents
                             # max_features=5000 # Maximum features to be extracted                    
                            )                        
project_ids = projects['Project ID'].tolist()
tfidf_matrix = vectorizer.fit_transform(text)
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix
def get_project_profile(project_id):
    idx = project_ids.index(project_id)
    project_profile = tfidf_matrix[idx:idx+1]
    return project_profile

def get_project_profiles(ids):
    project_profiles_list = [get_project_profile(x) for x in np.ravel([ids])]
    project_profiles = scipy.sparse.vstack(project_profiles_list)
    return project_profiles

def build_donors_profile(donor_id, donations_indexed_df):
    donations_donor_df = donations_indexed_df.loc[donor_id]
    donor_project_profiles = get_project_profiles(donations_donor_df['Project ID'])
    donor_project_strengths = np.array(donations_donor_df['eventStrength']).reshape(-1,1)
    #Weighted average of project profiles by the donations strength
    donor_project_strengths_weighted_avg = np.sum(donor_project_profiles.multiply(donor_project_strengths), axis=0) / (np.sum(donor_project_strengths)+1)
    donor_profile_norm = sklearn.preprocessing.normalize(donor_project_strengths_weighted_avg)
    return donor_profile_norm

from tqdm import tqdm

def build_donors_profiles(): 
    donations_indexed_df = donations_full_df[donations_full_df['Project ID'].isin(projects['Project ID'])].set_index('Donor ID')
    donor_profiles = {}
    for donor_id in tqdm(donations_indexed_df.index.unique()):
        donor_profiles[donor_id] = build_donors_profile(donor_id, donations_indexed_df)
    return donor_profiles

donor_profiles = build_donors_profiles()
print("# of donors with profiles: %d" % len(donor_profiles))
mydonor1 = "6d5b22d39e68c656071a842732c63a0c"
mydonor2 = "0016b23800f7ea46424b3254f016007a"
mydonor1_profile = pd.DataFrame(sorted(zip(tfidf_feature_names, 
                        donor_profiles[mydonor1].flatten().tolist()), 
                        key=lambda x: -x[1])[:10],
                        columns=['token', 'relevance'])
mydonor2_profile = pd.DataFrame(sorted(zip(tfidf_feature_names, 
                        donor_profiles[mydonor2].flatten().tolist()), 
                        key=lambda x: -x[1])[:10],
                        columns=['token', 'relevance'])
mydonor1_profile
mydonor2_profile
class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, projects_df=None):
        self.project_ids = project_ids
        self.projects_df = projects_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_projects_to_donor_profile(self, donor_id, topn=1000):
        #Computes the cosine similarity between the donor profile and all project profiles
        cosine_similarities = cosine_similarity(donor_profiles[donor_id], tfidf_matrix)
        #Gets the top similar projects
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar projects by similarity
        similar_projects = sorted([(project_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_projects
        
    def recommend_projects(self, donor_id, projects_to_ignore=[], topn=10, verbose=False):
        similar_projects = self._get_similar_projects_to_donor_profile(donor_id)
        #Ignores projects the donor has already donated
        similar_projects_filtered = list(filter(lambda x: x[0] not in projects_to_ignore, similar_projects))
        
        recommendations_df = pd.DataFrame(similar_projects_filtered, columns=['Project ID', 'recStrength']).head(topn)

        recommendations_df = recommendations_df.merge(self.projects_df, how = 'left', 
                                                    left_on = 'Project ID', 
                                                    right_on = 'Project ID')[['recStrength', 'Project ID', 'Project Title', 'Project Essay']]


        return recommendations_df

cbr_model = ContentBasedRecommender(projects)
cbr_model.recommend_projects(mydonor1)
cbr_model.recommend_projects(mydonor2)
#Creating a sparse pivot table with donors in rows and projects in columns
donors_projects_pivot_matrix_df = donations_full_df.pivot(index='Donor ID', 
                                                          columns='Project ID', 
                                                          values='eventStrength').fillna(0)

# Transform the donor-project dataframe into a matrix
donors_projects_pivot_matrix = donors_projects_pivot_matrix_df.as_matrix()

# Get donor ids
donors_ids = list(donors_projects_pivot_matrix_df.index)

# Print the first 5 rows of the donor-project matrix
donors_projects_pivot_matrix[:5]
# Performs matrix factorization of the original donor-project matrix
# Here we set k = 20, which is the number of factors we are going to get
# In the definition of SVD, an original matrix A is approxmated as a product A ≈ UΣV 
# where U and V have orthonormal columns, and Σ is non-negative diagonal.
U, sigma, Vt = svds(donors_projects_pivot_matrix, k = 20)
sigma = np.diag(sigma)

# Reconstruct the matrix by multiplying its factors
all_donor_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 

#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_donor_predicted_ratings, 
                           columns = donors_projects_pivot_matrix_df.columns, 
                           index=donors_ids).transpose()
cf_preds_df.head()
class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, projects_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.projects_df = projects_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_projects(self, donor_id, projects_to_ignore=[], topn=10):
        # Get and sort the donor's predictions
        sorted_donor_predictions = self.cf_predictions_df[donor_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={donor_id: 'recStrength'})

        # Recommend the highest predicted projects that the donor hasn't donated to
        recommendations_df = sorted_donor_predictions[~sorted_donor_predictions['Project ID'].isin(projects_to_ignore)] \
                               .sort_values('recStrength', ascending = False) \
                               .head(topn)

 
        recommendations_df = recommendations_df.merge(self.projects_df, how = 'left', 
                                                          left_on = 'Project ID', 
                                                          right_on = 'Project ID')[['recStrength', 'Project ID', 'Project Title', 'Project Essay']]


        return recommendations_df

cfr_model = CFRecommender(cf_preds_df, projects)
cfr_model.recommend_projects(mydonor1)
cfr_model.recommend_projects(mydonor2)
class HybridRecommender:
    
    MODEL_NAME = 'Hybrid'
    
    def __init__(self, cb_rec_model, cf_rec_model, projects_df):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.projects_df = projects_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_projects(self, donor_id, projects_to_ignore=[], topn=10):
        #Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_projects(donor_id, projects_to_ignore=projects_to_ignore, 
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        
        #Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_projects(donor_id, projects_to_ignore=projects_to_ignore,  
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        
        #Combining the results by Project ID
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how = 'inner', 
                                   left_on = 'Project ID', 
                                   right_on = 'Project ID')
        
        #Computing a hybrid recommendation score based on CF and CB scores
        recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
        
        #Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        recommendations_df = recommendations_df.merge(self.projects_df, how = 'left', 
                                                    left_on = 'Project ID', 
                                                    right_on = 'Project ID')[['recStrengthHybrid', 
                                                                              'Project ID', 'Project Title', 
                                                                              'Project Essay']]


        return recommendations_df
    
hybrid_model = HybridRecommender(cbr_model, cfr_model, projects)

hybrid_model.recommend_projects(mydonor1)
hybrid_model.recommend_projects(mydonor2)
def get_projects_donated(donor_id, donations_df):
    # Get the donor's data and merge in the movie information.
    try:
        donated_projects = donations_df.loc[donor_id]['Project ID']
        return set(donated_projects if type(donated_projects) == pd.Series else [donated_projects])
    except KeyError:
        return []

#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_PROJECTS = 100

class ModelEvaluator:

    def get_not_donated_projects_sample(self, donor_id, sample_size, seed=42):
        donated_projects = get_projects_donated(donor_id, donations_full_indexed_df)
        all_projects = set(projects['Project ID'])
        non_donated_projects = all_projects - donated_projects

        #random.seed(seed)
        non_donated_projects_sample = random.sample(non_donated_projects, sample_size)
        return set(non_donated_projects_sample)

    def _verify_hit_top_n(self, project_id, recommended_projects, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_projects) if c == project_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_donor(self, model, donor_id):
        #Getting the projects in test set
        donated_values_testset = donations_test_indexed_df.loc[donor_id]
        if type(donated_values_testset['Project ID']) == pd.Series:
            donor_donated_projects_testset = set(donated_values_testset['Project ID'])
        else:
            donor_donated_projects_testset = set([donated_values_testset['Project ID']])  
        donated_projects_count_testset = len(donor_donated_projects_testset) 

        #Getting a ranked recommendation list from a model for a given donor
        donor_recs_df = model.recommend_projects(donor_id, 
                                               projects_to_ignore=get_projects_donated(donor_id, 
                                                                                    donations_train_indexed_df), 
                                               topn=100000000)

        hits_at_3_count = 0
        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each project the donor has donated in test set
        for project_id in donor_donated_projects_testset:
            #Getting a random sample (100) projects the donor has not donated 
            #(to represent projects that are assumed to be no relevant to the donor)
            non_donated_projects_sample = self.get_not_donated_projects_sample(donor_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_PROJECTS, 
                                                                              seed=42)

            #Combining the current donated project with the 100 random projects
            projects_to_filter_recs = non_donated_projects_sample.union(set([project_id]))

            #Filtering only recommendations that are either the donated project or from a random sample of 100 non-donated projects
            valid_recs_df = donor_recs_df[donor_recs_df['Project ID'].isin(projects_to_filter_recs)]                    
            valid_recs = valid_recs_df['Project ID'].values
            #Verifying if the current donated project is among the Top-N recommended projects
            hit_at_3, index_at_3 = self._verify_hit_top_n(project_id, valid_recs, 3)
            hits_at_3_count += hit_at_3
            hit_at_5, index_at_5 = self._verify_hit_top_n(project_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(project_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the donated projects that are ranked among the Top-N recommended projects, 
        #when mixed with a set of non-relevant projects
        recall_at_3 = hits_at_3_count / float(donated_projects_count_testset)
        recall_at_5 = hits_at_5_count / float(donated_projects_count_testset)
        recall_at_10 = hits_at_10_count / float(donated_projects_count_testset)

        donor_metrics = {'hits@3_count':hits_at_3_count, 
                         'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'donated_count': donated_projects_count_testset,
                          'recall@3': recall_at_3,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return donor_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for donors')
        people_metrics = []
        for idx, donor_id in enumerate(list(donations_test_indexed_df.index.unique().values)):
            #if idx % 100 == 0 and idx > 0:
            #    print('%d donors processed' % idx)
            donor_metrics = self.evaluate_model_for_donor(model, donor_id)  
            donor_metrics['_donor_id'] = donor_id
            people_metrics.append(donor_metrics)
        print('%d donors processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('donated_count', ascending=False)
        
        global_recall_at_3 = detailed_results_df['hits@3_count'].sum() / float(detailed_results_df['donated_count'].sum())
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['donated_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['donated_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@3': global_recall_at_3,
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()
print('Evaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(cbr_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df = cb_detailed_results_df[['_donor_id', 'donated_count', "hits@3_count", 'hits@5_count','hits@10_count', 
                                                'recall@3','recall@5','recall@10']]
cb_detailed_results_df.head(10)
print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cfr_model)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df = cf_detailed_results_df[['_donor_id', 'donated_count', "hits@3_count", 'hits@5_count','hits@10_count', 
                                                'recall@3','recall@5','recall@10']]
cf_detailed_results_df.head(10)
print('Evaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_model)
print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
hybrid_detailed_results_df = hybrid_detailed_results_df[['_donor_id', 'donated_count', "hits@3_count", 'hits@5_count','hits@10_count', 
                                                'recall@3','recall@5','recall@10']]
hybrid_detailed_results_df.head(10)
global_metrics_df = pd.DataFrame([cf_global_metrics, 
                                  cb_global_metrics, 
                                  hybrid_global_metrics]).set_index('modelName')
global_metrics_df
donor = "a0e1d358aa17745ff3d3f4e4909356f3"
project_list = get_projects_donated(donor, donations_test_indexed_df)
donated = projects[projects["Project ID"].isin(project_list)]["Project Title"].tolist()

cbr_rec = cbr_model.recommend_projects(donor).head(7)["Project Title"].tolist()
cfr_rec = cfr_model.recommend_projects(donor).head(7)["Project Title"].tolist()
hybrid_rec = hybrid_model.recommend_projects(donor).head(7)["Project Title"].tolist()

d = {'Donated': donated, 
     'Content-Based': cbr_rec,
    'Collaborative-Filtering': cfr_rec,
    'Hybrid': hybrid_rec}
pd.DataFrame(data=d)