import os

from time import time

import datetime as dt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
input_dir = "../input"

test_dir = os.path.join(input_dir,'cv-machine-learning-gbdts')

test_x = pd.read_csv('{}/test_x.gz'.format(test_dir), index_col=0, compression='gzip')



print('Number of test instances: {}'.format(test_x.shape))

test_x.sample(3)
test_y = pd.read_csv('{}/test_y.gz'.format(test_dir), header=None, index_col=0, compression='gzip')

print('Number of test instances: {}'.format(test_y.shape))

test_y.columns = ['Matched']



test_y.sample(3)
print('The number of unmatched instances is {}'.format(sum(test_y['Matched']==0)))

print('The number of matched instances is {}'.format(sum(test_y['Matched']==1)))
test_y = test_x[['questions_id']].merge(test_y, left_index=True, right_index=True)
gbdts_dir = os.path.join(input_dir,'cv-machine-learning-gbdts')

gbdts_y = pd.read_csv('{}/predicted_test_y.gz'.format(gbdts_dir), index_col=0, compression='gzip')

gbdts_y.columns = ['GBDTs_Prob']



print('Number of GBDTs predicted scores: {}'.format(gbdts_y.shape))

gbdts_y.head(3)
predicted_scores= test_y.merge(gbdts_y, left_index=True, right_index=True)
predicted_scores.sample(3)
clr_dir = os.path.join(input_dir,'cv-machine-learning-conditional-lr')

print(os.listdir(clr_dir))
clr_y = pd.read_csv('{}/ConditionalLR_predicted_y.csv'.format(clr_dir), index_col=0)

print('Number of CLR predicted scores: {}'.format(clr_y.shape))

clr_y.columns = ['CLR_Prob']

clr_y.index = predicted_scores.index

clr_y.head(3)
predicted_scores= predicted_scores.merge(clr_y, left_index=True, right_index=True)
predicted_scores.head(3)
def compute_min_rank(rows, col_name):

    rows['Rank'] = rows[col_name].rank(ascending=False)

    rows=rows[rows['Matched']==1]

    return rows['Rank'].min()
def compute_max_rank(rows, col_name):

    rows['Rank'] = rows[col_name].rank(ascending=False)

    rows=rows[rows['Matched']==1]

    return rows['Rank'].max()
cut_points = range(1,20+1)

def compute_recall(cut_points, ranked_results):

    cut_results = {'Top K': [],

                   'GBDTs': [],

                   'CLR': [],

                  }

    for cut_point in cut_points:

        cut_results['Top K'].append(cut_point)

        cut_results['GBDTs'].append(ranked_results[ranked_results['GBDTs'] <= cut_point].shape[0])

        cut_results['CLR'].append(ranked_results[ranked_results['CLR'] <= cut_point].shape[0])



    cut_results = pd.DataFrame(cut_results)

    cut_results['GBDTs'] = cut_results['GBDTs'] / ranked_results.shape[0]

    cut_results['CLR'] = cut_results['CLR'] / ranked_results.shape[0]

    return cut_results
min_ranked_results = pd.DataFrame({'GBDTs': predicted_scores.groupby('questions_id').apply(compute_min_rank, col_name='GBDTs_Prob'),

                                   'CLR': predicted_scores.groupby('questions_id').apply(compute_min_rank, col_name='CLR_Prob'),

                                   'Matches': predicted_scores.groupby('questions_id')['Matched'].sum(),

                                   'Recommendations': predicted_scores.groupby('questions_id')['Matched'].count()})

min_ranked_results = min_ranked_results[min_ranked_results['Matches'] > 0]
min_ranked_results.sample(5)
min_rank_cut_results = compute_recall(cut_points, min_ranked_results)

min_rank_cut_results.set_index('Top K').plot()

plt.ylabel('At Least One Recall')

plt.title('At Least One Recall Performance Comparison:\nConditional Logistic Regression vs GBDTs')

plt.savefig('min_rank_recommendation_recall.jpg')
max_ranked_results = pd.DataFrame({'GBDTs': predicted_scores.groupby('questions_id').apply(compute_max_rank, col_name='GBDTs_Prob'),

                                   'CLR': predicted_scores.groupby('questions_id').apply(compute_max_rank, col_name='CLR_Prob'),

                                   'Matches': predicted_scores.groupby('questions_id')['Matched'].sum(),

                                   'Recommendations': predicted_scores.groupby('questions_id')['Matched'].count()})

max_ranked_results = max_ranked_results[max_ranked_results['Matches'] > 0]
max_ranked_results.sample(5)
max_rank_cut_results = compute_recall(cut_points, max_ranked_results)

max_rank_cut_results.set_index('Top K').plot()

plt.ylabel('Full Recall')

plt.title('Full Recall Performance Comparison:\nConditional Logistic Regression vs GBDTs')

plt.savefig('max_rank_recommendation_recall.jpg')
predicted_scores['GBDTs_Rank'] = predicted_scores.groupby('questions_id')['GBDTs_Prob'].rank(ascending=False)
print('The total number of correct original recommendations: {}'.format(predicted_scores[

    (predicted_scores['Matched']==1)].shape[0]))



print('The total number of correct ML top-20 recommendations: {}'.format(

    predicted_scores[((predicted_scores['GBDTs_Rank'] <= 20) & (predicted_scores['Matched']==1))].shape[0]))



print('The ML top-20 accuracy is {}%'.format(np.round(100 * predicted_scores[

    ((predicted_scores['GBDTs_Rank'] <= 20) & (predicted_scores['Matched']==1))].shape[0] / predicted_scores[(predicted_scores['Matched']==1)].shape[0],1)))
print('The total number of original recommendations: {}'.format(predicted_scores.shape[0]))



print('The total number of ML top-20 recommendations: {}'.format(predicted_scores[

    predicted_scores['GBDTs_Rank'] <= 20].shape[0]))



print('The decrease in the number of sent recommendations is {} folds: '.format(

    np.round(predicted_scores.shape[0] / predicted_scores[

    predicted_scores['GBDTs_Rank'] <= 20].shape[0], 1)

))
predicted_scores = test_x[['answer_user_id', 'emails_date_sent']].merge(

    predicted_scores, left_index=True, right_index=True)
before_ml_counts = predicted_scores.groupby(['answer_user_id', 'emails_date_sent'])['questions_id'].count()

before_ml_counts[before_ml_counts <= 30].hist(bins=30)

plt.ylabel('Emails')

plt.title('The mean value is {}'.format(round(before_ml_counts.mean(),1)))

plt.savefig('before_ml_questions_in_each_email.jpg')
after_ml_counts = predicted_scores[predicted_scores['GBDTs_Rank'] <= 20

                ].groupby(['answer_user_id', 'emails_date_sent'])['questions_id'].count()

after_ml_counts[after_ml_counts <= 30].hist(bins=30)

plt.ylabel('Emails')

plt.title('The mean value is {}'.format(round(after_ml_counts.mean(),1)))

plt.savefig('after_ml_questions_in_each_email.jpg')
predicted_scores.to_csv('predicted_scores.csv.gz', compression='gzip')



min_ranked_results.to_csv('min_ranked_results.csv.gz', compression='gzip')

min_rank_cut_results.to_csv('min_rank_cut_results.csv.gz', compression='gzip')



max_ranked_results.to_csv('max_ranked_results.csv.gz', compression='gzip')

max_rank_cut_results.to_csv('max_rank_cut_results.csv.gz', compression='gzip')
os.listdir()