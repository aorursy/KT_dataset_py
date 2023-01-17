import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

import seaborn as sns



%matplotlib inline
def simulate_n_submissions(n_submissions, public_ratio, full_mask_ratio, score_scale=0.361):

    """

    Simulates the submissions that we are performing on the public leaderboard

    

    Parameters

    ----------

    n_submissions : int

        Number of submissions to be made

    public_ratio : float

        Fraction of the test set that belongs to public leaderboard

    full_mask_ratio : float

        Fraction of the submission that will use full_mask

    score_scale : float

        A factor for scaling the score. Typical value is 0.361, which is the score for using

        full mask on all the submission instances

    """

    # Create the synthetic test set

    test_size = int(1e5)

    test_set = np.concatenate((np.ones(int(test_size*public_ratio)), 

                               np.zeros(int(test_size*(1-public_ratio)))))

    test_set /= np.sum(test_set)

    # Do the submissions and collect the scores

    score_list = []

    for _ in tqdm(range(n_submissions)):

        sampling_mask = (np.random.rand(test_size) < full_mask_ratio).astype(np.int)

        score = np.sum(sampling_mask*test_set)*score_scale

        

        score_list.append(score)

    return score_list
plt.figure(figsize=(12, 6))

score_list = []

p_range = [0.01, 0.05, 0.5]

labels = ['test set public ratio= %.2f' % i for i in p_range]

for p in p_range:

    score_list.append(simulate_n_submissions(1000, public_ratio=p, full_mask_ratio=0.2, score_scale=0.361))

    

for values, label in zip(score_list, labels):

    sns.distplot(values, label=label)

plt.xlabel('Submission score')

plt.legend();
submission_scores = [0.075, 0.074, 0.068, 0.076, 0.069, 0.075,0.07,0.071,0.076,0.071,0.071,0.07,0.073,0.075,0.076,0.065,0.072,0.077,0.063,0.074,0.067,0.068,0.065,0.069]
plt.figure(figsize=(12, 6))

score_list = [submission_scores]

p_range = [0.01, 0.05, 0.5]

labels = ['real submission'] + ['test set public ratio= %.3f' % i for i in p_range]

for p in p_range:

    score_list.append(simulate_n_submissions(1000, public_ratio=p, full_mask_ratio=0.2, score_scale=0.361))

    

for values, label in zip(score_list, labels):

    sns.distplot(values, label=label)

plt.xlabel('Submission score')

plt.legend();
plt.figure(figsize=(12, 6))

score_list = [submission_scores]

p_range = [ 0.005, 0.01, 0.02]

labels = ['real submission'] + ['test set public ratio= %.3f' % i for i in p_range]

for p in p_range:

    score_list.append(simulate_n_submissions(10000, public_ratio=p, full_mask_ratio=0.2, score_scale=0.361))

    

for values, label in zip(score_list, labels):

    sns.distplot(values, label=label)

plt.xlabel('Submission score')

plt.legend();