import random

import numpy as np

import seaborn as sns

from tqdm import tqdm_notebook as tqdm

from scipy.stats import truncnorm

import matplotlib.pyplot as plt



def get_truncated_normal(mean=0, sd=1, low=0, upp=10):

    return truncnorm(

        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
n_people = 10000

n_choice = 12



sample = np.random.uniform(low=0.0, high=1.0, size=n_choice)

sns.distplot(sample, bins=50)
chance = np.random.uniform(low=0.0, high=1.0, size=n_choice)

evaluation = np.random.uniform(low=0.0, high=1.0, size=n_choice)

success = (evaluation < (chance)).astype(np.byte)

score = np.where(success==1)

score = 1-chance[score]

sns.distplot(score, bins=50)

if len(score) != 0:

    score = np.max(score)

else:

    score = 0

for c, e, s in zip(chance, evaluation, success):

    print("Chance: {} Evaluation: {} Success: {}".format(c, e, s))

print("Final happiness = {}".format(score))
pbar = tqdm(range(n_people))

scores = []

for p in pbar:

    chance = np.random.uniform(low=0.0, high=1.0, size=n_choice) 

    evaluation = np.random.uniform(low=0.0, high=1.0, size=n_choice)

    success = (evaluation < (chance)).astype(np.byte)

    score = np.where(success==1)

    score = 1-chance[score]

    if len(score) != 0:

        score = np.max(score)

    else:

        score = 0

    scores.append(score)

#     pbar.set_description("{}".format(score))

scores = np.array(scores)

print("Score: {} STD: {}".format(scores.mean(), scores.std()))

sns.distplot(scores, bins=50)
# in uniform distribution, what `high` is the best?

scoreses = []

stds = []

highs = []



pbar = tqdm(np.linspace(0.3, 1.0, 70-1))

for high in pbar:

    scores = []

    for p in range(n_people):

    #     chance = np.random.uniform(low=0.0, high=1.0, size=n_choice) # Score: 0.6498417210963432 STD: 0.17268462601679235

    #     chance = get_truncated_normal(mean=0.5, sd=1, low=0, upp=1).rvs(n_choice) # Score: 0.6520318920635775 STD: 0.16939947534968955

    #     chance = np.random.uniform(low=0.0, high=0.1, size=n_choice) # Score: 0.4257803296876376 STD: 0.46683676301815763

    #     chance = np.random.uniform(low=0.0, high=0.01, size=n_choice) # Score: 0.060698758035326296 STD: 0.23794155238700707

        chance = np.random.uniform(low=0.0, high=high, size=n_choice) # Score: 0.7364615192835041 STD: 0.17269687224170607

        evaluation = np.random.uniform(low=0.0, high=1.0, size=n_choice)

        success = (evaluation < (chance)).astype(np.byte)

        score = np.where(success==1)

        score = 1-chance[score]

        if len(score) != 0:

            score = np.max(score)

        else:

            score = 0

        scores.append(score)

    #     pbar.set_description("{}".format(score))

    scores = np.array(scores)

    

    scoreses.append(scores.mean())

    stds.append(scores.std())

    highs.append(high)
plt.scatter(highs, scoreses)

plt.show()
plt.scatter(highs, stds)

plt.show()
pbar = tqdm(range(n_people))

scores = []

for p in pbar:

    chance = np.random.uniform(low=0.0, high=0.4, size=n_choice) # Score: 0.7364615192835041 STD: 0.17269687224170607

    evaluation = np.random.uniform(low=0.0, high=1.0, size=n_choice)

    success = (evaluation < (chance)).astype(np.byte)

    score = np.where(success==1)

    score = 1-chance[score]

    if len(score) != 0:

        score = np.max(score)

    else:

        score = 0

    scores.append(score)

#     pbar.set_description("{}".format(score))

scores = np.array(scores)

print("Score: {} STD: {}".format(scores.mean(), scores.std()))

sns.distplot(scores, bins=50)

print("You have {}% chance not getting any schools".format(len(np.where(scores==0))/len(scores)*100))
scoreses = []

stds = []

means = []



pbar = tqdm(np.linspace(0.0, 1.0, 10+1))

for mean in pbar:

    scores = []

    for p in range(n_people):

        chance = get_truncated_normal(mean=mean, sd=1, low=0, upp=1).rvs(n_choice)

        evaluation = np.random.uniform(low=0.0, high=1.0, size=n_choice)

        success = (evaluation < (chance)).astype(np.byte)

        score = np.where(success==1)

        score = 1-chance[score]

        if len(score) != 0:

            score = np.max(score)

        else:

            score = 0

        scores.append(score)

    #     pbar.set_description("{}".format(score))

    scores = np.array(scores)

    

    scoreses.append(scores.mean())

    stds.append(scores.std())

    means.append(mean)
plt.scatter(means, scoreses)

plt.show()
plt.scatter(means, stds)

plt.show()
sns.distplot(get_truncated_normal(mean=mean, sd=1, low=0, upp=1).rvs(n_people))
scoreses = []

stds = []

sds = []



pbar = tqdm(np.linspace(0.1, 1.0, 10+1))

for sd in pbar:

    scores = []

    for p in range(n_people):

        chance = get_truncated_normal(mean=0, sd=sd, low=0, upp=1).rvs(n_choice)

        evaluation = np.random.uniform(low=0.0, high=1.0, size=n_choice)

        success = (evaluation < (chance)).astype(np.byte)

        score = np.where(success==1)

        score = 1-chance[score]

        if len(score) != 0:

            score = np.max(score)

        else:

            score = 0

        scores.append(score)

    #     pbar.set_description("{}".format(score))

    scores = np.array(scores)

    

    scoreses.append(scores.mean())

    stds.append(scores.std())

    sds.append(sd)
plt.scatter(means, scoreses)

plt.show()
plt.scatter(means, stds)

plt.show()
means = np.linspace(0.1, 0.3, 5)

sds = np.linspace(0.001, 0.1, 5)



best_means = -1

best_sds = -1

best_score = 0



for mean in means:

    for sd in sds:

        scores = []

        dis = get_truncated_normal(mean=0.2, sd=0.05, low=0, upp=1)

        for p in range(n_people):

            chance = dis.rvs(n_choice)

            evaluation = np.random.uniform(low=0.0, high=1.0, size=n_choice)

            success = (evaluation < (chance)).astype(np.byte)

            score = np.where(success==1)

            score = 1-chance[score]

            if len(score) != 0:

                score = np.max(score)

            else:

                score = 0

            scores.append(score)

        #     pbar.set_description("{}".format(score))

        scores = np.array(scores)

        scores = scores.mean()



        if scores > best_score:

            best_score = scores

            best_means = mean

            best_sds = sd

    

print("Mean: {} Sd: {} gives {}".format(best_means, best_sds, best_score))
pbar = tqdm(range(n_people))

scores = []

dis = get_truncated_normal(mean=0.2, sd=0.056, low=0, upp=1)

for p in pbar:

    chance = dis.rvs(n_choice)

    evaluation = np.random.uniform(low=0.0, high=1.0, size=n_choice)

    success = (evaluation < (chance)).astype(np.byte)

    score = np.where(success==1)

    score = 1-chance[score]

    if len(score) != 0:

        score = np.max(score)

    else:

        score = 0

    scores.append(score)

#     pbar.set_description("{}".format(score))

scores = np.array(scores)

print("Score: {} STD: {}".format(scores.mean(), scores.std()))

sns.distplot(scores, bins=50)

print("You have {}% chance not getting any schools".format(len(np.where(scores==0))/len(scores)*100))
sns.distplot(get_truncated_normal(mean=0.15, sd=0.02575, low=0, upp=1).rvs(n_people)).set(xlim=(0, 1))
pbar = tqdm(range(n_people))

scores = []

for p in pbar:

    chance = np.array([0.2] * n_choice)

    evaluation = np.random.uniform(low=0.0, high=1.0, size=n_choice)

    success = (evaluation < (chance)).astype(np.byte)

    score = np.where(success==1)

    score = 1-chance[score]

    if len(score) != 0:

        score = np.max(score)

    else:

        score = 0

    scores.append(score)

#     pbar.set_description("{}".format(score))

scores = np.array(scores)

print("Score: {} STD: {}".format(scores.mean(), scores.std()))

sns.distplot(scores, bins=50)

print("You have {}% chance not getting any schools".format(len(np.where(scores==0))/len(scores)*100))
pbar = tqdm(range(n_people))

scores = []

for p in pbar:

    chance = np.array([0.2] * int(10) + [0.5]* int(2))

    evaluation = np.random.uniform(low=0.0, high=1.0, size=n_choice)

    success = (evaluation < (chance)).astype(np.byte)

    score = np.where(success==1)

    score = 1-chance[score]

    if len(score) != 0:

        score = np.max(score)

    else:

        score = 0

    scores.append(score)

#     pbar.set_description("{}".format(score))

scores = np.array(scores)

print("Score: {} STD: {}".format(scores.mean(), scores.std()))

sns.distplot(scores, bins=50)

print("You have {}% chance not getting any schools".format(len(np.where(scores==0))/len(scores)*100))