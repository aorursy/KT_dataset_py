from tqdm import tqdm as tqdm

import numpy as np

users_visits = {}



train_data = open('../input/train.csv').read().splitlines()[1:]

for line in tqdm(train_data):

    ids_visits = line.split()

    visits = list(map(int, ids_visits[1:]))

    idx = int(ids_visits[0][:-1]) - 1

    users_visits[idx] = np.array(visits) - 1
from collections import defaultdict



answers = []

for user_visits in tqdm(users_visits.values()):

    scores = np.zeros(7)

    for day in user_visits:

        scores[day % 7]  += 1 + int(day / 31)

    scores_sum = scores.sum()

    for day in range(7):

        scores[day] /= scores_sum

    for day in range(7):

        for prev_day in range(day):

            scores[day] *= 1 - scores[prev_day]

    answers.append(np.argmax(scores))
from matplotlib import pyplot as plt



plt.hist(answers, bins=np.arange(0, 7, 0.5))

plt.show()
with open("answers.csv", "w") as f:

        print("id,nextvisit", file=f)

        for idx, day in enumerate(answers):

            print("{}, {}".format(idx + 1, day + 1), file=f)