import numpy as np

import pandas as pd

from tqdm import tqdm



DATA_PATH = '../input/train.csv'

DAYS_IN_WEEK = 7

DAYS_IN_DATASET = 1099
def write_prediction_to_file(prediction):

    header = "id,nextvisit"

    with open("submission.csv", "w") as f:

        print(header, file=f)

        for idx, p in enumerate(prediction, 1):

            print("{}, {}".format(idx, p), file=f)
def get_week_number(visit):

    return visit // DAYS_IN_WEEK + (1 if visit % DAYS_IN_WEEK else 0) 



def get_probs(visits, delta=1.0, alpha=0.5, coef=None, recount=True):

    weekday_probs = np.zeros(DAYS_IN_WEEK, dtype=np.float64)

    # compute week weights

    for v in visits:

        weekday = (v - 1) % 7

        weekday_probs[weekday] += np.power(get_week_number(v), delta)

    # normalization

    if coef:

        weekday_probs = weekday_probs / coef

    # copy probabilities for ensemble

    naive_probs = np.array(weekday_probs)

    # recount probabilities

    if recount:

        negative_probs = 1 - weekday_probs

        for i in range(1, DAYS_IN_WEEK):

            weekday_probs[i] *= np.prod(negative_probs[:i])

            negative_probs[i] = 1 - weekday_probs[i]

    # ensemble

    for i in range(len(weekday_probs)):

        weekday_probs[i] = alpha * weekday_probs[i] + (1 - alpha) * naive_probs[i]

    return weekday_probs
delta = 1.15

alpha = 0.9



data = pd.read_csv(DATA_PATH)

data.visits = data.visits.apply(lambda row: np.fromstring(row, dtype=int, sep=" "))

weeks_num = DAYS_IN_DATASET // DAYS_IN_WEEK # 1099 / 7 = 157

normalization_coef = np.sum([np.power(i, delta) for i in range(1, weeks_num + 1)])



prediction = list()

for row in tqdm(data.visits.values):

    probs = get_probs(row, delta, alpha, coef=normalization_coef)

    next_day_visit = np.argmax(probs) + 1

    prediction.append(next_day_visit)



write_prediction_to_file(prediction)