from collections import Counter



import pandas as pd

import sklearn.utils.extmath as extmath
DAYS_IN_WEEK = 7
data = pd.read_csv("../input/train.csv")

data.visits = data.visits.transform(lambda s: list(sorted(map(int, s.split()))))
def max_interval(sequence):

    return max(b - a for a, b in zip(sequence, sequence[1:]))





def day_of_week_index(day_number):

    return (day_number - 1) % 7 + 1





class NextVisitPredictor(object):

    def __init__(self, decay_base=0.997, visits_from=1099 + 1):

        self.decay_base = decay_base

        self.visits_from = visits_from



    def apply_decay(self, visit):

        return self.decay_base ** (self.visits_from - visit)

    

    def predict(self, past_visits):

        max_gap = max_interval(visits)

        if self.visits_from - visits[-1] > max(2 * max_gap, 2 * DAYS_IN_WEEK):

            return 0



        days_of_week = list(map(day_of_week_index, past_visits))

        day_weights = list(map(self.apply_decay, past_visits))

        mode, _ = extmath.weighted_mode(days_of_week, day_weights)

        return int(mode[0])
next_visit_predictor = NextVisitPredictor()



with open("solution.csv", "w") as f:

    print("id,nextvisit", file=f)

    for row in data.itertuples():

        idx, visits = row.id, row.visits

        next_visit = next_visit_predictor.predict(visits)

        print(f"{idx}, {next_visit}", file=f)