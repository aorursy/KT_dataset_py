# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Any results you write to the current directory are saved as output.
rscr = pd.read_csv("../input/RegularSeasonCompactResults.csv")
tcr = pd.read_csv("../input/TourneyCompactResults.csv")
teams = pd.read_csv("../input/Teams.csv")
submission = pd.read_csv("../input/SampleSubmission.csv")
seeds = pd.read_csv("../input/TourneySeeds.csv")
submit_to_test = pd.DataFrame(submission.Id.str.split('_').tolist(), columns=["Season","Team1","Team2"]).astype("int64")
class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]
def seed_str_to_num(seed):
    return int(float(''.join(i for i in seed if i.isdigit())))

seed_str_to_num = Memoize(seed_str_to_num)
def pred_winner_from_seed(row):
    seed1 = seed_str_to_num(seeds[(seeds.Season == row.Season) & (seeds.Team == row.Team1)].iloc[0].Seed)
    seed2 = seed_str_to_num(seeds[(seeds.Season == row.Season) & (seeds.Team == row.Team2)].iloc[0].Seed)
    return 0.5-(0.03*(seed1-seed2))
    #if (seed1 == seed2):
    #    return 0.5
    #elif (seed1 > seed2):
    #    return 0.25
    #else:
    #    return 0.75
winner = submit_to_test.apply(pred_winner_from_seed, 1)
submission["Pred"] = winner
submission.to_csv("submission.csv", index=False, index_label=False)