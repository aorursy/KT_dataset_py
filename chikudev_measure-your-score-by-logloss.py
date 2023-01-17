import pandas as pd

import numpy as np
def logloss(true_label, predicted, eps=1e-15):

    p = np.clip(predicted, eps, 1 - eps)

    if true_label == 1:

        return -np.log(p)

    return -np.log(1 - p)
tourney_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')

submission_result = pd.read_csv('../input/submission/submission.csv')



logloss_results = []



for column_name, item in submission_result.iterrows():

    ids = item['ID'].split('_')



    result = -1



    record = tourney_result[

        (tourney_result['Season'] == int(ids[0])) & 

        (tourney_result['WTeamID'] == int(ids[1])) & 

        (tourney_result['LTeamID'] == int(ids[2]))]

    if record.shape[0] > 0:

        result = 1



    record = tourney_result[

        (tourney_result['Season'] == int(ids[0])) & 

        (tourney_result['WTeamID'] == int(ids[2])) & 

        (tourney_result['LTeamID'] == int(ids[1]))]

    if record.shape[0] > 0:

        result = 0

        



    if result > -1:

        logloss_results.append(round(logloss(result, item['Pred']), 6))

        



print('score', sum(logloss_results)/len(logloss_results))
