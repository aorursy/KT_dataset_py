import pandas as pd 

import os
SUBMISSIONS_PATH = '/kaggle/input/submissions/'
submissions_all = []

for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):

    for filename in filenames:

        submissions_all.append(os.path.join(dirname, filename))

submissions_all.sort()

print( submissions_all)
def ensemble(submissions_all, sub_idx,weights=[]):

    submission_with_weight = []

    for i in range(len(sub_idx)):

        print(f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}")

        submission = pd.read_csv(submissions_all[sub_idx[i]])

        submission = submission.loc[:, ['healthy', 'multiple_diseases', 'rust', 'scab']].values 

        submission_with_weight.append(submission*weights[i])

    submission_avg = sum(submission_with_weight)

    return submission_avg   
def make_submission_file(submission_avg, submissions_all):

    submission_df = pd.read_csv(submissions_all[0])

    submission_df.iloc[:, 1:] = 0

    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = submission_avg

    submission_df.to_csv('submission.csv', index=False)
submission_avg = ensemble(submissions_all,[0,2,5], [0.1, 0.8,0.1])

make_submission_file(submission_avg,submissions_all)