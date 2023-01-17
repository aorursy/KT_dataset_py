import pandas as pd
sub = pd.read_csv("../input/google-quest-qa-subm-files/final_submission.csv")

sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
id_in_sub = set(sub.qa_id)

id_in_sample_submission = set(sample_submission.qa_id)

diff = id_in_sample_submission - id_in_sub



sample_submission = pd.concat([

    sub,

    sample_submission[sample_submission.qa_id.isin(diff)]

]).reset_index(drop=True)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)