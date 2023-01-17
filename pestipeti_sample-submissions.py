import pandas as pd
DIR_INPUT = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"



SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"

MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"
# Single mode

sample_submission = pd.read_csv(SINGLE_MODE_SUBMISSION)



# Multi mode

# sample_submission = pd.read_csv(MULTI_MODE_SUBMISSION)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)