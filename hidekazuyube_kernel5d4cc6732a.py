import pandas as pd

df_gender_submission = pd.resd_csv('../input/gender_submission.csv')

df_gender_submission.to_csv('gender_submission.csv', index=False)