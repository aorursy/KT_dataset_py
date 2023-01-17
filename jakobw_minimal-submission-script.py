import os

import numpy as np

import pandas as pd



DATA_PATH = '/kaggle/input/prostate-cancer-grade-assessment/test_images'

TEST_CSV = '/kaggle/input/prostate-cancer-grade-assessment/test.csv'

SAMPLE_CSV = '/kaggle/input/prostate-cancer-grade-assessment/sample_submission.csv'



if os.path.exists(DATA_PATH):

    subject_ids = list(pd.read_csv(TEST_CSV).image_id)

    

    print(f"length subject_ids: {len(subject_ids)}")

    

    preds = []

    for subject_id in subject_ids:

        score = 0

        preds.append(score)



    sub_df = pd.DataFrame({'image_id': subject_ids, 'isup_grade': preds})

    sub_df.to_csv('submission.csv', index=False)

    print(sub_df.head())

else:

    print("test images not available (v3)")

    sub_df = pd.read_csv(SAMPLE_CSV)

    sub_df.to_csv('submission.csv', index=False)