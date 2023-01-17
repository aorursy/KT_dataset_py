import pandas as pd

import numpy as np

import glob, os

PATH = "../input/rsna-str-pulmonary-embolism-detection/"



train_df = pd.read_csv(PATH + "train.csv")

test_df = pd.read_csv(PATH + "test.csv")



TRAIN_PATH = PATH + "train/"

TEST_PATH = PATH + "test/"

sub = pd.read_csv(PATH + "sample_submission.csv")

train_image_file_paths = glob.glob(TRAIN_PATH + '/*/*/*.dcm')

test_image_file_paths = glob.glob(TEST_PATH + '/*/*/*.dcm')



print(f'Train dataframe shape  :{train_df.shape}')

print(f'Test dataframe shape   :{test_df.shape}')



print(f'Number of train images : {len(train_image_file_paths)}')

print(f'Number of test images  : {len(test_image_file_paths)}')
exam_level_features = ['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',

                       'leftsided_pe',         'chronic_pe',        'rightsided_pe', 

                       'acute_and_chronic_pe', 'central_pe',        'indeterminate']
sub.info()
from tqdm.notebook import tqdm

prediction_counts = {}

for idx in tqdm(range(sub.shape[0])):

    if len(sub['id'][idx][13:]) > 1:

        key = sub['id'][idx][13:]

    else:

        key = 'pe_present_on_image'

    prediction_counts[key] = prediction_counts.get(key, 0) + 1

print(f'Total row count in submission: {sub.shape[0]}')

prediction_counts
N_img = len(test_image_file_paths)

N_exams = len(os.listdir(TEST_PATH))

N_exam_level_features = len(exam_level_features)



total_rows_submission = N_img + (N_exams * N_exam_level_features)

print(f'Total row count in submission: {total_rows_submission}')
StudyInstanceUIDs = os.listdir(TEST_PATH)

SOPInstanceUIDs = [filename[-16:-4] for filename in test_image_file_paths]



submission_rows = []



for exam_level_feature in exam_level_features:

    for StudyInstanceUID in StudyInstanceUIDs:

        submission_rows.append(StudyInstanceUID+'_'+exam_level_feature)



submission_rows = submission_rows + SOPInstanceUIDs 

print(f'Total row count in submission: {len(submission_rows)}')
len(submission_rows)
submission_file = pd.DataFrame({'id': submission_rows, 'label': (np.zeros(len(submission_rows))+0.35)})
submission_file.head()

submission_file.to_csv('submission.csv', index = False)