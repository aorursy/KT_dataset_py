import numpy as np
import pandas as pd
train_df = pd.read_csv('../input/binary_histology_students_training.csv')
train_df.sample(3)
test_df = pd.read_csv('../input/binary_histology_student_testing.csv')
test_df.sample(3)
submission_df = test_df[['index']].copy()
submission_df['cancer']=False # never cancer
submission_df.to_csv('submission.csv', index=False)
submission_df.sample(3)
