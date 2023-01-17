# Language : Python

# Environment : Preferences

# Accelerator : None

# Internet : Off

#

# (1) Push [Save Version] -> Select [Save & Run All (Commit)] -> Push [Save]

# (2) Push [Submit] from [Output]



import pandas as pd



submission = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

submission['toxic'] = 0.0

submission.to_csv('submission.csv', index=False)