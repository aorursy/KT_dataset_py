import pandas as pd 

from random import uniform as rdm
submission = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")

submission['label'] = submission['label'].apply(lambda x: rdm(0.49, 0.51))

submission.to_csv('submission.csv', index=False)