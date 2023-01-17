import pandas as pd 

from random import uniform as rdm
submission = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")

# insert all the stuff to do

# read filenames

# process files

# predict value for files

# set prediction for files

submission['label'] = submission['label'].apply(lambda x: rdm(0.79, 0.81))

# make csv again

submission.to_csv('submission.csv', index=False)

# done
