import os

print(os.listdir('../input/'))
!ls ../input
import pandas as pd

sampleSubmission = pd.read_csv('../input/sampleSubmission.csv')
sampleSubmission.head()
sampleSubmission.to_csv('./my_output.csv', index=False)
print('output ok!')
print(os.listdir('../input/'))
print(os.listdir('./'))
