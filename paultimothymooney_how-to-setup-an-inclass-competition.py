import numpy as np
import pandas as pd 
import os
#print(os.listdir("../input/"))
diabetes = pd.read_csv('../input/scratchpad/diabetes.csv')
diabetes['Id'] = diabetes.index
diabetes.head()
diabetes['split'] = np.random.randn(diabetes.shape[0], 1)
msk = np.random.rand(len(diabetes)) <= 0.7
train = diabetes[msk]
test = diabetes[~msk]

train = train.drop(columns='split')
train.head()
test2 = test.drop(columns='Outcome')
test2 = test2.drop(columns='split')
test2.head()
answer_key = test['Outcome']
answer_key = pd.DataFrame(answer_key)
answer_key['Id'] = answer_key.index
answer_key = answer_key[['Id', 'Outcome']]
answer_key.head()
sample_submission = test['Outcome']
sample_submission = sample_submission.replace(0)
sample_submission = pd.DataFrame(sample_submission)
sample_submission['Id'] = sample_submission.index
sample_submission = sample_submission[['Id', 'Outcome']]
sample_submission.head()
train.to_csv('train.csv', index=False)
test2.to_csv('test.csv', index=False)
answer_key.to_csv('answer_key.csv', index=False)
sample_submission.to_csv('sample_submission.csv', index=False)