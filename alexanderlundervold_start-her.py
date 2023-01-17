%matplotlib inline
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sampleSubmission = pd.read_csv('../input/sampleSubmission.csv')
demography = pd.read_csv('../input/demography.csv')
#train = pd.read_csv('https://github.com/MMIV-ML/ELMED219x/raw/master/data/project/train.csv')
#test = pd.read_csv('https://github.com/MMIV-ML/ELMED219x/raw/master/data/project/test.csv')
#sampleSubmission = pd.read_csv('https://github.com/MMIV-ML/ELMED219x/raw/master/data/project/sampleSubmission.csv')
#demography = pd.read_csv('https://github.com/MMIV-ML/ELMED219x/raw/master/data/project/demography.csv')
train.head()
test.head()
demography.head()
train.info()
test.info()
demography.info()
train.describe()
merged = pd.merge(demography, train, on='IXI_ID')
merged.head()
list(merged.columns)
fig = sns.boxplot(x='SEX_ID (1=m, 2=f)', y='EstimatedTotalIntraCranialVol', data=merged)
fig = sns.boxplot(x='AGE_GROUP', y='EstimatedTotalIntraCranialVol', data=merged)
predicted_age_groups = [0 for i in test['IXI_ID']]
len(predicted_age_groups)
predicted_age_groups
submission = pd.DataFrame({'IXI_IDX': test['IXI_ID'], 'label': predicted_age_groups})
submission.head()
submission.to_csv('submission.csv', index=False)