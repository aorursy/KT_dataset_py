%matplotlib inline

import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

from pathlib import Path
DATA = Path('../input/mmiv-healthhack-2019')

list(DATA.iterdir())
train = pd.read_csv(DATA/'train.csv')

test = pd.read_csv(DATA/'test.csv')

sampleSubmission = pd.read_csv(DATA/'sampleSubmission.csv')
#train = pd.read_csv('https://github.com/MMIV-ML/MMIVHealthHack2019/raw/master/train.csv')

#test = pd.read_csv('https://github.com/MMIV-ML/MMIVHealthHack2019/raw/master/test.csv')

#sampleSubmission = pd.read_csv('https://github.com/MMIV-ML/MMIVHealthHack2019/raw/master/sampleSubmission.csv')

#demography = pd.read_csv('https://github.com/MMIV-ML/MMIVHealthHack2019/raw/master/demography.csv')
train.info()
test.info()
train.head()
test.head()
train.Age.hist()

plt.show()
fig = sns.boxplot(x='Sex', y='EstimatedTotalIntraCranialVol', data=train)
# To illustrate the process, we need some predictions. Let's use the mean:

predicted_ages = [train.Age.mean() for i in test.SubjectID]
len(predicted_ages), predicted_ages[:5]
submission = pd.DataFrame({'SubjectID': test.SubjectID, 'label': predicted_ages})
submission.head()
submission.to_csv('submission.csv', index=False)