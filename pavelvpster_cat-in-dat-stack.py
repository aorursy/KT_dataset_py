import numpy as np

import pandas as pd



import warnings

warnings.simplefilter('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
submission = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv', index_col='id')
kernels = pd.read_csv('../input/cat-in-dat-kernels/kernels.csv', index_col='id')
kernels.head()
import glob



def make_filename(idx):

    return glob.glob('../input/cat-in-dat-kernels/' + str(idx) + '__submission__*.csv')[0]



def read_predictions(idx):

    temp = pd.read_csv(make_filename(idx), index_col='id')

    temp.columns = [str(idx)]

    return temp





predictions = pd.concat([read_predictions(idx) for idx in kernels.index], axis=1)

predictions.shape
predictions.head()
# From https://seaborn.pydata.org/examples/many_pairwise_correlations.html



import seaborn as sns

import matplotlib.pyplot as plt



corr = predictions.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(12, 12))



sns.heatmap(corr, mask=mask, cmap='Blues', vmin=0.95, center=0, linewidths=1, annot=True, fmt='.4f')
submission['target'] = predictions.mean(axis=1)

submission.to_csv('stack-mean.csv')
submission.head()
scores = kernels['score']



sum_scores = sum(scores)



weights = [x / sum_scores for x in scores]
sum_predictions = predictions.dot(pd.Series(weights, index=predictions.columns))
sum_predictions.head()
submission['target'] = sum_predictions

submission.to_csv('stack-weighted-sum.csv')
N = 3



selected = kernels.sort_values('score', ascending=False).head(N)
print('Max selected score =', selected['score'].max())

print('Min selected score =', selected['score'].min())
filter_predictions = predictions.loc[:,selected.index.values.astype(str)]
filter_predictions.head()
submission['target'] = filter_predictions.mean(axis=1)

submission.to_csv('stack-filtered.csv')