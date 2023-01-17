import numpy as np

import pandas as pd



import warnings

warnings.simplefilter('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv', index_col='id')
kernels = pd.read_csv('../input/cat-in-dat-2-public-kernels/kernels.csv', index_col='id')
kernels.head(10)
import glob



def make_filename(idx):

    return glob.glob('../input/cat-in-dat-2-public-kernels/' + str(idx) + '__submission.csv')[0]



def read_predictions(idx):

    temp = pd.read_csv(make_filename(idx), index_col='id')

    temp.columns = [str(idx)]

    return temp





predictions = pd.concat([read_predictions(idx) for idx in kernels.index], axis=1)

predictions.shape
predictions.head()
import seaborn as sns

import matplotlib.pyplot as plt





plt.figure(figsize=(10,10))



for column in predictions.columns:

    sns.kdeplot(predictions[column], label=column)



plt.show()
# From https://seaborn.pydata.org/examples/many_pairwise_correlations.html



import seaborn as sns

import matplotlib.pyplot as plt



corr = predictions.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(12, 12))



sns.heatmap(corr, mask=mask, cmap='Blues', vmin=0.95, center=0, linewidths=1, annot=True, fmt='.4f')



plt.show()
submission['target'] = predictions.mean(axis=1)

submission.to_csv('stack-mean.csv')
submission.head()
scores = kernels['score']



sum_scores = sum(scores)



weights = [x / sum_scores for x in scores]
weighted_sum_prediction = predictions.dot(pd.Series(weights, index=predictions.columns))
weighted_sum_prediction.head()
submission['target'] = weighted_sum_prediction

submission.to_csv('stack-weighted-sum.csv')
scores = kernels['score']



sum_scores = sum(scores)



weights = [x / sum_scores for x in scores]
from scipy.stats import rankdata





def blend_by_ranking(data, weights):

    out = np.zeros(data.shape[0])

    for idx,column in enumerate(data.columns):

        out += weights[idx] * rankdata(data[column].values)

    out /= np.max(out)

    return out
blend_by_ranking_prediction = blend_by_ranking(predictions, weights)
blend_by_ranking_prediction
submission['target'] = blend_by_ranking_prediction

submission.to_csv('stack-blend-by-ranking.csv')