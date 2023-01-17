import numpy as np

import pandas as pd

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

from scipy.stats import skew, boxcox

import statsmodels.formula.api as smf



# Load Training Data

train = pd.read_csv('../input/train.csv', dtype={'id': np.int32})



# Load Test Data

test = pd.read_csv('../input/test.csv', dtype={'id': np.int32})
plt.hist(train['loss'], 30, normed=1)

plt.xlabel('Loss')

plt.ylabel('Probability')

plt.title('Distribution of Losses')

plt.show()
skew(train['loss'])
train['log_loss'] = np.log(train['loss'])



plt.hist(train['log_loss'], 30, normed=1)

plt.xlabel('Log(Loss)')

plt.ylabel('Probability')

plt.title('Distribution of Log(Loss)es')

plt.show()
features_numeric = test.dtypes[test.dtypes != "object"].index

features_skewed = train[features_numeric].apply(lambda x: skew(x.dropna()))

features_skewed
features_skewed = features_skewed[features_skewed > 0.2]

for feat in features_skewed.index:

    train[feat], lam = boxcox(train[feat] + 1)

    test[feat] = boxcox(test[feat] + 1, lam)



features_skewed = train[features_numeric].apply(lambda x: skew(x.dropna()))

features_skewed
features_categorical = [feat for feat in test.columns if 'cat' in feat]



for feat in features_categorical:

    a = pd.DataFrame(train['log_loss'].groupby([train[feat]]).mean())

    a[feat] = a.index

    train[feat] = pd.merge(left=train, right=a, how='left', on=feat)['log_loss_y']

    test[feat] = pd.merge(left=test, right=a, how='left', on=feat)['log_loss']



features_categorical = test.dtypes[test.dtypes == "object"].index
counts = train.count()

len(counts[counts < train.shape[0]])
counts = test.count()

len(counts[counts < test.shape[0]])
temp = test.dropna(1)

counts = temp.count()

len(counts[counts < temp.shape[0]])
model = smf.ols('log_loss ~ ' + ' + '.join(temp.columns), data=train).fit()

model.summary()
yhat = np.exp(model.predict(test))
result = pd.DataFrame({'id': test['id'].values, 'loss': yhat})

result = result.set_index('id')

result.to_csv('simplelmprediction.csv', index=True, index_label='id')