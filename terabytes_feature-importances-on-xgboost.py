import pandas as pd

import xgboost as xgb



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_X = train_df.drop(['Activity'], axis=1).values

train_y = train_df.Activity.values



test_X = test_df.drop(['Activity'], axis=1).values

test_y = test_df.Activity.values
model = xgb.XGBClassifier()
model.fit(train_X, train_y)
score=model.score(test_X, test_y)

print('Test accuracy:', score)
importance = model.booster().get_fscore().items()

plot_x = train_df.drop(['Activity'], axis=1).columns.values

plot_y = [0] * 562

for key, value in importance:

    plot_y[int(key[1:])] = value
plot_x = [k for k, v in zip(plot_x, plot_y) if v > 20]

plot_y = [v for v in plot_y if v > 20]
g = sns.barplot(

    x=plot_x,

    y=plot_y)

plt.xticks(rotation=90)

plt.show()