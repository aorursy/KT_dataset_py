import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import numpy as np

 
train_df = pd.read_csv('../input/train.csv')

train_df.head()
test_df = pd.read_csv('../input/test.csv')

test_df.head()
columns_in = [f"sensor#{i}" for i in range(12)]

columns_out = "oil_extraction"



X = train_df[columns_in].values

train_y = train_df[columns_out].values

test_x = test_df[columns_in].values



def transform(x):

    diffs = 25

    diff = [np.zeros(x.shape) for i in range(diffs)]

    n = len(x)

    diff[0][1:,:] = x[:n-1,:]

    for i in range(1, diffs):

        diff[i][1:,:] = diff[i-1][:n-1,:]

    return np.hstack([x, *diff])



train_x = np.hstack([transform(X), transform(X**2)])
lin_reg = LinearRegression()

lin_reg.fit(train_x, train_y)

train_predictions = lin_reg.predict(train_x)



n = 500

plt.plot(train_df["timestamp"].values[:n], train_y[:n], label='oil extraction')

plt.plot(train_df["timestamp"].values[:n], train_predictions[:n], label='prediction')

plt.legend(loc='upper right')

plt.show()
def predictions_to_submission_file(predictions):

    submission_df = pd.DataFrame(columns=['Expected', 'Id'])

    submission_df['Expected'] = predictions

    submission_df['Id'] = range(len(predictions))

    submission_df.to_csv('submission.csv', index=False)



test_predictions = lin_reg.predict(np.hstack([transform(test_x), transform(test_x**2)]))

predictions_to_submission_file(test_predictions)
plt.plot(train_df["timestamp"].values[:1000], train_y[:1000], label='oil extraction')

plt.show()

plt.plot(train_df["timestamp"].values[:1000], train_df["sensor#0"].values[:1000], label='oil extraction')

plt.show()