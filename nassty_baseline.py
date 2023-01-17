import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
train_df = pd.read_csv('../input/train.csv')

train_df.head()
test_df = pd.read_csv('../input/test.csv')

test_df.head()
columns_in = [f"sensor#{i}" for i in range(12)]

columns_out = "oil_extraction"



train_x = train_df[columns_in].values

train_y = train_df[columns_out].values

test_x = test_df[columns_in].values
lin_reg = LinearRegression()

lin_reg.fit(train_x, train_y)

train_predictions = lin_reg.predict(train_x)



plt.plot(train_df["timestamp"].values, train_y, label='oil extraction')

plt.plot(train_df["timestamp"].values, train_predictions, label='prediction')

plt.legend(loc='upper right')

plt.show()
def predictions_to_submission_file(predictions):

    submission_df = pd.DataFrame(columns=['Expected', 'Id'])

    submission_df['Expected'] = predictions

    submission_df['Id'] = range(len(predictions))

    submission_df.to_csv('submission.csv', index=False)



test_predictions = lin_reg.predict(test_x)

predictions_to_submission_file(test_predictions)