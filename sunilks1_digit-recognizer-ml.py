import numpy as np
import pandas as pd
import os

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
train.head()


# To check data types
train.dtypes.head()
test.dtypes.head()
sample_submission.dtypes
# MISSING VALUE
mv=train.isna().sum()
mv.head()
train.describe()
import matplotlib.pyplot as plt
first_digit =train.iloc[3].drop('label').values.reshape(28,28)
plt.imshow(first_digit)
from sklearn.model_selection import train_test_split
train_1, validation = train_test_split(train,
                               test_size = 0.3,
                               random_state=500)
print(train_1.shape)
print(validation.shape)



train_1_y = train_1['label']
validation_y = validation['label']

train_1_x = train_1.drop('label', axis=1)
validation_x = validation.drop('label', axis=1)

train_1_x.shape
from sklearn.ensemble import RandomForestClassifier
Model = RandomForestClassifier (random_state=100, n_estimators=300)
Model.fit(train_1_x, train_1_y)

# predicting on test data
test_pred = Model.predict(validation_x)

df_pred = pd.DataFrame({'actual': validation_y,
                         'predicted': test_pred})

df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
df_pred.head()
df_pred['pred_status'].value_counts() / df_pred.shape[0]* 100
test_pred = Model.predict(test)
df_test_pred = pd.DataFrame(test_pred, columns=['Label'])
df_test_pred['ImageId'] = test.index + 1
df_test_pred[['ImageId', 'Label']].to_csv('submission.csv', index=False)

pd.read_csv('submission.csv').head(3)

sample_submission.head()
