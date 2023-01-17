import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd
# reading test data

df_test = pd.read_csv('/kaggle/input/goodreads-quote-tagging/test.csv')

df_test.head()
# creating dataset of all false predictions for tags

df_labels = df_test.drop('quote', axis=1)

df_preds = df_labels.replace(to_replace=True, value=False)

df_preds.head()
# comparing values between dataframes and getting average tag accuracy per quote



def accuracy(df_preds, df_labels):

    diff = df_preds.values != df_labels.values

    accuracy = 1 - diff.flatten().sum() / df_labels.shape[0] / df_labels.shape[1]

    print('{:.2f}%'.format(accuracy * 100))

    

accuracy(df_preds, df_labels)
for column in df_preds:

    df_preds[column] = np.where(df_test['quote'].str.contains(column, case=False, na=False), True, False)

    

df_preds.head()
accuracy(df_preds, df_labels)