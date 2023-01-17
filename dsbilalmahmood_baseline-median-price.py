## Libraries used

import numpy as np 

import pandas as pd
## Reading test data

df = pd.read_csv("../input/test.tsv",sep = "\t")

## Median price from training data

median_price = 17.0

## Predicted price vector

pred_price_hat  = np.ones(shape = (df.shape[0] ,1)) * median_price

## Predicted dataframe

df_prediction = df[["test_id"]]

df_prediction["price"] = pred_price_hat 

## Saving the predictions

df_prediction.to_csv("sample_submission.csv",

                     index = False)