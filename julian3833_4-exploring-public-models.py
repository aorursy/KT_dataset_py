#https://www.kaggle.com/paulorzp/one-line-base-model-lb-0-847
import os; 
import pandas as pd

pd.read_csv('../input/sample_submission.csv', 
            converters = {'EncodedPixels': lambda p: None}).to_csv('submission_paulorzp.csv', index=False)