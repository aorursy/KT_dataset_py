import pandas as pd

import numpy as np



train = pd.read_csv('../input/shopee-sentiment-analysis/train.csv', index_col='review_id')

test = pd.read_csv('../input/shopee-sentiment-analysis/test.csv', index_col='review_id')

test_labelled = pd.read_csv('../input/test-labelled/test_labelled.csv', index_col='review_id')
test
test_labelled
def dataframe_similarity(df1, df2, which='both'):

    """Find rows which are the same between two DataFrames."""

    comparison_df = df1.merge(df2,

                              indicator=True,

                              how='outer')

    diff_df = comparison_df[comparison_df['_merge'] == which]

    return diff_df
testsimilarities = dataframe_similarity(test, test_labelled)

testsimilarities
train
newtrain = train.append(test_labelled, ignore_index=True)
newtrain
newtrain = newtrain.sample(frac=1).reset_index(drop=True)
newtrain