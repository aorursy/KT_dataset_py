import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
# read the sample_submisison file into a dataframe, for sake of clarity let's limit the number of rows to 5
df_samples = pd.read_csv('../input/sample_submission.csv', nrows=5)
df_samples.head()
# let's unpivot the dataframe to produce the shape regression models usually expect
# consisting of pairs `parcelid`, `transactiondate``

def reshape_for_model(df):
    """Unipivot the submission data and apply some renamings"""
    df = pd.melt(df, ['ParcelId'])
    df.drop('value', axis=1, inplace=True)
    df.columns = ['parcelid', 'transactiondate']
    df['transactiondate'] = df['transactiondate'].apply(
        lambda date_str: "%s-%s-01" %(date_str[:4], date_str[-2:]))
    return df
df_reshaped = reshape_for_model(df_samples)
df_reshaped.head()
df_reshaped['prediction'] = pd.Series(data=np.random.rand(len(df_reshaped)))
df_reshaped.head()
# to get the original shape of our submission sample file, we can now pivot the table again
# and remove the column names
def reshape_for_submission(df):
    """Reformat the transactiondate and pivot the data"""
    df['transactiondate'] = df['transactiondate'].apply(lambda td: "%s%s" %(td[:4],td[5:7]))
    df = df.pivot(index='parcelid', columns='transactiondate', values='prediction'
                 ).reset_index().rename_axis(None,1)
    df = df.rename(index=str, columns={"parcelid": "ParcelId"})
    return df
reshape_for_submission(df_reshaped)