import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

trans = pd.read_csv('../input/transactions.csv')
sample = trans.loc[trans.msno.isin(['QA7uiXy8vIbUSPOkCf9RwQ3FsT8jVq2OxDr8zqa7bRQ=','waLDQMmcOu2jLDaV1ddDkgCrB/jl6sD66Xzs0Vqax1Y=']),:].copy(deep=True)
def identify_churn(trans):

    trans["transaction_date"] = pd.to_datetime(trans["transaction_date"], format='%Y%m%d')

    trans["membership_expire_date"] = pd.to_datetime(trans["membership_expire_date"], format='%Y%m%d')

    trans = trans.sort_values(by=['msno', 'transaction_date']).reset_index(drop=True)

    trans["next_trans"] =trans.groupby("msno")["transaction_date"].shift(-1)

    trans["day_diff"] = trans.groupby("msno").apply(lambda trans: trans["next_trans"] - trans["membership_expire_date"]).reset_index(drop=True)

    threshold = pd.Timedelta('31 days')

    trans["churn_flag"] = trans["day_diff"]>threshold

    #trans['churn_date'] = trans["membership_expire_date"] + pd.Timedelta('31 days')

    return trans

sample = identify_churn(sample)
sample[sample.msno=='QA7uiXy8vIbUSPOkCf9RwQ3FsT8jVq2OxDr8zqa7bRQ=']