# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def df_stats(df):

    df_report = pd.DataFrame([], 

                             columns=["columns", "record", "nan", "nan_rate", 

                                      "nuniques", "nunique_ratio", 

                                      "unique_values", "dtype", "MB" 

                                     ])



    for col in df:

        dftrg = df[col]

        row = [col, len(dftrg), dftrg.isna().sum(), dftrg.isna().mean(), dftrg.nunique(),  dftrg.nunique()/len(dftrg), dftrg.unique() , dftrg.dtypes, dftrg.memory_usage(deep=True, index=False)/1024/1024]



        s = pd.DataFrame(row, index=df_report.columns).T

        df_report = df_report.append(s,  ignore_index=True)



    return df_report
DEBUG=True
test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

sample_submission = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")
df_all_dict = {

    "test_features": test_features,

    "train_features": train_features,

    "train_targets_scored": train_targets_scored,

    "train_targets_nonscored": train_targets_nonscored,

    "sample_submission": sample_submission,

}



df_all_list = [test_features, train_features, train_targets_scored, train_targets_nonscored, sample_submission]
for name, df in df_all_dict.items():

    display("## {}".format(name))

    display(df.head(3))
display(train_features.head(3))

df_stats(train_features)
train_features.agg(["min", "max"])
train_features.columns.values
display(test_features.head(3))

df_stats(test_features)
display(train_targets_scored.head(3))

df_stats(train_targets_scored)

display(train_targets_scored.columns.values)
display(train_targets_nonscored.head(3))

df_stats(train_targets_nonscored)
display(train_targets_nonscored.columns[:20])
display(sample_submission.head(3))

df_stats(sample_submission)
submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



usecols = submission.columns.tolist()

usecols.remove("sig_id")

assert "sig_id" not in usecols



submission.loc[:, usecols] = (np.random.rand(3982, 206) > 0.5).astype(int)
submission


submission.to_csv('submission.csv', index=False)