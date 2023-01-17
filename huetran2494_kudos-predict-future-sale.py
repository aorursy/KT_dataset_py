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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

default_path = '/kaggle/input/competitive-data-science-predict-future-sales/'
train_df = pd.read_csv(default_path+'sales_train.csv')
items_df = pd.read_csv(default_path+'items.csv')
item_categories_df = pd.read_csv(default_path+'item_categories.csv')
shops_df = pd.read_csv(default_path+'shops.csv')

sample_submission_df = pd.read_csv(default_path+'sample_submission.csv')
test_df = pd.read_csv(default_path+'test.csv')
print(train_df.shape, test_df.shape)
train_df.head()
train_df.info()
train_df.shape
print('Total records:', train_df.shape[0]/(10**6))
print('Total columns:',train_df.shape[1])
def exploring_stats(df_input):
    total_rows = df_input.shape[0]
    total_columns = df_input.shape[1]
    # check data type
    name = []
    sub_type = []
    for n, t in df_input.dtypes.iteritems():
        name.append(n)
        sub_type.append(t)

    # check distinct
    # cname is column name
    check_ndist = []
    for cname in df_input.columns:
        ndist = df_input[cname].nunique()
        pct_dist = ndist * 100.0 / total_rows
        check_ndist.append("{} ({:0.2f}%)".format(ndist, pct_dist))
    # check missing
    check_miss = []
    for cname in df_input.columns:
        nmiss = df_input[cname].isnull().sum()
        pct_miss = nmiss * 100.0 / total_rows
        check_miss.append("{} ({:0.2f}%)".format(nmiss, pct_miss))
    # check zeros
    check_zeros = []
    for cname in df_input.columns:
        try:
            nzeros = (df_input[cname] == 0).sum()
            pct_zeros = nzeros * 100.0 / total_rows
            check_zeros.append("{} ({:0.2f}%)".format(nzeros, pct_zeros))
        except:
            check_zeros.append("{} ({:0.2f}%)".format(0, 0))
            continue
    # check negative
    check_negative = []
    for cname in df_input.columns:
        try:
            nneg = (df_input[cname].astype("float") < 0).sum()
            pct_neg = nneg * 100.0 / total_rows
            check_negative.append("{} ({:0.2f}%)".format(nneg, pct_neg))
        except:
            check_negative.append("{} ({:0.2f}%)".format(0, 0))
            continue
    data = {"column_name": name, "data_type": sub_type, "n_distinct": check_ndist, "n_miss": check_miss, "n_zeros": check_zeros,
            "n_negative": check_negative, }
    # check stats
    df_stats = df_input.describe().transpose()
    check_stats = []
    for stat in df_stats.columns:
        data[stat] = []
        for cname in df_input.columns:
            try:
                data[stat].append(df_stats.loc[cname, stat])
            except:
                data[stat].append(0.0)
    # col_ordered = ["name", "sub_type", "n_distinct", "n_miss", "n_negative", "n_zeros",
    #                "25%", "50%", "75%", "count", "max", "mean", "min", "std"]  # + list(pdf_sample.columns)
    df_data = pd.DataFrame(data)
    # df_data = pd.concat([df_data, df_sample], axis=1)
    # df_data = df_data[col_ordered]
    return df_data
exploring_stats(train_df)
train_df['date'] = pd.to_datetime(train_df['date'])
train_df.head()
