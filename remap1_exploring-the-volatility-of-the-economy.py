# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)
train_df = pd.read_csv("../input/train.csv")

train_df = train_df[['id','timestamp','price_doc', 'full_sq', 'num_room']]

train_df = train_df[train_df['full_sq'] < 450]

train_df = train_df[train_df['num_room'] > 0]

macro_df = pd.read_csv("../input/macro.csv")

macro_df = macro_df[['timestamp','usdrub','eurrub','mortgage_rate', 'mortgage_value', 'deposits_rate', 'salary_growth', 'cpi', 'ppi', 'overdue_wages_per_cap']]
def visualize_feature_over_time(df, feature):

    df['date_column'] = pd.to_datetime(df['timestamp'])

    df['mnth_yr'] = df['date_column'].apply(lambda x: x.strftime('%B-%Y'))



    df = df[[feature,"mnth_yr"]]

    df_vis = df.groupby('mnth_yr')[feature].mean()

    df_vis = df_vis.reset_index()

    df_vis['mnth_yr'] = pd.to_datetime(df_vis['mnth_yr'])

    df_vis.sort_values(by='mnth_yr')

    df_vis.plot(x='mnth_yr', y=feature)



    plt.figure()

    plt.show()
visualize_feature_over_time(train_df, "price_doc")
visualize_feature_over_time(macro_df, "usdrub")
visualize_feature_over_time(macro_df, "eurrub")
visualize_feature_over_time(macro_df, "cpi")
visualize_feature_over_time(macro_df, "ppi")
visualize_feature_over_time(macro_df, "mortgage_rate")
visualize_feature_over_time(macro_df, "deposits_rate")
visualize_feature_over_time(macro_df, "salary_growth")
visualize_feature_over_time(macro_df, "overdue_wages_per_cap")