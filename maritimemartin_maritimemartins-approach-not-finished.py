# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

test_data.columns
train_data.columns
train_data.head()
surv_mask = train_data.apply(lambda x: True if x["Survived"]==1 else False, axis=1)

survived = train_data[surv_mask]

killed = train_data[~surv_mask]

def get_series_entropie(series):

    series_prob_distr = series.value_counts(normalize=True)

    series_calc = series_prob_distr.apply(lambda x: x*math.log(x,2))

    return -series_calc.sum()



def get_df_entropie(df):

    return df.apply(lambda x: get_series_entropie(x))

    
entr_diff = ((get_df_entropie(survived)+ get_df_entropie(killed))/2) - get_df_entropie(test_data) 

entr_diff.sort_values()

get_df_entropie(test_data)
get_df_entropie(survived)
get_df_entropie(killed)