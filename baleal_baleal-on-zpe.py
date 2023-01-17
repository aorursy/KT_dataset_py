# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train_2016_v2.csv", parse_dates = ["transactiondate"])

train_df.shape
train_df['transaction_day'] = train_df['transactiondate'].dt.weekday



cnt_srs = train_df['transaction_day'].value_counts()

plt.figure(figsize=(10,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Day of transaction', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
prop_df = pd.read_csv("../input/properties_2016.csv")

prop_df.shape
if train_df.shape[1] == 3: 

    train_df = pd.merge(train_df, prop_df, on='parcelid', how='left' )

train_df.shape

def NanPercent(daf):

    var, c_nan, p_nan = [], [], [];

    for i in range (0,len(list(daf))):

            count_nan = daf.shape[0] - daf[list(daf)[i]].count()

            percent_nan = (count_nan / daf.shape[0]) * 100

            var.append(list(daf)[i]),c_nan.append(count_nan), p_nan.append(percent_nan)

    

    Nanpercent_df = pd.DataFrame(

        {'Variable': var,

         'Nr of Nans': c_nan,

         '% of Nans': p_nan

        })

    return Nanpercent_df.sort_values(['% of Nans'])

NanPercent(train_df)
#Data PreProcessing

#propertyzoningdesc => string

train_df['propertyzoningdesc'] = train_df['propertyzoningdesc'].astype(str)

#hashottuborspa => TRUE ==> 1

train_df.hashottuborspa.replace('True',1, inplace=True)

train_df['hashottuborspa'] = train_df['hashottuborspa'].astype('float64')

#propertycountylandusecode ==> string

train_df['propertycountylandusecode'] = train_df['propertycountylandusecode'].astype(str)

#fireplaceflag ==> TRUE ==> 1

train_df.fireplaceflag.replace('True',1, inplace=True)

train_df['fireplaceflag'] = train_df['fireplaceflag'].astype('float64')

#taxdelinquencyflag ==> y

train_df.taxdelinquencyflag.replace('Y',1, inplace=True)

train_df['taxdelinquencyflag'] = train_df['taxdelinquencyflag'].astype('float64')



train_df.dtypes
sns.set(context="paper", font="monospace")

corrmat = train_df.corr()

f, ax = plt.subplots(figsize=(12,9))

sns.heatmap(corrmat,vmax=1, square=True)

f.tight_layout()