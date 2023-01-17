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
import pandas as pd

import numpy as np

import seaborn as sns

sns.set_style('whitegrid')

from scipy import stats

import matplotlib.pyplot as plt



%matplotlib inline


prop_df = pd.read_csv('../input/properties_2016.csv')

train_df = pd.read_csv("../input/train_2016.csv", parse_dates=["transactiondate"])

#features_name_dict = pd.read_excel("../input/zillow_data_dictonary.xlsx")
print(prop_df.shape)

print(train_df.shape)
train_df.head()
prop_df.head()
#print(features_name_dict)
#features = features_name_dict.to_dict(orient='records')

features = features_name_dict.to_records()

#print(features)
merged_df= pd.merge(prop_df,train_df,on="parcelid")
merged_df.head(2).transpose()