!pip install pandas-profiling
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/german-credit/german_credit_data.csv')

df.head()
df.profile_report(style={'full_width':True})
import seaborn as sns 

sns.set_color_codes("pastel")

sns.catplot(hue="Checking_account", x="Saving_accounts",y ="Duration",kind="bar",data=df,height=10)
sns.scatterplot(x="Duration", y="Credit_amount",

                     hue="Age", size="Age",

                    sizes=(20, 300),

                    alpha=0.8,

                    palette="Set2",

                    #style=df.Job,

                     data=df)