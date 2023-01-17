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
iptrain = pd.read_csv('../input/train-with-ipdata/ipaddresstrain.csv')

countrycount_df = iptrain.groupby('country_name').count()[['id']]

countrycount_df.columns = ['count']

countrycount_df

country_toxic_df = iptrain.groupby('country_name').mean()[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]

df= pd.concat([countrycount_df,country_toxic_df],axis=1)

df.sort_values(by=['toxic'],ascending=False)