# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

#from subprocess import check_output





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

smsData = pd.read_csv('/kaggle/input/spam.csv',encoding ="ISO-8859-1")

type(smsData)



smsData.head(10)



import matplotlib.pyplot as plt

#ts = pd.Series(np.random.randint(0,1000), index=pd.date_range('1/1/2000', periods=1000))



#ts = ts.cumsum()



#ts.plot()



#lens = smsData.v2.str.len()

#df.loc[df['column_name'] == some_value]

hamSms = smsData.loc[smsData['v1'] == 'ham']

spamSms = smsData.loc[smsData['v1'] == 'spam']

hamLens = hamSms.v2.str.len()

spamLens = spamSms.v2.str.len()

#print(hamSms)

#print(lens)

#lens.hist(bins = 100)

sns.set(color_codes=True)



# x = np.random.normal(size=100)

sns.distplot(hamLens, label='length ham of message', color='green');



sns.distplot(spamLens, label='length spam of message', color='red');





#smsData.plot(x='v1',y='v2')

#smsData.show()






