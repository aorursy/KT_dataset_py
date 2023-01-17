# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
trainusers=pd.read_csv("../input/train_users_2.csv")

testusers=pd.read_csv("../input/test_users.csv")

users=pd.concat((trainusers,testusers),axis=0,ignore_index=True)
users.drop('id',axis=1,inplace=True)

users.head(30)
users.gender.replace('-unknown-',np.nan,inplace=True)

users.first_browser.replace('-unknown-',np.nan,inplace=True)

users.head(30)
users_nan = (users.isnull().sum() / users.shape[0]) * 100

users_nan[users_nan > 0].drop('country_destination')
users['age'].describe()
women = sum(users['gender'] == 'FEMALE')

men = sum(users['gender'] == 'MALE')



female_destinations = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts() / women * 100

male_destinations = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts() / men * 100



male_destinations.plot(kind='bar', width=0.4,  position=0, label='men')

female_destinations.plot(kind='bar',color='#ff9966', width=0.4, position=1, label='women')

plt.legend()

plt.xlabel('Destination Country')

plt.ylabel('Percentage per gender')

plt.show()