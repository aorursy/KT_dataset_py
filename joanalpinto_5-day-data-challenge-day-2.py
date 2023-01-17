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



df_parties = pd.read_csv("../input/party_in_nyc.csv")



#numeric variables' statistics

print(df_parties.describe())

print("\n")



#remove missing values from latitude varible

print('Number of missing values in latitude: %d.' % df_parties.Latitude.isnull().sum())

mean_latitude = df_parties.Latitude.mean()

df_parties.Latitude = df_parties.Latitude.fillna(mean_latitude)



#pick Latitude varible

print('%s' % "Latitude")

latitude = df_parties["Latitude"]

print(latitude)



#plot Latitude Histogram

plt.hist(latitude)


