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

#read csv

h1b = pd.read_csv('../input/h1b_kaggle.csv')

print(h1b.head())

h1b.dropna()



#select out wages and plot

wages = h1b['PREVAILING_WAGE']

print(wages.shape) #to determine the range

plt.scatter(range(3002458),wages) #Plot the graph - looks like we have a big outlier

plt.xlabel("H1b Petitions")

plt.show()



#use a function to filter the column

def reject_outliers(data, m=2):

    return data[abs(data - np.mean(data)) < m * np.std(data)]



fwages = reject_outliers(wages,m=3) #filter outliers more than 3 std deviations

print(fwages.shape)



#plot again

plt.scatter(range(fwages.shape[0]),fwages) #Plot the graph - looks like we have a big outlier

plt.xlabel("H1b Petitions")

plt.ylabel("Salary")

plt.show()



print(fwages.describe())



#Let's find out who employs the most H1B's in San Diego

sdiego=h1b[h1b['WORKSITE']=='SAN DIEGO, CALIFORNIA']

sdiego['EMPLOYER_NAME'].value_counts().head(25)
