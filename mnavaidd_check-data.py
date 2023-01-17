# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

BPD = pd.read_csv("../input/crime-in-baltimore/BPD_Part_1_Victim_Based_Crime_Data.csv")
BPD.head()
BPD.head(n=100)
BPD.describe()
missing_data = BPD.isnull()

missing_data
for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")
BPD.drop('CrimeCode', axis=1, inplace=True)
BPD
BPD.drop('Post', axis=1, inplace=True)

BPD.drop('Neighborhood', axis=1, inplace=True)
BPD
BPD.CrimeDate.astype('datetime64')
BPD['CrimeDate']= pd.to_datetime(BPD['CrimeDate']) 

BPD['CrimeDate']
BPD
def extract_date(BPD,column):

    BPD[column+"_year"] = BPD[column].apply(lambda x: x.year)

    BPD[column+"_month"] = BPD[column].apply(lambda x: x.month)

extract_date(BPD, 'CrimeDate')

BPD.head()
BPD['CrimeDate_year'].value_counts()
import matplotlib.pylab as plt

plt.hist(BPD['CrimeDate_year'])

plt.title("Yearly Crime")

plt.xlabel("Crime Year")

plt.ylabel("Count")

plt.show()
import matplotlib.pylab as plt

plt.hist(BPD['CrimeDate_month'])

plt.title("Yearly Crime")

plt.xlabel("Crime Year")

plt.ylabel("Count")

plt.show()
BPD['Location'].value_counts()