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
datag = pd.read_csv("../input/student-grades-record/grade-records.csv")
datag.head()
datag['First Name']
datag.tail()
datag.columns
datag.shape
datag.info()
datag.dtypes

datag.head()





#removes spaces form the columns:

datag = datag.rename(columns={c: c.replace(' ', '') for c in datag.columns})

datag.head()



# convert percentage to float

datag['CW1'] = datag['CW1'].apply(lambda x: x.replace('%', '')).astype('float') / 100

#

datag['Finalexam'] = datag['Finalexam'].apply(lambda x: x.replace('%', '')).astype('float') / 100

datag['CW2'] = datag['CW2'].apply(lambda x: x.replace('%', '')).astype('float') / 100

datag['Mid-termexams'] = datag['Mid-termexams'].apply(lambda x: x.replace('%', '')).astype('float') / 100



datag.head()



# convert to integer

datag['CW1'] = (datag['CW1']*100).astype('int')

datag['CW2'] = (datag['CW2']*100).astype('int')

datag['Finalexam'] = (datag['Finalexam']*100).astype('int')

datag['Mid-termexams'] = (datag['Mid-termexams']*100).astype('int')



datag.head()
datag.head()
print(datag['Grade'].value_counts(dropna=False))
datag.describe()
import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool
datag.head()



datag.boxplot(column='Mid-termexams', by = 'Finalexam')
datagnew = datag.head(10)



datagnew
# melting:



melted1 = pd.melt(frame=datagnew, id_vars='Grade', value_vars= ['Finalexam', 'Mid-termexams'])

melted1
# toplam aldırıp yaptık, farklı komut ile, çünkü duplicate verdi, sum yapmak gerekti.:

table2 = pd.pivot_table(melted1, values='value', index=['Grade'], columns=['variable'], aggfunc=np.sum)

table2



# concenate-rows:

datag.head(10)



datag1 = datag.head()

datag2 = datag.tail()



conc_datag_rows = pd.concat([datag1, datag2], axis=0, ignore_index=True)

conc_datag_rows

# concenate-columns:

datag3 = datag['FirstName'].head()

datag4 = datag['CW1'].head()



conc_datag_cols = pd.concat([datag3,datag4], axis=1)



conc_datag_cols

datag.dtypes
datag.head(10)
datag['CW2'] = datag['CW2'].astype('float')

datag.dtypes

datag.head(10)
datag.info()
datag['CW1'].value_counts(dropna=False)
datag5 = datag

datag5.head(10)



datag5['CW1'].dropna(inplace=True)  #nas değerleri at



assert datag5['CW1'].notnull().all()



datag5['CW1'].fillna('empty', inplace=True) # na ları, empty ile doldur



assert datag5['CW1'].notnull().all()
assert datag5.columns[1] == 'FirstName'