# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
rswDataDf = pd.read_csv('../input/train.csv')
X_ = rswDataDf['X']

Y_ = rswDataDf['Y']
plt.scatter(X_,Y_)
print('The number of examples in the train dataset: ',len(rswDataDf))

print(rswDataDf.columns.values)

uniqueCats = rswDataDf['Category'].unique()



for i in range(len(uniqueCats)):

    print(i, '.', uniqueCats[i])
IncidentByCat = rswDataDf[['Category','Dates']].groupby('Category').count().reset_index()
IncidentByCat.rename(columns = {'Dates':'count'}, inplace=True)
index = np.arange(0,len(IncidentByCat))

plt.bar(index,IncidentByCat['count'],align='center', alpha=0.5)

plt.xticks(index,IncidentByCat['Category'], rotation='vertical')

plt.show()