# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
SuicideRate = pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")
SuicideRate.head()
SuicideRate.tail()
pearsoncorr = SuicideRate.corr(method='pearson')

pearsoncorr
def correlation(dataset, threshold):

    col_corr = set() # Set of all the names of deleted columns

    corr_matrix = dataset.corr(method='pearson')

    for i in range(len(corr_matrix.columns)):

        for j in range(i):

            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):

                colname = corr_matrix.columns[i] # getting the name of column

                col_corr.add(colname)

                if colname in dataset.columns:

                    del dataset[colname] # deleting the column from the dataset



    print(col_corr)
correlation(SuicideRate, 0.3)
SuicideRate.head()