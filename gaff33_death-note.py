# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# Any results you write to the current directory are saved as output.
#Read in the death records to pandas df
deaths = pd.read_csv('../input/DeathRecords.csv')
codes = pd.read_csv('../input/Icd10Code.csv')
manners = pd.read_csv('../input/MannerOfDeath.csv')
icd10 = pd.read_csv('../input/Icd10Code.csv')
deaths[deaths['MannerOfDeath']==0]['Age'].hist(bins=range(102))

top10 = deaths[['Icd10Code', 'Id']]\
    .groupby(['Icd10Code'])\
    .count()\
    .sort_values(['Id'], ascending=False)\
    .head(10)
top10 = pd.merge(top10, icd10, left_index=True, right_on=['Code'])
top10
top10.plot(kind='bar', x='Description')
