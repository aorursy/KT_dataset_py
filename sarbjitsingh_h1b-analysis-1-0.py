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
import pandas as pd
df = pd.read_csv('../input/h1b_kaggle.csv')
df.head()
df.info()
df.dropna(inplace=True)

df.isnull().sum()
%matplotlib inline

yearly = df.groupby('YEAR')

yearly['CASE_STATUS'].count()

yearly['CASE_STATUS'].count().plot()
dataengg = df.loc[df['JOB_TITLE'].str.contains('DATA')]
deyearly = dataengg.groupby('YEAR')

deyearly['CASE_STATUS'].count()

deyearly['CASE_STATUS'].count().plot()
hengg = df.loc[df['JOB_TITLE'].str.contains('HARDWARE ENG')]

hw = hengg.groupby('WORKSITE')

ss = hw['CASE_STATUS'].count()

ss.sort_values(ascending=False).head().plot(kind='bar')
xx = dataengg.groupby('SOC_NAME')

yy = xx['CASE_STATUS'].count()

yy.sort_values(ascending=False).head().plot(kind='bar')