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
df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
df.head()
df.info()
df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis',cbar=False)
100*df.isnull().sum()/len(df)
df=df.dropna()
sns.heatmap(df.isnull(),cmap='viridis',cbar=False,yticklabels=False)
df=df.drop('Uniq Id',axis=1)
df=df.drop('Crawl Timestamp',axis=1)
df['Job Title'].value_counts().head(10)
df['Job Salary'].value_counts()
df['Location'].value_counts().head(10)

df[df['Location']=='Bengaluru']['Job Title'].value_counts().head()

df[df['Location']=='Bengaluru']['Industry'].value_counts().head()
df[df['Location']=='Bengaluru']['Functional Area'].value_counts().head()
df['Role'].value_counts().head(10)
df[df['Role']=='Software Developer']['Functional Area'].value_counts().head()
df[df['Role']=='Software Developer']['Industry'].value_counts().head()
