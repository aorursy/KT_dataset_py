# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from plotly import __version__
import cufflinks as cf
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
df.head()
df.info
df.count()
df["Uniq Id"].nunique()
df.dtypes

df = df[df['Role'].str.len() <= 50]
df = df[df['Job Title'].str.len() <= 50]
df = df[df['Job Salary'].str.len() <= 50]
df = df[df['Job Experience Required'].str.len() <= 50]
df = df[df['Key Skills'].str.len() <= 100]
df = df[df['Role Category'].str.len() <= 50]
df = df[df['Location'].str.len() <= 50]
df = df[df['Functional Area'].str.len() <= 100]
df = df[df['Industry'].str.len() <= 100]


df=df.where(df.apply(lambda x: x.map(x.value_counts()))>=10, "other")
df['Job Title'].value_counts(),
df['Job Salary'].value_counts()
df['Job Experience Required'].value_counts(),
df['Key Skills'].value_counts()
df['Role Category'].value_counts(),
df['Location'].value_counts()
df['Functional Area'].value_counts(),
df['Industry'].value_counts()
df['Role'].value_counts()
#df.groupby('Job Title').nunique().plot(kind='bar')
#plt.show()

df['Job Title'].value_counts().plot(kind='bar')
plt.show()