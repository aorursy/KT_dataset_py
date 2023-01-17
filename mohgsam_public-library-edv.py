# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/public-libraries/libraries.csv').sample(50)
df
df.columns
df_feacher = [ 'Library ID', 'Submission Year', 'Library Name','City','County', 'County Population',
       'Employees', 'Total Staff',  'Salaries', 'Benefits',
       'Total Staff Expenditures', 'Print Collection Expenditures',
       'Print Collection', 'Digital Collection', 'Audio Collection',
       'Downloadable Audio', 'Physical Video', 'Downloadable Video','Hours Open',
       'Library Visits', 'Registered Users','Library Programs',
       'Children’s Programs', 'Young Adult Programs',
       'Library Program Audience', 'Children’s Program Audience',
       'Young Adult Program Audience', 'Public Internet Computers']
df = df[df_feacher]
df
j = 0
for i in df['Total Staff']:
    print(j,'>',i)
    j = j +1
j = 0
for i in df['Hours Open']:
    print(j,'>',i)
    j = j +1
#df = df.drop([6095], axis=0)
df['Total Staff'] = pd.to_numeric(df['Total Staff'])
df['Hours Open'] = pd.to_numeric(df['Hours Open'])
plt.figure(figsize=(16,9))
sns.barplot(y= df['Library Name'],x=df['Total Staff'])
plt.title("image 1")
df['Submission Year'].dtypes
# df['Submission Year'] = pd.to_numeric(df['Submission Year'])
drop_None_Value = df.drop(index=df[df['Submission Year'] == 'None'].index)
new_sf_Dataset = drop_None_Value

df = new_sf_Dataset[['Submission Year','Benefits','Salaries']]

ddf = df.set_index(['Submission Year'])
plt.figure(figsize=(16,9))
sns.lineplot(data=ddf)




