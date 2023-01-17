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


data = pd.read_csv('../input/pdb_data_no_dups.csv')
data_seq = pd.read_csv('../input/pdb_data_seq.csv')
print(data.shape) 
data.head()
print(data_seq.shape)
data_seq.head()
data.describe(include='all').T
data_seq.describe(include='all').T
data.columns
year_df = data.groupby(['publicationYear']).count()['structureId'].reset_index()
year_df = year_df[year_df['publicationYear']!=2017]
plt.figure(figsize=(10,7))
plt.plot(year_df['publicationYear'], year_df['structureId'])


plt.xlabel('Proteins entered into PDB')
plt.ylabel('Publication Year')
plt.title('Growth of the PDB over time')
plt.show()


