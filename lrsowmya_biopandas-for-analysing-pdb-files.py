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
%pip install biopandas
from biopandas.pdb import PandasPdb

ppdb_df =  PandasPdb().read_pdb('/kaggle/input/6nzk.pdb')

type(ppdb_df.df)
ppdb_df.df.keys()
atom_df = ppdb_df.df['ATOM']
atom_df.head()
het_df = ppdb_df.df['HETATM']
het_df.head()
import matplotlib.pyplot as plt
atom_df['b_factor'].plot(kind='hist')
plt.title('B-Factor of human coronavirus')
plt.xlabel('B-Factor')
atom_df.element_symbol.unique()
atom_df['element_symbol'].value_counts().plot(kind='bar')
plt.title('Element symbol distribution')
plt.ylabel('Count')
plt.xlabel('Element symbol')
atom_df['atom_name'].value_counts().plot(kind='bar', figsize=(10,8))
plt.title('Atom name distribution')
plt.ylabel('Count')
plt.xlabel('atom_name symbol')
ppdb_df =  PandasPdb().read_pdb('/kaggle/input/6lu7.pdb')
catom_df = ppdb_df.df['ATOM']
chtm_df = ppdb_df.df['HETATM']
catom_df.head()
catom_df['b_factor'].plot(kind='hist')
plt.title('B-Factor of COVID-19')
plt.xlabel('B-Factor')
catom_df['element_symbol'].value_counts().plot(kind='bar')
plt.title('Element symbol distribution')
plt.ylabel('Count')
plt.xlabel('Element symbol')
catom_df['atom_name'].value_counts().plot(kind='bar', figsize=(10,8))
plt.title('Atom name distribution')
plt.ylabel('Count')
plt.xlabel('atom_name symbol')