# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/the-insurance-company-tic-benchmark/tic_2000_train_data.csv', delimiter=',')
target =  pd.read_csv('/kaggle/input/the-insurance-company-tic-benchmark/tic_2000_target_data.csv', delimiter=',')
nRow, nCol = df.shape
tRow, tCol = target.shape
print(f'Train: There are {nRow} rows and {nCol} columns')
print(f'Test: There are {tRow} rows and {tCol} columns')
df.head()
df['target'] = df.CARAVAN
df['home_ownserhip'] = np.where(df.MHKOOP == 1, 'Own', np.where(df.MHHUUR == 1, 'Rent', 'Oth'))
df['home_ownserhip'].value_counts()

sns.lmplot("MKOOPKLA", "target", df, x_bins=4, logistic=True, truncate=True);
sns.set_style("dark")

fg = sns.FacetGrid(df, hue="target", aspect=3)
fg.map(sns.kdeplot, "MHKOOP", shade=True)