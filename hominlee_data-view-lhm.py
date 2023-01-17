# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1 = pd.read_json('/kaggle/input/clothing-fit-dataset-for-size-recommendation/renttherunway_final_data.json', lines=True)

df2 = pd.read_json('/kaggle/input/clothing-fit-dataset-for-size-recommendation/modcloth_final_data.json', lines=True)
import seaborn as sns
sns.distplot(df1['size'])
sns.distplot(np.log1p(df1['size']))
df1.loc[df1['M'] == 'October', 'M']