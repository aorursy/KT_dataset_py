# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
database_df = pd.read_csv('../input/database.csv', index_col='peak_id')
database_df.rename(index=str, columns={"id": "pmid"}, inplace=True)
database_df
features_df = pd.read_csv('../input/features.csv', index_col='pmid')
features_df.describe()
features_df.mean(axis=0).sort_values(ascending=False)
