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
schema = pd.read_csv('../input/survey_results_schema.csv')
public = pd.read_csv('../input/survey_results_public.csv')
schema.head()
public.head()
for col in public.columns:
    if public[col].dtypes not in ['int64','float64']:
        print(public.groupby(col).size().sort_values(ascending=False).head(20))
        print()
