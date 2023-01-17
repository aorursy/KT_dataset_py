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
df = pd.read_csv('../input/en.openfoodfacts.org.products.tsv',sep='\t')
df.sample()
df.columns.values
df.shape
df.loc[(df['categories'].notnull()) & (df['product_name'].notnull()),['product_name', 'generic_name', 'categories', 'categories_en', 'labels', 'labels_en']]
