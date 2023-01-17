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
#df = pd.read_csv("/kaggle/input/kensho-derived-wikimedia-data/item.csv")
#df.shape
#import re

#pat = re.compile("politic.*position.*", re.I)

#idx = df.en_label.apply(lambda x: True if type(x) is str and pat.match(x) else False)
#df.loc[idx,:]
print("""
item_id \t en_label \t\t\t\t en_description
7225059 \t Political positions of Barack Obama \t NaN
""")
#del df
df = pd.read_csv("/kaggle/input/kensho-derived-wikimedia-data/statements.csv")
df.shape
df.loc[df.source_item_id.apply(lambda x: True if x == 7225059 else False),:]
df.dtypes
