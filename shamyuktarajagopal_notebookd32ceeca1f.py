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
import numpy as np # linear algebra
import pandas as pd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
netflix = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")


netflix
netflix.head()
netflix.info()
netflix.describe()
netflix.columns.values
lis = ['show_id','release_year']
print(f"{'Attribute':18} {'Mean':10} {'Median':10} {'Mode':10} {'Std':<14} {'Variance':10}")
for col in lis:
    print(f"{col:18}{round(netflix[col].mean(),3):10}{round(netflix[col].median(),3):10}{round(netflix[col].mode(),3)[0]:10}{round(netflix[col].std(),3):14}{'':3} {round(netflix[col].var(),3):10}")
sns.set_style('whitegrid')
sns.scatterplot(netflix['title'],netflix['release_year'])
sns.scatterplot(netflix['show_id'],netflix['release_year'])

sns.boxplot(netflix['release_year'],orient='v')
sns.countplot(netflix['title'])
sns.countplot(netflix['show_id'])
sns.countplot(netflix['type'])
sns.distplot(netflix['type'],color='salmon',kde=False)
sns.distplot(netflix['type'],color='salmon',kde=True)
sns.distplot(netflix['release_year'],color='salmon',kde=False)
print(f"The dataset has about {netflix.shape[0]} rows and {netflix.shape[1]} columns")

