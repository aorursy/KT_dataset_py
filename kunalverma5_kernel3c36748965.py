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
import pandas as pd
reviews = pd.read_csv("../input/malaria-dataset/estimated_numbers.csv", index_col=0)
reviews.head(3)
reviews['WHO Region'].value_counts().head(10).plot.bar()
(reviews['WHO Region'].value_counts().head(10) / len(reviews)).plot.bar()
reviews['No. of deaths_max'].value_counts().sort_index().plot.bar()
reviews['No. of deaths_max'].value_counts().sort_index().plot.line()
reviews['No. of deaths_max'].value_counts().sort_index().plot.area()
reviews[reviews['No. of deaths_max'] < 200]['No. of deaths_max'].plot.hist()
reviews['No. of deaths_max'].plot.hist()
reviews[reviews['No. of deaths_max'] > 1500]
reviews['No. of deaths_max'].plot.hist()