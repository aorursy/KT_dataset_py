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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = '/kaggle/input/goodreadsbooks/books.csv'
df = pd.read_csv(path,error_bad_lines = False)
df.head()
df.drop(df.columns[[0,4,5]],axis=1, inplace = True)

print(df.count())
df['title'].value_counts().count()
df.sort_values('ratings_count', ascending= False, inplace=True, na_position='first')
df.head()

text_reviews = df['text_reviews_count']
label = df['title']
index = np.arange(len(text_reviews))
# plotting purpose
plt.barh(label, index)
plt.ylabel('Books')
plt.xlabel("Ratings' count")
plt.title('Text review count of the books')
plt.show()


