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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('fashion-mnist_train.csv')
df_test = pd.read_csv('new_test.csv')
df.info() # Выборки имеют тип int64
df.isnull().any().sum() # ровно 0 пропущенных значений
print(df.shape)
df.head()
X = np.array(df.loc[:, 'pixel1':])
y = np.array(df.loc[:, 'label'])
some_ten_images = df.groupby('label').first()

plt.figure(figsize=(30,10))
i = 1
for j in range(10):
    ax = plt.subplot(2, 5, i)
    pixels = np.array(some_ten_images.iloc[j]).reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    i += 1
plt.show()
sns.countplot(y)