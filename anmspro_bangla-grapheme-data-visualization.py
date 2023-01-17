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
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
par_file = '../input/bengaliai-cv19/train_image_data_0.parquet'
df = pd.read_parquet(par_file, engine='pyarrow')
print(df)
df.describe()
df.info()
sns.countplot(df['0'])
HEIGHT = 137
WIDTH = 236
n = 3
plt.figure(figsize=(12, 12))
#fig = plt.subplots(n, 2, figsize=(10, 5*n))

for idx in range(n):
    #somehow the original input is inverted
    img = 255 - df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)

    plt.subplot(1, n, idx+1)
    plt.imshow(img)
plt.show()
