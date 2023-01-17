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
fd = '../input/cee-498-project8-pore-in-concrete/train.xlsx'
batch1 = pd.DataFrame(pd.read_excel(fd, sheet_name = 0))
batch2 = pd.DataFrame(pd.read_excel(fd, sheet_name = 'Batch2'))
batch1
batch1['porosity(%)'].mean()
batch2
batch2['porosity(%)'].mean()
import matplotlib.pyplot as plt
batch1['porosity(%)'].hist()
batch2['porosity(%)'].hist()
plt.xlabel('Image Id')
plt.ylabel('Porosity')
plt.title('Porosity Distribution of Batches')

batch1_mean=[]
img_id=[]
for i in range(1,101):
    batch1_mean.append(batch1['porosity(%)'][0:i].mean())
    img_id.append(i)
df1=pd.DataFrame(batch1_mean)
df2=pd.DataFrame(img_id)

batch2_mean=[]
img_id=[]
for i in range(1,101):
    batch2_mean.append(batch2['porosity(%)'][0:i].mean())
    img_id.append(i)
df3=pd.DataFrame(batch2_mean)
df4=pd.DataFrame(img_id)

import matplotlib.pyplot as plt
plt.scatter(df2,df1)
plt.scatter(df4,df3)
plt.xlabel('Number of Images')
plt.ylabel('Mean Porosity')


import PIL
import glob
for f in glob.glob('../input/cee-498-project8-pore-in-concrete/batch1/batch1/*.png'):
    im1=PIL.Image.open(f)
    print(f)
im1