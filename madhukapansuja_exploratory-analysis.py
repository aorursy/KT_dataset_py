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
train = pd.read_csv("/kaggle/input/st4035-2020-inclass-1/train_data.csv")
test = pd.read_csv("/kaggle/input/st4035-2020-inclass-1/test_data.csv")
train = train.drop(columns=['ID'])
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

import seaborn as sns 
sns.set(color_codes=True)
print(train.describe())
# Create the default pairplot
sns.pairplot(train)

plt.figure(figsize=(10,5))
c= train.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c
train['X1L'] = np.log(train['X1'])
train['X1S'] = np.square(train['X1'])
train['X1P'] = np.power(train['X1'], 3)
# Plot Histogram on x
x = train['X1']
plt.hist(x, bins=20)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
# Plot Histogram on x
x = train['X1S']
plt.hist(x, bins=20)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
train['X2L'] = np.log(train['X2'])
train['X2S'] = np.square(train['X2'])
train['X2P'] = np.power(train['X2'], 3)
# Plot Histogram on x
x = train['X2']
plt.hist(x, bins=20)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
# Plot Histogram on x
x = train['X2P']
plt.hist(x, bins=20)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
# Plot Histogram on x
x = train['X3']
plt.hist(x, bins=20)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
train['X4L'] = np.log(train['X4'])
train['X4S'] = np.square(train['X4'])
train['X4P'] = np.power(train['X4'], 4)
# Plot Histogram on x
x = train['X4']
plt.hist(x, bins=20)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
# Plot Histogram on x
x = train['X4P']
plt.hist(x, bins=20)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
