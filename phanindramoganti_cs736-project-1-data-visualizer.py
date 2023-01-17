# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fp15=open('/kaggle/input/cs736p1/output.4.15.0-custom.csv').readlines()

fp6=open('/kaggle/input/cs736p1/output.4.6.0-custom-nopatch.csv').readlines()

fp6patch=open('/kaggle/input/cs736p1/output.4.6.0-custom-patched.csv').readlines()

fp6config=open('/kaggle/input/cs736p1/output.4.6.0-custom.csv').readlines()

fp15 = [x.split(',') for x in fp15][2:]

fp6 = [x.split(',') for x in fp6][2:]

fp6patch = [x.split(',') for x in fp6patch][2:]

fp6config = [x.split(',') for x in fp6config][2:]

fp15vals=[float(x[1]) for x in fp15]

fp6vals=[float(x[1]) for x in fp6]

fp6patchvals=[float(x[1]) for x in fp6patch]

fp6configvals=[float(x[1]) for x in fp6config]

testLabels = [x[0][:-8].strip() for ind,x in enumerate(fp15)]

df = pd.DataFrame({"v4.15":fp15vals,"v4.6":fp6vals,"v4.6-patch":fp6patchvals,"v4.6-config":fp6configvals},index=testLabels)

fig, axes = plt.subplots(11, 4,figsize=(25,30))

for i, test_name in enumerate(testLabels):

    if i%2==0:

        row = df.iloc[i:i+2,:]

        row.plot(kind='bar', ax=axes[np.unravel_index(i//2, axes.shape)],rot=0)

        axes[np.unravel_index(i//2, axes.shape)].set(ylabel=testLabels[i])

        axes[np.unravel_index(i//2, axes.shape)].set_xticklabels(["KBEST","AVERAGE"])

        

axes[np.unravel_index(42, axes.shape)].set_visible(False)

axes[np.unravel_index(43, axes.shape)].set_visible(False)

handles, labels = plt.gca().get_legend_handles_labels()

fig.legend(handles, labels, loc='upper center')



fig.tight_layout()

plt.show()

        

    
df2 = df.sub(df.min(axis=1), axis=0).div(df.min(axis=1),axis=0)*100
df3=df2.iloc[6::2]

df4 = df3.drop(['big send','big recv','fork    Chil','thr create    Chil','big fork    Chil','huge fork','huge fork    Chil'])

df4=df4.drop(['huge write','huge read','huge mmap','huge munmap', 'huge page fault','mid page fault'])

ax = df4['v4.15'].plot(kind='bar',figsize=(20,5))

plt.legend(loc='upper right')

ax.set(ylabel="Slowdown Percentage")
ax = df4['v4.6'].plot(kind='bar',figsize=(20,5), color='orange')

plt.legend(loc='upper right')

ax.set(ylabel="Slowdown Percentage")
ax = df4['v4.6-patch'].plot(kind='bar',figsize=(20,5), color='green')

plt.legend(loc='upper right')

ax.set(ylabel="Slowdown Percentage")
ax = df4['v4.6-config'].plot(kind='bar',figsize=(20,5), color='red')

plt.legend(loc='upper right')

ax.set(ylabel="Slowdown Percentage")