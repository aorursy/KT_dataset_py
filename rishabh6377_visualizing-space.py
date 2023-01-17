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
import matplotlib.pyplot as plt

import seaborn as sns
space = pd.read_csv("/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv")
space.head()
space['Company Name'].value_counts()
plt.figure(figsize=(25,10))

space['Company Name'].value_counts().plot.bar() #very easy way to create a bar

plt.xlabel("Space Agencies")

plt.ylabel("Total launches by different agencies")

for i,v in enumerate(space['Company Name'].value_counts().values): #for bar values

    plt.text(i-0.2,v+0.9,str(v)) #first two parameter are used to adjust position of text

plt.show()
plt.figure(figsize=(14,8))

plt.pie(space['Company Name'].value_counts(),radius=2,labels=space['Company Name'].value_counts().index,labeldistance=1.01,autopct="%.1f%%")

plt.show()
companies = np.array(space['Company Name'].value_counts().index).reshape(28,2) #we will create subplots of N*2 That's why i reshaped it like this
import warnings

warnings.filterwarnings('ignore')
fig,axes=plt.subplots(28,2,figsize=(20,100))

for i in range(28):

    for j in range(2):

        space[space['Company Name'] == companies[i,j]]['Status Mission'].value_counts().plot(kind="pie",ax=axes[i,j],radius=2.,autopct="%.1f%%")

        axes[i,j].set_xlabel(companies[i,j],labelpad=50,fontsize=20)

        axes[i,j].set_ylabel("")

        plt.axis("equal")

plt.tight_layout()

plt.show()
fig,axes=plt.subplots(28,2,figsize=(20,300))

for i in range(28):

    for j in range(2):

        sns.barplot(x=space[space['Company Name'] == companies[i,j]]['Location'].value_counts().index,y=space[space['Company Name'] == companies[i,j]]['Location'].value_counts().values,ax=axes[i,j])

        axes[i,j].set_xlabel("")

        axes[i,j].set_ylabel("")

        axes[i,j].set_title("Launches by {0}".format(companies[i,j]))

        axes[i,j].tick_params(axis='x',rotation=90)

fig.subplots_adjust(hspace=2)

plt.show()