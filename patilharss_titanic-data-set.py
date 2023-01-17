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

%matplotlib inline
train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train.describe()
train.info()
sns.heatmap(train.isnull(),yticklabels=False)
train.drop('Cabin',axis='columns',inplace=True)

train.head()
sns.set_style('darkgrid')
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)
def show_values_on_bars(axs, h_v="v", space=0.4):

    def _show_on_single_plot(ax):

        if h_v == "v":

            for p in ax.patches:

                _x = p.get_x() + p.get_width() / 2

                _y = p.get_y() + p.get_height()

                value = int(p.get_height())

                ax.text(_x, _y, value, ha="center") 

        elif h_v == "h":

            for p in ax.patches:

                _x = p.get_x() + p.get_width() + float(space)

                _y = p.get_y() + p.get_height()

                value = int(p.get_width())

                ax.text(_x, _y, value, ha="left")



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)
fig=plt.figure(figsize=(20,9))

axes=fig.add_axes([0,0,1,1])

x=sns.countplot(x='Age',data=train.dropna(),order=train['Age'].value_counts().index)

show_values_on_bars(axes,"v",0.3)

x.set_xticklabels(x.get_xticklabels(),rotation=90)

plt.show()

sns.countplot(x='SibSp',hue='Pclass',data=train)
train['Fare'].hist(bins=50,figsize=(16,9))
import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist',bins=30)