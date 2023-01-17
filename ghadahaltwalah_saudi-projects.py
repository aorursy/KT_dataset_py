# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style('whitegrid')

%matplotlib inline

%config InlineBackend.figure_format ='retina'

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Saudi_projects_dataset.csv')
# inspecting dataframe head 

data.head()
# inspecting dataframe shape

data.shape
# checking null values

data.isnull().sum()
# exploring some features 



data.region_project.value_counts()
# how many unique values in each column

data.nunique()
plt.figure(figsize=(16, 6))

sns.barplot(x='sectors', y='sector_budgets', data= data , palette='winter_r');

plt.title('Sectors Budgets',fontsize=16);
# explore the precent of each status



labels = data["status_project"].unique()

sizes =  data["status_project"].value_counts().values

explode=[0.1,0,0,0,0,0,0]

parcent = 100.*sizes/ sizes.sum()

labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, parcent)]



# colors = ['yellowgreen', 'gold', 'lightblue', 'lightcoral','blue']

colors = sns.color_palette("GnBu_d", 10)

patches, texts= plt.pie(sizes, colors=colors, explode=explode, shadow=True,startangle=90)

plt.legend(patches, labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)







plt.title("Status Classification");

plt.show();
# Here we explore the relationship between status_project and sector number projects:



plt.figure(figsize = (12, 6));

sns.boxplot(x = 'status_project', y = 'sector_num_projects',  data = data, palette='winter_r' );

xt = plt.xticks(rotation=45);

plt.title('Project Status ',fontsize=16);