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

import seaborn as sns

import matplotlib.pyplot as plt
dataset = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
dataset.head(5)
dataset.info()
dataset.describe(include='O')
for data in [dataset]:

    data['status'] = data['status'].map({'Placed':1,'Not Placed':0}).astype(int)
dataset.groupby(['status','gender'])['gender','status'].count()
sns.set(style="darkgrid")





g = sns.FacetGrid(data=dataset, row="gender", col="status", margin_titles=True)

bins = np.linspace(0, 60, 13)

g.map(plt.hist, "status", color="brown", bins=bins)
dataset.groupby(['ssc_b','status'])['status'].count()
dataset.groupby(['hsc_b','status'])['status'].count()
# train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

dataset[['hsc_b','status']].groupby(['hsc_b']).mean().sort_values(by='status', ascending=False)
dataset.groupby(['degree_t','status'])[['status']].count()
dataset.groupby(['status','workex'])[['status']].count()
sns.set(style="darkgrid")





g = sns.FacetGrid(data=dataset, row="workex", col="status",hue='gender', margin_titles=True)

bins = np.linspace(0, 60, 13)

(g.map(plt.hist, "status",bins=bins).add_legend())
# ratio of no of female and male students

dataset[dataset['status']==1].groupby(['gender'])['gender'].count()
dataset[dataset['status']==1].groupby(['gender'])['gender'].count().plot.bar(color='purple')
dataset[dataset['status']==1].groupby(['specialisation','gender'])[['specialisation']].count()
sns.set(style="darkgrid")



g = sns.FacetGrid(data=dataset, row="specialisation", col="status",hue='gender', margin_titles=True)

bins = np.linspace(0, 60, 13)

(g.map(plt.hist, "status",bins=bins).add_legend())
data = dataset[dataset['status']==1]

data[data['specialisation']=='Mkt&HR'].groupby('gender')[['status']].count().plot.bar(color='orange')
dataset['salary'].dropna().plot(color='green')
dataset['salary'].dropna().plot.box()
dataset[['salary']].dropna().describe()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

data = dataset

data['gender'] = le.fit_transform(data['gender'])

data.tail(20)
# dataset['degree_p'].plot.hist()

sns.catplot(y="degree_p",x='status',color='lightgreen', hue="gender", kind="bar", data=dataset)
# dataset['degree_p'].plot.hist()

sns.catplot(y="degree_p",x='status', hue="specialisation", kind="bar", data=dataset)
sns.catplot(y="hsc_p",x='gender',color='yellow', hue="status", kind="bar", data=dataset)
from scipy.stats import pearsonr

corr, _ = pearsonr(dataset['degree_p'], dataset['hsc_p'])

corr