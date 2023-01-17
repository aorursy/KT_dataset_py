# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/german-credit/german_credit_data.csv")

data.head()
data.drop("Unnamed: 0", inplace=True, axis=1)

data.head()
job_dictionary = {0:"unskilled and non-resident", 1:"unskilled and resident", 2:"skilled", 3:"higly skilled"}

data = data.replace({"Job":job_dictionary})

data.head()
data.shape
data.columns
data.info()
data.describe()
data.isnull().sum() / data.shape[0]
data.Age.hist()
data["Credit amount"].hist()
data.Duration.hist()
corr = data[["Age","Credit amount", "Duration"]].corr()

corr
cmap = sns.diverging_palette(250, 0, as_cmap=True)

sns.heatmap(corr, cmap=cmap, square=True, linewidths=.5, vmax=1, vmin=-.2)
sns.regplot(x=data["Credit amount"], y=data["Duration"],order=3, line_kws={"color":"orange"})
df_cat = data[['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account','Purpose']]
for i in df_cat.columns:

    cat_num = df_cat[i].value_counts()

    title = cat_num.name

    cat_num.name = "# of applicants"

    chart = sns.barplot(x=cat_num.index, y=cat_num)

    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

    

    chart.y ="# of applicants"

    plt.title(title)

    plt.show()
data.groupby("Sex").mean()["Credit amount"].T.plot(kind="bar")
data.groupby("Sex").mean()[["Age", "Duration"]].T.plot(kind="bar")
data.groupby("Job").mean()["Credit amount"].T.plot(kind="bar")
data.groupby("Job").mean()[["Age", "Duration"]].T.plot(kind="bar")
data.groupby("Housing").mean()["Credit amount"].T.plot(kind="bar")
data.groupby("Housing").mean()[["Age", "Duration"]].T.plot(kind="bar")
data.groupby("Saving accounts").mean()["Credit amount"].T.plot(kind="bar")
data.groupby("Saving accounts").mean()[["Age", "Duration"]].T.plot(kind="bar")
data.groupby("Checking account").mean()["Credit amount"].T.plot(kind="bar")
data.groupby("Checking account").mean()[["Age", "Duration"]].T.plot(kind="bar")
data.groupby("Purpose",sort=True).mean()["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Housing","Purpose"], sort=True).count().loc["free"]["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Housing","Purpose"], sort=True).count().loc["own"]["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Housing","Purpose"], sort=True).count().loc["rent"]["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Saving accounts","Purpose"], sort=True).mean().loc["little"]["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Saving accounts","Purpose"], sort=True).mean().loc["moderate"]["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Saving accounts","Purpose"], sort=True).mean().loc["rich"]["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Saving accounts","Purpose"], sort=True).mean().loc["quite rich"]["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Sex","Purpose"], sort=True).mean().loc["male"]["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Sex","Purpose"], sort=True).mean().loc["female"]["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Job","Purpose"], sort=True).mean().loc["skilled"]["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Job","Purpose"], sort=True).mean().loc["higly skilled"]["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Job","Purpose"], sort=True).mean().loc["unskilled and resident"]["Credit amount"].T.sort_values().plot(kind="barh")
data.groupby(["Job","Purpose"], sort=True).mean().loc["unskilled and non-resident"]["Credit amount"].T.sort_values().plot(kind="barh")