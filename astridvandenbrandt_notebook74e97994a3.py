# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set_style('whitegrid')



titanic_df = pd.read_csv('../input/train.csv')

titanic_df.info()

titanic_df.head()
g = sns.factorplot(x="Survived", hue="Sex", kind="count",

                   data=titanic_df,size=6,palette="muted")

g.set_ylabels("Number of Passengers")
g = sns.factorplot(x="SibSp", y="Survived", data=titanic_df,

                   size=8, kind="bar")

g.set_ylabels("survival probability")
g = sns.factorplot(x="SibSp", y="Survived", hue='Pclass',data=titanic_df,

                   size=8, kind="bar")

g.set_ylabels("survival probability")
g = sns.factorplot(x="Parch", y="Survived", data=titanic_df,

                   size=8, kind="bar")

g.set_ylabels("survival probability")
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

g = sns.factorplot(x="Embarked", y="Survived", hue='SibSp',data=titanic_df,

                   size=8, kind="bar")

g.set_ylabels("survival probability")
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

g = sns.factorplot(x="Embarked", y="Survived", hue='Parch',data=titanic_df,

                   size=8, kind="bar")

g.set_ylabels("survival probability")
g = sns.factorplot(x="Pclass", y="Survived", hue='Parch',data=titanic_df,

                   size=8, kind="bar")

g.set_ylabels("survival probability")
g = sns.factorplot(x="Pclass", y="Survived", hue='SibSp',data=titanic_df,

                   size=8, kind="bar")

g.set_ylabels("survival probability")
g = sns.factorplot(x="Pclass", y="Survived", data=titanic_df,

                   size=8, kind="bar")

g.set_ylabels("survival probability")
g = sns.factorplot(x="Survived", hue="Age", kind="count",

                   data=titanic_df,size=6,palette="muted")

g.set_ylabels("Number of Passengers")