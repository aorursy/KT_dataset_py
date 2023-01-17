# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
print(train.columns)

print(test.columns)
plot_1 = train.groupby(['Sex','Survived']).size().unstack().plot.bar(stacked=True,color=['r','g'])

plot_1.set_ylabel("No. of Survivers/Deceased")

plt.show()
plot_2 = train.groupby(['Pclass','Survived']).size().unstack().plot.bar(stacked=True,color=['r','g'])

plot_2.set_ylabel("No. of Survivers/Deceased")

plt.show()
plot_2 = train.groupby(['Pclass','Sex','Survived']).size().unstack().plot.bar(stacked=True,color=['r','g'])

plot_2.set_ylabel("No. of Survivers/Deceased")

plt.show()
age_group = train.groupby([pd.cut(train["Age"], np.arange(0, 100, 10)),'Sex']).size()

age_group2 = train.groupby([pd.cut(train["Age"], np.arange(0, 100, 10))]).size()

age_group_perct = (age_group*100/age_group2)

plot_3 = age_group_perct.unstack().plot.bar(stacked=True,color=['r','g'])

plot_3.set_ylabel("% of Male/Female")

plt.show()

age_group = train.groupby([pd.cut(train["Age"], np.arange(0, 100, 10)),'Survived']).size()

age_group2 = train.groupby([pd.cut(train["Age"], np.arange(0, 100, 10))]).size()

age_group_perct = (age_group*100/age_group2)

plot_3 = age_group_perct.unstack().plot.bar(stacked=True,color=['r','g'])

plot_3.set_ylabel("% of Survivers/Deceased")

plt.show()

age_group = train.groupby([pd.cut(train["Age"], np.arange(0, 100, 10)),'Sex','Survived']).size()

age_group2 = train.groupby([pd.cut(train["Age"], np.arange(0, 100, 10))]).size()

age_group_perct = (age_group*100/age_group2)

plot_4 = age_group_perct.unstack().plot.bar(stacked=True,color=['r','g'])

plot_4.set_ylabel("% of Survivers/Deceased")

plt.show()
sibsp_group = train.groupby(['SibSp','Survived']).size()

sibsp_group2 = train.groupby(['SibSp']).size()

sibsp_plot = sibsp_group*100/sibsp_group2

plot_5 = sibsp_plot.unstack().plot.bar(stacked=True,color=['r','g'])

plot_5.set_ylabel("% of Survivers/Deceased")

plt.show()

plot_6 = sibsp_group.unstack().plot.bar(stacked=True,color=['r','g'])

plot_6.set_ylabel("No. of Survivers/Deceased")

plt.show()

Parch_group = train.groupby(['Parch','Survived']).size()

Parch_group2 = train.groupby(['Parch']).size()

Parch_plot = Parch_group*100/Parch_group2

plot_6 = Parch_plot.unstack().plot.bar(stacked=True,color=['r','g'])

plot_6.set_ylabel("% of Survivers/Deceased")

plt.show()

plot_7 = Parch_group.unstack().plot.bar(stacked=True,color=['r','g'])

plot_7.set_ylabel("Np. of Survivers/Deceased")

plt.show()