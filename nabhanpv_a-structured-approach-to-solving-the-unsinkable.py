# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



DATA_DIR = '../input'



train_data = os.path.join(DATA_DIR, 'train.csv')

test_data = os.path.join(DATA_DIR, 'test.csv')



df_train = pd.read_csv(train_data)

df_test = pd.read_csv(test_data)



df_train.head()
# let's see the features available

print(df_train.columns.values)
df_train.info()
cabin_missing = (df_train['Cabin'].isnull().sum()/len(df_train)) * 100

print(f'Cabin column has {cabin_missing:.1f}% values missing')
df_train.describe()
df_train.head()
survived = df_train['Survived'] # or df_train.survived or df_train.iloc[:, 1] or df_train.loc[:, ['Survived']]

no_of_survivors = survived.value_counts().values[1]



print(f"Values in survived: {survived.unique()}")

print(f"People survived: ({no_of_survivors}/{len(survived)}) - {no_of_survivors/len(survived) * 100:.1f}%\n", "*"*40)

print(survived.describe())
pid = df_train['PassengerId'] 



print(f"No. of unique passenger id's: {pid.nunique()}")
print(f"Passenger ids ranges from: {pid.min()} to {pid.max()}")
import matplotlib.pyplot as plt



plt.scatter(pid, survived);
df_train.drop(columns='PassengerId', inplace=True)
pclass = df_train['Pclass']



print(f"Different passenger classes are: {pclass.unique()}")
import seaborn as sns





ax = sns.countplot(x=pclass, hue=survived, order=[1, 2, 3])



def plot_with_and_without_hue(plot=sns.countplot, x=[], y=[], hue=[], data=[], axes=[], figsize=(8, 4)):

    if data != []:

        if type(x) == str:

            x = data[x]

        if type(y) == str:

            y = data[y]

        if type(hue) == str:

            hue = data[hue]

    

    if not axes:

        fig, axes = plt.subplots(1, 2, figsize=figsize)

    

    plot(x=x, y=y, data=data, ax=axes[0])

    plot(x=x, y=y, hue=hue, data=data, ax=axes[0])

        

def set_legend_text(texts):

    L=plt.legend()

    L.get_texts()[0].set_text(texts[0])

    L.get_texts()[1].set_text(texts[1])



def annotate_axes(ax):

    for patch in ax.patches:

        height = patch.get_height()

        ax.annotate('{}'.format(height),

                    xy=(patch.get_x() + patch.get_width() / 2, height),

                       horizontalalignment='center')



set_legend_text(["Didn't survive", "Survived"])

annotate_axes(ax)



df_train[['Pclass', 'Survived']].groupby('Pclass').mean()
names = df_train['Name']

print(f"No.of unique names: {len(names.unique())}/{len(df_train)}")
df_train.drop(columns='Name', inplace=True)
sex = df_train['Sex']



print(f"These are the different sex's in our data: {sex.unique()}")
ax = sns.countplot(sex)

annotate_axes(ax);
ax = sns.countplot(sex, hue=survived)

annotate_axes(ax)



set_legend_text(["Didn't survive", "Survived"])
# male_survivors = len(df_train[df_train['Sex']=='Male' and df_train['Survived']==1])

# # print(f"Male passengers survived: {male_survivors/len(df_train)}")