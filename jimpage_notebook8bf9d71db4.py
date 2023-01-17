# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
train_df.head()
type(train_df['Cabin'][0])
def cabin_letter(Cabin):

    if isinstance(Cabin,str):

        letters = ['A','B','C','D','E','F','G','T']

        for letter in letters:

            if letter in Cabin:

                return letter

train_df['cabin_letter'] = train_df['Cabin'].map(cabin_letter)
bar_df = train_df.groupby('cabin_letter').mean()[['Survived']]

plt.bar(range(len(bar_df.index)), bar_df['Survived'])

plt.xticks(range(len(bar_df.index)), bar_df.index)

pd.options.display.max_rows = 999

bar_df
cat_vars=['Pclass','Sex','SibSp','Parch','Cabin','Embarked']

#cat_vars=['Pclass']



plt.figure(1)

for num, cat_var in enumerate(cat_vars):

    print(num, cat_var)

    plt.subplot(3,2,num+1)

    bar_df = train_df.groupby(cat_var).mean()[['Survived']]

    plt.bar(range(len(bar_df.index)), bar_df['Survived'])

    plt.xticks(range(len(bar_df.index)),bar_df.index)

plt.subplots_adjust(hspace=.5)

plt.show()
plt.plot(range(0,10),[x**3 for x in range(0,10)],'bH-.',mec='r',mfc='g',linewidth=10,ms=20)

plt.xlabel('0-9')

plt.ylabel('cubes 0-9')

plt.axis([0.,8,0,600])
plt.figure()

plt.subplot(421)

plt.plot([1,2,3,4])

plt.subplot(222)

plt.plot([1,2,3,4],'r')
