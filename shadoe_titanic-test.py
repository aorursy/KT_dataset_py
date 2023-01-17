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
import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# preview the data

titanic_df.head()
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')

sns.factorplot('Embarked','Survived',data=titanic_df,size=3,aspect=3)
#可模板化

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

#单，离散值，value——countplot，percent——factorplot

sns.countplot(x='Embarked', data=titanic_df, ax=axis1)

#双，离散值——离散值，value——countplot，percent——factorplot

sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)

# group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)