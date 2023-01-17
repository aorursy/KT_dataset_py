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
tips=sns.load_dataset("tips")

df=tips.copy()

df.head()
#I can see all argumans with these code, dont memorize, learn the easy way to learn

# ?sns.boxplot

sns.boxplot
sns.boxplot(x="total_bill",data=df);
sns.boxplot(x="total_bill",data=df,orient="v");
df.describe().T
#In which days we earn much

sns.boxplot(x="day",y="total_bill",data=df);
sns.boxplot(x="time",y="total_bill",data=df)
sns.boxplot(x="size",y="total_bill",data=df);
sns.boxplot(x="day",y="total_bill",hue="sex",data=df);
sns.catplot(x="day",y="total_bill",hue="sex",kind="violin", data=df);
sns.scatterplot(x="total_bill",y="tip",data=df);
sns.scatterplot(x="total_bill",y="tip",hue="sex",data=df);
sns.scatterplot(x="total_bill",y="tip",hue="time",data=df);
sns.scatterplot(x="total_bill",y="tip",hue="time",style="time",data=df);
sns.scatterplot(x="total_bill",y="tip",hue="day",style="day",data=df);
sns.scatterplot(x="total_bill",y="tip",hue="size",size="size",data=df);
sns.lmplot(x="total_bill",y="tip",data=df);
sns.lmplot(x="total_bill",y="tip",hue="smoker",data=df);
sns.lmplot(x="total_bill",y="tip",hue="smoker",col="time",row="sex",data=df);
#scatter plot

iris=sns.load_dataset("iris")

df=iris.copy()

df.head()
df.dtypes
df.shape
sns.pairplot(df);
sns.pairplot(df,hue="species");
sns.pairplot(df,hue="species",markers=["o","s","D"]);
sns.pairplot(df,hue="species",kind="reg");