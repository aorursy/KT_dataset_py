# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

import scipy.stats as stats

import warnings

warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook



iris_df = pd.read_csv('../input/iris/Iris.csv')
iris_df.info()
iris_df["Species"].value_counts()
iris_df.drop(['Id'],axis=1,inplace=True)

iris_df.head()
iris_df.describe().plot(kind = "area",fontsize=15, figsize = (20,10), table = True,colormap="coolwarm") 

plt.title("High-Level Statistics",fontsize = 20,weight='bold')

ax1 = plt.axes()

x_axis = ax1.axes.get_xaxis()

x_axis.set_visible(False)

plt.show()
colors = ['#ff9999','#66b3ff','#99ff99']

plt.figure(figsize = (8,8))

patches,_, _ = plt.pie(iris_df['Species'].value_counts(), explode=(0.01,0.01,0.01), labels=iris_df['Species'].unique(),

                       colors=colors, autopct='%1.1f%%', textprops={'fontsize': 15,'weight':'bold'})

for pie_wedge in patches:

    pie_wedge.set_edgecolor('black')

plt.title("Iris Species Count",fontsize = 20,weight='bold')

plt.show()  
#Calculate averages of columns based on species.

avg_columns=iris_df.groupby('Species',as_index=False).mean()



# melt() function unpivots a DataFrame from wide to long format.

# a specific format of the data frame object where one or more columns work as identifiers.

# all the remaining columns are treated as values.

avg_result_df = pd.melt(avg_columns,id_vars="Species",var_name="feature",value_name='average')

avg_result_df.head()
#  bar plot of averages of features by species

plt.figure(figsize=(10,8))

ax=sns.barplot(x="feature", y="average",hue="Species", data=avg_result_df,palette = "magma", ci=None)



#value of each bar

for i in ax.patches:

    # get_x pulls left or right; get_height pushes up or down

    ax.text(i.get_x()+.06, i.get_height()-.2, '{:.2f}'.format(i.get_height()), fontsize=9,

                color='white',weight='bold')



plt.xlabel("Feature Names",fontsize = 15,weight='bold')

plt.ylabel("Average (cm)",fontsize = 15,weight='bold')



plt.title("Averages of Features by Species",fontsize = 20,weight='bold')

plt.show()
# Also can be => kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }  

g = sns.jointplot(x='SepalLengthCm',y='SepalWidthCm',data=iris_df, kind='kde', color='orchid')

g.fig.set_figwidth(8)

g.fig.set_figheight(8)

g.annotate(stats.pearsonr)

g.set_axis_labels('SepalLengthCm','SepalWidthCm', fontsize=15,weight='bold')

plt.show()
g=sns.lmplot(x='SepalLengthCm',y='SepalWidthCm',data=iris_df,col='Species', hue = 'Species', palette="spring")

g.fig.set_figwidth(15)

g.fig.set_figheight(5)



axes = g.axes.flatten()

axes[0].set_ylabel("sepal width (cm)", fontsize=15,weight='bold')

for i,ax in enumerate(axes):

    ax.set_title(iris_df['Species'].unique()[i],fontsize = 20,weight='bold')

    ax.set_xlabel("sepal length (cm)", fontsize=15,weight='bold')



plt.show()
plt.figure(figsize = (15,10))

for i,value in enumerate(iris_df.columns[:-1]):

    plt.subplot(2,2,i+1)

    sns.violinplot(x='Species', y=value,data=iris_df, inner="points",palette="plasma")

    plt.xlabel('Species',fontsize = 15,weight='bold')

    plt.ylabel(value,fontsize = 15,weight='bold')



plt.show()
data = pd.melt(iris_df,id_vars="Species",var_name="feature",value_name='value')



plt.figure(figsize = (15,10))

sns.swarmplot(x="feature", y="value", hue="Species", data=data,palette="inferno")  

plt.xlabel("Features",fontsize = 15,weight='bold')

plt.ylabel("Value (cm)",fontsize = 15,weight='bold')

plt.show()
sns.pairplot(iris_df, hue='Species',palette="husl",diag_kws=dict(shade=True))

plt.show()