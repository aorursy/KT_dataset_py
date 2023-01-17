# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
%matplotlib inline
from matplotlib.ticker import FormatStrFormatter


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.dtypes
df['Outcome'] = df['Outcome'].astype(bool)  #changing Outcome label to Boolean datatype

df.describe().T
preg_proportion = np.array(df['Pregnancies'].value_counts())
preg_month = np.array(df['Pregnancies'].value_counts().index)
preg_proportion_percentage = np.array(preg_proportion/sum(preg_proportion)*100,dtype = 'int64')
preg = pd.DataFrame({'month':preg_month,'count_of_preg_group':preg_proportion,'proportion':preg_proportion_percentage})
preg.set_index('month',inplace=True)
preg.head()
fig, axes = plt.subplots(nrows=3 , ncols=2, dpi=120,figsize =(10,6))

plot00 = sns.countplot('Pregnancies', data=df, ax=axes[0][0],color='green')
axes[0][0].set_title('count',fontsize=8)
axes[0][0].set_xlabel('Month Of pregnant', fontsize=8)
axes[0][0].set_ylabel('count',fontsize=8)
plt.tight_layout()

plot01 = sns.countplot('Pregnancies', data=df,hue='Outcome',ax=axes[0][1])
axes[0][1].set_title('dia Vs.Non-diab',fontsize=8)
axes[0][0].set_xlabel('Month Of pregnant', fontsize=8)
axes[0][0].set_ylabel('count',fontsize=8)
plt.tight_layout()

plot01 = sns.distplot(df['Pregnancies'], ax=axes[1][0])
axes[0][1].set_title('Pregnancies distribution',fontsize=8)
axes[0][0].set_xlabel('Month Of pregnant', fontsize=8)
axes[0][0].set_ylabel('frequency',fontsize=8)
plt.tight_layout()

plot11_1 = df[df["Outcome"]==False]["Pregnancies"].plot.hist(ax=axes[1][1],label='Non-diab')
plot11_2 = df[df["Outcome"]==True]["Pregnancies"].plot.hist(ax=axes[1][1],label='diab')
axes[1][1].set_title("diab Vs.Non Diab",fontsize=8)
axes[0][0].set_xlabel('Month Of pregnant', fontsize=8)
axes[0][0].set_ylabel('count',fontsize=8)
plot11_1.axes.legend(loc=1)
# plt.setp(axes[1][1].get_legend().get_texts())
# plt.setp(axes[1][1].get_legend().get_title())
plt.tight_layout()

plot20 = sns.boxplot(df['Pregnancies'],ax=axes[2][0],orient='v')
axes[2][0].set_title('pregnancies')
axes[2][0].set_xlabel('pregnancy')
axes[2][1].set_ylabel('five point summary')
plt.tight_layout()

plot21 = sns.boxplot(x='Outcome',y="Pregnancies", data=df, ax=axes[2][1])
axes[2][1].set_title('diab Vs.Non Diab')
axes[2][1].set_xlabel('pregnancy')
axes[2][1].set_ylabel('five point summary')
plt.tight_layout()
plt.show()

df.Glucose.describe()
figure, axes = plt.subplots(nrows=2,ncols=2, dpi=120,figsize=(10,8))

plot00 = sns.distplot(df["Glucose"], ax=axes[0][0], color='green')
axes[0][0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axes[0][0].set_title('distribution of glucose')
axes[0][0].set_xlabel('Glucose')
axes[0][0].set_ylabel('count')
plt.tight_layout()

plot01 = sns.distplot(df[df["Outcome"]==True]["Glucose"],ax=axes[0][1],label = 'Diabitics')
sns.distplot(df[df["Outcome"]==False]["Glucose"],ax=axes[0][1],label = 'Non Diabitics')
axes[0][1].set_title('distribution of glucose')
axes[0][1].set_xlabel('Glucose')
axes[0][1].set_ylabel('count')
plot01.axes.legend(loc=1)
plt.tight_layout()


plot10 = sns.boxplot(df["Glucose"],ax=axes[1][0],orient='v')
axes[1][0].set_title('boxplot of glucose')
axes[1][0].set_xlabel('Glucose')
axes[1][0].set_ylabel('five point summary')

plot11 = sns.boxplot(x='Outcome' , y='Glucose', data=df,ax=axes[1][1])
axes[1][1].set_title(r'Numerical Summary (Outcome)',fontdict={'fontsize':8})
axes[1][1].set_ylabel(r'Five Point Summary(Glucose)',fontdict={'fontsize':7})
plt.xticks(ticks=[0,1],labels=['Non-Diab.','Diab.'],fontsize=7)
axes[1][1].set_xlabel('Category',fontdict={'fontsize':7})
plt.tight_layout()

