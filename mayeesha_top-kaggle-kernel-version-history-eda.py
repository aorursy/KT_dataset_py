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
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
kernels = pd.read_csv("../input/voted-kaggle-kernels.csv")
kernels.head()
kernels.describe()
kernels[kernels["Version History"].isnull() == True].head()
kernels = kernels.dropna(subset=["Version History"])
versions = kernels["Version History"].str.split("|").dropna()
print(kernels.shape)
print(len(versions))
versions[0]
kernels["number_of_versions"] = versions.map(len)
sns.heatmap(kernels[["Votes","Comments","Views","Forks","number_of_versions"]].corr(),annot=True)
kernels["number_of_versions"].describe()
plt.figure(figsize=(8,6))
sns.boxplot(y=kernels["number_of_versions"])
sns.pairplot(kernels[["Votes","Comments","Views","Forks","number_of_versions","Language"]].dropna(),kind="reg",diag_kind="kde",hue="Language");
def remove_outliers(df):
    from scipy import stats
    df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
from scipy import stats
kernels_outliers_removed = kernels[stats.zscore(kernels["number_of_versions"])<3]
sns.heatmap(kernels_outliers_removed[["Votes","Comments","Views","Forks","number_of_versions"]].corr(),annot=True)
def extract_dates(x):
    temp = [y.split(",")[1] for y in x]
    return temp
     
def extract_unique_dates(dates):
    return pd.to_datetime(dates).unique()
kernels["days_worked"] = versions.map(extract_dates)

kernels["unique_days_worked"] = kernels["days_worked"].map(extract_unique_dates)
kernels["number_unique_days_worked"] = kernels["unique_days_worked"].map(len)
kernels[["number_of_versions","days_worked","unique_days_worked","number_unique_days_worked"]].head()
sns.heatmap(kernels[["Votes","Comments","Views","Forks","number_of_versions","number_unique_days_worked"]].corr(),annot=True)
kernels["year"] = kernels["unique_days_worked"].map(lambda x:x[0].year)
kernels["month"] = kernels["unique_days_worked"].map(lambda x:x[0].month)
kernels["year"].value_counts().sort_index().plot(kind="bar",figsize=(10,6),color='darkgray')
plt.xlabel("Year")
plt.ylabel("Number of Kernels")
months = ['January','February',"March","April","May","June","July","August","September","October","November","December"]
ax = kernels["month"].value_counts().sort_index().plot(kind="bar",color=['r','r']+['darkgray']*8+['r','r'],figsize=(10,6),rot=70)
ax.set_xticklabels(months);
ax.set_xlabel("Months");
ax.set_ylabel("Number of Kernels");
ax.set_title("Kernels Published Monthly Aggregate(2015-2018)")
