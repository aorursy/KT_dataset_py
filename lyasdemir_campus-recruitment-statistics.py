# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.columns
data.head(10)
data.info()
data.describe()
def bar_plot(variable):
    """
        input: variable ex: "Sex"
        output: bar plot & value count
    """
    # get feature
    var = data[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
category1 = ["gender","ssc_b", "hsc_b","hsc_s","degree_t", "workex", "specialisation","status"]
for c in category1:
    bar_plot(c)
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(data[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
numericVar = ["ssc_p","hsc_p","degree_p", "etest_p","mba_p", "salary","sl_no"]
for n in numericVar:
    plot_hist(n)
data1=data
data1.workex.value_counts()
a=0
for i in data1.workex:
    if i=="Yes":
        data1.workex[a]="1"
        a+=1
    elif i=="No":
        data1.workex[a]="0"
        a+=1
        
data1.workex.value_counts()
a=0
for i in data1.gender:
    if i=="M":
        data1.gender[a]="1"
        a+=1
    elif i=="F":
        data1.gender[a]="0"
        a+=1
data1.gender.value_counts()
data.workex=data.workex.astype(int)
data.workex
# gender vs workex
data1[["gender","workex"]].groupby(["gender"], as_index = False).mean().sort_values(by="workex",ascending = False)
# degree_t vs workex
data1[["degree_t","workex"]].groupby(["degree_t"], as_index = False).mean().sort_values(by="workex",ascending = False)
data2=data1[data1.degree_p>=60]
data2[["degree_t","workex"]].groupby(["degree_t"], as_index = False).mean().sort_values(by="workex",ascending = False)
# degree_p vs workex
data1[["degree_p","workex"]].groupby(["degree_p"], as_index = False).mean().sort_values(by="workex",ascending = False)
data2.head()
data2.corr()
data2['degree_t'].value_counts()
dataSci_Tech=data[data['degree_t']=='Sci&Tech']
dataSci_Tech[["workex","etest_p"]].groupby(["workex"], as_index = False).mean().sort_values(by="etest_p",ascending = False)
dataComm_Mgmt=data[data['degree_t']=='Comm&Mgmt']
dataComm_Mgmt[["workex","etest_p"]].groupby(["workex"], as_index = False).mean().sort_values(by="etest_p",ascending = False)
dataSci_Tech=data[data['degree_t']=='Others']
dataSci_Tech[["workex","etest_p"]].groupby(["workex"], as_index = False).mean().sort_values(by="etest_p",ascending = False)
dataPlaced=data[data['status']=='Placed']
dataPlaced[["workex","etest_p"]].groupby(["workex"], as_index = False).mean().sort_values(by="etest_p",ascending = False)
dataPlaced[["gender","workex"]].groupby(["gender"], as_index = False).mean().sort_values(by="workex",ascending = False)
played_sci_tech=dataPlaced[dataPlaced['degree_t']=='Sci&Tech']['degree_t'].count()
print("140 out of",played_sci_tech)
played_Comm_Mgmt=dataPlaced[dataPlaced['degree_t']=='Comm&Mgmt']['degree_t'].count()
print("140 out of",played_Comm_Mgmt)
played_others=dataPlaced[dataPlaced['degree_t']=='others']['degree_t'].count()
print("140 out of",played_others)
data.isnull().sum()
data2
data.degree_t.value_counts()
data.boxplot(column="salary",by = "degree_t")
plt.show()
data2.degree_t
data2[data2.degree_t=='Comm&Mgmt']= data2[data2.degree_t=='Comm&Mgmt'].fillna("280000")
data2[data2.degree_t=='Others']= data2[data2.degree_t=='Others'].fillna("270000")
data2[data2.degree_t=='Sci&Tech']= data2[data2.degree_t=='Sci&Tech'].fillna("290000")
data2.salary.value_counts()
data2.columns[data2.isnull().any()]
data2.isnull().sum()
data2.isnull().any()
#there is no empty value anymore.
data2
list1 = ["ssc_p", "hsc_p", "degree_p", "etest_p","mba_p"]
sns.heatmap(data2[list1].corr(), annot = True, fmt = ".2f")
plt.show()

g = sns.catplot(x="hsc_s", y="workex", hue="gender", data=data2,
                height=8, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("işe secilenler")
g = sns.catplot(x="degree_t", y="workex", hue="gender", data=data2,
                height=8, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("işe secilenler")
g = sns.catplot(x="specialisation", y="workex", hue="gender", data=data2,
                height=8, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("işe secilenler")
g = sns.catplot(x="gender", y="workex", data=data2,
                height=8, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("işe secilenler")
g = sns.FacetGrid(data, col = "gender")
g.map(sns.distplot, "degree_p", bins = 25)
plt.show()
g = sns.FacetGrid(data, col = "gender")
g.map(sns.distplot, "etest_p", bins = 25)
plt.show()
g = sns.FacetGrid(data, col = "gender")
g.map(sns.distplot, "mba_p", bins = 25)
plt.show()
g = sns.FacetGrid(data, col = "specialisation")
g.map(sns.distplot, "mba_p", bins = 25)
plt.show()
sns.factorplot(x = "gender", y = "salary", hue = "degree_t",data = data, kind = "box")
plt.show()
sns.factorplot(x = "specialisation", y = "salary", hue = "degree_t",data = data, kind = "box")
plt.show()
sns.factorplot(x = "hsc_s", y = "salary", data = data, kind = "box")
sns.factorplot(x = "degree_t", y = "salary", data = data, kind = "box")
plt.show()
data
datasalary=data[data.status=="Placed"]
datasalary
ortalama1=datasalary.ssc_p.sum()/len(datasalary.ssc_p)
ortalama2=datasalary.hsc_p.sum()/len(datasalary.hsc_p)
ortalama3=datasalary.degree_p.sum()/len(datasalary.degree_p)
ortalama4=datasalary.etest_p.sum()/len(datasalary.etest_p)
ortalama5=datasalary.mba_p.sum()/len(datasalary.mba_p)
dizi=[]
dizi=[ortalama1,ortalama2,ortalama3,ortalama4,ortalama5]
dizi
monitoring_studies = [ortalama1,ortalama2,ortalama3,ortalama4,ortalama5]
headlines = ["ssc_p","hsc_p","degree_p","etest_p","mba_p"]
colors = ['g','y',"orange","r","purple"]

plt.pie(monitoring_studies,
labels=headlines,
colors=colors,
  startangle=90,
  shadow= True,
  explode=(0,0,0,0.1,0.1),
  autopct='%1.1f%%'
    )
 
plt.title('percentiles')
plt.show()
x = ["ssc_p","hsc_p","degree_p","etest_p","mba_p"]
y = dizi

plt.bar(x,y)
plt.title("bar plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

from collections import Counter
dict(Counter(datasalary.salary).most_common(5))

sorted(datasalary.salary)[-15:]
datatop15=datasalary[datasalary.salary>=393000.0]
datatop15.info()
datatop15
def bar_plot(variable):
    """
        input: variable ex: "Sex"
        output: bar plot & value count
    """
    # get feature
    var = datatop15[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    print("{}: \n {}".format(variable,varValue))
    print(format(varValue))
    

category1 = ["ssc_b","hsc_b","hsc_s","degree_t","specialisation"]
for c in category1:
    bar_plot(c)
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(datatop15[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
numericVar = ["ssc_p","hsc_p","degree_p", "etest_p","mba_p", "salary","sl_no"]
for n in numericVar:
    plot_hist(n)
corr=datatop15.corr()
corr
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(datatop15.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()