# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
data.head()
data.info()
data.describe()
data.head()
data.drop(columns="writing score")
data.info()
def bar_plot(variable):
    """
    input: variable = ex:'gender'
    output: bar plot & value count
    """
    # get feature
    var = data[variable]
    # count number of categorical variable(value)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Fre")
    plt.title("Variable")
    plt.show()
    
    print("{}: \n {}".format(variable,varValue))
category1 = ["gender","race/ethnicity","parental level of education","lunch","test preparation course"]
for i in category1:
    bar_plot(i)
def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(data[variable],bins=50)
    plt.xlabel(variable)
    plt.ylabel("Fre")
    plt.title("{} distribution with hist".format(variable))
numericVar = ["math score","writing score","reading score"]
for i in numericVar:
    plot_hist(i)
# race/ethnicity vs math score

data[["race/ethnicity","math score"]].groupby(["race/ethnicity"],as_index=False).mean().sort_values(by="math score",ascending=False)
# gender vs math score

data[["gender","math score"]].groupby(["gender"],as_index=False).mean().sort_values(by="math score",ascending=False)
# parental level of education vs math score

data[["parental level of education","math score"]].groupby(["parental level of education"],as_index=False).mean().sort_values(by="math score",ascending=False)
# test preparation course vs math score

data[["test preparation course","math score"]].groupby(["test preparation course"],as_index=False).mean().sort_values(by="math score",ascending=False)
data['test preparation course'].replace(["completed"],1,inplace=True)
data['test preparation course'].replace(["none"],0,inplace=True)

data = data.rename(columns={'math score':'math_score'})
data['math_score'] = data['math_score'].astype(float)

area_list = list(data['race/ethnicity'].unique())

math_score_new = []

for i in area_list:
    x = data[data['race/ethnicity'] == i]
    math_score_rate = sum(x.math_score)/len(x)
    math_score_new.append(math_score_rate)

data_new = pd.DataFrame({'area_list': area_list,'math_score_ratio': math_score_new})
new_index = (data_new['math_score_ratio'].sort_values(ascending=True)).index.values
sorted_data2 = data_new.reindex(new_index)

plt.figure(figsize = (15,10))
sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['math_score_ratio'])
plt.xticks(rotation=90)
plt.xlabel('Groups', fontsize=15)
plt.ylabel('Math Scores', fontsize=15)
plt.title('Groups and Math Scores Exams',fontsize=16,style='italic')
plt.show()
data.head()
data = data.rename(columns={'reading score':'reading_score'})
data['reading_score'] = data['reading_score'].astype(float)

area_list = list(data['race/ethnicity'].unique())

reading_score_new = []

for i in area_list:
    x = data[data['race/ethnicity'] == i]
    reading_score_rate = sum(x.reading_score)/len(x)
    reading_score_new.append(reading_score_rate)

data1_new = pd.DataFrame({'area_list': area_list,'reading_score_ratio': reading_score_new})
new1_index = (data1_new['reading_score_ratio'].sort_values(ascending=True)).index.values
sorted_data1 = data1_new.reindex(new1_index)

plt.figure(figsize = (15,10))
sns.barplot(x=sorted_data1['area_list'], y=sorted_data1['reading_score_ratio'])
plt.xticks(rotation=90)
plt.xlabel('Groups', fontsize=15)
plt.ylabel('Reading Scores', fontsize=15)
plt.title('Groups and Reading Scores Exams',fontsize=16,style='italic')
plt.show()
sorted_data1['reading_score_ratio'] = sorted_data1['reading_score_ratio']/max(sorted_data1['reading_score_ratio'])
sorted_data2['math_score_ratio'] = sorted_data2['math_score_ratio']/max(sorted_data2['math_score_ratio'])

data0 = pd.concat([sorted_data1,sorted_data2['math_score_ratio']],axis=1)
data0.sort_values('math_score_ratio',inplace=True)

f, ax = plt.subplots(figsize = (20,10))
sns.pointplot(x='area_list',y='reading_score_ratio',data=data0,color='lime',alpha=0.7)
sns.pointplot(x='area_list',y='math_score_ratio',data=data0,color='red',alpha=0.7)
plt.grid()
plt.text(3.5,0.90,"reading score ratio",fontsize=15,color='lime',style='italic')
plt.text(3.5,0.89,"math score ratio",fontsize=15,color='red',style='italic')
plt.xlabel("Groups")
plt.ylabel("Values")
plt.title('Reading Score Ratio vs Math Score Ratio',fontsize=20,color='blue')
data0.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code
# joint kernel density
# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.
# If it is zero, there is no correlation between variables
# Show the joint distribution using kernel density estimation 

g = sns.jointplot(data0.reading_score_ratio,data0.math_score_ratio,kind='kde',size=7)
plt.savefig("graph.png")
plt.show()
g = sns.jointplot(data0.reading_score_ratio,data0.math_score_ratio,data=data0,size=7,ratio=5,color='r')
