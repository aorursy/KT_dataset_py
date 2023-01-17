# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

MCR=pd.read_csv('../input/multipleChoiceResponses.csv',encoding="ISO-8859-1")

FFR=pd.read_csv('../input/freeformResponses.csv')

print(MCR.info())

print(FFR.info())
print(MCR.columns)

print(MCR.isnull().sum().sort_values(ascending=False))
MCR['Age'].fillna(int(MCR['Age'].mean()),inplace=True)

print(MCR['Age'].value_counts(bins=20))
print(MCR['Country'].value_counts())
MCR['Country'].replace('Republic of China', 'China',inplace=True)

MCR['Country'].replace("People 's Republic of China", 'China',inplace=True)


#plt.hist(MCR['Age'],bins=5)

f1=plt.figure()

f1.set_size_inches(20, 10)

plt.hist([MCR['Age']], color=['r'], alpha=0.5)

f1=plt.figure()

f1.set_size_inches(20, 20)

#sns.violinplot(y='Age',data=MCR,x='GenderSelect',split=True)

#plt.legend(loc=7)

g = sns.factorplot(x="Country", y="Age",row="GenderSelect",data=MCR, kind="violin", size=20, aspect=0.8,row_order=['Male','Female'])

g.set(ylim=(10,50))





f1=plt.figure()

f1.set_size_inches(20, 10)

#sns.countplot('Country',data=MCR)

MCR['Country'].value_counts().plot(kind='bar',color=['g'],alpha=0.6)

plt.xticks( rotation='vertical')


f1=plt.figure()

f1.set_size_inches(10, 10)

sns.countplot(y='Country',data=MCR,hue='GenderSelect')

plt.legend(loc=7)


g = sns.factorplot(y="Country", x="Age",col='GenderSelect',data=MCR,kind="bar",size=10,aspect=1,col_wrap=2,ci=None)

#sns.violinplot(y='Age',x='GenderSelect',data=MCR)

#plt.xticks(fontsize=20)                    

#plt.yticks(fontsize=14)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

sns.countplot(x="CareerSwitcher",data=MCR,ax=ax1)

sns.countplot(x="EmploymentStatus",data=MCR,ax=ax2)

f.set_size_inches(20, 10)

plt.xticks( rotation='vertical')
f1=plt.figure()

f1.set_size_inches(8, 8)

sns.countplot(x="EmploymentStatus", hue="CareerSwitcher", data=MCR)

plt.xticks( rotation='vertical')
f1=plt.figure()

f1.set_size_inches(15, 10)

ax = sns.countplot(x="CurrentJobTitleSelect", hue="EmploymentStatus", data=MCR)

plt.xticks( rotation='vertical')

#print(MCR.groupby('EmploymentStatus',as_index=False)['CareerSwitcher'].sum())
f1=plt.figure()

f1.set_size_inches(8, 8)

sns.countplot( x="LearningDataScience", hue='EmploymentStatus',data=MCR)

plt.xticks( rotation='vertical')
f1=plt.figure()

f1.set_size_inches(8, 8)

sns.countplot(x="Country", hue="CareerSwitcher", data=MCR)

plt.xticks( rotation='vertical')
#sns.countplot('CareerSwitcher',data=MCR)

f1=plt.figure()

f1.set_size_inches(8,20)

Job_colors=['#78C850','#F08030', '#6890F0', '#9b59b6','#95a5a6','#2ecc71','#F8D030','#E0C068', '#EE99AC',

                    '#C03028','#F85888','#B8A038','#705898', '#98D8D8', '#34495e' ]

sns.countplot(y="MLToolNextYearSelect", hue="CurrentJobTitleSelect", data=MCR,palette=Job_colors, linewidth=5,saturation=1)

plt.xticks( rotation='vertical')

plt.legend(loc=7)


