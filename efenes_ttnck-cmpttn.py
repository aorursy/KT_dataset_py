# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
surv=pd.read_csv('../input/titanic/gender_submission.csv',encoding="windows-1252")

data2=pd.read_csv('../input/titanic/train.csv',encoding="windows-1252")

data3=pd.read_csv('../input/titanic/test.csv',encoding="windows-1252")
surv.head()
surv.info()
# Show the joint distribution using kernel density estimation

g = sns.jointplot(surv.PassengerId, surv.Survived, kind="kde", size=7)

plt.show()
#Correlation

f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data2.corr(), annot=True, linewidths=0.5,linecolor="yellow", fmt= '.1f',ax=ax)

plt.show()
data3.head()
f,ax = plt.subplots(figsize=(10, 10))

sns.swarmplot(x=data3.Sex, y=data3.Age,hue=data2.Survived, ax=ax)

plt.ylabel('Age',color = 'purple',fontsize=15)

plt.xlabel('Sex',color = 'purple',fontsize=15)

plt.show()
sns.countplot(data3.Sex)

plt.title("Sum_Gender",color = 'blue',fontsize=15)

plt.show()
data5 = surv.Survived

data6= data3.Sex

conc_data_row = pd.concat((data5,data6),axis =1,ignore_index =True) 

datag=conc_data_row

datag.rename(columns = {0: "Survival", 1:"Gender"}, inplace = True)

data7=datag.groupby('Gender')['Survival'].sum().reset_index()

data7


