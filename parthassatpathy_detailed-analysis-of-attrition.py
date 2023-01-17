# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Import and suppress warnings

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

data.info()
data.head()
attritionCount = data['Attrition'].value_counts()

print(attritionCount)

labels = ['No','Yes']

colors = ['gold', 'lightskyblue']

explode = (0.1,0)



plt.pie(attritionCount,explode = explode, labels = labels, colors = colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

       

plt.show()
age_yes = data['Age'].ix[data['Attrition']=='Yes'].values

age_no = data['Age'].ix[data['Attrition']=='No'].values



plt.subplot(1,2,1)

sns.distplot(age_yes)

plt.xlabel('Age')

plt.title('People leaving the Company')



plt.subplot(1,2,2)

sns.distplot(age_no)

plt.xlabel('Age')

plt.title('People staying in the Company')

plt.show()
sns.countplot(x= 'BusinessTravel',hue= 'Attrition', data= data)

plt.ylabel('Attrition')

plt.xlabel('Business Travel Done')

plt.title('Business Travels Done per Attrition labels')

plt.show()
plt.subplot(1,2,1)

sns.distplot(data.ix[data['Attrition']=='Yes']['DailyRate'].values)

plt.xlabel('DailyRate')

plt.title('People leaving the Company')

plt.subplot(1,2,2)

sns.distplot(data.ix[data['Attrition']=='No']['DailyRate'].values)

plt.xlabel('DailyRate')

plt.title('People staying in the Company')

plt.show()
sns.countplot(x= 'Department',hue= "Attrition", data= data, palette= 'Greens_d')

plt.show()
colIndex = [4,6,7,10,11,13,14,15,16,17,20,22,25,27,29,30]

for j in range(16):

    col = data.columns.values[colIndex[j]]

    #col = 'Department'

    Count_yes = (data.ix[data['Attrition']=='Yes'][col]).value_counts()

    Count = data[col].value_counts()

    for i in range(len(Count_yes)):

        Count_yes.values[i] = Count_yes.values[i] * 100/ Count.values[i]

    #plt.subplot(4,4,j+1)

    sns.barplot(x=Count_yes.index,y=Count_yes.values)

    plt.xlabel(col)

    plt.ylabel('Attrition Rate')

    plt.title('Attrition Rate of Different ' + col)

    plt.show()