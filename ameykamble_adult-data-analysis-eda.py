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

import matplotlib 

import matplotlib.pyplot as plt

import csv
census_data = pd.read_csv('../input/inputdata/adult.csv')

census_data.head()
census_data.isnull().sum()
census_data.describe()
census_data.info()
census_data.workclass = census_data.workclass.replace({'?':'Not-Known'})

census_data.occupation = census_data.occupation.replace({'?':'Not-Known'})

census_data = census_data.rename(columns = {'education.num':'education_num'})

census_data = census_data.rename(columns ={'marital.status':'marital_status'})

census_data = census_data.rename(columns ={'capital.gain':'capital_gain'})

census_data = census_data.rename(columns ={'capital.loss':'capital_loss'})

census_data = census_data.rename(columns = {'hours.per.week':'hours_per_week'})

census_data = census_data.rename(columns ={'native.country':'native_country'})

census_data.head()
sns.countplot(x = 'sex',data = census_data)

plt.title("Gender")
census_data4 =census_data.groupby('workclass').sex.count().sort_values()

plt.title('Work - class')

census_data4.plot.bar()
census_data5 =census_data.groupby('occupation').sex.count().sort_values()

plt.title('Occupation')

census_data5.plot.bar()
census_data6 =census_data.groupby('marital_status').sex.count().sort_values()

plt.title('Marital_status')

census_data6.plot.bar()
census_data7 =census_data.groupby('relationship').sex.count().sort_values()

plt.title('Relationship')

census_data7.plot.bar()
census_data8 =census_data.groupby('race').sex.count().sort_values()

plt.title('Race')

census_data8.plot.bar()
census_data9 = pd.crosstab(census_data.sex , census_data.income)

print("Following is contigency table")

census_data9
a1 = [9592,1179]

a2 = [15128,6662]



a3 = np.array([a1,a2])



from scipy import stats

stats.chi2_contingency(a3)



chi2_stat, p_val, dof, ex = stats.chi2_contingency(a3)

print("Chisquare test value is : ",chi2_stat)

print("\nDegree of freedom is : ",dof)

print("\nP-Value is : ",p_val)

print("\nExpected observation contiggency table\n")

print(ex)

x,y,z = a3[0][1]+a3[0][0],a3[1][1]+a3[1][0],a3[0][0]+a3[1][0]+a3[0][1]+a3[1][1]

print('Number of female earning less than <=50K is ',a3[0][0])

print('Number of female observation is ',a3[0][1]+a3[0][0])

print('Number of male ',a3[1][1]+a3[1][0])

print('Total observation is ',a3[0][0]+a3[1][0]+a3[0][1]+a3[1][1])

print("Value of evaluation metric is ",((x*y)/z))
census_data10 = (census_data.groupby(['sex','income']).workclass.count()/census_data.groupby(['sex']).workclass.count())*100

census_data10
census_data11 = (census_data.groupby(['sex','income','workclass']).workclass.count()/census_data.groupby(['sex','income']).workclass.count())*100

census_data11
census_data11 = (census_data.groupby(['sex','income','marital_status']).workclass.count()/census_data.groupby(['sex','income']).workclass.count())*100

census_data11