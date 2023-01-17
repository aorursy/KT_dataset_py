# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
kiva_loans = pd.read_csv ('../input/kiva_loans.csv')
kiva_mpi_region_locations = pd.read_csv ('../input/kiva_mpi_region_locations.csv')
loan_theme_ids = pd.read_csv ('../input/loan_theme_ids.csv')
loan_themes_by_region = pd.read_csv ('../input/loan_themes_by_region.csv')
kiva_loans.head()
kiva_loans.info()
kiva_loans.sector.value_counts()
kiva_loans.sector.unique()
kiva_loans.loan_amount.describe()
sector_list = list(kiva_loans.sector.unique())
sector_loan_amount = []

for i in sector_list:
    x = kiva_loans[kiva_loans.sector == i]
    sectorLoanAmount = sum(x.loan_amount) / len(x)
    sector_loan_amount.append(sectorLoanAmount)
    
data = pd.DataFrame({'sector_list': sector_list, 'sector_loan_amount':sector_loan_amount})
new_index = (data['sector_loan_amount'].sort_values(ascending = False)).index.values
sorted_data = data.reindex(new_index)

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['sector_list'], y=sorted_data['sector_loan_amount'])
plt.xticks(rotation= 90)
plt.xlabel('Sector List')
plt.ylabel('Loan Amount')
plt.title('SectorList-LoanAmount')

#also
sns.barplot (x = kiva_loans.sector, y = kiva_loans.loan_amount)
plt.xticks(rotation = 90)
plt.show()
country_list = list(kiva_loans.country.unique())
country_list_new = country_list[0:10]
country_list_new

term_months = []

for i in country_list_new:
    x = kiva_loans[kiva_loans.country == i]
    termMonths = sum(x.term_in_months)
    term_months.append(termMonths)
    
#sorting

data = pd.DataFrame({'country_list': country_list_new , 'term_months': term_months })
new_index = (data['term_months'].sort_values(ascending = True)).index.values
sorted_data2 = data.reindex(new_index)

#visualization

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['term_months'], y=sorted_data2['country_list'])
plt.xticks(rotation= 90)
plt.xlabel('Term in Months')
plt.ylabel('Country List')
plt.title('Term in Months - Country List')


kiva_loans.info()
country_list = list(kiva_loans.country.unique())
new_country_list = country_list[0:10]

funded_amount = []
term_in_months = []
lender_count = []

for i in new_country_list:
    x = kiva_loans[kiva_loans.country == i]
    funded_amount.append(sum(x.funded_amount)/len(x))
    term_in_months.append(sum(x.term_in_months)/1000)
    lender_count.append(sum(x.lender_count)/1000)
    
f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x = funded_amount, y = new_country_list, color = 'blue', alpha = 0.6 , label = 'fundedamount')
sns.barplot(x = term_in_months, y = new_country_list, color = 'green', alpha = 0.7 , label = 'terminmonths')
sns.barplot(x = lender_count, y = new_country_list, color = 'cyan', alpha = 0.5 , label = 'lendercount')    
    
ax.legend(loc = 'upper right', frameon = True)
ax.set(xlabel = 'Country', ylabel = 'Numeric Values', title= "Country - Numeric Values")



country_list = list(kiva_loans.country.unique())
new_country_list = country_list[0:10]

country_loan_amount = []

for i in new_country_list:
    x = kiva_loans[kiva_loans.country == i]
    country_loan_amount.append(sum(x.loan_amount)/len(x))

#sorted

data3 = pd.DataFrame({'country_list': country_list_new , 'country_amount': country_loan_amount })
new_index3 = (data3['country_amount'].sort_values(ascending = True)).index.values
sorted_data3 = data3.reindex(new_index3)    

#something like normalization

sorted_data2 ['term_months'] = sorted_data2 ['term_months'] / max ( sorted_data2 ['term_months'])
sorted_data3 ['country_amount'] = sorted_data3 ['country_amount'] / max ( sorted_data3 ['country_amount'])

data4 = pd.concat([sorted_data2,sorted_data3['country_amount']],axis=1)
data4.sort_values('country_amount', inplace = True) 

#visualization

f,ax1 = plt.subplots(figsize = (20,20))
sns.pointplot(x = 'country_list', y = 'country_amount', data = data4, color = 'red', alpha = 0.7)
sns.pointplot(x = 'country_list', y = 'term_months', data = data4, color = 'blue', alpha = 0.7)
plt.text(20,0.6,'term in months', color = 'blue', fontsize = 15, style = 'normal')
plt.text(20,0.5, 'loan amount', color ='red', fontsize = 15, style = 'normal')
plt.xlabel ('Country', fontsize = 15 , color = 'magenta')
plt.ylabel ('Values', fontsize = 15 , color = 'magenta')
plt.title('Term in Months and Loan Amount', fontsize = 20 , color = 'lime')
plt.grid()
g = sns.jointplot(data4.term_months, data4.country_amount, kind = 'kde', size = 7)
plt.savefig('graph.png')
plt.show()
g = sns.jointplot(data4.term_months, data4.country_amount,size = 5, ratio = 3)
label = kiva_loans.activity.value_counts().index
label1 = label[0:5]
sizes = kiva_loans.activity.value_counts().values
sizes1 = sizes[0:5]

colors = ['red','green','blue','magenta','lime']
explode = [0,0,0,0,0]
#visualization

plt.figure(figsize = (7,7))
plt.pie(sizes1, explode = explode , labels = label1, colors = colors)

sns.lmplot (x = "term_months", y= "country_amount", data = data4)
plt.show()
sns.kdeplot(data4.term_months,data4.country_amount, shade = True, cut = 3)
plt.show()
data4.corr()
f,ax = plt.subplots(figsize = (5,5))
sns.heatmap(data4.corr(), annot = True, linewidths = .5, fmt = '.1f', ax = ax)
plt.show()
sns.countplot(kiva_loans.repayment_interval)
plt.title('Repayment Interval')
sectors = kiva_loans.sector.value_counts()

sns.barplot(x = sectors[0:5].index , y = sectors[0:5].values)
plt.xlabel("Sectors")
plt.ylabel("Values")
different_sectors = ['agricultere' if i == 'Agriculture' else 'other' for i in kiva_loans.sector]
df_train = pd.DataFrame({'sector':different_sectors})
sns.countplot(df_train.sector)
