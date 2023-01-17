from IPython.display import HTML

HTML('''
<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>
''')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load 
import matplotlib.pyplot as plt                   #For graphics
import seaborn as sns #For better looking graphics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading the Train data
train = pd.read_csv('../input/train.csv')
train.head()
# Total number of unique customers
train.card_id.nunique()
# Getting the right format of date
train['first_active_month'] = pd.to_datetime(train['first_active_month'],format='%Y%m',infer_datetime_format=True)
# Checking the data type
train.info()
# Converting features to String type
train['feature_1'] = train['feature_1'].apply(lambda x:str(x))
train['feature_2'] = train['feature_2'].apply(lambda x:str(x))
train['feature_3'] = train['feature_3'].apply(lambda x:str(x))
temp_series = train[['first_active_month','card_id']].groupby('first_active_month').aggregate({'card_id':'count'}).reset_index()
plt.plot(temp_series['first_active_month'],temp_series['card_id'])
plt.xlabel(('Time'))
plt.ylabel(('Count of Customers'))
plt.title('Popularity over time for ELO products')
plt.show()
temp_series1 = train[['first_active_month','target']].groupby('first_active_month').aggregate({'target':'sum'}).reset_index()
plt.plot(temp_series1['first_active_month'],temp_series1['target'])
plt.xlabel('Time')
plt.ylabel('customers loyalty score')
plt.title('Outlook of ELO Products')
plt.show()
temp_series4 = train[['feature_3','card_id']].groupby('feature_3').aggregate({'card_id':'count'}).reset_index()
temp_series4 = temp_series4.sort_values('card_id',ascending=False)
plt.bar(temp_series4['feature_3'],temp_series4['card_id'])
plt.xlabel(('Economy Card    &     Premium Card'))
plt.ylabel(('Count of unique card id'))
plt.title('Two type in Feature 3')
plt.show()
temp_series2 = train[['feature_1','card_id']].groupby('feature_1').aggregate({'card_id':'count'}).reset_index()
temp_series2 = temp_series2.sort_values('card_id',ascending=False)
plt.bar(temp_series2['feature_1'],temp_series2['card_id'])
plt.xlabel(('Type of Feature 1'))
plt.ylabel(('count of card id'))
plt.title('Distribution of Feature 1')
plt.show()
# Where feature 3 is Economy what is feature 1
train[train['feature_3']=='1'].feature_1.unique()
## And lets check Premium card for Feature 1 labels 
train[train['feature_3']=='0'].feature_1.unique()
# Frequncy of each feature 1 in Premium Cards
train[train['feature_3']=='0'].feature_1.value_counts()
temp_series3 = train[['feature_2','card_id']].groupby('feature_2').aggregate({'card_id':'count'}).reset_index()
temp_series3 = temp_series3.sort_values('card_id',ascending=False)
plt.bar(temp_series3['feature_2'],temp_series3['card_id'])
plt.xlabel('Silver      Gold       Platinum')
plt.ylabel('Count of card id')
plt.title('Premium Card Type in Feature 1')
plt.show()
# Frequncy of each feature 2 in Economy Cards
eco_fe2 = train[train['feature_3']=='1'].feature_2.value_counts().reset_index().rename(columns={'feature_2':'Economy'})
# Frequncy of each feature 2 in Premium Cards
prem_fe2 = train[train['feature_3']=='0'].feature_2.value_counts().reset_index().rename(columns={'feature_2':'Premium'})
#merge the two
eco_prem = pd.merge(eco_fe2,prem_fe2,on='index',how='inner')
ind = eco_prem.index.tolist()  # the x locations for the groups

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

yvals = eco_prem.Economy.tolist()
rects1 = ax.bar(ind, yvals, width= 0.25, color='r')
zvals = eco_prem.Premium.tolist()
rects2 = ax.bar(ind, zvals, width=0.25, color='g')

ax.set_ylabel('Frequency')
ax.set_xlabel('Feature_2')
ax.set_xticks(ind)
ax.set_xticklabels( ('1', '2', '3') )
ax.set_title('Feature 2 type distribution in Economy and Premium Card')
ax.legend(('Economy', 'Premium') )
train['year'] = train['first_active_month'].dt.year
tarin_back = train
train = train[train['year']!=2018]
# Loyalty Card behaviour by Students, Working Professionals and Entreprenuers
work_pro = train[train['feature_1']=='1'][['year','target']].groupby('year').aggregate({'target':'sum'}).reset_index()

stud = train[train['feature_1']=='2'][['year','target']].groupby('year').aggregate({'target':'sum'}).reset_index()
entre = train[train['feature_1']=='3'][['year','target']].groupby('year').aggregate({'target':'sum'}).reset_index()
entre = entre.rename(columns={'target':'entrepreneur'})
work_pro = work_pro.rename(columns={'target':'working professional'})
stud = stud.rename(columns={'target':'student'})
entre_work = pd.merge(entre,work_pro,on='year',how='outer')
entre_work_stud = pd.merge(entre_work,stud,on='year',how='outer')
entre_work_stud.head()
entre_work_stud.set_index('year',inplace=True)
plt.figure(figsize=(15,7))
plt.xlabel('Target for New Joins by Feature 2')

ax1 = entre_work_stud['working professional'].plot(color='blue', grid=True, label='Working Professional')
ax2 = entre_work_stud['student'].plot(color='red', grid=True, secondary_y=True, label='Student')
ax3 = entre_work_stud['entrepreneur'].plot(color='green', grid=True, secondary_y=True, label='Entrepreneur')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
h3, l3 = ax3.get_legend_handles_labels()

plt.legend(h1+h2, l1+l2, loc=2)
plt.show()
