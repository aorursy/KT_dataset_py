import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))
base_data = pd.read_csv('../input/ru-elections-2018/uiks-utf8.csv')
base_data['Kalpi'] = 1
add_data = pd.read_csv('../input/additional-data/AddData_Income_and_violations.csv')
data = base_data.merge(add_data,left_on='region_name',right_on='region_name')
data.tail(2)
for y in range (25,32):
    data[data.columns[y]] = ([pd.to_numeric(i.replace(",", "").replace(" ", "").replace("-",""))for i in data[data.columns[y]]])
registered_to_vote = np.sum(data['registered_voters'])
valid_votes = np.sum(data['valid_ballots'])
print('Total registered to vote: ', format(registered_to_vote,',d'))
print('Valid counted votes: {:,.0f} wich is {:.2%}. (According to official results: 67.5%)'.format(valid_votes,valid_votes/registered_to_vote))
print('Number of regions: ',len(set(data['region_name'])))
regions_sum = data.groupby(['English_region_name','region_name'])[['Kalpi','registered_voters','valid_ballots']].sum()
regions_sum['Votes_per'] = (regions_sum['valid_ballots']/regions_sum['registered_voters'])*100
regions_sum['Region_weight'] = (regions_sum['valid_ballots']/valid_votes)*100
print('5 top regions by votes percentage:')
regions_sum.sort_values('Votes_per',ascending=False).head()
print('5 bottom regions by votes percentage:')
regions_sum.sort_values('Votes_per',ascending=False).tail()
fig = plt.figure()
fig.suptitle('Votes Percentage \n')
ax1 = fig.add_subplot(121)
ax1.hist(regions_sum['Votes_per'], bins=10, color='blue',alpha = 0.7)
ax1.axvline((valid_votes/registered_to_vote)*100, color='k', linestyle='dashed', linewidth=1)
ax1.set(xlabel="participate percentage", ylabel="number of regions")
ax1.set_title('Histogram')
ax2 = fig.add_subplot(122)
ax2.hist(regions_sum['Votes_per'], bins=10, color='blue',alpha = 0.7,cumulative=True)
ax2.axvline((valid_votes/registered_to_vote)*100, color='k', linestyle='dashed', linewidth=1)
ax2.set(xlabel="participate percentage")
ax2.set_title('Cumulative Histogram')

plt.show()
print('In more then half regions votes persantage was lass then avarage:')
x,y = ((regions_sum['Votes_per']<=(valid_votes/registered_to_vote)*100).value_counts())
labels = 'Less then avg', 'More then avg'
explode = ( 0, 0.1)
sizes = [x,y]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

plt.show()
print('Higher vote persantage registered in regions with lower votes weight:')
sns.regplot(x=regions_sum['Region_weight'], y=regions_sum['Votes_per'])
candidates = ['baburin','grudinin','zhirinovsky','putin','sobchak','suraykin','titov','yavlinsky']
data[candidates].sum().sort_values(ascending = True).plot(kind = 'barh')
#Cheking of there is UIK with more counted then registered votes 
data[data['valid_ballots']>data['registered_voters']].shape
#Cheking of there is UIK with only Putin votes
data[data['valid_ballots']==data['putin']].shape
#Show examples of UIKs with only Putin votes
only_putin_voters = data[data['valid_ballots']==data['putin']]
only_putin_voters.sort_values('putin',ascending = False).head()
#Group UIK with only Putin votes by TIK
tik_sum = data.groupby(['English_region_name','region_name','tik_name'])[['Kalpi','registered_voters','valid_ballots']].sum()
GR_only_putin_voters = only_putin_voters.groupby(['English_region_name','region_name','tik_name'])['Kalpi','received_ballots','valid_ballots','putin'].sum()\
.sort_values('putin',ascending = False).rename(index = str,columns={'Kalpi':'P_CountOfUIKs','received_ballots':'P_received_ballots','valid_ballots':'P_valid_ballots','putin':'P_putin'})
GR_merge = GR_only_putin_voters.merge(tik_sum,how = 'left', left_on=['English_region_name','region_name','tik_name'],right_on=['English_region_name','region_name','tik_name'])
GR_merge['UIKs_per'] = GR_merge['P_CountOfUIKs'] / GR_merge['Kalpi']
#GR_merge.sort_values('English_region_name').head(10)
GR_merge.sort_values('UIKs_per',ascending = False).head(10)
data['others'] = (data['baburin']).fillna(0) + (data['grudinin']).fillna(0)+(data['zhirinovsky']).fillna(0)+ (data['sobchak']).fillna(0)+(data['suraykin']).fillna(0)+ (data['titov']).fillna(0)+(data['yavlinsky']).fillna(0)
Region_Gr = data.groupby(['English_region_name','region_name'])[['Kalpi','registered_voters','valid_ballots','putin','others','found_onsite_ballots','found_offsite_ballots']].sum()
Region_Gr['Votes_per'] = (Region_Gr['valid_ballots']/Region_Gr['registered_voters'])*100
Region_Gr['Region_weight'] = (Region_Gr['valid_ballots']/valid_votes)*100
Region_Gr['Putin_per'] = (Region_Gr['putin']/Region_Gr['valid_ballots']*100)
Region_Gr['Others_per'] = (Region_Gr['others']/Region_Gr['valid_ballots']*100)
Region_Gr['Offsite_per'] = (Region_Gr['found_offsite_ballots']/(Region_Gr['found_offsite_ballots']+Region_Gr['found_onsite_ballots']))*100
Region_Gr = Region_Gr.drop(columns = ['putin','others','registered_voters','valid_ballots','found_offsite_ballots','found_onsite_ballots'])
#Region_Gr.sort_values('Putin_per',ascending=True).head()

Income_Gr = data[data.region_name.str.contains("9") == False].groupby(['English_region_name','region_name'])[[' AvgRegionIncome2015 ',' AvgRegionIncome2016 ',' AvgRegionIncomePlace ',' Violations ']].max()
Income_Gr['Income_change_2015-2016'] = (Income_Gr[' AvgRegionIncome2016 ']/Income_Gr[' AvgRegionIncome2015 '] - 1)*100
#Income_Gr.sort_values(' AvgRegionIncomePlace ').tail()

Income_check = Income_Gr.merge(Region_Gr, how = 'left', left_on='English_region_name',right_on='English_region_name')
Income_check['Violations_rate'] = (Income_check[' Violations ']/Income_check['Kalpi'])*100
Income_check = Income_check.drop(columns = ' Violations ')
#Income_check.head()

corr = Income_check.drop(columns = [' AvgRegionIncome2015 ',' AvgRegionIncome2016 ','Kalpi','Others_per']).corr()
corr.style.background_gradient().set_precision(2)
fig = plt.figure(figsize=(15,3))
ax1 = fig.add_subplot(131)
ax1 = sns.regplot(x=Income_check[' AvgRegionIncomePlace '], y=Income_check['Putin_per'])
ax2 = fig.add_subplot(132)
ax2 = sns.regplot(x=Income_check['Votes_per'], y=Income_check['Putin_per'])
ax2 = fig.add_subplot(133)
ax2 = sns.regplot(x=Income_check['Violations_rate'], y=Income_check['Putin_per'])
print('Show outsiders:')
Income_check[(Income_check['Putin_per']>90) & (Income_check['Votes_per']<80)]
def leading_digit(x,dig=1):
    x = str(x)
    if float(x)>0: 
        return int(x[dig-1])
    else:
        return 0

BL_FD_list=[]
for c in candidates:
    new_c = 'FD_'+ c
    BL_FD_list.append(new_c)
    data[new_c] = data.apply(lambda row: leading_digit(row[c]),axis=1)
t = 240
fig = plt.figure(figsize=(10,3))
fig.suptitle('Benford Law by candidate \n')
for BLF in BL_FD_list: 
    t+=1
    ax = fig.add_subplot(t)
    Fst_digit = data.groupby(BLF)['Kalpi'].sum()
    Fst_digit = Fst_digit.to_frame().merge(pd.DataFrame({'digit':[1,2,3,4,5,6,7,8,9],'expected_rate':[0.301,0.176,0.125,0.097,0.079,0.067,0.058,0.051,0.046]}), left_index=True, right_on='digit')#.loc[1:9]
    Fst_digit['actual_count'] = Fst_digit['Kalpi']/sum(Fst_digit['Kalpi'])
    Fst_digit.set_index('digit')
    Fst_digit['actual_count'].plot(kind='bar',title = BLF)
    ax = Fst_digit['expected_rate'].plot(secondary_y=True)
    
    
data['first_digit_registered'] = data.apply(lambda row: leading_digit(row['registered_voters']),axis=1)

Fst_digit_reg = data.groupby('first_digit_registered')['Kalpi'].size()
Fst_digit_reg = Fst_digit_reg.to_frame().merge(pd.DataFrame({'digit':[1,2,3,4,5,6,7,8,9],'expected_rate':[0.301,0.176,0.125,0.097,0.079,0.067,0.058,0.051,0.046]}), left_index=True, right_on='digit')#.loc[1:9]
#Fst_digit_reg.set_index(np.array([1,2,3,4,5,6,7,8,9]), inplace=True)
Fst_digit_reg['actual_count'] = Fst_digit_reg['Kalpi']/sum(Fst_digit_reg['Kalpi'])
Fst_digit_reg['actual_count'].plot(kind='bar',title = 'Check Benford law first digit destribution of total registered votes')
Fst_digit_reg['expected_rate'].plot(secondary_y=True)