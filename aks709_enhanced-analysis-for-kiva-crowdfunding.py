#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Akshay Sharma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec

plt.style.use('dark_background')
loans_data = pd.read_csv('../input/kiva_loans.csv')


one_million = 1000000
one_thousand = 1000
one_hundred = 100

print(loans_data.keys())
loans_data.head()
#From these attributes we can see that there are 18 features are available for the given data
#Out of these 18 features we can majorly try to look data in terms of 6 features
# sector,country,currency,partern_id,borrower-genders,repayment_interval
# So we'll show the data based on these features
#Right now i've projected data only based on 2 features other features i'll project soon

def GenderDataForSectors(sectors):
    total_count = 0
    sector_male = []
    sector_female = []
    for sector in sectors:
       
        current_sector = groupby_sectors.get_group(sector)['borrower_genders']
        sector_data = countBorrowers(current_sector)
       
        total_count = total_count + sector_data[0]
        sector_male.append(sector_data[1])
        sector_female.append(sector_data[2])
    return [total_count,sector_male,sector_female]

def countBorrowers(sector):
    item_count = 0
    male_count = 0
    female_count = 0
    for item in sector.dropna() :
        array = [s.replace(' ','') for s in item.split(',')]
        temp_array = np.array(array)
        item_count = item_count + len(temp_array)
        male_count = male_count + len(temp_array[temp_array == 'male'])
        female_count = female_count + len(temp_array[temp_array == 'female'])

    return (item_count,male_count,female_count)


#Sector-wise data visualization
groupby_sectors = loans_data.groupby('sector')
sectors = np.unique(loans_data['sector'])

fig,ax = plt.subplots(2,2,figsize=(10,9))

#Funded amount for different sectors (in Million $)
fund_amt = groupby_sectors.sum()['funded_amount']/one_million
ax[0][0].bar(sectors,fund_amt)
ax[0][0].set_title("Funded Amount (million $)")
for ticks in ax[0][0].get_xticklabels():
    ticks.set_rotation(90)

#Loan given to different sectors (in Million $)
loan_amt = groupby_sectors.sum()['loan_amount']/one_million
ax[0][1].bar(sectors,loan_amt)
ax[0][1].set_title("Loan Amount (million $)")
for ticks in ax[0][1].get_xticklabels():
    ticks.set_rotation(90)

#Loan amount based on gender of borrowers
bar_width = 0.4
x_ticks = np.arange(len(sectors))
borrower_count,male_count,female_count = GenderDataForSectors(sectors)
total_male = np.sum(male_count)
total_female = np.sum(female_count)
ax[1][0].bar(x_ticks,np.divide(male_count,one_hundred),width=bar_width,color="red",label="Male")    
ax[1][0].bar(x_ticks+bar_width,np.divide(female_count,one_hundred),width=bar_width,color="blue",label="Female")
ax[1][0].legend()

ax[1][0].set_xticks(x_ticks)
ax[1][0].set_xticklabels(sectors)

ax[1][0].set_title("Gender Distribution (in hundreds)")
for ticks in ax[1][0].get_xticklabels():
        ticks.set_rotation(90)
        
#Male and female division of borrowers
ax[1][1].set_title("Male Female distribution ")
ax[1][1].pie([total_male,total_female],labels=['male','female'],colors=['red','blue'],autopct = '%1.1f%%',shadow=True)

fig.tight_layout(h_pad=0.8)    
fig.suptitle("Projections Sector-wise")


    



#Country-wise data visualization
plt.figure(figsize=(22,15))
groupby_country = loans_data.groupby('country')
countries = np.unique(loans_data['country'])
x_ticks = np.arange(len(countries))
bar_width = 0.6
bar_space = 0.2
ax_c1 = plt.subplot2grid((3,3),(0,0),colspan=2)
ax_c2 = plt.subplot2grid((3,3),(1,0),colspan=2)
ax_c3 = plt.subplot2grid((3,3),(2,0),colspan=2)

#Total number of loans given to borrowers of each country
loan_count_country = groupby_country.count()['id']
ax_c1.bar(x_ticks,np.divide(loan_count_country,one_thousand),width = bar_width)
ax_c1.set_xticks([])
ax_c1.set_title("Number of loans given per country (in thousands)")

#Total amount of loan given to borrowers of each country
loan_amt_country = groupby_country.sum()['loan_amount']    
ax_c2.bar(countries,np.divide(loan_amt_country,one_million),width=bar_width)
ax_c2.set_xticks([])
ax_c2.set_title("Amount of loan given per country (in millions)")

#Maximum duration of loan repayment period for each country
max_loan_tenure_country = groupby_country.max()['term_in_months']
ax_c3.bar(countries,max_loan_tenure_country,width=bar_width)
ax_c3.set_title("Max Loan Tenure (in months)")


for ticks in ax_c3.get_xticklabels():
    ticks.set_fontsize(14)
    ticks.set_rotation(90)
    

plt.tight_layout()
plt.subplots_adjust(top=0.96)
plt.suptitle('Country-wise Projection')



#Based on repayment-interval

repayment_interval_grp = loans_data.groupby('repayment_interval')

#Different loan repayment interval contribution to the whole dataset
total_loans = loans_data.shape[0]
loans_payment_type = np.unique(loans_data['repayment_interval'])
loan_count_payment_type = [loans_data[loans_data['repayment_interval']==payment_type].shape[0] for payment_type in loans_payment_type]

plt.figure(figsize=(11,8))
gs = gridspec.GridSpec(4,3)
ax_r1 = plt.subplot(gs[:,:2])
ax_r1.pie(loan_count_payment_type,labels=loans_payment_type,autopct="%1.1f%%",colors=['red','blue','green','yellow'])
ax_r1.set_title('Contribution based on repayment-interval')

#Top 5 countries who had maximum irregular repayment of loan
ax_r2 = plt.subplot(gs[:2,-1])
bar_width = 0.3
irregular_payments = repayment_interval_grp.get_group('irregular')
irregular_groupby_country_count = irregular_payments.groupby('country').count()['repayment_interval']
irregular_groupby_countries = irregular_payments.groupby('country').count().index
irregular_data = pd.DataFrame({'country':irregular_groupby_countries,
                              'irregular_loans':np.array(irregular_groupby_country_count)})
irregular_data.sort_values(by='irregular_loans',inplace=True)

ax_r2.bar(irregular_data.country[-5:],irregular_data.irregular_loans[-5:]/one_thousand,width=bar_width,color=['red','orange','yellow','green','blue'])
ax_r2.set_title('Top 5 countries having irregular payment')
ax_r2.set_ylabel('No. of loans(in thousand)')

#Gender distribution for male and female based on repayment interval
ax_r3 = plt.subplot(gs[-2:,-1])
male_count_by_paymenttype = []
female_count_by_paymenttype = []
for pmt_type in loans_payment_type :
    pmt_type_data = loans_data[loans_data['repayment_interval'] == pmt_type]
    pmt_type_borrower_gender = pmt_type_data['borrower_genders']
    _,male_cnt,female_cnt =countBorrowers(pmt_type_borrower_gender)
    male_count_by_paymenttype.append(male_cnt)
    female_count_by_paymenttype.append(female_cnt)


xticks = np.arange(len(loans_payment_type))
ax_r3.bar(xticks,np.divide(male_count_by_paymenttype,one_thousand),label='male',width=bar_width,color='red')
ax_r3.bar(xticks+(bar_width),np.divide(female_count_by_paymenttype,one_thousand),label='female',width=bar_width,color='blue')
ax_r3.set_xticklabels(['','bullet','monthly','irregular','weekly'])
ax_r3.set_title('Gender dist in different repayment interval')
ax_r3.set_ylabel('No.of borrowers(in thousand)')
ax_r3.legend()
plt.tight_layout()

#Visualization based on activities
grpby_activity = loans_data.groupby('activity').sum()
grpby_activity_loan_data = pd.DataFrame({'activity':grpby_activity.index,'total_loan_amt':grpby_activity['loan_amount']})
grpby_activity_loan_data.sort_values(by='total_loan_amt',inplace=True)

#Last 10 activities for which loan has been borrowed
plt.figure(figsize=(10,5))
plt.bar(grpby_activity_loan_data.activity[:10],np.divide(grpby_activity_loan_data['total_loan_amt'][:10],one_thousand),width=bar_width,color='red')
plt.xticks(rotation=90)
plt.yticks(np.arange(0,50,5))
plt.suptitle('Bottom 10 Activities (in loan amount)')
plt.ylabel('Total Loan Amount (in thousand dollars $')

#Top 10 activities for which loan has been borrowed
plt.figure(figsize=(10,5))
plt.bar(grpby_activity_loan_data.activity[-10:],np.divide(grpby_activity_loan_data['total_loan_amt'][-10:],one_million),width=bar_width,color='green')
plt.xticks(rotation=90)
plt.suptitle('Top 10 Activities (in loan amount)')
plt.ylabel('total Loan Amount (in million dollars $)')
#plt.xticks(range(5),last_five)