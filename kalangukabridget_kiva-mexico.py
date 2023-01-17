import os
for dirname,_,filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname,filename))
import pandas as pd
import numpy as np
kiva = pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
kiva.head(2)
loans = kiva.groupby('country')['loan_amount'].sum().sort_values(ascending = False).reset_index()
loans
sector = kiva.groupby('sector')['loan_amount'].sum().sort_values(ascending = False).reset_index().head(10)
sector
Mexico = kiva[kiva['country'] == 'Mexico'].reset_index(drop = True)
Mexico.head(2)
Mexico.info()
Mexico.isna().sum()
Mexico_sector = Mexico.groupby('sector')['loan_amount'].sum().sort_values(ascending = False).reset_index().head(10)
Mexico_sector
def gender_lead(gender):
    gender = str(gender)
    if gender.startswith('f'):
        gender = 'female'
    else:
        gender = 'male'
    return gender
kiva['gender_lead'] = kiva['borrower_genders'].apply(gender_lead)
kiva['gender_lead'].nunique()
f = kiva['gender_lead'].value_counts()[0]
m = kiva['gender_lead'].value_counts()[1]

print('{} females ({}%) vs {} males ({}%) got loans'.format(f,round(f*100/(f+m),2),m,round(m*100/(f+m)),2))
Mexico['gender_lead'] = Mexico['borrower_genders'].apply(gender_lead)
Mexico['gender_lead'].nunique()
f = Mexico['gender_lead'].value_counts()[0]
m = Mexico['gender_lead'].value_counts()[1]

print('{} females ({}%) vs {} males ({}%) got loans'.format(f,round(f*100/(f+m),2),m,round(m*100/(f+m)),2))
Mexico_activity = Mexico.groupby('activity')['loan_amount'].sum().sort_values(ascending = False).reset_index().head(10)
Mexico_activity
Mexico_region = Mexico.groupby('region')['loan_amount'].sum().sort_values(ascending = False).reset_index().head(10)
Mexico_region
Mexico_laonPaymentMethod = Mexico.groupby('repayment_interval')['loan_amount'].sum().sort_values(ascending = False).reset_index().head(10)
Mexico_laonPaymentMethod
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
Mexico.head(2)
sector_df = Mexico.groupby('sector')['loan_amount', 'lender_count', 'funded_amount',].sum()\
         .sort_values(by = 'loan_amount', ascending = False).reset_index().head(10)

sector_df
activity_df = Mexico.groupby('activity')['loan_amount', 'lender_count', 'funded_amount'].sum()\
         .sort_values(by = 'loan_amount', ascending = False).reset_index().head(10)

activity_df
sector = sector_df['sector']
activity = activity_df['activity']
loan = sector_df['loan_amount']
fund = sector_df['funded_amount']
lender = sector_df['lender_count']
lender_1 = activity_df['lender_count']
plt.figure(figsize = (10,5))

plt.title('Loan Amount by Sector', fontsize = 20)
plt.xlabel('Sector', fontsize = 15)
plt.ylabel('Loan Amount', fontsize = 15)

plt.xticks(rotation = 75)

plt.plot(sector, loan)

plt.show()
plt.figure(figsize = (10,5))

plt.title('Loan Amount and Funded Amount by activity', fontsize = 15)
plt.xlabel('activity', fontsize = 15)
plt.ylabel('Loan Amount', fontsize = 15)

plt.xticks(rotation = 75)

plt.plot(activity, fund, c = 'k', label = 'Funded Amount')
plt.bar(activity, loan, label = 'Loan Amount')

plt.legend()

plt.show()
plt.figure(figsize = (10,5))

plt.title('Loan Amount by Sector', fontsize = 15)
plt.ylabel('Sector', fontsize = 15)
plt.xlabel('Loan Amount', fontsize = 15)

plt.xticks(rotation = 75)

sector_l = list(sector)
loan_l = list(loan)

sector_l.reverse()
loan_l.reverse()

plt.barh(sector_l, loan_l)

plt.show()
plt.figure(figsize = (10,5))

plt.title('Loan Amount vs Lender Count', fontsize = 20)
plt.xlabel('Loan Amount', fontsize = 15)
plt.ylabel('Lender Count', fontsize = 15)

colour = np.arange(len(sector))

plt.xticks(rotation = 75)

plt.scatter(loan, lender, c = colour, cmap = 'Blues', marker = 'o', s = loan/1000, edgecolor = 'k', alpha = 1.0)

plt.show()
plt.figure(figsize = (20,20))
plt.subplot(2,2,1)
plt.title('Loan Amount by Sector')

plt.xticks(rotation = 45)

plt.plot(sector, loan)


plt.subplot(2,2,2)
plt.title('Distribution of Term in Months')

plt.xticks(rotation = 45)

plt.hist(Mexico['term_in_months'], edgecolor = 'k', bins = 15)


plt.subplot(2,2,3)
plt.title('Lender Count by Sector')

plt.bar(sector, lender)


plt.subplot(2,2,4)
colours = np.arange(len(sector_df['sector']))

plt.title('Loan Amount vs Lender Count')

plt.scatter(sector, lender, c = colours, cmap = 'Blues',
            marker = 'o', edgecolor = 'k', alpha = 0.75, s = sector_df['loan_amount']/1000)


plt.savefig('plot.png')
plt.show()
Gender2 = ['Females', 'Males'] 
 
data = [57.1, 43.0] 
  
fig = plt.figure(figsize =(10, 7)) 
plt.pie(data, labels = Gender2) 
  
plt.show() 
