import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import matplotlib
#matplotlib.rc['font.size'] = 9.0
matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)
import seaborn as sns

%matplotlib inline
data = pd.read_csv("../input/citywide-payroll-data-fiscal-year.csv")
data.sample(10)
data = data.drop(['Last Name','First Name'], axis=1)
data['Total Pay'] = data['Regular Gross Paid'] + data['Total OT Paid']
data['Fiscal Year'] = data['Fiscal Year'].astype(str)
plt.figure(figsize=(8,5))
g = sns.FacetGrid(data, hue='Fiscal Year', size=10, hue_order=['2014',
                                                              '2015',
                                                              '2016','2017'], palette="Paired")
g.map(sns.kdeplot, "Total Pay", shade=True)
g.set_xticklabels(rotation=45)
g.add_legend()
plt.show()
data['Pay Basis'].unique()
data_per_annum = data[data['Pay Basis'].isin([' per Annum',
                                           ' Prorated Annual',
                                           'per Annum','Prorated Annual'])].drop('Pay Basis',
                                                                                axis=1)
data_per_hour = data[data['Pay Basis'].isin([' per Hour',
                                           'per Hour'])].drop('Pay Basis', axis=1)
data_per_day = data[data['Pay Basis'].isin([' per Day',
                                           'per Day'])].drop('Pay Basis', axis=1)

print ("Per Annum Basis --> ",data_per_annum.shape,
       "\nPer Day Basis -- >", data_per_day.shape,
       "\nPer Hour Basis -- >", data_per_hour.shape)
dist_pay_type = [data_per_annum.shape[0], data_per_day.shape[0], data_per_hour.shape[0]]
plt.figure(figsize=(7,7))
plt.pie(dist_pay_type, labels=['Per Annum','Per Day','Per Hour'],
                  autopct='%1.1f%%', shadow=True, startangle=90,
             colors=['#66b3ff','#ff9999','#99ff99'])
plt.title("Pay Type Basis in the city of New York")
plt.show()
def plot_high_low_pay(col, count, pay_basis):
    
    if (pay_basis=='Annum'):
        highest_paying_annum = data_per_annum.groupby(str(col))['Total Pay'].mean().reset_index().sort_values('Total Pay', ascending=False).head(count)
        lowest_paying_annum = data_per_annum.groupby(str(col))['Total Pay'].mean().reset_index().sort_values('Total Pay', ascending=True).head(count)

        f, ax = plt.subplots(2,1,figsize=(20,25))
        ax1 = sns.barplot(x='Total Pay', y=str(col), data=highest_paying_annum, 
                      orient='h', ax=ax[0])
        ax1 = sns.barplot(x='Total Pay', y=str(col), data=lowest_paying_annum, 
                      orient='h', ax=ax[1])
        ax[0].set_xlabel("Average Total Pay")
        ax[1].set_xlabel("Average Total Pay")
        plt.show()
    elif (pay_basis == 'Day'):
        highest_paying_day = data_per_day.groupby(str(col))['Total Pay'].mean().reset_index().sort_values('Total Pay', ascending=False).head(count)
        lowest_paying_day = data_per_day.groupby(str(col))['Total Pay'].mean().reset_index().sort_values('Total Pay', ascending=True).head(count)

        f, ax = plt.subplots(2,1,figsize=(20, 25))
        ax1 = sns.barplot(x='Total Pay', y=str(col), 
                          data=highest_paying_day, orient='h', ax=ax[0])
        ax1 = sns.barplot(x='Total Pay', y=str(col), 
                          data=lowest_paying_day, orient='h', ax=ax[1])
        ax[0].set_xlabel("Average Total Pay")
        ax[1].set_xlabel("Average Total Pay")
        plt.show()
    elif (pay_basis=='Hour'):
        highest_paying_hour = data_per_hour.groupby(str(col))['Total Pay'].mean().reset_index().sort_values('Total Pay', ascending=False).head(count)
        lowest_paying_hour = data_per_hour.groupby(str(col))['Total Pay'].mean().reset_index().sort_values('Total Pay', ascending=True).head(count)

        f, ax = plt.subplots(2,1,figsize=(20, 25))
        ax1 = sns.barplot(x='Total Pay', y=str(col), 
                          data=highest_paying_hour, orient='h', ax=ax[0])
        ax1 = sns.barplot(x='Total Pay', y=str(col), 
                          data=lowest_paying_hour, orient='h', ax=ax[1])
        ax[0].set_xlabel("Average Total Pay")
        ax[1].set_xlabel("Average Total Pay")
        plt.show()
plot_high_low_pay(col='Agency Name', count=10, pay_basis='Annum')
plot_high_low_pay(col='Agency Name', count=10, pay_basis='Day')
plot_high_low_pay(col='Agency Name', count=10, pay_basis='Hour')
plot_high_low_pay(col='Title Description', count=20, pay_basis='Annum')
plot_high_low_pay(col='Title Description', count=20, pay_basis='Day')
plot_high_low_pay(col='Title Description', count=20, pay_basis='Hour')
data['Work Location Borough'] = data['Work Location Borough'].str.strip().str.upper()
location_pay = data.groupby('Work Location Borough')['Total Pay'].mean().reset_index().sort_values('Total Pay',ascending=False)
sns.set_style("whitegrid")
plt.figure(figsize=(20,7))
sns.boxplot(x=data['Work Location Borough'], y=data['Total Pay'],
           data=location_pay, palette="coolwarm_r")
plt.xticks(rotation=90)
plt.show()
data['Agency Name'] = data['Agency Name'].str.strip().str.upper()
ot_ = data.groupby('Agency Name')['OT Hours'].mean().reset_index().sort_values('OT Hours',ascending=False)
ot_ = ot_.head(10)
ot_pay = data.groupby('Agency Name')['Total OT Paid'].mean().reset_index().sort_values('Total OT Paid',ascending=False)
ot_pay = ot_pay.head(10)
#sns.set_style("whitegrid")
f, ax = plt.subplots(2,1, figsize=(20,13))
sns.barplot(y=ot_['Agency Name'], x=ot_['OT Hours'],
           data=ot_, palette="BuGn_r", orient='h',ax=ax[0])
sns.barplot(x=ot_pay['Agency Name'], y=ot_pay['Total OT Paid'],
           data=ot_pay, palette="ocean_r", ax=ax[1])
ax[0].set_xlabel("Average Over Time Hours")
ax[1].set_ylabel("Average Over Time Salary")
plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=90)
plt.show()