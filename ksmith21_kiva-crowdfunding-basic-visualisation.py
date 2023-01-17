# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read in data with Pandas
kiva_loans = pd.read_csv("../input/kiva_loans.csv")
kiva_regions = pd.read_csv("../input/kiva_mpi_region_locations.csv")
#Let's take a look at the loans dataset
kiva_loans.head(5)
kiva_loans.describe()
#Loan/fund aggregate per country
kl_loan_amount = kiva_loans.loan_amount.tolist()
kl_fund_amount = kiva_loans.funded_amount.tolist()
kl_country = kiva_loans.country.tolist()
#Find aggregates of countries
kl_unique_country = []
for i in kl_country:
    if not i in kl_unique_country:
        kl_unique_country.append(i)
print("Unique countries:", len(kl_unique_country))

kl_unique_loan=np.zeros(len(kl_unique_country))
kl_unique_funded=np.zeros(len(kl_unique_country))
for i, loan in enumerate(kl_loan_amount):
    country = kl_country[i]
    dex = kl_unique_country.index(country)
    #loan:
    kl_unique_loan[dex]=kl_unique_loan[dex] + loan
    #funding:
    fund = kl_fund_amount[i]
    kl_unique_funded[dex]=kl_unique_funded[dex] + fund

total_fund=0
total_loan=0
for i, loan in enumerate(kl_loan_amount):
    total_loan=total_loan + loan
    total_fund=total_fund + kl_fund_amount[i]
    
print("Total Loaned:", total_loan)    
print("Total Funded:", total_fund)
print("Fraction of loan amounts funded:", (total_fund/total_loan)*100, "%")

#Sort country list by loan amount list
kl_loan_country_dict = dict(zip(kl_unique_country, kl_unique_loan))
kl_fund_country_dict = dict(zip(kl_unique_country, kl_unique_funded))
kl_country_sorted = sorted(kl_loan_country_dict, key=kl_loan_country_dict.get, reverse=False)
kl_loan_sorted = []
kl_fund_sorted = []
dummy = []
for i, country in enumerate(kl_country_sorted):
    kl_loan_sorted.append(kl_loan_country_dict.get(country))
    kl_fund_sorted.append(kl_fund_country_dict.get(country))
    dummy.append(i)

#plotting
f, ax = plt.subplots(1,1, figsize=(12,24))
ax.set_title("Loans & Funding per Country w/ Labelled Loan Amounts")
ax.set_xlabel("Amount Loaned/Funded (USD)")
ax.barh(y=dummy, width=kl_loan_sorted, color='c', label='Loan')
ax.barh(y=dummy, width=kl_fund_sorted, color='b', label='Funding')
ax.set_xlim([0, max(kl_loan_sorted)*1.1]) 
plt.yticks(dummy, kl_country_sorted)
ax.legend(prop={"size" : 14})

#Bar Labels
for i, v in enumerate(kl_loan_sorted):
    ax.text(v + max(kl_loan_sorted)/100, i-0.25, str(round(float(v)/1000000,2))+ "M", color='k', fontweight='bold')
plt.show()
#Loand 
kl_sector = kiva_loans.sector.tolist()

#Organize the data
kl_sector_unique = []
kl_sector_count = []
kl_sector_loan = []
for i, sector_name in enumerate(kl_sector):
    if sector_name in kl_sector_unique:
        dex = kl_sector_unique.index(sector_name)
        kl_sector_count[dex] += 1
        kl_sector_loan[dex] += kl_loan_amount[i]
    else:
        kl_sector_unique.append(sector_name)
        kl_sector_count.append(1)
        kl_sector_loan.append(kl_loan_amount[i])


#sort the data
kl_sector_loan_dict = dict(zip(kl_sector_unique, kl_sector_loan))
kl_sector_count_dict = dict(zip(kl_sector_unique, kl_sector_count))
kl_sector_sorted = sorted(kl_sector_loan_dict, key=kl_sector_loan_dict.get, reverse=True)
kl_sector_loan_sorted = []
kl_sector_count_sorted = []
dummy_arr = []
for i, country in enumerate(kl_sector_sorted):
    kl_sector_loan_sorted.append(kl_sector_loan_dict.get(country))
    kl_sector_count_sorted.append(kl_sector_count_dict.get(country))
    dummy_arr.append(i)
        
        
#plotting      
bar_width = 0.4
bar_locations_1 = [x-bar_width/2 for x in range(len(kl_sector_unique))]
f, ax1 = plt.subplots(1,1, figsize=(18,8))
ax1.set_title("Loans per Sector")
ax1.set_xlabel("Sector")
ax1.set_ylabel("Amount Loaned/Funded (USD)")
label_1 = ax1.bar(bar_locations_1, kl_sector_loan_sorted, color='c', align='center', label='Loan Amount', width=bar_width)
plt.xticks(dummy_arr, kl_sector_sorted, rotation = '45')

bar_locations_2 = [x+bar_width/2 for x in range(len(kl_sector_unique))]
ax2 = ax1.twinx()
ax2.set_ylabel("Number of Loans")
label_2 = ax2.bar(bar_locations_2, kl_sector_count_sorted, color='b', align='center', label='Loan Number', width=bar_width)


ax2.legend((label_1[0], label_2[0]), ("Loan Amount (USD)", "Number of Loans"), prop={"size":16})
plt.show()
#Finding the total amount loaned by male vs female
kl_gender = kiva_loans.borrower_genders.tolist()

loancounts = np.zeros(4)  #0=male, 1=female, 2=NA
gender_loans=np.zeros(4)
for i, gender in enumerate(kl_gender):
#Only the first gender is counted for each submission
    if type(gender) == str:
        if gender[:5] =='male,' or gender[:7] == 'female,':
            kl_gender[i] = 'group'
            gender_loans[2] = gender_loans[0] + kl_loan_amount[i]
            loancounts[2]+=1
        elif gender[:4]=='male':
            kl_gender[i] = 'male'
            gender_loans[0] = gender_loans[0] + kl_loan_amount[i]
            loancounts[0]+=1
        else:
            kl_gender[i] = 'female'
            gender_loans[1] = gender_loans[1] + kl_loan_amount[i]
            loancounts[1]+=1
    else:
        kl_gender[i] = 'NA'
        gender_loans[3] = gender_loans[3] + kl_loan_amount[i]
        loancounts[3]+=1

bar_width = 0.4
bar_locations_1 = [x-bar_width/2 for x in [1,2,3,4]]
f1, ax1 = plt.subplots(1,1, figsize=(12,8))
ax1.set_title("Loans per Gender")
ax1.set_ylabel("Amount (USD)")
ax1.set_xlabel("Gender")
label_1 = ax1.bar(bar_locations_1, gender_loans, color='c', align='center', label='Loan Amount', width=bar_width)
ax1.set_xticks([1,2,3,4])
ax1.set_xticklabels(('Male', 'Female', 'Group', 'N/A'))

bar_locations_2 = [x+bar_width/2 for x in [1,2,3,4]]
ax2 = ax1.twinx()
ax2.set_title("Loans per Gender")
ax2.set_ylabel("Number of Loans")
ax2.set_xlabel("Gender")
label_2 = ax2.bar(bar_locations_2, loancounts, color='b', align='center', label='Loan Number', width=bar_width)
ax2.set_xticklabels(('Male', 'Female', 'Group', 'N/A'))
ax2.legend((label_1[0], label_2[0]), ("Loan Amount (USD)", "Number of Loans"), prop={"size":14})

plt.show()

# Histogram of loan amounts below $1000
nbins=21
bins=[]
for i in range(nbins):
    bins.append((i)*50)
    
f, ax = plt.subplots(1,1, figsize=(12,8))
ax.set_title("Viva Loans - Capped at $1000")
ax.set_ylabel("Frequency")
ax.set_xlabel("Loan Amount (USD)")
ax.hist(kl_loan_amount, bins, color='c', rwidth=0.90)
plt.show()

#Taking a look at the distribution of small vs large loans
nbins=51
bins=[]
for i in range(nbins):
    bins.append((i)*100)

f, ax = plt.subplots(1,1, figsize=(12,8))
ax.set_title("Viva Loans - Capped at $5000")
ax.set_ylabel("Frequency")
ax.set_xlabel("Loan Amount (USD)")
ax.hist(kl_loan_amount, bins, color='c', rwidth=0.90)
plt.show()
#plot total amounts over time
kl_loan_dates = kiva_loans.date.tolist()

kl_loan_year = []
kl_loan_month = []
kl_loan_day = []
for i in kl_loan_dates:
    kl_loan_year.append(str(i)[0:4])
    kl_loan_month.append(str(i)[5:7])
    kl_loan_day.append(str(i)[8:10])


loans_per_month = [0]
x=[0]
current_month=int(kl_loan_month[0])
j=0
for i, loan in enumerate(kl_loan_amount):
    new_month=int(kl_loan_month[i])
    if new_month!=current_month:
        x.append(j+1) #just for bar graph coordinates
        loans_per_month.append(0)
        j+=1
    loans_per_month[j]=loans_per_month[j]+loan
    current_month=new_month

dates = []
for year in ["2014", "2015", "2016", "2017"]:
    for month in ["1","2","3","4","5","6","7","8","9","10","11","12"]:
        dates.append(month + "/" + year)
dates=dates[:len(x)]    

f, ax = plt.subplots(1,1, figsize=(12,8))
ax.set_title("Viva Loan History")
ax.set_ylabel("Loan Amount (USD)")
ax.bar(x=x, height=loans_per_month, color='c', align='center', label='Loan Aggregate')
plt.xticks(x, dates, rotation = 'vertical')
plt.plot(x, loans_per_month, color='b')

#Trend Line
z = np.polyfit(x[:-1], loans_per_month[:-1], 1) #ignoring the last point
p = np.poly1d(z)
plt.plot(x, p(x), "r--", label="linear Fit")
plt.legend(prop = {"size" : 14})
print("Linear fit:", "y = %.1fx + %.1f"%(z[0],z[1]))
plt.show()
kiva_regions.head(5)
#Visualizing the MPI on a world map

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

lats = kiva_regions.lat.tolist()
lons = kiva_regions.lon.tolist()
MPI = kiva_regions.MPI.tolist()
MPI_sorted = sorted(MPI)

f, ax = plt.subplots(1, 1, figsize=(24,14))
mp = Basemap(projection='merc', \
            llcrnrlat=min(lats)-5, urcrnrlat=max(lats)+5, \
            llcrnrlon=min(lons)-5, urcrnrlon=max(lons)+5, \
            lat_ts=20, \
            resolution='c')


mp.shadedrelief(scale=0.2)
mp.drawcountries(linewidth=0.3)
mp.drawcoastlines(linewidth=0.5)

x, y = mp(lons, lats)  # transform coordinates
im = ax.scatter(x, y, s=20, c=MPI, cmap = 'magma', alpha = 0.7) 
ax.set_title("World Map with MPI Visualization", fontsize='18')

#Colour bar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.show()