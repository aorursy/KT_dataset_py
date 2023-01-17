import sqlite3
import pandas as pd
import statsmodels.api as sm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import scipy.stats as st
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import warnings
warnings.filterwarnings('ignore')
import sqlite3 
conn = sqlite3.connect('Northwind_small.sqlite')
cur = conn.cursor()
# joining sql tables and selecting data columns

cur.execute(""" SELECT od.ID, od.ProductID, od.UnitPrice, od.Quantity, od.Discount, o.OrderDate, e.Title, 
                r.RegionDescription, s.CompanyName, o.Freight
                FROM 'Order' o
                JOIN 'OrderDetail' od
                ON od.OrderID = o.ID
                JOIN Employee e
                ON o.EmployeeID = e.ID
                JOIN EmployeeTerritory as et
                ON e.ID = et.EmployeeID
                JOIN Territory as t
                ON et.TerritoryID = t.ID
                JOIN Region as r
                ON t.RegionID = r.ID
                JOIN Shipper s
                ON o.ShipVia = s.ID
                GROUP BY od.ID
                """)
# creating a dataframe from the sql query

df = pd.DataFrame(cur.fetchall())

# adding the dataframe columns
 
df.columns = [x[0] for x in cur.description]
# checking the data

df.head()
# checking the missing values

df.info()
# checking missing values

df.isna().sum()
#checking the top 5 values in each column

for col in df.columns:
    print(col, '\n', df[col].value_counts(normalize=True).head(), '\n\n' )
# checking number of unique values

for col in df.columns:
    print(col, '\n', len(df[col].unique()))
# Checking what are the unique discount percentages and how many records each percentage has

discounts = df['Discount'].unique()
discounts.sort()

for i in range(len(discounts)):
    
    discount = discounts[i]
    s1 = df[df['Discount']== discount]['Quantity']
    print("{}:  ".format(discount), len(s1))
# changing the 1%, 2%, 3%, 4% and 6% discounts as 5%

x = [0.01, 0.02, 0.03, 0.04, 0.06]

for i in x:
    df['Discount'][df['Discount']== i] = 0.05
discounts = df['Discount'].unique()
discounts.sort()

for i in range(len(discounts)):
    
    discount = discounts[i]
    s1 = df[df['Discount']== discount]['Quantity']
    print("{}:  ".format(discount), len(s1))
#checking outliers

df.describe()
# Creating a subset with product id 11

df1 = df[df['ProductId']==11]

df1.head()
#adding revenue column

df['Revenue'] = (df['UnitPrice'] * (1 - df['Discount'])) * df['Quantity']

# checking if the newly added column is there

df.head()
#adding season column

df['OrderDate'] = pd.to_datetime(df['OrderDate'])
df['Season'] = (df['OrderDate'].dt.month%12 + 3)//3

df['Season'][df['Season']== 1] = 'Winter'
df['Season'][df['Season']== 2] = 'Spring'
df['Season'][df['Season']== 3] = 'Summer'
df['Season'][df['Season']== 4] = 'Autumn'
# checking if the season column has been added

df.head()
#adding average freight column

df['Avg_Freight'] = (df['Freight'])/(df['Quantity'])
# checking if the average freight column has been added

df.head()
# Creating two subsets of Quantity based on whether discount is applied or not

no_discount = df[df['Discount']==0.0]['Quantity']
discount = df[df['Discount']!=0.0]['Quantity']
# checking if the two subsets add up to the total count

print(len(no_discount))
print(len(discount))
print(len(no_discount)+len(discount))
# Checking the distribution plot of no_discount

sns.distplot(no_discount)
st.normaltest(no_discount)
sns.distplot(discount)
st.normaltest(discount)
# Writing a function to take a random sample with sample size = n

def get_sample(data, n):
    sample = []
    while len(sample) != n:
        x = np.random.choice(data)
        sample.append(x)
    
    return sample
# Writing a function to calculate the sample mean

def get_sample_mean(sample):
    return sum(sample) / len(sample)
# Writing a function to create the sampling distribution

def create_sample_distribution(data, dist_size=100, n=30):
    sample_dist = []
    while len(sample_dist) != dist_size:
        sample = get_sample(data, n)
        sample_mean = get_sample_mean(sample)
        sample_dist.append(sample_mean)
    
    return sample_dist
# Creating the sampling distribution from no_discount

no_discount_sample = create_sample_distribution(no_discount, 1000, 30)
# Checking the distribution plot of no_discount's sampling distriubtion

sns.distplot(no_discount_sample)
# Calculating the p-value of no_discount's sampling distriubtion

st.normaltest(no_discount_sample)
# Creating the sampling distribution from discount

discount_sample = create_sample_distribution(discount, 1000, 30)
# Checking the distribution plot of discount's sampling distriubtion

sns.distplot(discount_sample)
# Calculating the p-value of discount's sampling distriubtion

st.normaltest(discount_sample)
# Creating test sampling distribution from no_discount 

no_discount_10 = create_sample_distribution(no_discount, 10, 30)
no_discount_100 = create_sample_distribution(no_discount, 100, 30)

# Printing the normal test p-values

print(st.normaltest(no_discount_10))
print(st.normaltest(no_discount_100))
# Creating test sampling distribution from discount

discount_10 = create_sample_distribution(discount, 10, 30)
discount_100 = create_sample_distribution(discount, 100, 30)

# Printing the normal test p-values

print(st.normaltest(discount_10))
print(st.normaltest(discount_100))
# Checking if the groups have equal variance

st.levene(no_discount_sample, discount_sample, center='mean')
# Writing a helper function to calculate the Welch's t-value

def welch_t(a, b):

    numerator = np.mean(a) - np.mean(b)
    
    denominator = np.sqrt((np.var(a, ddof=1)/len(a)) + (np.var(b, ddof=1)/len(b)))
    
    return np.abs(numerator/denominator)

# Writing a helper function to calculate the Welch's df

def welch_df(a, b):
    
    s1 = np.var(a, ddof=1) 
    s2 = np.var(b, ddof=1)
    n1 = len(a)
    n2 = len(b)
    
    numerator = (s1/n1 + s2/n2)**2
    denominator = (s1/ n1)**2/(n1 - 1) + (s2/ n2)**2/(n2 - 1)
    
    return numerator/denominator

# Writing a function to calculate the Welch's p-value

def p_value(a, b, two_sided=False):

    t = welch_t(a, b)
    df = welch_df(a, b)
    
    p = 1-st.t.cdf(np.abs(t), df)
    
    if two_sided:
        return 2*p
    else:
        return p
print("Median Values: \ts1: {} \ts2: {}".format(round(np.median(no_discount_sample), 2), round(np.median(discount_sample), 2)))
print("Mean Values: \ts1: {} \ts2: {}".format(round(np.mean(no_discount_sample), 2), round(np.mean(discount_sample), 2)))
print('Sample sizes: \ts1: {} \ts2: {}'.format(len(no_discount_sample), len(discount_sample)))
print("Welch's t-test p-value:", p_value(no_discount_sample, discount_sample, two_sided=True))
# Checking the boxplot of s1_sample to detect outliers

sns.boxplot(no_discount_sample)
# Checking the boxplot of s2_sample to detect outliers

sns.boxplot(discount_sample)
# Writing a function to remove outliers

def remove_outliers(s):
    Q1 = np.quantile(s, q=0.25)
    Q3 = np.quantile(s, q=0.75)
    IQR = Q3 - Q1
    lower_range = Q1-(1.5*IQR)
    upper_range = Q3+(1.5*IQR)
    
    s_test = []
    
    for i in s:
        if lower_range <= i <= upper_range:
            s_test.append(i)
        else:
            continue
    
    return s_test
# Removing outliers from s1_sample

no_discount_no_outliers = remove_outliers(no_discount_sample)

# Checking if the outliers have been removed

len(no_discount_no_outliers)
# Removing outliers from s2_sample

discount_no_outliers = remove_outliers(discount_sample)

# Checking if the outliers have been removed

len(discount_no_outliers)
# Reviewing the distribution plots

sns.distplot(no_discount_no_outliers)
# Reviewing the distribution plots

sns.distplot(discount_no_outliers)
# Re-running the Welch's t-test and printing out the values

print("Median Values: \ts1: {} \ts2: {}".format(round(np.median(no_discount_no_outliers), 2), round(np.median(discount_no_outliers), 2)))
print("Mean Values: \ts1: {} \ts2: {}".format(round(np.mean(no_discount_no_outliers), 2), round(np.mean(discount_no_outliers), 2)))
print('Sample sizes: \ts1: {} \ts2: {}'.format(len(no_discount_no_outliers), len(discount_no_outliers)))
print("Welch's t-test p-value:", p_value(no_discount_no_outliers, discount_no_outliers, two_sided=True))
# Dividing the data into subsets

zero = df[df['Discount']==0.0]['Quantity']
five = df[df['Discount']==0.05]['Quantity']
ten = df[df['Discount']==0.1]['Quantity']
fifteen = df[df['Discount']==0.15]['Quantity']
twenty = df[df['Discount']==0.2]['Quantity']
twenty_five = df[df['Discount']==0.25]['Quantity']
# Checking the distribution plot and normal test of each group

discounts = [zero, five, ten, fifteen, twenty, twenty_five]
discount_names = ['0%', '5%', '10%', '15%', '20%', '25%']

for i in range(len(discounts)):

    sns.distplot(discounts[i], label=discount_names[i])
    plt.legend()
    plt.show()
    print(st.normaltest(discounts[i]))
    print ("-------------------------------------------------------------------------------------")
# Creating sampling distributions

zero = create_sample_distribution(zero, 1000, 30)
five = create_sample_distribution(five, 1000, 30)
ten = create_sample_distribution(ten, 1000, 30)
fifteen = create_sample_distribution(fifteen, 1000, 30)
twenty = create_sample_distribution(twenty, 1000, 30)
twenty_five = create_sample_distribution(twenty_five, 1000, 30)

discounts = [zero, five, ten, fifteen, twenty, twenty_five]
# Reviewing the distribution plots

for i in range(len(discounts)):

    sns.distplot(discounts[i], label=discount_names[i])
    plt.legend()
    plt.show()
    print(st.normaltest(discounts[i]))
    print ("-------------------------------------------------------------------------------------")
# Confirming that there are outliers

for i in range(len(discounts)):
    
    print('The boxplot of {} distribution'.format(discount_names[i]))
    sns.boxplot(discounts[i])
    plt.show()
    print ("-------------------------------------------------------------------------------------")
# Removing outliers

zero = remove_outliers(zero)
five = remove_outliers(five)
ten = remove_outliers(ten)
fifteen = remove_outliers(fifteen)
twenty = remove_outliers(twenty)
twenty_five = remove_outliers(twenty_five)

discounts = [zero, five, ten, fifteen, twenty, twenty_five]
# Reviewing the distribution plots

for i in range(len(discounts)):

    sns.distplot(discounts[i], label=discount_names[i])
    plt.legend()
    plt.show()
    print(st.normaltest(discounts[i]))
    print ("-------------------------------------------------------------------------------------")
# checking if variances are equal

st.levene(zero, five, ten, fifteen, twenty, twenty_five, center='mean')
# performing Welch's t-test between 0% and 5%

print(p_value(zero, five))

# checking the mean values

print('0%:', np.mean(zero))
print('5%:', np.mean(five))
# performing Welch's t-test between 5% and 10%

print(p_value(five, ten))

# checking the mean values

print('5%:', np.mean(five))
print('10%:', np.mean(ten))
from scipy.stats import t

t = welch_t(five, ten)
df = welch_df(five, ten)

st.t.cdf(t, df)
# performing Welch's t-test between 10% and 15%

print(p_value(ten, fifteen))

# checking the mean values

print('10%:', np.mean(ten))
print('15%:', np.mean(fifteen))
# performing Welch's t-test between 15% and 20%

print(p_value(fifteen, twenty))

# checking the mean values

print('15%:', np.mean(fifteen))
print('20%:', np.mean(twenty))
# performing Welch's t-test between 20% and 25%

print(p_value(twenty, twenty_five))

# checking the mean values

print('20%:', np.mean(twenty))
print('25%:', np.mean(twenty_five))
# Creating subsets based on season

all_seasons = df['Revenue']
spring = df[df['Season']=='Spring']['Revenue']
summer = df[df['Season']=='Summer']['Revenue']
autumn = df[df['Season']=='Autumn']['Revenue']
winter = df[df['Season']=='Winter']['Revenue']
# checking if the total number of each season adds up to all seasons

len(all_seasons) == len(spring) + len(summer) + len(autumn) + len(winter)
# checking the distribution plots and p-values of normal test for each group

seasons = [all_seasons, spring, summer, autumn, winter]
season_names = ['All Seasons', 'Spring', 'Summer', 'Autumn', 'Winter']

for i in range(len(seasons)):

    sns.distplot(seasons[i], label=season_names[i])
    plt.legend()
    plt.show()
    print(st.normaltest(seasons[i]))
    print ("-------------------------------------------------------------------------------------")
# creating sampling distribution for each group

all_season_samples = create_sample_distribution(all_seasons, 1000, 30)
spring_sample = create_sample_distribution(spring, 1000, 30)
summer_sample = create_sample_distribution(summer, 1000, 30)
autumn_sample = create_sample_distribution(autumn, 1000, 30)
winter_sample = create_sample_distribution(winter, 1000, 30)
# checking the distribution plots and p-values again

seasons = [all_season_samples, spring_sample, summer_sample, autumn_sample, winter_sample]
season_names = ['All Seasons', 'Spring', 'Summer', 'Autumn', 'Winter']

for i in range(len(seasons)):

    sns.distplot(seasons[i], label=season_names[i])
    plt.legend()
    plt.show()
    print(st.normaltest(seasons[i]))
    print ("-------------------------------------------------------------------------------------")
# Confirming that there are outliers

for i in range(len(seasons)):
    
    print('The boxplot of {} distribution'.format(season_names[i]))
    sns.boxplot(seasons[i])
    plt.show()
    print ("-------------------------------------------------------------------------------------")
# Removing outliers

all_season_samples = remove_outliers(all_season_samples)
spring_sample = remove_outliers(spring_sample)
summer_sample = remove_outliers(summer_sample)
autumn_sample = remove_outliers(autumn_sample)
winter_sample = remove_outliers(winter_sample)

seasons = [all_season_samples, spring_sample, summer_sample, autumn_sample, winter_sample]
# Reviewing the distribution plots

for i in range(len(seasons)):

    sns.distplot(seasons[i], label=season_names[i])
    plt.legend()
    plt.show()
    print(st.normaltest(seasons[i]))
    print ("-------------------------------------------------------------------------------------")
# checking if variances are equal

st.levene(all_season_samples, spring_sample, summer_sample, autumn_sample, winter_sample, center='mean')
seasons = [spring_sample, summer_sample, autumn_sample, winter_sample]
season_names = ['Spring', 'Summer', 'Autumn', 'Winter']

for i in range(len(seasons)):
    
    print("Testing Between:{} \t{}".format('All Seasons', season_names[i]))
    print("Median Values: \ts1: {} \ts2: {}".format(round(np.median(all_season_samples), 2), round(np.median(seasons[i]), 2)))
    print("Mean Values: \ts1: {} \ts2: {}".format(round(np.mean(all_season_samples), 2), round(np.mean(seasons[i]), 2)))
    print('Sample sizes: \ts1: {} \ts2: {}'.format(len(all_season_samples), len(seasons[i])))
    print("Welch's t-test p-value:", p_value(all_season_samples, seasons[i], two_sided=True))
    print ("-------------------------------------------------------------------------------------")
# performing Welch's t-test between spring and summer

print(p_value(summer_sample, spring_sample, two_sided=True))

# checking the mean values

print('Summer:', np.mean(summer_sample))
print('Spring:', np.mean(spring_sample))
# performing Welch's t-test between spring and autumn

print(p_value(autumn_sample, spring_sample, two_sided=True))

# checking the mean values

print('Autumn:', np.mean(autumn_sample))
print('Spring:', np.mean(spring_sample))
# performing Welch's t-test between spring and winter

print(p_value(winter_sample, spring_sample, two_sided=True))

# checking the mean values

print('Winter:', np.mean(winter_sample))
print('Spring:', np.mean(spring_sample))
# performing Welch's t-test between summer and autumn

print(p_value(autumn_sample, summer_sample, two_sided=True))

# checking the mean values

print('Autumn:', np.mean(autumn_sample))
print('Summer:', np.mean(summer_sample))
# performing Welch's t-test between summer and winter

print(p_value(winter_sample, summer_sample, two_sided=True))

# checking the mean values

print('Winter:', np.mean(winter_sample))
print('Summer:', np.mean(summer_sample))
# performing Welch's t-test between autumn and winter

print(p_value(winter_sample, autumn_sample, two_sided=True))

# checking the mean values

print('Winter:', np.mean(winter_sample))
print('Autumn:', np.mean(autumn_sample))
v = np.concatenate([spring_sample, summer_sample, autumn_sample, winter_sample])
labels = ['spring'] * len(spring_sample) + ['summer'] * len(summer_sample) + ['autumn'] * len(autumn_sample) + ['winter'] * len(winter_sample)
print(pairwise_tukeyhsd(v, labels, 0.05))
# Checking the unique regions

df['RegionDescription'].unique()
# Dividing the revenue based on regions

all_regions = df['Revenue']
eastern = df[df['RegionDescription']=='Eastern']['Revenue']
western = df[df['RegionDescription']=='Western']['Revenue']
southern = df[df['RegionDescription']=='Southern']['Revenue']
northern = df[df['RegionDescription']=='Northern']['Revenue']
# Checking if all the subsets add up to the total population

len(all_regions) == len(eastern) + len(western) + len(southern) + len(northern)
regions = [all_regions, eastern, western, southern, northern]
region_names = ['All Regions', 'Eastern', 'Western', 'Southern', 'Northern']

# Checking the distribution plots

for i in range(len(regions)):

    sns.distplot(regions[i], label=region_names[i])
    plt.legend()
    plt.show()
    print(st.normaltest(regions[i]))
    print ("-------------------------------------------------------------------------------------")
# Creating sampling distributions

all_regions_sample = create_sample_distribution(all_regions, 1000, 30)
eastern_sample = create_sample_distribution(eastern, 1000, 30)
western_sample = create_sample_distribution(western, 1000, 30)
northern_sample = create_sample_distribution(northern, 1000, 30)
southern_sample = create_sample_distribution(southern, 1000, 30)
regions = [all_regions_sample, eastern_sample, western_sample, southern_sample, northern_sample]
region_names = ['All Regions', 'Eastern', 'Western', 'Southern', 'Northern']

# Reviewing the distribution of all_regions

for i in range(len(regions)):

    sns.distplot(regions[i], label=region_names[i])
    plt.legend()
    plt.show()
    print(st.normaltest(regions[i]))
    print ("-------------------------------------------------------------------------------------")
for i in range(len(regions)):
    
    print('The boxplot of {} distribution'.format(region_names[i]))
    sns.boxplot(regions[i])
    plt.show()
    print ("-------------------------------------------------------------------------------------")
# removing outliers in each group

all_regions_sample = remove_outliers(all_regions_sample)
eastern_sample = remove_outliers(eastern_sample)
western_sample = remove_outliers(western_sample)
northern_sample = remove_outliers(northern_sample)
southern_sample = remove_outliers(southern_sample)

regions = [all_regions_sample, eastern_sample, western_sample, southern_sample, northern_sample]
# Reviewing the distribution plots

for i in range(len(regions)):

    sns.distplot(regions[i], label=region_names[i])
    plt.legend()
    plt.show()
    print(st.normaltest(regions[i]))
    print ("-------------------------------------------------------------------------------------")
# checking if variances are equal

st.levene(all_regions_sample, eastern_sample, western_sample, southern_sample, northern_sample, center='mean')
regions = [eastern_sample, western_sample, southern_sample, northern_sample]
region_names = ['Eastern', 'Western', 'Southern', 'Northern']

# Checking the Welch's t-test p-values

for i in range(len(regions)):
    
    print("Testing Between:{} \t{}".format('All Regions', region_names[i]))
    print("Median Values: \ts1: {} \ts2: {}".format(round(np.median(all_regions_sample), 2), round(np.median(regions[i]), 2)))
    print("Mean Values: \ts1: {} \ts2: {}".format(round(np.mean(all_regions_sample), 2), round(np.mean(regions[i]), 2)))
    print('Sample sizes: \ts1: {} \ts2: {}'.format(len(all_regions_sample), len(regions[i])))
    print("Welch's t-test p-value:", p_value(all_regions_sample, regions[i], two_sided=True))
    print ("-------------------------------------------------------------------------------------")
# performing Welch's t-test between eastern and western

print(p_value(western_sample, eastern_sample, two_sided=True))

# checking the mean values

print('Western:', np.mean(western_sample))
print('Eastern:', np.mean(eastern_sample))
# performing Welch's t-test between eastern and northern

print(p_value(northern_sample, eastern_sample, two_sided=True))

# checking the mean values

print('Northern:', np.mean(northern_sample))
print('Eastern:', np.mean(eastern_sample))
# performing Welch's t-test between eastern and southern

print(p_value(southern_sample, eastern_sample, two_sided=True))

# checking the mean values

print('Southern:', np.mean(southern_sample))
print('Eastern:', np.mean(eastern_sample))
# performing Welch's t-test between western and northern

print(p_value(northern_sample, western_sample, two_sided=True))

# checking the mean values

print('Northern:', np.mean(northern_sample))
print('Western:', np.mean(western_sample))
# performing Welch's t-test between western and southern

print(p_value(southern_sample, western_sample, two_sided=True))

# checking the mean values

print('Southern:', np.mean(southern_sample))
print('Western:', np.mean(western_sample))
# performing Welch's t-test between northern and southern

print(p_value(southern_sample, northern_sample))

# checking the mean values

print('Southern:', np.mean(southern_sample))
print('Northern:', np.mean(northern_sample))
v = np.concatenate([eastern_sample, western_sample, southern_sample, northern_sample])
labels = ['eastern'] * len(eastern_sample) + ['western'] * len(western_sample) + ['southern'] * len(southern_sample) + ['northern'] * len(northern_sample)
print(pairwise_tukeyhsd(v, labels, 0.05))
# Checking what are the shipping companies

df['CompanyName'].unique()

# Dividing the avg_freight data into subsets

all_companies = df['Avg_Freight']
federal = df[df['CompanyName'] == 'Federal Shipping']['Avg_Freight']
speedy = df[df['CompanyName'] == 'Speedy Express']['Avg_Freight']
united = df[df['CompanyName'] == 'United Package']['Avg_Freight']
# Checking if the subset populations add up to the total population

len(all_companies) == len(federal) + len(speedy) + len(united)
companies = [all_companies, federal, speedy, united]
company_names = ['All Company', 'Federal Shipping', 'Speedy Express', 'United Package']

# Checking the distribution plots

for i in range(len(companies)):

    sns.distplot(companies[i], label=company_names[i])
    plt.legend()
    plt.show()
    print(st.normaltest(companies[i]))
    print ("-------------------------------------------------------------------------------------")
# Creating Sampling Distributions

all_companies_sample = create_sample_distribution(all_companies, 1000, 30)
federal_sample = create_sample_distribution(federal, 1000, 30)
speedy_sample = create_sample_distribution(speedy, 1000, 30)
united_sample = create_sample_distribution(united, 1000, 30)
companies = [all_companies_sample, federal_sample, speedy_sample, united_sample]
company_names = ['All Company', 'Federal Shipping', 'Speedy Express', 'United Package']

# Reviewing the distribution plots

for i in range(len(companies)):

    sns.distplot(companies[i], label=company_names[i])
    plt.legend()
    plt.show()
    print(st.normaltest(companies[i]))
    print ("-------------------------------------------------------------------------------------")
# Checking the box plots

for i in range(len(companies)):
    
    print('The boxplot of {} distribution'.format(company_names[i]))
    sns.boxplot(companies[i])
    plt.show()
    print ("-------------------------------------------------------------------------------------")
# Removing outliers from each group

all_companies_sample = remove_outliers(all_companies_sample)
federal_sample = remove_outliers(federal_sample)
speedy_sample = remove_outliers(speedy_sample)
united_sample = remove_outliers(united_sample)

companies = [all_companies_sample, federal_sample, speedy_sample, united_sample]

# Reviewing the distribution plots

for i in range(len(companies)):

    sns.distplot(companies[i], label=company_names[i])
    plt.legend()
    plt.show()
    print(st.normaltest(companies[i]))
    print ("-------------------------------------------------------------------------------------")
# checking the variance of each group

st.levene(all_companies_sample, federal_sample, speedy_sample, united_sample, center='mean')
companies = [federal_sample, speedy_sample, united_sample]
company_names = ['Federal Shipping', 'Speedy Express', 'United Package']

# Checking the Welch's t-test p-values

for i in range(len(companies)):
    
    print("Testing Between:{} \t{}".format('All Companies', company_names[i]))
    print("Median Values: \ts1: {} \ts2: {}".format(round(np.median(all_companies_sample), 2), round(np.median(companies[i]), 2)))
    print("Mean Values: \ts1: {} \ts2: {}".format(round(np.mean(all_companies_sample), 2), round(np.mean(companies[i]), 2)))
    print('Sample sizes: \ts1: {} \ts2: {}'.format(len(all_companies_sample), len(companies[i])))
    print("Welch's t-test p-value:", p_value(all_companies_sample, companies[i], two_sided=True))
    print ("-------------------------------------------------------------------------------------")
# performing Welch's t-test between federal and speedy

print(p_value(speedy_sample, federal_sample, two_sided=True))

# checking the mean values

print('Speedy Express:', np.mean(speedy_sample))
print('Federal Shipping:', np.mean(federal_sample))
# performing Welch's t-test between federal and united

print(p_value(united_sample, federal_sample, two_sided=True))

# checking the mean values

print('United Package:', np.mean(united_sample))
print('Federal Shipping:', np.mean(federal_sample))
# performing Welch's t-test between speedy and united

print(p_value(united_sample, speedy_sample, two_sided=True))

# checking the mean values

print('United Package:', np.mean(united_sample))
print('Speedy Express:', np.mean(speedy_sample))
v = np.concatenate([federal_sample, speedy_sample, united_sample])
labels = ['Federal Shipping'] * len(federal_sample) + ['Speedy Express'] * len(speedy_sample) + ['United Package'] * len(united_sample)
print(pairwise_tukeyhsd(v, labels, 0.05))