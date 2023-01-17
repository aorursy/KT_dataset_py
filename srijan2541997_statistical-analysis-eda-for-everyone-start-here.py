import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns 

%matplotlib inline
data = pd.read_csv('../input/Walmart_Store_sales.csv')

display(data.head())

data1 = data.copy()

display(data.tail()) #Quater 4 is incomplete. We have data till october 2012 !
display(data.shape)                      # Shape of data 

display(data.info())                     # Info 

display(data.describe())                 # Description 

display(data.isnull().sum())             # Null values check,found none ! 
max_sales = data.groupby('Store')['Weekly_Sales'].sum()

max_sales.idxmax()
plt.figure(figsize=(20,15))

sns.barplot(x=data.Store, y = data.Weekly_Sales)
std_all = data.groupby('Store')['Weekly_Sales'].std()

#print(std_all,'\n')

print("And Maximum Standard Deviation is for the Store No. {} ".format(std_all.idxmax()))
cv_all = ((data.groupby('Store')['Weekly_Sales'].std())/(data.groupby('Store')['Weekly_Sales'].mean()))*100

#print(cv_all,"\n")

print("And Maximum Coefficient of variation is for the Store No. {} ".format(cv_all.idxmax()))
stores = data.groupby('Store')

store_35 = stores.get_group(35)

plt.figure(figsize=(10,10))

sns.distplot(store_35.Weekly_Sales)
sns.boxplot(store_35.Weekly_Sales)
plt.figure(figsize=(50,15))

sns.lineplot(x='Date', y = 'Weekly_Sales', data = store_35)
plt.figure(figsize=(20,20))

sns.scatterplot(x='Date', y = 'Weekly_Sales', hue = 'Holiday_Flag', data = store_35.head(20), s=250)
growth = data.copy()

growth['Date'] = pd.to_datetime(growth.Date,format='%d-%m-%Y')

growth['Year'], growth['Month'] = growth['Date'].dt.year, growth['Date'].dt.month

growth
# Now lets group data with year = 2012



growth_group = growth.groupby('Year',sort=False)

growth_group_2012 = growth_group.get_group(2012)

growth_group_2012
growth_group_2012_Quaters = growth_group_2012.groupby('Month')

growth_group_2012_Q1_1 = growth_group_2012_Quaters.get_group(1)

growth_group_2012_Q1_2 = growth_group_2012_Quaters.get_group(2)

growth_group_2012_Q1_3 = growth_group_2012_Quaters.get_group(3)



Quater_1 = growth_group_2012_Q1_1.append(growth_group_2012_Q1_2)

Quater_1 = Quater_1.append(growth_group_2012_Q1_3)

display(Quater_1.head())







growth_group_2012_Q2_4 = growth_group_2012_Quaters.get_group(4)

growth_group_2012_Q2_5 = growth_group_2012_Quaters.get_group(5)

growth_group_2012_Q2_6 = growth_group_2012_Quaters.get_group(6)

Quater_2 = growth_group_2012_Q2_4.append(growth_group_2012_Q2_5)

Quater_2 = Quater_2.append(growth_group_2012_Q2_6)

display(Quater_2.head())







growth_group_2012_Q3_7 = growth_group_2012_Quaters.get_group(7)

growth_group_2012_Q3_8 = growth_group_2012_Quaters.get_group(8)

growth_group_2012_Q3_9 = growth_group_2012_Quaters.get_group(9)

Quater_3 = growth_group_2012_Q3_7.append(growth_group_2012_Q3_8)

Quater_3 = Quater_3.append(growth_group_2012_Q3_9)

display(Quater_3.head())





# The last and minimum Quater ! 

growth_group_2012_Q4_10 = growth_group_2012_Quaters.get_group(10)

Quater_4 = growth_group_2012_Q4_10

display(Quater_4.head())



plt.figure(figsize=(30,15))

sns.barplot(x='Store', y = 'Weekly_Sales', data = Quater_1)
df2 = pd.DataFrame(Quater_1.groupby('Store')['Weekly_Sales'].sum())

df2["Quater1_Sales"] = pd.DataFrame(Quater_1.groupby('Store')['Weekly_Sales'].sum())

df2["Quater2_Sales"] = pd.DataFrame(Quater_2.groupby('Store')['Weekly_Sales'].sum())

df2["Quater3_Sales"] = pd.DataFrame(Quater_3.groupby('Store')['Weekly_Sales'].sum())

df2["Quater4_Sales"] = pd.DataFrame(Quater_4.groupby('Store')['Weekly_Sales'].sum())

df2.drop('Weekly_Sales', axis = 1, inplace = True)
df2['Q3 - Q2'] = df2['Quater3_Sales'] - df2['Quater2_Sales']

df2['Overall Growth Rate in 2012 Q3 %'] = (df2['Q3 - Q2']/df2['Quater2_Sales'])*100

df2['Q2 - Q1'] = df2['Quater2_Sales'] - df2['Quater1_Sales']

df2['Overall Growth Rate in 2012 Q2 %'] = (df2['Q2 - Q1']/df2['Quater1_Sales'])*100

df2.head() #Displaying Few Rows..
print("The Store which has good growth in Quater 3 in 2012 is : ")

display(df2['Overall Growth Rate in 2012 Q3 %'].idxmax())

plt.figure(figsize=(20,10))

sns.barplot(x=df2.index, y = 'Overall Growth Rate in 2012 Q3 %', data = df2)
#display(df2['Overall Growth Rate in 2012 Q2 %'])

#print('\n')

print("The Store which has good growth in Quater 2 in 2012 is : ")

display(df2['Overall Growth Rate in 2012 Q2 %'].idxmax())
plt.figure(figsize=(20,10))

sns.barplot(x=df2.index, y = 'Overall Growth Rate in 2012 Q2 %', data = df2)
#Some holidays have a negative impact on sales. Find out holidays which have higher sales than the mean sales in

#non-holiday season for all stores together

#------------------------------------------------------------------------------------------------------------------



data1['Date'] = pd.to_datetime(data1.Date, format = '%d-%m-%Y')

data1['Year'], data1['Month'] = data1['Date'].dt.year, data1['Date'].dt.month

holiday_group = data1.groupby('Holiday_Flag',sort=False)

holiday_week = holiday_group.get_group(1)

display(holiday_week.shape)

display(holiday_week.info())

display(holiday_week.describe())

display(holiday_week.head())
non_holiday_week = holiday_group.get_group(0)

non_holiday_week.head()
plt.figure(figsize=(10,10))

sns.lineplot(x='Date', y = 'Weekly_Sales', data = holiday_week.head(10), sort=False)

sns.lineplot(x='Date', y = 'Weekly_Sales', data = non_holiday_week.head(100), sort = False)



#Holidays.



# Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13 

# Labour Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13 

# Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13 

# Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13
v=holiday_week.groupby(['Month', 'Year'],sort=False)['Weekly_Sales'].mean()

v1 = pd.DataFrame(v)

v1
val = non_holiday_week.groupby(['Month', 'Year'],sort=False)['Weekly_Sales'].mean()

val1 = pd.DataFrame(val)

val1.head()
#MonthWise



groups = growth.groupby('Year')[['Month','Store', 'Weekly_Sales', 'Date']]

group2010 = groups.get_group(2010)

gr = group2010.groupby('Month')

month_2010 = [2,3,4,5,6,7,8,9,10,11,12]

sale_monthWise = []



for i in month_2010:

    val = gr.get_group(i)['Weekly_Sales'].sum()

    sale_monthWise.append(val)

 

month_fallSem  = [7,8,9,10,11,12]

month_springsem = [2,3,4,5,6]

total_spring = sum(sale_monthWise[0:5])

total_fall   = sum(sale_monthWise[5:])

semwise =[total_fall,total_spring]

semval = ['Fall', 'Spring']



plt.figure(figsize=(8,5))

plt.xlabel("Month")

plt.ylabel("Sales_MonthWise in 2010")

plt.title('Sales MonthWise in Year 2010')

sns.barplot(x=month_2010,y=sale_monthWise)

plt.figure()

sns.barplot(x=semval, y = semwise)

plt.xlabel("Month")

plt.ylabel("Sales_MonthWise in 2010")

plt.title('Sales MonthWise in Year 2010')

plt.figure()



#Grouping by month 12 to analyse more which date has more weekly_Sales.

group2010Months = group2010.groupby('Month')[['Weekly_Sales', 'Store', 'Date']]

group2010month12 = group2010Months.get_group(12)

plt.figure(figsize=(20,5))

sns.lineplot(x='Date', y = 'Weekly_Sales', data = group2010month12.head(10))

plt.figure()

plt.figure(figsize=(20,5))

sns.barplot(x='Date', y = 'Weekly_Sales', data = group2010month12)
#MonthWise



groups = growth.groupby('Year')[['Month','Store', 'Weekly_Sales', 'Date']]

group2011 = groups.get_group(2011)

gr = group2011.groupby('Month')

month_2011 = [1,2,3,4,5,6,7,8,9,10,11,12]

sale_monthWise = []



for i in month_2011:

    val = gr.get_group(i)['Weekly_Sales'].sum()

    sale_monthWise.append(val)

 

month_fallSem  = [7,8,9,10,11,12]

month_springsem = [1,2,3,4,5,6]

total_spring = sum(sale_monthWise[0:5])

total_fall   = sum(sale_monthWise[5:])

semwise =[total_fall,total_spring]

semval = ['Fall', 'Spring']



plt.figure(figsize=(8,5))

plt.xlabel("Month")

plt.ylabel("Sales_MonthWise in 2011")

plt.title('Sales MonthWise in Year 2011')

sns.barplot(x=month_2011,y=sale_monthWise)

plt.figure()

sns.barplot(x=semval, y = semwise)

plt.xlabel("Month")

plt.ylabel("Sales_MonthWise in 2011")

plt.title('Sales MonthWise in Year 2011')

plt.figure()



#Grouping by month 12 to analyse more which date has more weekly_Sales.

group2011Months = group2011.groupby('Month')[['Weekly_Sales', 'Store', 'Date']]

group2011month12 = group2011Months.get_group(12)

plt.figure(figsize=(20,5))

sns.lineplot(x='Date', y = 'Weekly_Sales', data = group2011month12.head(10))

plt.figure()

plt.figure(figsize=(20,5))

sns.barplot(x='Date', y = 'Weekly_Sales', data = group2011month12)



#MonthWise



groups = growth.groupby('Year')[['Month','Store', 'Weekly_Sales', 'Date']]

group2012 = groups.get_group(2012)

gr = group2012.groupby('Month')

month_2012 = [1,2,3,4,5,6,7,8,9,10]

sale_monthWise = []



for i in month_2012:

    val = gr.get_group(i)['Weekly_Sales'].sum()

    sale_monthWise.append(val)

 

month_fallSem  = [7,8,9,10]

month_springsem = [1,2,3,4,5,6]

total_spring = sum(sale_monthWise[0:5])

total_fall   = sum(sale_monthWise[5:])

semwise =[total_fall,total_spring]

semval = ['Fall', 'Spring']



plt.figure(figsize=(8,5))

plt.xlabel("Month")

plt.ylabel("Sales_MonthWise in 2012")

plt.title('Sales MonthWise in Year 2012')

sns.barplot(x=month_2012,y=sale_monthWise)

plt.figure()

sns.barplot(x=semval, y = semwise)

plt.xlabel("Month")

plt.ylabel("Sales_MonthWise in 2012")

plt.title('Sales MonthWise in Year 2012')



#Grouping by month 7 to analyse more which date has more weekly_Sales.

group2012Months = group2012.groupby('Month')[['Weekly_Sales', 'Store', 'Date']]

group2012month7 = group2012Months.get_group(7)

plt.figure(figsize=(20,5))

sns.lineplot(x='Store', y = 'Weekly_Sales', data = group2012month7)

plt.figure()

plt.figure(figsize=(20,5))

sns.barplot(x='Store', y = 'Weekly_Sales', data = group2012month7)

#let's Group the data.



hypothesis = growth.groupby('Store')[['Fuel_Price','Unemployment', 'CPI','Weekly_Sales', 'Holiday_Flag']]

factors  = hypothesis.get_group(1)

day_arr = [1]

for i in range (1,len(factors)):

    day_arr.append(i*7)

    

factors['Day'] = day_arr.copy()
factors
sns.heatmap(factors.corr(), annot = True)
sns.lmplot(x='Fuel_Price', y = 'Unemployment', data = factors)

plt.figure()

sns.lmplot(x='CPI', y = 'Unemployment', data = factors)
from scipy import stats

ttest,pval = stats.ttest_rel(factors['Weekly_Sales'],factors['CPI'])

sns.distplot(factors.CPI)

plt.figure()

print(pval)

if pval<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")

    

sns.scatterplot(x='CPI', y = 'Weekly_Sales', data = factors, hue = 'Holiday_Flag')

plt.figure()

sns.lmplot(x='CPI', y = 'Weekly_Sales', data = factors, hue = 'Holiday_Flag')

plt.figure()

sns.lineplot(x='CPI', y = 'Weekly_Sales', data = factors)
from scipy import stats

ttest,pval = stats.ttest_rel(factors['Weekly_Sales'],factors['Fuel_Price'])

sns.distplot(factors.Fuel_Price)

plt.figure()

print(pval)

if pval<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")

    

sns.scatterplot(x='Fuel_Price', y = 'Weekly_Sales', data = factors, hue = 'Holiday_Flag')

plt.figure()

sns.lmplot(x='Fuel_Price', y = 'Weekly_Sales', data = factors, hue = 'Holiday_Flag')

plt.figure()

sns.lineplot(x='Fuel_Price', y = 'Weekly_Sales', data = factors)
from scipy import stats

ttest,pval = stats.ttest_rel(factors['Weekly_Sales'],factors['Unemployment'])

sns.distplot(factors.Unemployment)

plt.figure()

print(pval)

if pval<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")

    

sns.scatterplot(x='Unemployment', y = 'Weekly_Sales', data = factors, hue = 'Holiday_Flag')

plt.figure()

sns.lmplot(x='Unemployment', y = 'Weekly_Sales', data = factors, hue = 'Holiday_Flag')

plt.figure()

sns.lineplot(x='Unemployment', y = 'Weekly_Sales', data = factors)
plt.figure(figsize=(30,10))

sns.barplot(x='Day', y = 'Weekly_Sales', data = factors.head(50), hue = 'Holiday_Flag')