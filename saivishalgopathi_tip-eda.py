# importing library for EDA

import numpy as np

import pandas as pd

# importing library for dala visualization

import matplotlib.pyplot as plt
df_tips = pd.read_excel('../input/tipsdataset/Tips_dataset.xlsx',sheet_name = 'tips') #importing the file and 'tips' sheet from the excel file
df_tips
df_tips.head(10).T # to view the first 10 rows of the set and .T is to veiw it in a transposed view
# checking the aggregation values of all the numeric columns in the data set

df_tips.describe() # we can write include = 'all' within the parentheses 
# here we are using a function of numpy to see the average value

overall_average_tip = df_tips['tip'].mean()
print("${r:1.2f} is the overall average tip by the customers".format(r=overall_average_tip))
tip_mean = df_tips['tip'].mean()

tip_median = df_tips['tip'].median()

print('mean = ',tip_mean)

print('meadian = ',tip_median)

# using matplotlib to create graphs

df_tips.boxplot('tip')

plt.show()
df_tips.boxplot('total_bill')

plt.show()
gender=df_tips.groupby('sex').sex.count()

gender
x=gender.index.tolist()

y=gender.values.tolist()
plt.pie(y,labels=x,autopct='%1.f%%')

plt.show()
# gender=(gender/gender.sum())*100

gender=gender/gender.sum()

gender
x=gender.index.tolist()

y=gender.values.tolist()
plt.bar(x,y,color=['r','b'])

plt.show()
averageTipByGender = df_tips.groupby('sex').tip.mean()

print(averageTipByGender)

plt.plot(averageTipByGender.index,averageTipByGender.values,'r^--')

plt.show()
averageTipByTime = df_tips.groupby('time').tip.mean()

print(averageTipByTime)

plt.bar(averageTipByTime.index,averageTipByTime.values,color=['r','k'])

plt.show()
averageTipBySize = df_tips.groupby('size').tip.mean()

print(averageTipBySize)

plt.plot(averageTipBySize.index,averageTipBySize.values,'ro--')

plt.show()
smokerBasedTipTotal = df_tips.groupby('smoker').tip.sum()

print("total",smokerBasedTipTotal)

x1 = smokerBasedTipTotal.index.tolist()

y1 = smokerBasedTipTotal.values.tolist()

print("-------------------------------")

plt.bar(x1,y1,color = ['g','r'])

plt.xlabel('Smoker')

plt.ylabel('Total TiP')

plt.title("Total tip based on smoker criteria")

plt.show()
smokerBasedTipAvg = df_tips.groupby('smoker').tip.mean()

print("Average",smokerBasedTipAvg)

x2 = smokerBasedTipAvg.index.tolist()

y2 = smokerBasedTipAvg.values.tolist()

print('-------------------------------------')

plt.bar(x2,y2,color = ['g','r'])

plt.xlabel('smoker')

plt.ylabel("average tip")

plt.title('Average tip based on smoking criteria')

plt.show()
genderSmokertip = df_tips.groupby(['sex','smoker']).tip.mean()

genderSmokertip
df_tips['pct_tip'] = df_tips.tip/df_tips.total_bill
df_tips.head(10).T
pctGender = df_tips.groupby('sex').pct_tip.sum()

print(pctGender)

plt.bar(pctGender.index,pctGender.values,color=['r','k'])

plt.show()
pctSize = df_tips.groupby('size').pct_tip.max()

print(pctSize)

plt.plot(pctSize.index,pctSize.values,'o--')

plt.show()
df_tips['pct_tip'].describe()
genderSmokerPct = df_tips.groupby(['sex','smoker']).pct_tip.mean()

genderSmokerPct
plt.scatter(df_tips['total_bill'],df_tips['tip'])

plt.xlabel("Total_Bill")

plt.ylabel("Tip")

plt.show()
plt.scatter(df_tips['total_bill'],df_tips['pct_tip'])

plt.xlabel('Total_Bill')

plt.ylabel('Pct_Tip')

plt.show()