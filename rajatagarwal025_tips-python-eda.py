import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df=pd.read_csv('../input/Tips Quick EDA exercise v0.1.csv')

df
#1.What is the overall average tip?



df['tip'].mean()



# 2.99 is the overall avg tip.
#2.Get a numerical summary for 'tip' - are the median and mean very different? What does this tell you about the field?



df['tip'].describe()
df['tip'].median(axis=0)



#Mean and Median are not different, they are exactly the same.
#3.Prepare a boxplot for 'tip', are there any outliers?





df.boxplot(column=['tip'],grid=False)

plt.show()



#yes their are multiple outliers.
#4.Prepare a boxplot for 'total_bill', are there any outliers?





df.boxplot(column=['total_bill'], grid=False)

plt.show()



#yes their are multiple outliers.
#5.Gender: what is the percent of females in the data? 



df['sex'].value_counts(normalize=True)*100



# Female percentage is 35.65.
#6.Prepare a bar plot with the bars representing the percentage of records for each gender.



gender_wise_percentage=df['sex'].value_counts(normalize=True)*100

gender_wise_percentage.plot(kind='bar',color=['b','g'])

plt.title('Bar plot with the bars representing the percentage of records for each gender')

plt.show()

#7.Does the average tip differ by gender? Does one gender tip more than the other?



df.pivot(columns ='sex',values='tip').mean()



# tip does differ on the basis of Gender (male & Female). female Tip is lesser than male.
#8.Does the average tip differ by the time of day?



df.pivot_table(index='day', columns='time', values='tip')





#Yes, tip varies as according to day and time, and highest is for'dinner' on sunday.
#9.Does the average tip differ by size (number of people at the table)? 



df.groupby(['size'])['tip'].mean()



#Yes, tip varies according to the size of the people.
#10.Do smokers tip more than non-smokers?



df.pivot(columns='smoker',values='tip').count()



# No, smoker tip is less than non-smoker .
#11.Gender vs. smoker/non-smoker and tip size - create a 2 by 2 and get the average tip size. Which group tips the most?



df.pivot_table(index='sex', columns='smoker', values='tip')



#Group of non-smoker and male tips the most.
#12.Create a new metric called 'pct_tip' = tip/ total_bill - this would be percent tip give, and should be a better measure of the tipping behaviour.



df['pct_tip']=df.tip/df.total_bill*100

df
#13.Does pct_tip differ by gender? Does one gender tip more than the other?



df.pivot(columns='sex', values='pct_tip').count()



# Male tip is greater than male tip.
#14.Does pct_tip differ by size (number of people at the table)? 



df.pivot(columns='size', values='pct_tip').count()



#Yes, pct_tip differ by size's/no of people, 2nd is the highest pct_tip.

#15.Make the gender vs. smoker view using pct_tip  - does your inference change?



df.pivot_table(index='sex', columns='smoker',values='pct_tip')



# Yes, female (smoker) has highest pct_tip

#16.Make a scatter plot of total_bill vs. tip.



df.plot.scatter('total_bill','tip')

plt.title('scatter plot of total_bill vs. tip')

plt.show()
#17.Make a scatter plot of total_bill vs. pct_tip.



df.plot.scatter('total_bill','pct_tip')

plt.title('Make a scatter plot of total_bill vs. pct_tip')

plt.show()